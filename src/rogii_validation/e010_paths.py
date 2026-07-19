"""Path primitives for E010 nonlinear candidate coverage.

Candidate generation uses only test-available inputs or frozen cross-fitted
parent paths. Hidden TVT is consumed only by the retrospective grid-oracle
scorer and never by HMM, DTW, or selector inference.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .gr_path import WellData, _calibration
from .harness import DataValidationError


@dataclass(frozen=True)
class GridOracle:
    family: str
    coefficients: tuple[float, ...]
    sse: float
    rmse: float
    candidate_index: int


@dataclass(frozen=True)
class StatePathResult:
    path: np.ndarray
    fallback_reason: str
    sampled_observations: int
    wall_seconds: float
    minimum_normalizer: float


def _finite(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"E010 invalid numeric {name}") from exc
    if not math.isfinite(parsed):
        raise DataValidationError(f"E010 non-finite numeric {name}")
    return parsed


def grid_values(spec: Mapping[str, Any]) -> np.ndarray:
    lower = _finite(spec.get("minimum"), "grid minimum")
    upper = _finite(spec.get("maximum"), "grid maximum")
    step = _finite(spec.get("step"), "grid step")
    if step <= 0.0 or upper < lower:
        raise DataValidationError("E010 malformed coefficient grid")
    values = np.arange(lower, upper + 0.5 * step, step, dtype=np.float64)
    if values.size == 0 or abs(values[0] - lower) > 1e-9 or abs(values[-1] - upper) > 1e-9:
        raise DataValidationError("E010 coefficient grid endpoints are not exact")
    return values


def basis_matrix(rows: int, family: Mapping[str, Any]) -> np.ndarray:
    if rows <= 0:
        raise DataValidationError("E010 basis requires positive row count")
    t = np.linspace(0.0, 1.0, rows, dtype=np.float64) if rows > 1 else np.zeros(1, dtype=np.float64)
    linear = 2.0 * t - 1.0 if rows > 1 else np.zeros(1, dtype=np.float64)
    kind = str(family.get("basis", ""))
    if kind == "affine":
        return np.column_stack((np.ones(rows), linear))
    if kind == "quadratic":
        shape = linear * linear
    elif kind == "hinge":
        location = _finite(family.get("location"), "hinge location")
        if not 0.0 <= location <= 1.0:
            raise DataValidationError("E010 hinge location must be in [0,1]")
        shape = np.maximum(0.0, t - location)
    elif kind == "jump":
        location = _finite(family.get("location"), "jump location")
        width = _finite(family.get("width"), "jump width")
        if not 0.0 <= location <= 1.0 or width <= 0.0:
            raise DataValidationError("E010 malformed jump basis")
        shape = np.tanh((t - location) / width)
    else:
        raise DataValidationError(f"E010 unknown basis {kind!r}")
    shape = shape - float(shape.mean())
    return np.column_stack((np.ones(rows), linear, shape))


def coefficient_grid(family: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
    grids = config["candidate_bank"]["coefficient_grids"]
    datum = grid_values(grids["datum_ft"])
    linear = grid_values(grids["linear_ft"])
    if str(family["basis"]) == "affine":
        mesh = np.meshgrid(datum, linear, indexing="ij")
    else:
        shape = grid_values(grids["shape_ft"])
        mesh = np.meshgrid(datum, linear, shape, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, len(mesh)).astype(np.float64, copy=False)


def valid_coefficients(family: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
    coefficients = coefficient_grid(family, config)
    reference_basis = basis_matrix(1001, family)
    maximum = _finite(config["candidate_bank"]["maximum_absolute_correction_ft"], "maximum correction")
    keep = np.zeros(coefficients.shape[0], dtype=bool)
    for start in range(0, coefficients.shape[0], 2048):
        block = coefficients[start : start + 2048]
        keep[start : start + len(block)] = np.max(np.abs(reference_basis @ block.T), axis=0) <= maximum + 1e-9
    output = coefficients[keep]
    if output.size == 0:
        raise DataValidationError(f"E010 family {family['name']} has no bounded candidates")
    return output


def candidate_count(config: Mapping[str, Any]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = len(config["candidate_bank"]["raw_anchors"])
    for family in config["candidate_bank"]["families"]:
        count = int(valid_coefficients(family, config).shape[0])
        counts[str(family["name"])] = count
        total += count
    return total, counts


def reconstruct_path(anchor: Sequence[float], family: Mapping[str, Any], coefficients: Sequence[float], maximum: float) -> np.ndarray:
    base = np.asarray(anchor, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64)
    matrix = basis_matrix(base.size, family)
    if coef.shape != (matrix.shape[1],) or not np.all(np.isfinite(base)) or not np.all(np.isfinite(coef)):
        raise DataValidationError("E010 malformed path reconstruction")
    correction = np.clip(matrix @ coef, -float(maximum), float(maximum))
    output = base + correction
    if not np.all(np.isfinite(output)):
        raise DataValidationError("E010 path reconstruction emitted non-finite values")
    return output


def grid_oracle(anchor: Sequence[float], truth: Sequence[float], family: Mapping[str, Any], coefficients: np.ndarray) -> GridOracle:
    base = np.asarray(anchor, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if base.ndim != 1 or target.shape != base.shape or base.size == 0:
        raise DataValidationError("E010 grid oracle path contract failed")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(target)):
        raise DataValidationError("E010 grid oracle received non-finite values")
    matrix = basis_matrix(base.size, family)
    if coefficients.ndim != 2 or coefficients.shape[1] != matrix.shape[1] or coefficients.shape[0] == 0:
        raise DataValidationError("E010 grid oracle coefficient contract failed")
    residual = base - target
    gram = matrix.T @ matrix
    linear = matrix.T @ residual
    constant = float(residual @ residual)
    best_sse = math.inf
    best_index = -1
    for start in range(0, coefficients.shape[0], 4096):
        block = coefficients[start : start + 4096]
        values = constant + 2.0 * (block @ linear) + np.einsum("ij,jk,ik->i", block, gram, block, optimize=True)
        local = int(np.argmin(values))
        value = float(values[local])
        absolute = start + local
        if value < best_sse - 1e-9 or (abs(value - best_sse) <= 1e-9 and (best_index < 0 or absolute < best_index)):
            best_sse = value
            best_index = absolute
    if best_index < 0 or not math.isfinite(best_sse):
        raise DataValidationError("E010 grid oracle failed to find a finite candidate")
    best_sse = max(0.0, best_sse)
    return GridOracle(str(family["name"]), tuple(float(v) for v in coefficients[best_index]), best_sse, math.sqrt(best_sse / base.size), best_index)


def _sample_indices(well: WellData, maximum: int) -> np.ndarray:
    finite = np.asarray([i for i in range(well.known_rows, len(well.md)) if well.gr[i] is not None], dtype=np.int64)
    if finite.size <= maximum:
        return finite
    positions = np.rint(np.linspace(0, finite.size - 1, maximum)).astype(np.int64)
    return np.unique(finite[positions])


def _calibrated_arrays(well: WellData, absolute_indices: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if absolute_indices.size < 8:
        return np.empty(0), np.empty(0), np.empty(0), "too_few_hidden_gr_samples"
    calibration = _calibration(well, well.typewell, 24)
    if calibration is None:
        return np.empty(0), np.empty(0), np.empty(0), "invalid_visible_calibration"
    hidden_indices = absolute_indices - well.known_rows
    horizontal = np.asarray([float(well.gr[int(i)]) for i in absolute_indices], dtype=np.float64)
    horizontal = float(calibration["sign"]) * (horizontal - float(calibration["h_mean"])) / max(float(calibration["h_std"]), 1e-9)
    sampled_base = base[hidden_indices]
    md = np.asarray([well.md[int(i)] for i in absolute_indices], dtype=np.float64)
    if not np.all(np.isfinite(horizontal)) or not np.all(np.isfinite(sampled_base)) or not np.all(np.isfinite(md)):
        return np.empty(0), np.empty(0), np.empty(0), "nonfinite_calibration"
    return horizontal, sampled_base, md, ""


def _reference_z(well: WellData, tvt: np.ndarray, calibration: Mapping[str, float]) -> np.ndarray:
    values = np.interp(tvt, np.asarray(well.typewell.tvt), np.asarray(well.typewell.gr), left=np.nan, right=np.nan)
    return (values - float(calibration["r_mean"])) / max(float(calibration["r_std"]), 1e-9)


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0.25:
        return np.asarray([1.0])
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return kernel / kernel.sum()


def _blur(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.size == 1:
        return values
    pad = kernel.size // 2
    return np.convolve(np.pad(values, (pad, pad), mode="edge"), kernel, mode="valid")


def _shift_forward(values: np.ndarray, shift_bins: float, kernel: np.ndarray) -> np.ndarray:
    size = values.size
    destination = np.arange(size, dtype=np.float64) + shift_bins
    left = np.floor(destination).astype(np.int64)
    fraction = destination - left
    output = np.zeros(size, dtype=np.float64)
    np.add.at(output, np.clip(left, 0, size - 1), values * (1.0 - fraction))
    np.add.at(output, np.clip(left + 1, 0, size - 1), values * fraction)
    return _blur(output, kernel)


def _shift_backward(values: np.ndarray, shift_bins: float, kernel: np.ndarray) -> np.ndarray:
    blurred = _blur(values, kernel)
    positions = np.arange(values.size, dtype=np.float64) + shift_bins
    return np.interp(positions, np.arange(values.size), blurred, left=blurred[0], right=blurred[-1])


def hmm_path(well: WellData, base_path: Sequence[float], variant: Mapping[str, Any], maximum_correction: float) -> StatePathResult:
    """Run deterministic forward-backward inference over offset and residual rate."""
    started = time.perf_counter()
    base = np.asarray(base_path, dtype=np.float64)
    if base.shape != (well.hidden_rows,) or not np.all(np.isfinite(base)):
        raise DataValidationError("E010 HMM base path contract failed")
    sampled = _sample_indices(well, int(variant["maximum_observations"]))
    horizontal, sampled_base, sampled_md, reason = _calibrated_arrays(well, sampled, base)
    calibration = _calibration(well, well.typewell, 24)
    if reason or calibration is None:
        return StatePathResult(base.copy(), reason or "invalid_visible_calibration", int(sampled.size), time.perf_counter() - started, 0.0)
    step = _finite(variant["grid_step_ft"], "HMM grid step")
    span = _finite(variant["position_span_ft"], "HMM position span")
    rate_states = int(variant["rate_states"])
    rate_span = _finite(variant["rate_span"], "HMM rate span")
    if step <= 0.0 or span <= 0.0 or rate_states < 3 or rate_span <= 0.0:
        raise DataValidationError("E010 HMM grid is degenerate")
    offsets = np.arange(-span, span + 0.5 * step, step, dtype=np.float64)
    center = 0.0
    if str(variant.get("rate_center", "zero")) == "visible_tail" and well.known_rows >= 3:
        start = max(0, well.known_rows - 32)
        visible_md = np.asarray(well.md[start : well.known_rows], dtype=np.float64)
        visible_tvt = np.asarray([float(v) for v in well.tvt_input[start : well.known_rows]], dtype=np.float64)
        hidden_md = np.asarray(well.md[well.known_rows :], dtype=np.float64)
        if visible_md[-1] > visible_md[0] and hidden_md.size > 1 and hidden_md[-1] > hidden_md[0]:
            visible_slope = float(np.polyfit(visible_md, visible_tvt, 1)[0])
            anchor_slope = float(np.polyfit(hidden_md, base, 1)[0])
            center = float(np.clip(visible_slope - anchor_slope, -rate_span, rate_span))
    rates = np.linspace(center - rate_span, center + rate_span, rate_states, dtype=np.float64)
    rate_sigma = _finite(variant["rate_sigma"], "HMM rate sigma")
    momentum = _finite(variant["momentum"], "HMM momentum")
    if rate_sigma <= 0.0 or not 0.0 <= momentum <= 1.0:
        raise DataValidationError("E010 HMM transition parameters are invalid")
    transition = np.empty((rate_states, rate_states), dtype=np.float64)
    for index, value in enumerate(rates):
        transition[index] = np.exp(-0.5 * ((rates - momentum * value) / rate_sigma) ** 2)
        total = float(transition[index].sum())
        transition[index] = transition[index] / total if total > 0.0 else np.eye(rate_states)[index]
    position_kernel = _gaussian_kernel(_finite(variant["position_sigma_ft"], "HMM position sigma") / step)
    emission_weight = _finite(variant["emission_weight"], "HMM emission weight")
    emissions = np.empty((sampled.size, offsets.size), dtype=np.float64)
    for index in range(sampled.size):
        reference = _reference_z(well, sampled_base[index] + offsets, calibration)
        difference = horizontal[index] - reference
        if str(variant["emission"]) == "student_t":
            degrees = _finite(variant.get("degrees_freedom", 4.0), "HMM degrees of freedom")
            cost = 0.5 * (degrees + 1.0) * np.log1p((difference * difference) / degrees)
        elif str(variant["emission"]) == "gaussian":
            cost = 0.5 * difference * difference
        else:
            raise DataValidationError("E010 HMM emission type is unknown")
        cost[~np.isfinite(cost)] = 50.0
        emissions[index] = np.maximum(np.exp(-emission_weight * np.minimum(cost, 100.0)), 1e-300)
    initial_position = np.exp(-0.5 * (offsets / max(step, 2.0)) ** 2)
    initial_rate = np.exp(-0.5 * ((rates - center) / max(rate_sigma * 4.0, rate_span / 4.0)) ** 2)
    alpha = initial_position[:, None] * initial_rate[None, :] * emissions[0, :, None]
    normalizer = float(alpha.sum())
    if normalizer <= 0.0 or not math.isfinite(normalizer):
        return StatePathResult(base.copy(), "initial_underflow", int(sampled.size), time.perf_counter() - started, 0.0)
    alpha /= normalizer
    minimum_normalizer = normalizer
    forward = np.empty((sampled.size, offsets.size, rate_states), dtype=np.float32)
    forward[0] = alpha.astype(np.float32)
    for index in range(1, sampled.size):
        dt = max(0.0, float(sampled_md[index] - sampled_md[index - 1]))
        mixed = alpha @ transition
        next_alpha = np.empty_like(alpha)
        for rate_index, rate in enumerate(rates):
            next_alpha[:, rate_index] = _shift_forward(mixed[:, rate_index], rate * dt / step, position_kernel)
        next_alpha *= emissions[index, :, None]
        normalizer = float(next_alpha.sum())
        minimum_normalizer = min(minimum_normalizer, normalizer)
        if normalizer <= 0.0 or not math.isfinite(normalizer):
            return StatePathResult(base.copy(), "forward_underflow", int(sampled.size), time.perf_counter() - started, minimum_normalizer)
        alpha = next_alpha / normalizer
        forward[index] = alpha.astype(np.float32)
    beta = np.ones((offsets.size, rate_states), dtype=np.float64)
    posterior_offsets = np.empty(sampled.size, dtype=np.float64)
    posterior = np.asarray(forward[-1], dtype=np.float64) * beta
    posterior /= max(float(posterior.sum()), 1e-300)
    posterior_offsets[-1] = float((posterior.sum(axis=1) * offsets).sum())
    for index in range(sampled.size - 2, -1, -1):
        dt = max(0.0, float(sampled_md[index + 1] - sampled_md[index]))
        weighted_next = emissions[index + 1, :, None] * beta
        position_back = np.empty_like(weighted_next)
        for rate_index, rate in enumerate(rates):
            position_back[:, rate_index] = _shift_backward(weighted_next[:, rate_index], rate * dt / step, position_kernel)
        beta = position_back @ transition.T
        scale = float(beta.max())
        if scale <= 0.0 or not math.isfinite(scale):
            return StatePathResult(base.copy(), "backward_underflow", int(sampled.size), time.perf_counter() - started, minimum_normalizer)
        beta /= scale
        posterior = np.asarray(forward[index], dtype=np.float64) * beta
        posterior /= max(float(posterior.sum()), 1e-300)
        posterior_offsets[index] = float((posterior.sum(axis=1) * offsets).sum())
    hidden_indices = sampled - well.known_rows
    correction = np.interp(np.arange(well.hidden_rows), hidden_indices, posterior_offsets, left=posterior_offsets[0], right=posterior_offsets[-1])
    correction = np.clip(correction, -float(maximum_correction), float(maximum_correction))
    output = base + correction
    if not np.all(np.isfinite(output)):
        raise DataValidationError("E010 HMM emitted non-finite path")
    return StatePathResult(output, "", int(sampled.size), time.perf_counter() - started, minimum_normalizer)


def _filled_hidden_gr(well: WellData) -> np.ndarray:
    values = np.asarray([np.nan if value is None else float(value) for value in well.gr[well.known_rows :]], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return values
    if finite.size == 1:
        values[:] = values[finite[0]]
        return values
    missing = np.flatnonzero(~np.isfinite(values))
    values[missing] = np.interp(missing, finite, values[finite])
    return values


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64) / window
    left = window // 2
    right = window - 1 - left
    return np.convolve(np.pad(values, (left, right), mode="edge"), kernel, mode="valid")


def dtw_path(well: WellData, base_path: Sequence[float], variant: Mapping[str, Any], maximum_correction: float) -> StatePathResult:
    """Run constrained offset-state dynamic programming around a legal anchor."""
    started = time.perf_counter()
    base = np.asarray(base_path, dtype=np.float64)
    if base.shape != (well.hidden_rows,) or not np.all(np.isfinite(base)):
        raise DataValidationError("E010 DTW base path contract failed")
    filled = _filled_hidden_gr(well)
    if filled.size == 0 or not np.any(np.isfinite(filled)):
        return StatePathResult(base.copy(), "all_hidden_gr_missing", 0, time.perf_counter() - started, 0.0)
    calibration = _calibration(well, well.typewell, 24)
    if calibration is None:
        return StatePathResult(base.copy(), "invalid_visible_calibration", 0, time.perf_counter() - started, 0.0)
    smoothed = _rolling_mean(filled, int(variant["smoothing_rows"]))
    maximum_observations = int(variant["maximum_observations"])
    if smoothed.size <= maximum_observations:
        hidden_indices = np.arange(smoothed.size, dtype=np.int64)
    else:
        hidden_indices = np.unique(np.rint(np.linspace(0, smoothed.size - 1, maximum_observations)).astype(np.int64))
    horizontal = float(calibration["sign"]) * (smoothed[hidden_indices] - float(calibration["h_mean"])) / max(float(calibration["h_std"]), 1e-9)
    lower = _finite(variant["offset_min_ft"], "DTW offset minimum")
    upper = _finite(variant["offset_max_ft"], "DTW offset maximum")
    step = _finite(variant["offset_step_ft"], "DTW offset step")
    transition_steps = int(variant["transition_steps"])
    if step <= 0.0 or upper < lower or transition_steps < 0:
        raise DataValidationError("E010 DTW state grid is malformed")
    offsets = np.arange(lower, upper + 0.5 * step, step, dtype=np.float64)
    references = np.empty((hidden_indices.size, offsets.size), dtype=np.float64)
    for index, hidden_index in enumerate(hidden_indices):
        references[index] = _reference_z(well, base[hidden_index] + offsets, calibration)
    difference = horizontal[:, None] - references
    emission = np.minimum(difference * difference, 100.0)
    emission[~np.isfinite(emission)] = 100.0
    derivative_weight = _finite(variant["derivative_weight"], "DTW derivative weight")
    if derivative_weight > 0.0 and hidden_indices.size > 1:
        horizontal_derivative = np.diff(horizontal, prepend=horizontal[0])
        reference_derivative = np.diff(references, axis=0, prepend=references[[0]])
        derivative_cost = np.minimum((horizontal_derivative[:, None] - reference_derivative) ** 2, 100.0)
        emission = (1.0 - derivative_weight) * emission + derivative_weight * derivative_cost
    transition_penalty = _finite(variant["transition_penalty"], "DTW transition penalty")
    state_count = offsets.size
    costs = emission[0] + 0.0025 * offsets * offsets
    back = np.empty((hidden_indices.size, state_count), dtype=np.int32)
    back[0] = np.arange(state_count, dtype=np.int32)
    for index in range(1, hidden_indices.size):
        next_cost = np.empty(state_count, dtype=np.float64)
        next_back = np.empty(state_count, dtype=np.int32)
        for state in range(state_count):
            start_state = max(0, state - transition_steps)
            end_state = min(state_count, state + transition_steps + 1)
            previous = np.arange(start_state, end_state)
            candidates = costs[start_state:end_state] + transition_penalty * (previous - state) ** 2
            local = int(np.argmin(candidates))
            next_back[state] = start_state + local
            next_cost[state] = float(candidates[local]) + emission[index, state]
        costs = next_cost
        back[index] = next_back
    states = np.empty(hidden_indices.size, dtype=np.int32)
    states[-1] = int(np.argmin(costs))
    for index in range(hidden_indices.size - 1, 0, -1):
        states[index - 1] = back[index, states[index]]
    sampled_offsets = offsets[states]
    correction = np.interp(np.arange(well.hidden_rows), hidden_indices, sampled_offsets, left=sampled_offsets[0], right=sampled_offsets[-1])
    correction = np.clip(correction, -float(maximum_correction), float(maximum_correction))
    output = base + correction
    if not np.all(np.isfinite(output)):
        raise DataValidationError("E010 DTW emitted non-finite path")
    return StatePathResult(output, "", int(hidden_indices.size), time.perf_counter() - started, float(np.min(costs)))
