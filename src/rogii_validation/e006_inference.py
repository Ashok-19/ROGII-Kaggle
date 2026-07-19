"""Pure-standard-library E006 PF-E004 fusion inference.

This module depends only on the names exported by ``e004_inference`` and the
Python standard library. The notebook builder removes the relative import and
executes the verified E004 runtime first in the same namespace.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .e004_inference import (  # removed by the notebook builder
    DeploymentDataError,
    SUBMISSION_DECIMAL_PLACES,
    extract_well_features,
    optional_float,
    predict_coefficients,
    read_horizontal,
    reconstruct_hidden,
)


@dataclass(frozen=True)
class E006TypewellCurve:
    tvt: tuple[float, ...]
    gr: tuple[float, ...]
    minimum_tvt: float
    maximum_tvt: float
    gr_mean: float
    gr_std: float

    @classmethod
    def read(cls, path: Path) -> "E006TypewellCurve":
        if not path.exists():
            raise DeploymentDataError(f"missing typewell {path}")
        pairs: list[tuple[float, float]] = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = {"TVT", "GR"} - set(reader.fieldnames or [])
            if missing:
                raise DeploymentDataError(f"{path}: missing typewell columns {sorted(missing)}")
            for row in reader:
                tvt = optional_float(row.get("TVT"))
                gr = optional_float(row.get("GR"))
                if tvt is not None and gr is not None:
                    pairs.append((tvt, gr))
        if len(pairs) < 2:
            raise DeploymentDataError(f"{path}: fewer than two finite TVT/GR pairs")
        pairs.sort()
        tvt_values: list[float] = []
        gr_values: list[float] = []
        start = 0
        while start < len(pairs):
            end = start + 1
            while end < len(pairs) and pairs[end][0] == pairs[start][0]:
                end += 1
            tvt_values.append(pairs[start][0])
            gr_values.append(sum(value for _, value in pairs[start:end]) / (end - start))
            start = end
        if len(tvt_values) < 2 or tvt_values[-1] <= tvt_values[0]:
            raise DeploymentDataError(f"{path}: typewell TVT support is degenerate")
        center = sum(gr_values) / len(gr_values)
        scale = math.sqrt(sum((value - center) ** 2 for value in gr_values) / len(gr_values))
        return cls(tuple(tvt_values), tuple(gr_values), tvt_values[0], tvt_values[-1], center, scale)

    def value(self, tvt: float) -> float | None:
        if not math.isfinite(tvt) or tvt < self.minimum_tvt or tvt > self.maximum_tvt:
            return None
        position = bisect.bisect_left(self.tvt, tvt)
        if position <= 0:
            return self.gr[0]
        if position >= len(self.tvt):
            return self.gr[-1]
        left = self.tvt[position - 1]
        right = self.tvt[position]
        if right <= left:
            return self.gr[position]
        fraction = (tvt - left) / (right - left)
        return self.gr[position - 1] * (1.0 - fraction) + self.gr[position] * fraction


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _sample_indices(indices: Sequence[int], maximum: int) -> list[int]:
    if len(indices) <= maximum:
        return list(indices)
    if maximum <= 1:
        return [indices[-1]]
    return sorted({indices[round(step * (len(indices) - 1) / (maximum - 1))] for step in range(maximum)})


def _huber(value: float, delta: float) -> float:
    absolute = abs(value)
    return 0.5 * absolute * absolute if absolute <= delta else delta * (absolute - 0.5 * delta)


def _calibration(
    gr: Sequence[float | None],
    tvt_input: Sequence[float | None],
    known_rows: int,
    curve: E006TypewellCurve,
    minimum: int,
) -> dict[str, float] | None:
    horizontal: list[float] = []
    reference: list[float] = []
    for index in range(known_rows):
        h = gr[index]
        tvt = tvt_input[index]
        if h is None or tvt is None:
            continue
        r = curve.value(float(tvt))
        if r is None:
            continue
        horizontal.append(float(h))
        reference.append(float(r))
    if len(horizontal) < minimum:
        return None
    h_mean, r_mean = _mean(horizontal), _mean(reference)
    h_std, r_std = _std(horizontal), _std(reference)
    if h_std <= 1e-9 or r_std <= 1e-9:
        return None
    covariance = sum((h - h_mean) * (r - r_mean) for h, r in zip(horizontal, reference))
    return {
        "h_mean": h_mean,
        "h_std": h_std,
        "r_mean": r_mean,
        "r_std": r_std,
        "sign": 1.0 if covariance >= 0.0 else -1.0,
    }


def _normalize(horizontal: float, reference: float, calibration: Mapping[str, float], clip_z: float) -> tuple[float, float]:
    left = float(calibration["sign"]) * (horizontal - float(calibration["h_mean"])) / float(calibration["h_std"])
    right = (reference - float(calibration["r_mean"])) / float(calibration["r_std"])
    return _clip(left, -clip_z, clip_z), _clip(right, -clip_z, clip_z)


def _path_loss(
    gr: Sequence[float | None],
    known_rows: int,
    hidden_rows: int,
    base: Sequence[float],
    curve: E006TypewellCurve,
    samples: Sequence[int],
    datum: float,
    toe: float,
    calibration: Mapping[str, float],
    alignment: Mapping[str, Any],
) -> float:
    clip_z = float(alignment["gr_clip_z"])
    delta = float(alignment["huber_delta_z"])
    total = 0.0
    usable = 0
    for absolute in samples:
        horizontal = gr[absolute]
        if horizontal is None:
            continue
        hidden_index = absolute - known_rows
        fraction = hidden_index / max(1, hidden_rows - 1)
        reference = curve.value(float(base[hidden_index]) + datum + toe * fraction)
        if reference is None:
            total += _huber(clip_z, delta)
            usable += 1
            continue
        left, right = _normalize(float(horizontal), float(reference), calibration, clip_z)
        total += _huber(left - right, delta)
        usable += 1
    if usable == 0:
        return math.inf
    prior = float(alignment["prior_weight"]) * (
        (datum / float(alignment["datum_prior_scale_ft"])) ** 2
        + (toe / float(alignment["toe_prior_scale_ft"])) ** 2
    )
    return total / usable + prior


def _apply_affine(base: Sequence[float], datum: float, toe: float, curve: E006TypewellCurve, alignment: Mapping[str, Any]) -> list[float]:
    maximum = float(alignment["maximum_absolute_correction_ft"])
    margin = float(alignment["typewell_margin_ft"])
    count = len(base)
    out: list[float] = []
    for index, value in enumerate(base):
        fraction = index / max(1, count - 1)
        correction = _clip(datum + toe * fraction, -maximum, maximum)
        out.append(_clip(float(value) + correction, curve.minimum_tvt - margin, curve.maximum_tvt + margin))
    return out


def particle_path(
    horizontal: Mapping[str, Any],
    curve: E006TypewellCurve,
    base: Sequence[float],
    alignment: Mapping[str, Any],
    particle: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    columns = horizontal["columns"]
    known_rows = int(horizontal["known_rows"])
    hidden_rows = int(horizontal["row_count"]) - known_rows
    gr = columns["GR"]
    tvt_input = columns["TVT_input"]
    hidden = gr[known_rows:]
    coverage = sum(value is not None for value in hidden) / max(1, hidden_rows)
    samples = _sample_indices(
        [index for index in range(known_rows, int(horizontal["row_count"])) if gr[index] is not None],
        int(alignment["maximum_gr_samples_per_well"]),
    )
    calibration = _calibration(gr, tvt_input, known_rows, curve, int(alignment["minimum_visible_calibration_samples"]))
    fallback = ""
    if coverage < float(alignment["minimum_hidden_gr_coverage"]):
        fallback = "low_hidden_gr_coverage"
    elif len(samples) < int(alignment["minimum_gr_samples"]):
        fallback = "too_few_hidden_gr_samples"
    elif curve.gr_std < float(alignment["flat_typewell_std_min"]):
        fallback = "flat_typewell_gr"
    elif calibration is None:
        fallback = "invalid_visible_calibration"
    if fallback:
        return list(base), {"fallback_reason": fallback, "coverage": coverage, "effective_fraction": 0.0}
    assert calibration is not None
    particles = [
        (float(datum), float(toe))
        for datum in alignment["datum_offsets_ft"]
        for toe in alignment["toe_offsets_ft"]
    ]
    weights = [1.0 / len(particles)] * len(particles)
    block_size = max(1, int(particle["block_samples"]))
    temperature = max(1e-9, float(particle["temperature"]))
    for start in range(0, len(samples), block_size):
        block = samples[start : start + block_size]
        losses = [
            _path_loss(gr, known_rows, hidden_rows, base, curve, block, datum, toe, calibration, alignment)
            for datum, toe in particles
        ]
        finite = [value for value in losses if math.isfinite(value)]
        if not finite:
            return list(base), {"fallback_reason": "particle_no_finite_weight", "coverage": coverage, "effective_fraction": 0.0}
        minimum = min(finite)
        updated = [weight * (math.exp(-(loss - minimum) / temperature) if math.isfinite(loss) else 0.0) for weight, loss in zip(weights, losses)]
        total = sum(updated)
        if total <= 0.0 or not math.isfinite(total):
            return list(base), {"fallback_reason": "particle_weight_collapse", "coverage": coverage, "effective_fraction": 0.0}
        weights = [value / total for value in updated]
    effective = 1.0 / sum(value * value for value in weights) / len(weights)
    datum = sum(weight * state[0] for weight, state in zip(weights, particles))
    toe = sum(weight * state[1] for weight, state in zip(weights, particles))
    return _apply_affine(base, datum, toe, curve, alignment), {
        "fallback_reason": "",
        "coverage": coverage,
        "effective_fraction": effective,
        "pf_datum": datum,
        "pf_toe": toe,
    }


def load_e006_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("schema_version") != 1 or model.get("experiment_id") != "E006":
        raise DeploymentDataError(f"{path}: unsupported E006 model schema")
    weight = float(model.get("fusion_weight", math.nan))
    if not math.isfinite(weight) or not (0.0 <= weight <= 1.0):
        raise DeploymentDataError(f"{path}: invalid E006 fusion weight")
    return model


def build_e006_prediction_map(model: Mapping[str, Any], test_dir: Path) -> tuple[dict[str, float], list[dict[str, Any]]]:
    e004_model = model["e004_model"]
    horizontal_files = sorted(test_dir.glob("*__horizontal_well.csv"))
    if not horizontal_files:
        raise DeploymentDataError(f"{test_dir}: no horizontal well files")
    predictions: dict[str, float] = {}
    well_rows: list[dict[str, Any]] = []
    weight = float(model["fusion_weight"])
    for horizontal_path in horizontal_files:
        well_id = horizontal_path.name.split("__", 1)[0]
        typewell_path = test_dir / f"{well_id}__typewell.csv"
        if not typewell_path.exists():
            raise DeploymentDataError(f"missing typewell for {well_id}")
        extracted = extract_well_features(
            horizontal_path,
            typewell_path,
            visible_slope_windows=e004_model["visible_slope_windows"],
            visible_backtest_fractions=e004_model["visible_backtest_fractions"],
            require_truth=False,
        )
        datum, trend = predict_coefficients(e004_model, extracted["features"])
        metadata = extracted["metadata"]
        base = reconstruct_hidden(metadata["last_visible_tvt"], metadata["hidden_rows"], datum, trend)
        horizontal = read_horizontal(horizontal_path, require_truth=False)
        curve = E006TypewellCurve.read(typewell_path)
        pf, diagnostic = particle_path(horizontal, curve, base, model["alignment"], model["particle_filter"])
        fused = [float(anchor) + weight * (float(particle_value) - float(anchor)) for anchor, particle_value in zip(base, pf)]
        if len(fused) != int(metadata["hidden_rows"]) or any(not math.isfinite(value) for value in fused):
            raise DeploymentDataError(f"{well_id}: invalid E006 fused path")
        for offset, value in enumerate(fused, start=int(metadata["known_rows"])):
            key = f"{well_id}_{offset}"
            if key in predictions:
                raise DeploymentDataError(f"duplicate prediction id {key}")
            predictions[key] = value
        well_rows.append({
            "well_id": well_id,
            "known_rows": metadata["known_rows"],
            "hidden_rows": metadata["hidden_rows"],
            "predicted_datum": datum,
            "predicted_trend": trend,
            "fusion_weight": weight,
            **diagnostic,
        })
    return predictions, well_rows


def write_e006_submission(model: Mapping[str, Any], test_dir: Path, sample_submission: Path, output_path: Path) -> dict[str, Any]:
    prediction_map, well_rows = build_e006_prediction_map(model, test_dir)
    sample_ids: list[str] = []
    seen: set[str] = set()
    with sample_submission.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != ["id", "tvt"]:
            raise DeploymentDataError(f"{sample_submission}: expected columns id,tvt")
        for row in reader:
            key = str(row.get("id") or "")
            if not key or key in seen:
                raise DeploymentDataError(f"{sample_submission}: empty or duplicate id {key!r}")
            seen.add(key)
            sample_ids.append(key)
    if set(sample_ids) != set(prediction_map):
        missing = sorted(set(sample_ids) - set(prediction_map))[:5]
        extra = sorted(set(prediction_map) - set(sample_ids))[:5]
        raise DeploymentDataError(f"sample/prediction ID mismatch; missing={missing}, extra={extra}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "tvt"], lineterminator="\n")
        writer.writeheader()
        for key in sample_ids:
            writer.writerow({"id": key, "tvt": format(prediction_map[key], f".{SUBMISSION_DECIMAL_PLACES}f")})
    return {
        "rows": len(sample_ids),
        "wells": len(well_rows),
        "minimum_prediction": min(prediction_map.values()),
        "maximum_prediction": max(prediction_map.values()),
        "well_predictions": well_rows,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "bytes": output_path.stat().st_size,
    }
