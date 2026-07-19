"""E005 clean typewell-horizontal GR candidate bank.

All predictor inputs are available in the competition test files. Hidden TVT is
read only by the evaluator. The implementation is standard-library only and
processes wells sequentially to keep memory bounded.
"""
from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .deployment import (
    _average_predictions as _e004_average_predictions,
    _crossfit as _e004_crossfit,
    _extract_training_records as _e004_extract_training_records,
    feature_family,
)
from .harness import (
    DataValidationError,
    ErrorAccumulator,
    OnlineCorrelation,
    WellMetric,
    _canonical_json,
    _fold_metrics,
    _quantile,
    _sha256,
    _summarize,
    _write_csv,
    _write_json,
)
from .learnability import _load_fold_maps

MISSING = {"", "nan", "NaN", "NA", "N/A", "null", "None"}
LEGAL_CANDIDATES = (
    "last_known_tvt",
    "e004_geometry_prefix",
    "align_affine",
    "align_visible_path",
    "pf_gr_path",
    "trellis_gr_path",
    "no_gr_geometry_prefix",
    "no_gr_typewell_affine",
)
DIAGNOSTIC_CANDIDATES = (
    "axis_confusion",
    "duplicate_align",
    "shuffled_typewell_gr",
    "oracle_target",
)
ALL_CANDIDATES = LEGAL_CANDIDATES + DIAGNOSTIC_CANDIDATES
ELIGIBLE = ("align_affine", "align_visible_path", "pf_gr_path", "trellis_gr_path")


def _optional(raw: str | None) -> float | None:
    if raw is None or raw.strip() in MISSING:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _required(raw: str | None, *, field: str, path: Path, row: int) -> float:
    value = _optional(raw)
    if value is None:
        raise DataValidationError(f"{path}:{row}: missing or non-finite {field}")
    return value


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    cx = _mean(xs)
    cy = _mean(ys)
    denominator = sum((value - cx) ** 2 for value in xs)
    if denominator <= 0.0:
        return 0.0
    return sum((x - cx) * (y - cy) for x, y in zip(xs, ys)) / denominator


def _sample_indices(indices: Sequence[int], maximum: int) -> list[int]:
    if len(indices) <= maximum:
        return list(indices)
    if maximum <= 1:
        return [indices[-1]]
    return sorted({indices[round(step * (len(indices) - 1) / (maximum - 1))] for step in range(maximum)})


def _huber(value: float, delta: float) -> float:
    absolute = abs(value)
    return 0.5 * absolute * absolute if absolute <= delta else delta * (absolute - 0.5 * delta)


@dataclass(frozen=True)
class TypewellCurve:
    tvt: tuple[float, ...]
    gr: tuple[float, ...]
    minimum_tvt: float
    maximum_tvt: float
    gr_mean: float
    gr_std: float
    digest: str

    @classmethod
    def read(cls, path: Path) -> "TypewellCurve":
        if not path.exists():
            raise DataValidationError(f"missing paired typewell {path}")
        pairs: list[tuple[float, float]] = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = {"TVT", "GR"} - set(reader.fieldnames or [])
            if missing:
                raise DataValidationError(f"{path}: missing typewell columns {sorted(missing)}")
            for row in reader:
                tvt = _optional(row.get("TVT"))
                gr = _optional(row.get("GR"))
                if tvt is not None and gr is not None:
                    pairs.append((tvt, gr))
        if len(pairs) < 2:
            raise DataValidationError(f"{path}: fewer than two finite TVT/GR pairs")
        pairs.sort()
        collapsed_tvt: list[float] = []
        collapsed_gr: list[float] = []
        start = 0
        while start < len(pairs):
            end = start + 1
            while end < len(pairs) and pairs[end][0] == pairs[start][0]:
                end += 1
            collapsed_tvt.append(pairs[start][0])
            collapsed_gr.append(_mean([value for _, value in pairs[start:end]]))
            start = end
        if len(collapsed_tvt) < 2 or collapsed_tvt[-1] <= collapsed_tvt[0]:
            raise DataValidationError(f"{path}: typewell TVT support is degenerate")
        return cls(
            tvt=tuple(collapsed_tvt),
            gr=tuple(collapsed_gr),
            minimum_tvt=collapsed_tvt[0],
            maximum_tvt=collapsed_tvt[-1],
            gr_mean=_mean(collapsed_gr),
            gr_std=_std(collapsed_gr),
            digest=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def value(self, tvt: float) -> float | None:
        if not math.isfinite(tvt) or tvt < self.minimum_tvt or tvt > self.maximum_tvt:
            return None
        position = bisect.bisect_left(self.tvt, tvt)
        if position <= 0:
            return self.gr[0]
        if position >= len(self.tvt):
            return self.gr[-1]
        left_tvt = self.tvt[position - 1]
        right_tvt = self.tvt[position]
        if right_tvt <= left_tvt:
            return self.gr[position]
        fraction = (tvt - left_tvt) / (right_tvt - left_tvt)
        return self.gr[position - 1] * (1.0 - fraction) + self.gr[position] * fraction


@dataclass(frozen=True)
class WellData:
    well_id: str
    md: tuple[float, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    z: tuple[float, ...]
    gr: tuple[float | None, ...]
    tvt_input: tuple[float | None, ...]
    truth: tuple[float, ...] | None
    known_rows: int
    typewell: TypewellCurve

    @property
    def hidden_rows(self) -> int:
        return len(self.md) - self.known_rows

    @property
    def last_visible_tvt(self) -> float:
        value = self.tvt_input[self.known_rows - 1]
        assert value is not None
        return value

    @property
    def hidden_gr_coverage(self) -> float:
        values = self.gr[self.known_rows :]
        return sum(value is not None for value in values) / max(1, len(values))


def read_well(horizontal: Path, typewell: Path, *, require_truth: bool) -> WellData:
    columns: dict[str, list[float | None]] = defaultdict(list)
    with horizontal.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"MD", "X", "Y", "Z", "GR", "TVT_input"} | ({"TVT"} if require_truth else set())
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{horizontal}: missing E005 columns {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            for name in ("MD", "X", "Y", "Z"):
                columns[name].append(_required(row.get(name), field=name, path=horizontal, row=row_number))
            columns["GR"].append(_optional(row.get("GR")))
            columns["TVT_input"].append(_optional(row.get("TVT_input")))
            if require_truth:
                columns["TVT"].append(_required(row.get("TVT"), field="TVT", path=horizontal, row=row_number))
    count = len(columns["MD"])
    if count < 4:
        raise DataValidationError(f"{horizontal}: fewer than four rows")
    md = [float(value) for value in columns["MD"] if value is not None]
    if any(second <= first for first, second in zip(md, md[1:])):
        raise DataValidationError(f"{horizontal}: MD must be strictly increasing; duplicate and reversed MD are rejected")
    known = 0
    hidden_seen = False
    for value in columns["TVT_input"]:
        visible = value is not None
        if visible and hidden_seen:
            raise DataValidationError(f"{horizontal}: TVT_input reappears after hidden suffix begins")
        if visible:
            known += 1
        else:
            hidden_seen = True
    if known < 3 or known >= count:
        raise DataValidationError(f"{horizontal}: requires at least three visible rows and a non-empty hidden suffix")
    if require_truth:
        for index in range(known):
            assert columns["TVT_input"][index] is not None and columns["TVT"][index] is not None
            if abs(float(columns["TVT_input"][index]) - float(columns["TVT"][index])) > 1e-9:
                raise DataValidationError(f"{horizontal}: visible TVT_input differs from TVT")
    return WellData(
        well_id=horizontal.name.split("__", 1)[0],
        md=tuple(md),
        x=tuple(float(value) for value in columns["X"] if value is not None),
        y=tuple(float(value) for value in columns["Y"] if value is not None),
        z=tuple(float(value) for value in columns["Z"] if value is not None),
        gr=tuple(columns["GR"]),
        tvt_input=tuple(columns["TVT_input"]),
        truth=tuple(float(value) for value in columns["TVT"] if value is not None) if require_truth else None,
        known_rows=known,
        typewell=TypewellCurve.read(typewell),
    )


def _calibration(well: WellData, curve: TypewellCurve, minimum: int) -> dict[str, float] | None:
    horizontal: list[float] = []
    reference: list[float] = []
    for index in range(well.known_rows):
        h = well.gr[index]
        tvt = well.tvt_input[index]
        if h is None or tvt is None:
            continue
        r = curve.value(tvt)
        if r is None:
            continue
        horizontal.append(h)
        reference.append(r)
    if len(horizontal) < minimum:
        return None
    h_mean, r_mean = _mean(horizontal), _mean(reference)
    h_std, r_std = _std(horizontal), _std(reference)
    if h_std <= 1e-9 or r_std <= 1e-9:
        return None
    covariance = sum((h - h_mean) * (r - r_mean) for h, r in zip(horizontal, reference))
    sign = 1.0 if covariance >= 0.0 else -1.0
    return {"h_mean": h_mean, "h_std": h_std, "r_mean": r_mean, "r_std": r_std, "sign": sign, "pairs": float(len(horizontal))}


def _normalize_pair(
    horizontal_gr: float,
    reference_gr: float,
    *,
    calibration: Mapping[str, float] | None,
    hidden_mean: float,
    hidden_std: float,
    curve: TypewellCurve,
    clip_z: float,
) -> tuple[float, float]:
    if calibration is None:
        left = (horizontal_gr - hidden_mean) / max(hidden_std, 1e-9)
        right = (reference_gr - curve.gr_mean) / max(curve.gr_std, 1e-9)
    else:
        left = float(calibration["sign"]) * (horizontal_gr - float(calibration["h_mean"])) / float(calibration["h_std"])
        right = (reference_gr - float(calibration["r_mean"])) / float(calibration["r_std"])
    return _clip(left, -clip_z, clip_z), _clip(right, -clip_z, clip_z)


def _hidden_samples(well: WellData, maximum: int) -> list[int]:
    indices = [index for index in range(well.known_rows, len(well.md)) if well.gr[index] is not None]
    return _sample_indices(indices, maximum)


def _apply_affine(base: Sequence[float], datum: float, toe: float, curve: TypewellCurve, config: Mapping[str, Any]) -> list[float]:
    margin = float(config["typewell_margin_ft"])
    maximum = float(config["maximum_absolute_correction_ft"])
    count = len(base)
    out: list[float] = []
    for index, value in enumerate(base):
        fraction = index / max(1, count - 1)
        correction = _clip(datum + toe * fraction, -maximum, maximum)
        out.append(_clip(float(value) + correction, curve.minimum_tvt - margin, curve.maximum_tvt + margin))
    return out


def _path_loss(
    well: WellData,
    base: Sequence[float],
    curve: TypewellCurve,
    samples: Sequence[int],
    datum: float,
    toe: float,
    *,
    calibration: Mapping[str, float] | None,
    config: Mapping[str, Any],
    axis_confusion: bool = False,
) -> float:
    hidden_values = [float(well.gr[index]) for index in samples if well.gr[index] is not None]
    if not hidden_values:
        return math.inf
    hidden_mean = _mean(hidden_values)
    hidden_std = _std(hidden_values)
    clip_z = float(config["gr_clip_z"])
    delta = float(config["huber_delta_z"])
    total = 0.0
    usable = 0
    for absolute_index in samples:
        horizontal_gr = well.gr[absolute_index]
        if horizontal_gr is None:
            continue
        hidden_index = absolute_index - well.known_rows
        fraction = hidden_index / max(1, well.hidden_rows - 1)
        lookup = well.md[absolute_index] + datum + toe * fraction if axis_confusion else float(base[hidden_index]) + datum + toe * fraction
        reference = curve.value(lookup)
        if reference is None:
            total += _huber(clip_z, delta)
            usable += 1
            continue
        left, right = _normalize_pair(
            horizontal_gr,
            reference,
            calibration=calibration,
            hidden_mean=hidden_mean,
            hidden_std=max(hidden_std, 1e-9),
            curve=curve,
            clip_z=clip_z,
        )
        total += _huber(left - right, delta)
        usable += 1
    if usable == 0:
        return math.inf
    prior = float(config["prior_weight"]) * (
        (datum / float(config["datum_prior_scale_ft"])) ** 2
        + (toe / float(config["toe_prior_scale_ft"])) ** 2
    )
    return total / usable + prior


def _grid_paths(
    well: WellData,
    base: Sequence[float],
    curve: TypewellCurve,
    samples: Sequence[int],
    *,
    calibration: Mapping[str, float] | None,
    config: Mapping[str, Any],
    axis_confusion: bool = False,
) -> tuple[list[float], dict[str, Any], list[tuple[float, float, float]]]:
    scored: list[tuple[float, float, float]] = []
    for datum in config["datum_offsets_ft"]:
        for toe in config["toe_offsets_ft"]:
            loss = _path_loss(
                well,
                base,
                curve,
                samples,
                float(datum),
                float(toe),
                calibration=calibration,
                config=config,
                axis_confusion=axis_confusion,
            )
            scored.append((loss, float(datum), float(toe)))
    scored.sort(key=lambda item: (item[0], abs(item[1]) + abs(item[2]), item[1], item[2]))
    best = scored[0]
    second = scored[1]
    if not math.isfinite(best[0]):
        return list(base), {"abstained": True, "reason": "no_finite_grid_path", "ambiguity_margin": 0.0}, scored
    margin = (second[0] - best[0]) / max(1e-12, abs(best[0]))
    return _apply_affine(base, best[1], best[2], curve, config), {
        "abstained": False,
        "reason": "",
        "datum": best[1],
        "toe": best[2],
        "loss": best[0],
        "ambiguity_margin": margin,
    }, scored


def _particle_path(
    well: WellData,
    base: Sequence[float],
    curve: TypewellCurve,
    samples: Sequence[int],
    calibration: Mapping[str, float],
    config: Mapping[str, Any],
    pf: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    particles = [(float(datum), float(toe)) for datum in config["datum_offsets_ft"] for toe in config["toe_offsets_ft"]]
    weights = [1.0 / len(particles)] * len(particles)
    block_size = max(1, int(pf["block_samples"]))
    temperature = max(1e-9, float(pf["temperature"]))
    for start in range(0, len(samples), block_size):
        block = samples[start : start + block_size]
        losses = [
            _path_loss(well, base, curve, block, datum, toe, calibration=calibration, config=config)
            for datum, toe in particles
        ]
        finite = [value for value in losses if math.isfinite(value)]
        if not finite:
            return list(base), {"abstained": True, "reason": "particle_no_finite_weight", "effective_fraction": 0.0}
        minimum = min(finite)
        updated = [weight * (math.exp(-(loss - minimum) / temperature) if math.isfinite(loss) else 0.0) for weight, loss in zip(weights, losses)]
        total = sum(updated)
        if total <= 0.0 or not math.isfinite(total):
            return list(base), {"abstained": True, "reason": "particle_weight_collapse", "effective_fraction": 0.0}
        weights = [value / total for value in updated]
    effective = 1.0 / sum(value * value for value in weights) / len(weights)
    datum = sum(weight * particle[0] for weight, particle in zip(weights, particles))
    toe = sum(weight * particle[1] for weight, particle in zip(weights, particles))
    if effective < float(pf["minimum_effective_fraction"]):
        # Low effective count is reported as ambiguity, but the bounded weighted
        # estimate remains deterministic and legal.
        reason = "low_effective_fraction"
    else:
        reason = ""
    return _apply_affine(base, datum, toe, curve, config), {
        "abstained": False,
        "reason": reason,
        "datum": datum,
        "toe": toe,
        "effective_fraction": effective,
        "maximum_weight": max(weights),
    }


def _trellis_path(
    well: WellData,
    base: Sequence[float],
    curve: TypewellCurve,
    samples: Sequence[int],
    calibration: Mapping[str, float],
    config: Mapping[str, Any],
    trellis: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    anchors = _sample_indices(samples, int(trellis["maximum_anchors"]))
    states = [float(value) for value in trellis["offset_states_ft"]]
    if not anchors or not states:
        return list(base), {"abstained": True, "reason": "empty_trellis"}
    hidden_gr = [float(well.gr[index]) for index in anchors if well.gr[index] is not None]
    hidden_mean, hidden_std = _mean(hidden_gr), max(_std(hidden_gr), 1e-9)
    clip_z = float(config["gr_clip_z"])
    delta = float(config["huber_delta_z"])

    def emission(absolute_index: int, state: float) -> float:
        horizontal = well.gr[absolute_index]
        assert horizontal is not None
        hidden_index = absolute_index - well.known_rows
        reference = curve.value(float(base[hidden_index]) + state)
        if reference is None:
            return _huber(clip_z, delta)
        left, right = _normalize_pair(horizontal, reference, calibration=calibration, hidden_mean=hidden_mean, hidden_std=hidden_std, curve=curve, clip_z=clip_z)
        return _huber(left - right, delta)

    max_transition = float(trellis["maximum_transition_ft"])
    transition_penalty = float(trellis["transition_penalty"])
    prior = float(trellis["offset_prior_penalty"])
    boundary = float(trellis["boundary_anchor_penalty"])
    costs = [emission(anchors[0], state) + boundary * abs(state) / max(1.0, max(abs(v) for v in states)) + prior * state * state for state in states]
    back: list[list[int]] = []
    for anchor in anchors[1:]:
        next_costs: list[float] = []
        next_back: list[int] = []
        for state_index, state in enumerate(states):
            choices: list[tuple[float, int]] = []
            for previous_index, previous in enumerate(states):
                difference = abs(state - previous)
                if difference <= max_transition + 1e-12:
                    choices.append((costs[previous_index] + transition_penalty * (difference / 5.0) ** 2, previous_index))
            if not choices:
                choices = [(costs[state_index], state_index)]
            best_cost, best_index = min(choices, key=lambda item: (item[0], item[1]))
            next_costs.append(best_cost + emission(anchor, state) + prior * state * state)
            next_back.append(best_index)
        costs = next_costs
        back.append(next_back)
    order = sorted(range(len(states)), key=lambda index: (costs[index], abs(states[index]), states[index]))
    final_index = order[0]
    second_cost = costs[order[1]] if len(order) > 1 else costs[final_index]
    chosen = [final_index]
    for links in reversed(back):
        chosen.append(links[chosen[-1]])
    chosen.reverse()
    anchor_hidden = [index - well.known_rows for index in anchors]
    anchor_offsets = [states[index] for index in chosen]
    correction: list[float] = []
    for hidden_index in range(well.hidden_rows):
        position = bisect.bisect_left(anchor_hidden, hidden_index)
        if position <= 0:
            value = anchor_offsets[0]
        elif position >= len(anchor_hidden):
            value = anchor_offsets[-1]
        else:
            left_i, right_i = anchor_hidden[position - 1], anchor_hidden[position]
            fraction = (hidden_index - left_i) / max(1, right_i - left_i)
            value = anchor_offsets[position - 1] * (1.0 - fraction) + anchor_offsets[position] * fraction
        correction.append(value)
    margin = (second_cost - costs[final_index]) / max(1e-12, abs(costs[final_index]))
    path = [
        _clip(float(value) + _clip(offset, -float(config["maximum_absolute_correction_ft"]), float(config["maximum_absolute_correction_ft"])), curve.minimum_tvt - float(config["typewell_margin_ft"]), curve.maximum_tvt + float(config["typewell_margin_ft"]))
        for value, offset in zip(base, correction)
    ]
    return path, {
        "abstained": False,
        "reason": "",
        "anchor_count": len(anchors),
        "minimum_offset": min(anchor_offsets),
        "maximum_offset": max(anchor_offsets),
        "ambiguity_margin": margin,
    }


def _no_gr_typewell_affine(well: WellData, config: Mapping[str, Any]) -> list[float]:
    window = min(well.known_rows, int(config["visible_window_rows"]))
    md = list(well.md[well.known_rows - window : well.known_rows])
    tvt = [float(value) for value in well.tvt_input[well.known_rows - window : well.known_rows] if value is not None]
    slope = _clip(_linear_slope(md, tvt), -float(config["slope_clip_ft_per_md"]), float(config["slope_clip_ft_per_md"]))
    slope *= float(config["damping"])
    margin = 20.0
    return [
        _clip(well.last_visible_tvt + slope * (well.md[index] - well.md[well.known_rows - 1]), well.typewell.minimum_tvt - margin, well.typewell.maximum_tvt + margin)
        for index in range(well.known_rows, len(well.md))
    ]


def validate_e005_config(config: Mapping[str, Any]) -> None:
    if list(config.get("candidate_order", [])) != list(ALL_CANDIDATES):
        raise DataValidationError("E005 candidate order differs from frozen implementation")
    if list(config.get("eligible_candidates", [])) != list(ELIGIBLE):
        raise DataValidationError("E005 eligible candidates differ from frozen implementation")
    for section, key in (("alignment", "datum_offsets_ft"), ("alignment", "toe_offsets_ft"), ("trellis", "offset_states_ft")):
        values = config.get(section, {}).get(key)
        if not isinstance(values, list) or not values:
            raise DataValidationError(f"E005 {section}.{key} must be a non-empty array")
        parsed = [float(value) for value in values]
        if any(not math.isfinite(value) for value in parsed) or len(set(parsed)) != len(parsed):
            raise DataValidationError(f"E005 {section}.{key} must contain unique finite values")
    states = [float(value) for value in config["trellis"]["offset_states_ft"]]
    if states != sorted(states):
        raise DataValidationError("E005 trellis offset states must be sorted")
    if float(config["trellis"]["maximum_transition_ft"]) < 0.0:
        raise DataValidationError("E005 maximum trellis transition must be non-negative")
    weights = config.get("diagnostic_blend_weights")
    if not isinstance(weights, list) or not weights or any(not (0.0 < float(value) < 1.0) for value in weights):
        raise DataValidationError("E005 diagnostic blend weights must be a non-empty array inside (0,1)")


def candidate_paths(
    well: WellData,
    e004_base: Sequence[float],
    shuffled_curve: TypewellCurve,
    config: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    validate_e005_config(config)
    if len(e004_base) != well.hidden_rows or not all(math.isfinite(float(value)) for value in e004_base):
        raise DataValidationError(f"{well.well_id}: invalid E004 base path")
    align = config["alignment"]
    samples = _hidden_samples(well, int(align["maximum_gr_samples_per_well"]))
    fallback_reason = ""
    if well.hidden_gr_coverage < float(align["minimum_hidden_gr_coverage"]):
        fallback_reason = "low_hidden_gr_coverage"
    elif len(samples) < int(align["minimum_gr_samples"]):
        fallback_reason = "too_few_hidden_gr_samples"
    elif well.typewell.gr_std < float(align["flat_typewell_std_min"]):
        fallback_reason = "flat_typewell_gr"
    calibration = _calibration(well, well.typewell, int(align["minimum_visible_calibration_samples"]))
    if not fallback_reason and calibration is None:
        fallback_reason = "invalid_visible_calibration"
    diagnostics: dict[str, Any] = {
        "well_id": well.well_id,
        "hidden_rows": well.hidden_rows,
        "hidden_gr_coverage": well.hidden_gr_coverage,
        "hidden_gr_samples": len(samples),
        "visible_calibration_pairs": int(calibration["pairs"]) if calibration else 0,
        "fallback_reason": fallback_reason,
        "typewell_digest": well.typewell.digest,
    }
    if fallback_reason:
        align_affine = list(e004_base)
        align_visible = list(e004_base)
        pf = list(e004_base)
        trellis = list(e004_base)
        diagnostics.update({
            "align_affine_abstained": True,
            "align_visible_abstained": True,
            "pf_abstained": True,
            "trellis_abstained": True,
            "ambiguity_margin": 0.0,
        })
    else:
        align_affine, raw_diag, _ = _grid_paths(well, e004_base, well.typewell, samples, calibration=None, config=align)
        assert calibration is not None
        align_visible, visible_diag, _ = _grid_paths(well, e004_base, well.typewell, samples, calibration=calibration, config=align)
        pf, pf_diag = _particle_path(well, e004_base, well.typewell, samples, calibration, align, config["particle_filter"])
        trellis, trellis_diag = _trellis_path(well, e004_base, well.typewell, samples, calibration, align, config["trellis"])
        diagnostics.update({
            "align_affine_abstained": bool(raw_diag.get("abstained")),
            "align_visible_abstained": bool(visible_diag.get("abstained")),
            "pf_abstained": bool(pf_diag.get("abstained")),
            "trellis_abstained": bool(trellis_diag.get("abstained")),
            "align_affine_datum": raw_diag.get("datum", 0.0),
            "align_affine_toe": raw_diag.get("toe", 0.0),
            "align_visible_datum": visible_diag.get("datum", 0.0),
            "align_visible_toe": visible_diag.get("toe", 0.0),
            "pf_datum": pf_diag.get("datum", 0.0),
            "pf_toe": pf_diag.get("toe", 0.0),
            "pf_effective_fraction": pf_diag.get("effective_fraction", 0.0),
            "trellis_minimum_offset": trellis_diag.get("minimum_offset", 0.0),
            "trellis_maximum_offset": trellis_diag.get("maximum_offset", 0.0),
            "ambiguity_margin": min(float(visible_diag.get("ambiguity_margin", 0.0)), float(trellis_diag.get("ambiguity_margin", 0.0))),
        })
    shuffled_calibration = _calibration(well, shuffled_curve, int(align["minimum_visible_calibration_samples"]))
    if fallback_reason or shuffled_calibration is None or shuffled_curve.gr_std < float(align["flat_typewell_std_min"]):
        shuffled = list(e004_base)
    else:
        shuffled, _, _ = _grid_paths(well, e004_base, shuffled_curve, samples, calibration=shuffled_calibration, config=align)
    axis = [
        _clip(well.md[index], well.typewell.minimum_tvt - float(align["typewell_margin_ft"]), well.typewell.maximum_tvt + float(align["typewell_margin_ft"]))
        for index in range(well.known_rows, len(well.md))
    ]
    no_gr = _no_gr_typewell_affine(well, config["no_gr_typewell_affine"])
    last = [well.last_visible_tvt] * well.hidden_rows
    paths = {
        "last_known_tvt": last,
        "e004_geometry_prefix": list(e004_base),
        "align_affine": align_affine,
        "align_visible_path": align_visible,
        "pf_gr_path": pf,
        "trellis_gr_path": trellis,
        "no_gr_geometry_prefix": list(e004_base),
        "no_gr_typewell_affine": no_gr,
        "axis_confusion": axis,
        "duplicate_align": list(align_visible),
        "shuffled_typewell_gr": shuffled,
    }
    if any(len(values) != well.hidden_rows for values in paths.values()):
        raise DataValidationError(f"{well.well_id}: candidate path length mismatch")
    if any(not math.isfinite(float(value)) for values in paths.values() for value in values):
        raise DataValidationError(f"{well.well_id}: non-finite candidate output")
    return paths, diagnostics


def validate_submission_ids(
    sample_submission: Path,
    predictions: Mapping[str, float],
    prediction_order: Sequence[str] | None = None,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    with sample_submission.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != ["id", "tvt"]:
            raise DataValidationError(f"{sample_submission}: expected columns id,tvt")
        for row in reader:
            key = str(row.get("id") or "")
            if not key or key in seen:
                raise DataValidationError(f"{sample_submission}: empty or duplicate id {key!r}")
            seen.add(key)
            ids.append(key)
    if prediction_order is not None and list(prediction_order) != ids:
        raise DataValidationError("prediction IDs are not in sample-submission order")
    if set(ids) != set(predictions):
        missing = sorted(set(ids) - set(predictions))[:5]
        extra = sorted(set(predictions) - set(ids))[:5]
        raise DataValidationError(f"sample/prediction ID mismatch; missing={missing}, extra={extra}")
    if any(not math.isfinite(float(predictions[key])) for key in ids):
        raise DataValidationError("submission contains non-finite predictions")
    return ids


def _write_gzip_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: (format(value, ".8f") if isinstance(value, float) else value) for key, value in row.items()})


def _e004_oof(root: Path, train_dir: Path, config: Mapping[str, Any]) -> tuple[dict[str, list[float]], dict[str, Any], list[dict[str, Any]]]:
    e004_config = json.loads((root / str(config["e004"]["config_path"])).read_text(encoding="utf-8"))
    records, feature_names, data_profile = _e004_extract_training_records(train_dir, e004_config)
    folds = _load_fold_maps(root, e004_config["fold_files"], str(e004_config["data_signature"]))
    selected_features = sorted(name for name in feature_names if feature_family(name) in {"geometry", "prefix"})
    map_predictions: dict[str, dict[str, list[float]]] = {}
    for fold_map in folds:
        predictions, _, _ = _e004_crossfit(records, fold_map["assignments"], int(fold_map["n_folds"]), selected_features, e004_config)
        map_predictions[str(fold_map["version"])] = predictions
    averaged = _e004_average_predictions(map_predictions)
    base_paths: dict[str, list[float]] = {}
    for well_id in sorted(records):
        n = int(records[well_id]["sufficient"]["rows"])
        last = float(records[well_id]["features"]["last_visible_tvt"])
        datum, trend = float(averaged[well_id][0]), float(averaged[well_id][1])
        base_paths[well_id] = [last + datum + trend * (index / max(1, n - 1) - 0.5 if n > 1 else 0.0) for index in range(n)]
    return base_paths, data_profile, folds


def _group_assignments(values: Mapping[str, tuple[float, ...]], bins: int) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: (*values[key], key))
    return {key: min(bins - 1, int(index * bins / len(ordered))) for index, key in enumerate(ordered)}


def _subset_summary(metrics: Mapping[str, WellMetric], ids: Sequence[str]) -> dict[str, Any]:
    selected = [metrics[well_id] for well_id in ids]
    return _summarize(selected) if selected else {"rows_scored": 0, "wells_scored": 0, "rmse": None}


def run_e005(
    *,
    root: Path,
    train_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: Mapping[str, Any],
    code_sha: str,
) -> dict[str, Any]:
    root = root.resolve()
    train_dir = train_dir.resolve()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    validate_e005_config(config)
    base_paths, data_profile, fold_maps = _e004_oof(root, train_dir, config)
    if data_profile["data_signature"] != str(config["data_signature"]) or len(base_paths) != int(config["expected_wells"]):
        raise DataValidationError("E005 data signature or well count mismatch")
    horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
    well_ids = [path.name.split("__", 1)[0] for path in horizontal_files]
    if set(well_ids) != set(base_paths):
        raise DataValidationError("E005/E004 well ID mismatch")
    curves = {well_id: TypewellCurve.read(train_dir / f"{well_id}__typewell.csv") for well_id in well_ids}
    shuffle_order = sorted(well_ids, key=lambda key: hashlib.sha256(f"e005-shuffle|{key}".encode()).hexdigest())
    shuffled_curves = {well_id: curves[shuffle_order[(index + 1) % len(shuffle_order)]] for index, well_id in enumerate(shuffle_order)}

    model_metrics: dict[str, dict[str, WellMetric]] = {candidate: {} for candidate in ALL_CANDIDATES}
    direct_sse = {candidate: 0.0 for candidate in ALL_CANDIDATES}
    correlations = {candidate: OnlineCorrelation() for candidate in ELIGIBLE + ("align_visible_path", "shuffled_typewell_gr")}
    blend_metrics: dict[str, dict[str, ErrorAccumulator]] = {
        f"{candidate}_w{str(weight).replace('.', 'p')}": {}
        for candidate in ELIGIBLE
        for weight in config["diagnostic_blend_weights"]
    }
    diagnostics_rows: list[dict[str, Any]] = []
    spatial_values: dict[str, tuple[float, ...]] = {}
    typewell_values: dict[str, tuple[float, ...]] = {}
    hidden_rows_by_well: dict[str, int] = {}
    missing_by_well: dict[str, float] = {}
    duplicate_delta = 0.0
    fallback_exact_delta = 0.0
    oof_path = artifact_dir / "oof_predictions.csv.gz"
    oof_fields = ["id", "well_id", "row_index", "hidden_index", "target", *ALL_CANDIDATES]

    def rows() -> Iterable[Mapping[str, Any]]:
        nonlocal duplicate_delta, fallback_exact_delta
        for horizontal in horizontal_files:
            well_id = horizontal.name.split("__", 1)[0]
            well = read_well(horizontal, train_dir / f"{well_id}__typewell.csv", require_truth=True)
            assert well.truth is not None
            paths, diagnostic = candidate_paths(well, base_paths[well_id], shuffled_curves[well_id], config)
            paths["oracle_target"] = list(well.truth[well.known_rows :])
            accumulators = {candidate: ErrorAccumulator() for candidate in ALL_CANDIDATES}
            blend_accumulators = {name: ErrorAccumulator() for name in blend_metrics}
            spatial_values[well_id] = (0.5 * (well.x[0] + well.x[-1]), 0.5 * (well.y[0] + well.y[-1]))
            typewell_values[well_id] = (well.typewell.gr_mean, well.typewell.gr_std, well.typewell.maximum_tvt - well.typewell.minimum_tvt)
            hidden_rows_by_well[well_id] = well.hidden_rows
            missing_by_well[well_id] = 1.0 - well.hidden_gr_coverage
            diagnostic["spatial_x_mid"] = spatial_values[well_id][0]
            diagnostic["spatial_y_mid"] = spatial_values[well_id][1]
            diagnostics_rows.append(diagnostic)
            for hidden_index in range(well.hidden_rows):
                absolute = well.known_rows + hidden_index
                target = float(well.truth[absolute])
                errors: dict[str, float] = {}
                for candidate in ALL_CANDIDATES:
                    prediction = float(paths[candidate][hidden_index])
                    error = prediction - target
                    errors[candidate] = error
                    direct_sse[candidate] += error * error
                    accumulators[candidate].add(error, float(hidden_index))
                duplicate_delta = max(duplicate_delta, abs(paths["duplicate_align"][hidden_index] - paths["align_visible_path"][hidden_index]))
                if diagnostic["fallback_reason"]:
                    fallback_exact_delta = max(fallback_exact_delta, max(abs(paths[candidate][hidden_index] - paths["e004_geometry_prefix"][hidden_index]) for candidate in ELIGIBLE))
                for candidate, accumulator in correlations.items():
                    accumulator.add(errors[candidate], errors["e004_geometry_prefix"])
                for candidate in ELIGIBLE:
                    for weight in config["diagnostic_blend_weights"]:
                        name = f"{candidate}_w{str(weight).replace('.', 'p')}"
                        prediction = float(weight) * paths[candidate][hidden_index] + (1.0 - float(weight)) * paths["e004_geometry_prefix"][hidden_index]
                        blend_accumulators[name].add(prediction - target, float(hidden_index))
                yield {
                    "id": f"{well_id}_{absolute}",
                    "well_id": well_id,
                    "row_index": absolute,
                    "hidden_index": hidden_index,
                    "target": target,
                    **{candidate: float(paths[candidate][hidden_index]) for candidate in ALL_CANDIDATES},
                }
            for candidate, accumulator in accumulators.items():
                model_metrics[candidate][well_id] = accumulator.finalize(well_id)
            for name, accumulator in blend_accumulators.items():
                blend_metrics[name][well_id] = accumulator

    _write_gzip_csv(oof_path, oof_fields, rows())
    summaries = {candidate: _summarize([model_metrics[candidate][well_id] for well_id in sorted(well_ids)]) for candidate in ALL_CANDIDATES}
    blend_summaries: dict[str, dict[str, Any]] = {}
    for name, by_well in blend_metrics.items():
        finalized = [by_well[well_id].finalize(well_id) for well_id in sorted(by_well)]
        blend_summaries[name] = _summarize(finalized)
    e004_rmse = float(summaries["e004_geometry_prefix"]["rmse"])
    baseline_rmse = float(summaries["last_known_tvt"]["rmse"])

    fold_rows: list[dict[str, Any]] = []
    fold_lookup: dict[tuple[str, str, int], float] = {}
    for candidate in ALL_CANDIDATES:
        for fold_map in fold_maps:
            for item in _fold_metrics(model_metrics[candidate], fold_map):
                row = {"candidate": candidate, **item}
                fold_rows.append(row)
                fold_lookup[(candidate, str(item["map"]), int(item["fold"]))] = float(item["rmse"])
    map_wins: dict[str, int] = {}
    map_cell_wins: dict[str, dict[str, int]] = {}
    for candidate in ELIGIBLE:
        wins = 0
        cells: dict[str, int] = {}
        for fold_map in fold_maps:
            version = str(fold_map["version"])
            count = sum(fold_lookup[(candidate, version, fold)] < fold_lookup[("e004_geometry_prefix", version, fold)] for fold in range(int(fold_map["n_folds"])))
            cells[version] = count
            wins += count >= 3
        map_wins[candidate] = wins
        map_cell_wins[candidate] = cells

    spatial = _group_assignments(spatial_values, int(config["stress"]["spatial_bins"]))
    type_clusters = _group_assignments(typewell_values, int(config["stress"]["typewell_clusters"]))
    long_threshold = _quantile([float(value) for value in hidden_rows_by_well.values()], float(config["stress"]["long_suffix_quantile"]))
    missing_threshold = _quantile(list(missing_by_well.values()), float(config["stress"]["high_missing_gr_quantile"]))
    ambiguity_threshold = float(config["stress"]["ambiguity_margin_threshold"])
    ambiguity = {str(row["well_id"]): float(row["ambiguity_margin"]) <= ambiguity_threshold for row in diagnostics_rows}
    stress_rows: list[dict[str, Any]] = []
    stress_lookup: dict[tuple[str, str, str], float] = {}
    for kind, assignments, groups in (
        ("spatial", spatial, range(int(config["stress"]["spatial_bins"]))),
        ("typewell_cluster", type_clusters, range(int(config["stress"]["typewell_clusters"]))),
    ):
        for group in groups:
            ids = [well_id for well_id in well_ids if assignments[well_id] == group]
            for candidate in ALL_CANDIDATES:
                summary = _subset_summary(model_metrics[candidate], ids)
                row = {"stress": kind, "group": str(group), "candidate": candidate, **summary}
                stress_rows.append(row)
                stress_lookup[(kind, str(group), candidate)] = float(summary["rmse"])
    special_sets = {
        "long_suffix": [well_id for well_id in well_ids if hidden_rows_by_well[well_id] >= long_threshold],
        "high_gr_missingness": [well_id for well_id in well_ids if missing_by_well[well_id] >= missing_threshold],
        "ambiguous_alignment": [well_id for well_id in well_ids if ambiguity[well_id]],
    }
    for name, ids in special_sets.items():
        for candidate in ALL_CANDIDATES:
            summary = _subset_summary(model_metrics[candidate], ids)
            stress_rows.append({"stress": name, "group": "selected", "candidate": candidate, **summary})

    correlation_rows = [
        {"candidate": candidate, "comparator": "e004_geometry_prefix", "residual_correlation": accumulator.value()}
        for candidate, accumulator in sorted(correlations.items())
    ]
    correlation_lookup = {row["candidate"]: row["residual_correlation"] for row in correlation_rows}
    pooled_difference = max(
        abs(direct_sse[candidate] - float(summaries[candidate]["sse"])) / max(1.0, direct_sse[candidate], float(summaries[candidate]["sse"]))
        for candidate in ALL_CANDIDATES
    )
    controls = {
        "data_integrity": {"pass": data_profile["data_signature"] == config["data_signature"] and len(well_ids) == int(config["expected_wells"]), "data_signature": data_profile["data_signature"], "wells": len(well_ids)},
        "e004_exact_comparator": {"pass": abs(e004_rmse - float(config["e004"]["expected_rmse"])) <= float(config["controls"]["e004_rmse_tolerance"]), "observed_rmse": e004_rmse, "expected_rmse": config["e004"]["expected_rmse"]},
        "pooled_sse_consistency": {"pass": pooled_difference <= float(config["controls"]["pooled_sse_relative_tolerance"]), "maximum_relative_difference": pooled_difference},
        "duplicate_alignment": {"pass": duplicate_delta <= float(config["controls"]["duplicate_maximum_prediction_delta"]), "maximum_prediction_delta": duplicate_delta},
        "shuffled_typewell_gr": {"pass": float(summaries["shuffled_typewell_gr"]["rmse"]) - float(summaries["align_visible_path"]["rmse"]) >= float(config["controls"]["shuffled_gr_minimum_rmse_loss"]), "unshuffled_rmse": summaries["align_visible_path"]["rmse"], "shuffled_rmse": summaries["shuffled_typewell_gr"]["rmse"]},
        "axis_confusion": {"pass": float(summaries["axis_confusion"]["rmse"]) > e004_rmse + 0.10, "rmse": summaries["axis_confusion"]["rmse"]},
        "oracle_leakage_sentinel": {"pass": float(summaries["oracle_target"]["rmse"]) <= float(config["controls"]["oracle_maximum_rmse"]), "rmse": summaries["oracle_target"]["rmse"], "eligible": False},
        "no_gr_fallback": {"pass": abs(float(summaries["no_gr_geometry_prefix"]["rmse"]) - e004_rmse) <= 1e-15 and fallback_exact_delta <= 1e-12, "fallback_wells": sum(bool(row["fallback_reason"]) for row in diagnostics_rows), "maximum_fallback_delta": fallback_exact_delta},
        "oof_identity": {"pass": sum(metric.rows_scored for metric in model_metrics["last_known_tvt"].values()) == int(data_profile["hidden_rows"]), "rows": sum(metric.rows_scored for metric in model_metrics["last_known_tvt"].values())},
    }
    candidate_rows: list[dict[str, Any]] = []
    for candidate in ALL_CANDIDATES:
        summary = summaries[candidate]
        candidate_rows.append({
            "candidate": candidate,
            "eligible": candidate in ELIGIBLE,
            "rmse": summary["rmse"],
            "gain_vs_last_known": baseline_rmse - float(summary["rmse"]),
            "gain_vs_e004": e004_rmse - float(summary["rmse"]),
            "median_well_rmse": summary["median_well_rmse"],
            "p90_well_rmse": summary["p90_well_rmse"],
            "p95_well_rmse": summary["p95_well_rmse"],
            "max_well_rmse": summary["max_well_rmse"],
            "worst_5pct_sse_share": summary["worst_5pct_sse_share"],
            "worst_10pct_sse_share": summary["worst_10pct_sse_share"],
            "mean_error_sse_share": summary["mean_error_sse_share"],
            "linear_trend_sse_share": summary["linear_trend_sse_share"],
            "shape_sse_share": summary["shape_sse_share"],
            "map_wins_vs_e004": map_wins.get(candidate, ""),
            "residual_correlation_to_e004": correlation_lookup.get(candidate),
        })
    selected = min(ELIGIBLE, key=lambda candidate: (float(summaries[candidate]["rmse"]), candidate))
    promotion = config["promotion"]
    spatial_min_gain = min(
        stress_lookup[("spatial", str(group), "e004_geometry_prefix")] - stress_lookup[("spatial", str(group), selected)]
        for group in range(int(config["stress"]["spatial_bins"]))
    )
    typewell_min_gain = min(
        stress_lookup[("typewell_cluster", str(group), "e004_geometry_prefix")] - stress_lookup[("typewell_cluster", str(group), selected)]
        for group in range(int(config["stress"]["typewell_clusters"]))
    )
    selected_summary = summaries[selected]
    gates = {
        "controls": all(bool(detail["pass"]) for detail in controls.values()),
        "gain_vs_last_known": baseline_rmse - float(selected_summary["rmse"]) >= float(promotion["minimum_gain_vs_last_known"]),
        "gain_vs_e004": e004_rmse - float(selected_summary["rmse"]) >= float(promotion["minimum_gain_vs_e004"]),
        "repeated_maps": map_wins[selected] >= int(promotion["minimum_map_wins"]),
        "p90_vs_last_known": float(selected_summary["p90_well_rmse"]) - float(summaries["last_known_tvt"]["p90_well_rmse"]) <= float(promotion["maximum_p90_deterioration_vs_each_comparator"]),
        "p90_vs_e004": float(selected_summary["p90_well_rmse"]) - float(summaries["e004_geometry_prefix"]["p90_well_rmse"]) <= float(promotion["maximum_p90_deterioration_vs_each_comparator"]),
        "worst5_vs_last_known": float(selected_summary["worst_5pct_sse_share"]) - float(summaries["last_known_tvt"]["worst_5pct_sse_share"]) <= float(promotion["maximum_worst5_sse_share_increase_vs_each_comparator"]),
        "worst5_vs_e004": float(selected_summary["worst_5pct_sse_share"]) - float(summaries["e004_geometry_prefix"]["worst_5pct_sse_share"]) <= float(promotion["maximum_worst5_sse_share_increase_vs_each_comparator"]),
        "spatial_stress": spatial_min_gain > 0.0 if bool(promotion["require_positive_spatial_gain_vs_e004"]) else True,
        "typewell_cluster_stress": typewell_min_gain > 0.0 if bool(promotion["require_positive_typewell_cluster_gain_vs_e004"]) else True,
        "diversity": correlation_lookup[selected] is not None and float(correlation_lookup[selected]) <= float(promotion["maximum_residual_correlation_to_e004_for_diversity"]),
    }
    promoted = all(gates.values())
    selected_well_rows: list[dict[str, Any]] = []
    for well_id in sorted(well_ids):
        metric = model_metrics[selected][well_id]
        diagnostic = next(row for row in diagnostics_rows if row["well_id"] == well_id)
        selected_well_rows.append({
            "well_id": well_id,
            "split": "oof",
            "candidate": selected,
            "rows_scored": metric.rows_scored,
            "rmse": metric.rmse,
            "mean_error": metric.mean_error,
            "sse": metric.sse,
            "regime": "fallback" if diagnostic["fallback_reason"] else "gr_active",
            "uncertainty": 1.0 / max(1e-12, float(diagnostic["ambiguity_margin"])),
            "datum_sse": metric.datum_sse,
            "trend_sse": metric.trend_sse,
            "shape_sse": metric.shape_sse,
            "trend_per_row": metric.trend_per_row,
        })
    summary = {
        "schema_version": 1,
        "experiment_id": "E005",
        "status": "promoted" if promoted else "rejected",
        "code_sha": code_sha,
        "data": data_profile,
        "selected_candidate": selected,
        "baseline_metrics": summaries["last_known_tvt"],
        "e004_metrics": summaries["e004_geometry_prefix"],
        "selected_candidate_metrics": selected_summary,
        "candidate_metrics": summaries,
        "diagnostic_blend_metrics": blend_summaries,
        "map_wins": map_wins,
        "map_cell_wins": map_cell_wins,
        "stress": {
            "spatial_minimum_gain_vs_e004": spatial_min_gain,
            "typewell_cluster_minimum_gain_vs_e004": typewell_min_gain,
            "long_suffix_threshold": long_threshold,
            "high_gr_missingness_threshold": missing_threshold,
            "ambiguous_wells": sum(ambiguity.values()),
        },
        "residual_correlations": {row["candidate"]: row["residual_correlation"] for row in correlation_rows},
        "controls": controls,
        "promotion": {"gates": gates, "promoted": promoted},
        "deployment": {
            "statistically_authorized": promoted,
            "local_package_built": False,
            "local_notebook_parity": False,
            "private_internet_disabled_kaggle_parity": False,
            "deployment_ready": False,
            "submission_created": False,
            "submission_made": False,
            "reason": "Run deployment packaging only after statistical promotion." if promoted else "Statistical promotion gates failed; Kaggle parity and submission are not authorized.",
        },
        "predictor_policy": {
            candidate: {"uses_hidden_target": candidate == "oracle_target", "eligible_for_promotion": candidate in ELIGIBLE}
            for candidate in ALL_CANDIDATES
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "candidate_metrics.csv", list(candidate_rows[0]), candidate_rows)
    _write_csv(output_dir / "fold_metrics.csv", list(fold_rows[0]), fold_rows)
    _write_csv(output_dir / "stress_metrics.csv", list(stress_rows[0]), stress_rows)
    _write_csv(output_dir / "alignment_diagnostics.csv", list(diagnostics_rows[0]), diagnostics_rows)
    _write_csv(output_dir / "residual_correlations.csv", list(correlation_rows[0]), correlation_rows)
    blend_rows = [{"blend": name, **metric} for name, metric in sorted(blend_summaries.items())]
    _write_csv(output_dir / "diagnostic_blends.csv", list(blend_rows[0]), blend_rows)
    _write_csv(output_dir / "selected_well_metrics.csv", list(selected_well_rows[0]), selected_well_rows)
    control_rows = [{"control": name, "status": "pass" if detail["pass"] else "fail", "detail_json": _canonical_json({key: value for key, value in detail.items() if key != "pass"})} for name, detail in sorted(controls.items())]
    _write_csv(output_dir / "control_metrics.csv", list(control_rows[0]), control_rows)
    result_files = [path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    artifact_manifest = {
        "schema_version": 1,
        "code_sha": code_sha,
        "files": [{"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size} for path in sorted(result_files, key=lambda item: item.name)],
        "external_artifacts": [{"path": str(oof_path.relative_to(root)), "sha256": _sha256(oof_path), "bytes": oof_path.stat().st_size, "tracked": False}],
        "config": {"path": "experiments/E005/config.json", "sha256": _sha256(root / "experiments/E005/config.json"), "bytes": (root / "experiments/E005/config.json").stat().st_size},
        "fold_files": [{"path": relative, "sha256": _sha256(root / relative), "bytes": (root / relative).stat().st_size} for relative in config["fold_files"]],
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return summary
