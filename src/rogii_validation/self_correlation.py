"""E007 horizontal GR self-correlation with controlled E006 placement.

The new evidence leg uses only the same horizontal well's GR, geometry, and
visible TVT prefix.  Typewell data never enters the predictor.  Frozen E006 OOF
predictions are used as an already cross-fitted anchor and comparator.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import resource
import statistics
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .fusion import (
    BlendSufficient,
    _context_shuffle,
    _correction_template,
    _group_assignments,
    _template_value,
)
from .gr_path import TypewellCurve
from .harness import (
    DataValidationError,
    ErrorAccumulator,
    WellMetric,
    _canonical_json,
    _quantile,
    _sha256,
    _summarize,
    _write_csv,
    _write_json,
)
from .learnability import _load_fold_maps

COMPARATORS = ("last_known_tvt", "e004_geometry_prefix", "e006_nested_fusion")
DIAGNOSTICS = (
    "selfcorr_raw_bounded",
    "crossfit_rmse_grid",
    "duplicate_conservative",
    "shuffled_selfcorr_conservative",
    "oracle_rowwise_best",
)
ELIGIBLE = (
    "visible_fixed_0p10",
    "crossfit_conservative",
    "crossfit_reliability_shrink",
    "crossfit_positive_gate",
)
ALL_CANDIDATES = COMPARATORS + DIAGNOSTICS[:1] + ELIGIBLE[:1] + DIAGNOSTICS[1:2] + ELIGIBLE[1:] + DIAGNOSTICS[2:]
TEMPLATE_POINTS = 64
MISSING = {"", "nan", "NaN", "NA", "N/A", "null", "None"}


@dataclass(frozen=True)
class HorizontalWell:
    well_id: str
    md: tuple[float, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    z: tuple[float, ...]
    gr: tuple[float | None, ...]
    tvt_input: tuple[float | None, ...]
    known_rows: int

    @property
    def hidden_rows(self) -> int:
        return len(self.md) - self.known_rows

    @property
    def last_visible_tvt(self) -> float:
        value = self.tvt_input[self.known_rows - 1]
        assert value is not None
        return float(value)

    @property
    def visible_gr_coverage(self) -> float:
        values = self.gr[: self.known_rows]
        return sum(value is not None for value in values) / max(1, len(values))

    @property
    def hidden_gr_coverage(self) -> float:
        values = self.gr[self.known_rows :]
        return sum(value is not None for value in values) / max(1, len(values))


@dataclass(frozen=True)
class MatchDiagnostic:
    template_states: int
    query_anchors: int
    median_best_distance: float
    median_margin: float
    fallback_reason: str


@dataclass(frozen=True)
class SelfCorrDiagnostic:
    visible_gr_coverage: float
    hidden_gr_coverage: float
    template_states: int
    query_anchors: int
    median_best_distance: float
    median_margin: float
    pseudo_gain: float
    pseudo_base_rmse: float
    pseudo_fixed_rmse: float
    fallback_reason: str
    maximum_absolute_correction: float


@dataclass(frozen=True)
class ReliabilityReference:
    pseudo_gain: tuple[tuple[float, str], ...]
    coverage: tuple[tuple[float, str], ...]
    distance: tuple[tuple[float, str], ...]
    margin: tuple[tuple[float, str], ...]


@dataclass(frozen=True)
class PlacementContext:
    key: str
    scope: str
    label: str
    outer_group: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


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
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def read_horizontal_selfcorr(path: Path) -> HorizontalWell:
    if not path.exists():
        raise DataValidationError(f"missing horizontal well {path}")
    columns: dict[str, list[float | None]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"MD", "X", "Y", "Z", "GR", "TVT_input"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{path}: missing E007 horizontal columns {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            for name in ("MD", "X", "Y", "Z"):
                columns[name].append(_required(row.get(name), field=name, path=path, row=row_number))
            columns["GR"].append(_optional(row.get("GR")))
            columns["TVT_input"].append(_optional(row.get("TVT_input")))
    count = len(columns["MD"])
    if count < 4:
        raise DataValidationError(f"{path}: fewer than four rows")
    md = [float(value) for value in columns["MD"] if value is not None]
    if any(second <= first for first, second in zip(md, md[1:])):
        raise DataValidationError(f"{path}: MD must be strictly increasing")
    known = 0
    hidden_seen = False
    for value in columns["TVT_input"]:
        visible = value is not None
        if visible and hidden_seen:
            raise DataValidationError(f"{path}: TVT_input reappears after hidden suffix begins")
        if visible:
            known += 1
        else:
            hidden_seen = True
    if known < 3 or known >= count:
        raise DataValidationError(f"{path}: requires a visible prefix and non-empty hidden suffix")
    return HorizontalWell(
        well_id=path.name.split("__", 1)[0],
        md=tuple(md),
        x=tuple(float(value) for value in columns["X"] if value is not None),
        y=tuple(float(value) for value in columns["Y"] if value is not None),
        z=tuple(float(value) for value in columns["Z"] if value is not None),
        gr=tuple(columns["GR"]),
        tvt_input=tuple(columns["TVT_input"]),
        known_rows=known,
    )


def validate_e007_config(config: Mapping[str, Any]) -> None:
    if list(config.get("candidate_order", [])) != list(ALL_CANDIDATES):
        raise DataValidationError("E007 candidate order differs from frozen implementation")
    if list(config.get("eligible_candidates", [])) != list(ELIGIBLE):
        raise DataValidationError("E007 eligible candidates differ from frozen implementation")
    if int(config.get("expected_wells", 0)) <= 0:
        raise DataValidationError("E007 expected_wells must be positive")
    signature = str(config.get("data_signature", ""))
    if len(signature) != 64 or any(character not in "0123456789abcdef" for character in signature):
        raise DataValidationError("E007 data signature must be a lowercase SHA-256")
    parent_sha = str(config.get("parents", {}).get("e006_oof_sha256", ""))
    if len(parent_sha) != 64 or any(character not in "0123456789abcdef" for character in parent_sha):
        raise DataValidationError("E007 parent OOF SHA-256 is malformed")
    fold_files = list(config.get("fold_files", []))
    if len(fold_files) != 5 or len(set(fold_files)) != 5:
        raise DataValidationError("E007 requires five unique fold files")
    grid = config.get("placement", {}).get("weight_grid")
    if not isinstance(grid, list) or not grid:
        raise DataValidationError("E007 placement weight grid must be non-empty")
    weights = [float(value) for value in grid]
    if weights != sorted(weights) or len(set(weights)) != len(weights) or weights[0] != 0.0 or weights[-1] != 1.0:
        raise DataValidationError("E007 placement weight grid must be unique, sorted, and include 0 and 1")
    if any(not math.isfinite(value) or not (0.0 <= value <= 1.0) for value in weights):
        raise DataValidationError("E007 placement weights must be finite and in [0,1]")
    fixed = float(config["placement"]["visible_fixed_weight"])
    if fixed not in weights:
        raise DataValidationError("E007 visible fixed weight must be present in the grid")
    selfcorr = config.get("self_correlation", {})
    integer_positive = (
        "minimum_common_fingerprint_features",
        "template_stride_rows",
        "query_stride_rows",
        "local_u_slope_radius_rows",
        "nearest_neighbors",
        "minimum_template_states",
        "minimum_query_anchors",
    )
    for name in integer_positive:
        try:
            value = int(selfcorr[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"E007 self_correlation.{name} is malformed") from exc
        if value <= 0:
            raise DataValidationError(f"E007 self_correlation.{name} must be positive")
    for name in ("maximum_absolute_u_slope_per_row", "correction_soft_cap_ft"):
        value = float(selfcorr.get(name, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise DataValidationError(f"E007 self_correlation.{name} must be positive and finite")
    for name in ("minimum_visible_gr_coverage", "minimum_hidden_gr_coverage"):
        value = float(selfcorr.get(name, math.nan))
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise DataValidationError(f"E007 self_correlation.{name} must be in [0,1]")
    for name in ("fingerprint_mean_radii_rows", "fingerprint_difference_radii_rows"):
        values = selfcorr.get(name)
        if not isinstance(values, list) or not values:
            raise DataValidationError(f"E007 self_correlation.{name} must be non-empty")
        parsed = [int(value) for value in values]
        if parsed != sorted(parsed) or len(set(parsed)) != len(parsed) or any(value < 0 for value in parsed):
            raise DataValidationError(f"E007 self_correlation.{name} must be unique sorted nonnegative integers")
    fractions = [float(value) for value in config.get("visible_only_design", {}).get("pseudo_boundary_fractions", [])]
    if not fractions or any(not math.isfinite(value) or not (0.0 < value < 1.0) for value in fractions):
        raise DataValidationError("E007 pseudo-boundary fractions are malformed")
    if not bool(config.get("selection", {}).get("no_post_score_candidate_additions")) or not bool(config.get("selection", {}).get("no_post_score_threshold_changes")):
        raise DataValidationError("E007 post-score mutation guards must remain enabled")
    deployment = config.get("deployment", {})
    if bool(deployment.get("internet")) or list(deployment.get("external_artifacts", [])):
        raise DataValidationError("E007 deployment must remain offline and self-contained")


def _rolling(values: Sequence[float | None]) -> tuple[list[float], list[int]]:
    sums = [0.0]
    counts = [0]
    for value in values:
        sums.append(sums[-1] + (0.0 if value is None else float(value)))
        counts.append(counts[-1] + int(value is not None))
    return sums, counts


def _window_mean(sums: Sequence[float], counts: Sequence[int], start: int, end: int) -> float | None:
    left = max(0, start)
    right = min(len(sums) - 1, end)
    count = counts[right] - counts[left]
    if count <= 0:
        return None
    return (sums[right] - sums[left]) / count


def _fingerprints(gr: Sequence[float | None], config: Mapping[str, Any]) -> list[tuple[float | None, ...]]:
    finite = [float(value) for value in gr if value is not None]
    if len(finite) < 2:
        return [tuple() for _ in gr]
    center = _mean(finite)
    scale = math.sqrt(sum((value - center) ** 2 for value in finite) / len(finite))
    if scale <= 1e-12 or not math.isfinite(scale):
        return [tuple() for _ in gr]
    normalized = [None if value is None else (float(value) - center) / scale for value in gr]
    sums, counts = _rolling(normalized)
    mean_radii = [int(value) for value in config["fingerprint_mean_radii_rows"]]
    difference_radii = [int(value) for value in config["fingerprint_difference_radii_rows"]]
    output: list[tuple[float | None, ...]] = []
    for index in range(len(normalized)):
        features: list[float | None] = []
        for radius in mean_radii:
            if radius == 0:
                features.append(normalized[index])
            else:
                features.append(_window_mean(sums, counts, index - radius, index + radius + 1))
        for radius in difference_radii:
            left = _window_mean(sums, counts, index - radius, index)
            right = _window_mean(sums, counts, index + 1, index + radius + 1)
            features.append(None if left is None or right is None else right - left)
        output.append(tuple(features))
    return output


def _fingerprint_distance(left: Sequence[float | None], right: Sequence[float | None], minimum_common: int) -> float:
    total = 0.0
    count = 0
    for first, second in zip(left, right):
        if first is None or second is None:
            continue
        difference = float(first) - float(second)
        total += difference * difference
        count += 1
    return math.inf if count < minimum_common else total / count


def _local_u_slope(values: Sequence[float], index: int, radius: int) -> float:
    left = index - radius
    right = index + radius
    if left < 0 or right >= len(values) or right <= left:
        raise DataValidationError("E007 local U slope window crosses the visible boundary")
    slope = (float(values[right]) - float(values[left])) / (right - left)
    if not math.isfinite(slope):
        raise DataValidationError("E007 local U slope is non-finite")
    return slope


def _match_slopes(
    *,
    gr: Sequence[float | None],
    visible_tvt: Sequence[float],
    z: Sequence[float],
    template_rows: int,
    total_rows: int,
    config: Mapping[str, Any],
) -> tuple[list[float], MatchDiagnostic]:
    if template_rows <= 0 or total_rows <= template_rows or total_rows > len(gr) or total_rows > len(z):
        raise DataValidationError("E007 invalid template/query boundary")
    if len(visible_tvt) < template_rows:
        raise DataValidationError("E007 visible TVT does not cover the template")
    radius = int(config["local_u_slope_radius_rows"])
    minimum_common = int(config["minimum_common_fingerprint_features"])
    fingerprints = _fingerprints(gr[:total_rows], config)
    visible_u = [float(visible_tvt[index]) + float(z[index]) for index in range(template_rows)]
    states = [
        index
        for index in range(radius, template_rows - radius, int(config["template_stride_rows"]))
        if sum(value is not None for value in fingerprints[index]) >= minimum_common
    ]
    anchors = list(range(template_rows, total_rows, int(config["query_stride_rows"])))
    if anchors and anchors[-1] != total_rows - 1:
        anchors.append(total_rows - 1)
    if not anchors:
        anchors = [total_rows - 1]
    if len(states) < int(config["minimum_template_states"]):
        return [0.0] * (total_rows - template_rows), MatchDiagnostic(len(states), len(anchors), math.inf, 0.0, "too_few_template_states")
    if len(anchors) < int(config["minimum_query_anchors"]):
        return [0.0] * (total_rows - template_rows), MatchDiagnostic(len(states), len(anchors), math.inf, 0.0, "too_few_query_anchors")
    slopes_by_state = {
        state: _clip(
            _local_u_slope(visible_u, state, radius),
            -float(config["maximum_absolute_u_slope_per_row"]),
            float(config["maximum_absolute_u_slope_per_row"]),
        )
        for state in states
    }
    anchor_slopes: list[float] = []
    best_distances: list[float] = []
    margins: list[float] = []
    neighbors = int(config["nearest_neighbors"])
    for anchor in anchors:
        matches = heapq.nsmallest(
            max(2, neighbors),
            (
                (_fingerprint_distance(fingerprints[anchor], fingerprints[state], minimum_common), state)
                for state in states
            ),
            key=lambda item: (item[0], item[1]),
        )
        matches = [item for item in matches if math.isfinite(item[0])]
        if not matches:
            return [0.0] * (total_rows - template_rows), MatchDiagnostic(len(states), len(anchors), math.inf, 0.0, "no_finite_fingerprint_match")
        selected = matches[:neighbors]
        weights = [math.exp(-min(20.0, distance)) for distance, _ in selected]
        total_weight = sum(weights)
        slope = (
            sum(weight * slopes_by_state[state] for weight, (_, state) in zip(weights, selected)) / total_weight
            if total_weight > 0.0
            else 0.0
        )
        slope = _clip(
            slope,
            -float(config["maximum_absolute_u_slope_per_row"]),
            float(config["maximum_absolute_u_slope_per_row"]),
        )
        anchor_slopes.append(slope)
        best = selected[0][0]
        second = matches[1][0] if len(matches) > 1 else best
        best_distances.append(best)
        margins.append((second - best) / max(1e-12, abs(best)))
    query_slopes: list[float] = []
    position = 0
    for absolute_index in range(template_rows, total_rows):
        while position + 1 < len(anchors) and anchors[position + 1] < absolute_index:
            position += 1
        if position + 1 >= len(anchors):
            value = anchor_slopes[-1]
        elif anchors[position] == absolute_index:
            value = anchor_slopes[position]
        else:
            left_index = anchors[position]
            right_index = anchors[position + 1]
            fraction = (absolute_index - left_index) / max(1, right_index - left_index)
            value = anchor_slopes[position] * (1.0 - fraction) + anchor_slopes[position + 1] * fraction
        query_slopes.append(float(value))
    return query_slopes, MatchDiagnostic(
        len(states),
        len(anchors),
        _median(best_distances),
        _median(margins),
        "",
    )


def _integrate_tvt(
    *,
    last_visible_tvt: float,
    last_visible_z: float,
    hidden_z: Sequence[float],
    slopes: Sequence[float],
) -> list[float]:
    if len(hidden_z) != len(slopes) or not slopes:
        raise DataValidationError("E007 hidden geometry and slope paths differ")
    current_u = float(last_visible_tvt) + float(last_visible_z)
    output: list[float] = []
    for z_value, slope in zip(hidden_z, slopes):
        if not math.isfinite(float(slope)):
            raise DataValidationError("E007 matched slope is non-finite")
        current_u += float(slope)
        output.append(current_u - float(z_value))
    return output


def _bounded_path(base: Sequence[float], raw: Sequence[float], cap: float) -> list[float]:
    if len(base) != len(raw) or not base:
        raise DataValidationError("E007 bounded path input lengths differ")
    if not math.isfinite(float(cap)) or cap <= 0.0:
        raise DataValidationError("E007 correction cap is invalid")
    output = [
        float(anchor) + cap * math.tanh((float(candidate) - float(anchor)) / cap)
        for anchor, candidate in zip(base, raw)
    ]
    if any(not math.isfinite(value) for value in output):
        raise DataValidationError("E007 bounded self-correlation path is non-finite")
    return output


def _pseudo_metrics(well: HorizontalWell, fraction: float, config: Mapping[str, Any]) -> dict[str, float | int | str]:
    visible_tvt = [float(value) for value in well.tvt_input[: well.known_rows] if value is not None]
    count = len(visible_tvt)
    cut = max(200, min(count - 100, round(count * float(fraction))))
    if cut <= 0 or cut >= count:
        return {"rows": 0, "base_sse": 0.0, "fixed_sse": 0.0, "gain": 0.0, "fallback_reason": "invalid_pseudo_boundary"}
    structural = [visible_tvt[index] + float(well.z[index]) for index in range(count)]
    radius = int(config["local_u_slope_radius_rows"])
    prior_indices = list(range(max(radius, cut - radius - 128), cut - radius))
    if not prior_indices:
        return {"rows": count - cut, "base_sse": 0.0, "fixed_sse": 0.0, "gain": 0.0, "fallback_reason": "no_pseudo_prior"}
    prior = _clip(
        _median([_local_u_slope(structural, index, radius) for index in prior_indices]),
        -float(config["maximum_absolute_u_slope_per_row"]),
        float(config["maximum_absolute_u_slope_per_row"]),
    )
    slopes, detail = _match_slopes(
        gr=well.gr[:count],
        visible_tvt=visible_tvt,
        z=well.z[:count],
        template_rows=cut,
        total_rows=count,
        config=config,
    )
    hidden_z = well.z[cut:count]
    current_u = structural[cut - 1]
    base: list[float] = []
    for z_value in hidden_z:
        current_u += prior
        base.append(current_u - float(z_value))
    if detail.fallback_reason:
        fixed = list(base)
    else:
        raw = _integrate_tvt(
            last_visible_tvt=visible_tvt[cut - 1],
            last_visible_z=well.z[cut - 1],
            hidden_z=hidden_z,
            slopes=slopes,
        )
        bounded = _bounded_path(base, raw, float(config["correction_soft_cap_ft"]))
        strength = 0.1
        fixed = [anchor + strength * (candidate - anchor) for anchor, candidate in zip(base, bounded)]
    truth = visible_tvt[cut:count]
    base_sse = sum((prediction - target) ** 2 for prediction, target in zip(base, truth))
    fixed_sse = sum((prediction - target) ** 2 for prediction, target in zip(fixed, truth))
    rows = len(truth)
    base_rmse = math.sqrt(base_sse / max(1, rows))
    fixed_rmse = math.sqrt(fixed_sse / max(1, rows))
    return {
        "rows": rows,
        "base_sse": base_sse,
        "fixed_sse": fixed_sse,
        "base_rmse": base_rmse,
        "fixed_rmse": fixed_rmse,
        "gain": base_rmse - fixed_rmse,
        "fallback_reason": detail.fallback_reason,
    }


def selfcorr_path(
    well: HorizontalWell,
    e006_base: Sequence[float],
    config: Mapping[str, Any],
    pseudo_primary: Mapping[str, Any] | None = None,
) -> tuple[list[float], SelfCorrDiagnostic]:
    if len(e006_base) != well.hidden_rows or not e006_base:
        raise DataValidationError(f"{well.well_id}: E006 base length differs from hidden suffix")
    if any(not math.isfinite(float(value)) for value in e006_base):
        raise DataValidationError(f"{well.well_id}: E006 base contains non-finite values")
    pseudo = dict(pseudo_primary or _pseudo_metrics(well, 0.75, config))
    fallback = ""
    if well.visible_gr_coverage < float(config["minimum_visible_gr_coverage"]):
        fallback = "low_visible_gr_coverage"
    elif well.hidden_gr_coverage < float(config["minimum_hidden_gr_coverage"]):
        fallback = "low_hidden_gr_coverage"
    visible_tvt = [float(value) for value in well.tvt_input[: well.known_rows] if value is not None]
    match = MatchDiagnostic(0, 0, math.inf, 0.0, fallback)
    if not fallback:
        slopes, match = _match_slopes(
            gr=well.gr,
            visible_tvt=visible_tvt,
            z=well.z,
            template_rows=well.known_rows,
            total_rows=len(well.md),
            config=config,
        )
        fallback = match.fallback_reason
    if fallback:
        path = list(float(value) for value in e006_base)
    else:
        raw = _integrate_tvt(
            last_visible_tvt=well.last_visible_tvt,
            last_visible_z=well.z[well.known_rows - 1],
            hidden_z=well.z[well.known_rows :],
            slopes=slopes,
        )
        path = _bounded_path(e006_base, raw, float(config["correction_soft_cap_ft"]))
    maximum = max(abs(float(candidate) - float(anchor)) for anchor, candidate in zip(e006_base, path))
    diagnostic = SelfCorrDiagnostic(
        visible_gr_coverage=well.visible_gr_coverage,
        hidden_gr_coverage=well.hidden_gr_coverage,
        template_states=match.template_states,
        query_anchors=match.query_anchors,
        median_best_distance=float(match.median_best_distance),
        median_margin=float(match.median_margin),
        pseudo_gain=float(pseudo.get("gain", 0.0)),
        pseudo_base_rmse=float(pseudo.get("base_rmse", 0.0)),
        pseudo_fixed_rmse=float(pseudo.get("fixed_rmse", 0.0)),
        fallback_reason=fallback,
        maximum_absolute_correction=maximum,
    )
    _validate_diagnostic(diagnostic, well.well_id)
    return path, diagnostic


def _validate_diagnostic(detail: SelfCorrDiagnostic, well_id: str) -> None:
    finite_values = (
        detail.visible_gr_coverage,
        detail.hidden_gr_coverage,
        detail.median_margin,
        detail.pseudo_gain,
        detail.pseudo_base_rmse,
        detail.pseudo_fixed_rmse,
        detail.maximum_absolute_correction,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise DataValidationError(f"{well_id}: non-finite E007 diagnostic")
    if not (0.0 <= detail.visible_gr_coverage <= 1.0 and 0.0 <= detail.hidden_gr_coverage <= 1.0):
        raise DataValidationError(f"{well_id}: invalid GR coverage")
    if detail.template_states < 0 or detail.query_anchors < 0 or detail.maximum_absolute_correction < 0.0:
        raise DataValidationError(f"{well_id}: invalid E007 diagnostic counts")
    if not detail.fallback_reason and not math.isfinite(float(detail.median_best_distance)):
        raise DataValidationError(f"{well_id}: active E007 path has non-finite match distance")


def _make_reference(diagnostics: Mapping[str, SelfCorrDiagnostic]) -> ReliabilityReference:
    if not diagnostics:
        raise DataValidationError("E007 reliability reference cannot be empty")
    for well_id, detail in diagnostics.items():
        _validate_diagnostic(detail, well_id)

    def pairs(getter: Any) -> tuple[tuple[float, str], ...]:
        return tuple(sorted((float(getter(detail)), well_id) for well_id, detail in diagnostics.items()))

    return ReliabilityReference(
        pseudo_gain=pairs(lambda item: item.pseudo_gain),
        coverage=pairs(lambda item: item.hidden_gr_coverage),
        distance=pairs(lambda item: item.median_best_distance if math.isfinite(item.median_best_distance) else 1e30),
        margin=pairs(lambda item: item.median_margin),
    )


def _percentile(reference: Sequence[tuple[float, str]], value: float, well_id: str, *, higher_better: bool) -> float:
    if len(reference) <= 1:
        return 1.0
    target = (float(value), str(well_id))
    low = 0
    high = len(reference)
    while low < high:
        middle = (low + high) // 2
        if reference[middle] < target:
            low = middle + 1
        else:
            high = middle
    rank = low / (len(reference) - 1)
    return _clip(rank if higher_better else 1.0 - rank, 0.0, 1.0)


def _reliability(detail: SelfCorrDiagnostic, reference: ReliabilityReference, well_id: str) -> float:
    distance = detail.median_best_distance if math.isfinite(detail.median_best_distance) else 1e30
    components = [
        _percentile(reference.pseudo_gain, detail.pseudo_gain, well_id, higher_better=True),
        _percentile(reference.coverage, detail.hidden_gr_coverage, well_id, higher_better=True),
        _percentile(reference.distance, distance, well_id, higher_better=False),
        _percentile(reference.margin, detail.median_margin, well_id, higher_better=True),
    ]
    return _clip(_mean(components), 0.0, 1.0)


def _effective_weight(
    candidate: str,
    base_weight: float,
    detail: SelfCorrDiagnostic,
    reference: ReliabilityReference,
    well_id: str,
    config: Mapping[str, Any],
) -> float:
    if not math.isfinite(float(base_weight)):
        raise DataValidationError(f"{well_id}: E007 base weight is non-finite")
    _validate_diagnostic(detail, well_id)
    base = _clip(float(base_weight), 0.0, 1.0)
    if candidate in {"crossfit_conservative", "duplicate_conservative"}:
        return base
    score = _reliability(detail, reference, well_id)
    if candidate == "crossfit_reliability_shrink":
        return base * score
    if candidate == "crossfit_positive_gate":
        gate = config["reliability"]
        return base if (
            detail.pseudo_gain > 0.0
            and detail.hidden_gr_coverage >= float(gate["positive_gate_minimum_hidden_gr_coverage"])
            and score >= float(gate["positive_gate_minimum_score"])
        ) else 0.0
    raise DataValidationError(f"unknown E007 reliability candidate {candidate}")


def _metric_for_path(well_id: str, truth: Sequence[float], prediction: Sequence[float]) -> WellMetric:
    if len(truth) != len(prediction) or not truth:
        raise DataValidationError(f"{well_id}: path metric lengths differ")
    accumulator = ErrorAccumulator()
    for index, (target, value) in enumerate(zip(truth, prediction)):
        if not math.isfinite(float(target)) or not math.isfinite(float(value)):
            raise DataValidationError(f"{well_id}: non-finite path metric input")
        accumulator.add(float(value) - float(target), float(index))
    return accumulator.finalize(well_id)


def _oracle_metric(well_id: str, truth: Sequence[float], base: Sequence[float], candidate: Sequence[float]) -> WellMetric:
    chosen = [
        float(anchor) if abs(float(anchor) - float(target)) <= abs(float(value) - float(target)) else float(value)
        for target, anchor, value in zip(truth, base, candidate)
    ]
    return _metric_for_path(well_id, truth, chosen)


def _parent_wells(path: Path) -> Iterator[dict[str, Any]]:
    required = {
        "id",
        "well_id",
        "row_index",
        "hidden_index",
        "target",
        "last_known_tvt",
        "e004_geometry_prefix",
        "nested_conservative_grid",
    }
    seen_wells: set[str] = set()
    current: dict[str, Any] | None = None
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{path}: missing E007 parent columns {sorted(missing)}")
        for row in reader:
            well_id = str(row["well_id"])
            if current is None or current["well_id"] != well_id:
                if current is not None:
                    seen_wells.add(current["well_id"])
                    yield current
                if well_id in seen_wells:
                    raise DataValidationError(f"{path}: parent OOF well rows are not contiguous")
                current = {
                    "well_id": well_id,
                    "ids": [],
                    "row_index": [],
                    "hidden_index": [],
                    "target": [],
                    "last": [],
                    "e004": [],
                    "e006": [],
                }
            assert current is not None
            hidden_index = int(row["hidden_index"])
            if hidden_index != len(current["hidden_index"]):
                raise DataValidationError(f"{well_id}: parent hidden indices are not contiguous")
            numeric = {
                "target": float(row["target"]),
                "last": float(row["last_known_tvt"]),
                "e004": float(row["e004_geometry_prefix"]),
                "e006": float(row["nested_conservative_grid"]),
            }
            if not all(math.isfinite(value) for value in numeric.values()):
                raise DataValidationError(f"{well_id}: parent OOF contains non-finite values")
            current["ids"].append(str(row["id"]))
            current["row_index"].append(int(row["row_index"]))
            current["hidden_index"].append(hidden_index)
            current["target"].append(numeric["target"])
            current["last"].append(numeric["last"])
            current["e004"].append(numeric["e004"])
            current["e006"].append(numeric["e006"])
    if current is not None:
        yield current


def _select_weights(stats: Mapping[str, BlendSufficient], config: Mapping[str, Any]) -> tuple[float, float, list[dict[str, Any]]]:
    if not stats:
        raise DataValidationError("E007 placement training set is empty")
    grid = [float(value) for value in config["placement"]["weight_grid"]]
    summaries = {weight: _summarize([stats[well_id].metric(weight) for well_id in sorted(stats)]) for weight in grid}
    base = summaries[0.0]
    rmse_weight = min(grid, key=lambda weight: (float(summaries[weight]["rmse"]), weight))
    passing = [
        weight
        for weight in grid
        if float(base["rmse"]) - float(summaries[weight]["rmse"]) >= float(config["placement"]["inner_minimum_gain_vs_e006"])
        and float(summaries[weight]["p90_well_rmse"]) - float(base["p90_well_rmse"]) <= float(config["placement"]["inner_maximum_p90_deterioration"])
        and float(summaries[weight]["worst_5pct_sse_share"]) - float(base["worst_5pct_sse_share"]) <= float(config["placement"]["inner_maximum_worst5_sse_share_increase"])
    ]
    if passing:
        best = min(passing, key=lambda weight: (float(summaries[weight]["rmse"]), weight))
        threshold = float(summaries[best]["rmse"]) + float(config["placement"]["conservative_rmse_slack"])
        conservative = min(weight for weight in passing if float(summaries[weight]["rmse"]) <= threshold)
    else:
        conservative = float(config["placement"]["fallback_weight"])
    rows = [
        {
            "weight": weight,
            "passes_conservative_constraints": weight in passing,
            "selected_rmse": weight == rmse_weight,
            "selected_conservative": weight == conservative,
            **summaries[weight],
        }
        for weight in grid
    ]
    return rmse_weight, conservative, rows


def _build_contexts(
    root: Path,
    well_ids: Sequence[str],
    fold_maps: Sequence[Mapping[str, Any]],
    spatial_assignments: Mapping[str, int],
    typewell_assignments: Mapping[str, int],
    config: Mapping[str, Any],
) -> tuple[list[PlacementContext], list[dict[str, Any]]]:
    contexts: list[PlacementContext] = []
    audits: list[dict[str, Any]] = []
    for fold_map in fold_maps:
        version = str(fold_map["version"])
        assignments = {well_id: int(value) for well_id, value in fold_map["assignments"].items()}
        for outer in range(int(fold_map["n_folds"])):
            test_ids = tuple(sorted(well_id for well_id in well_ids if assignments[well_id] == outer))
            train_ids = tuple(sorted(set(well_ids) - set(test_ids)))
            key = f"repeated:{version}:{outer}"
            contexts.append(PlacementContext(key, "repeated", version, outer, train_ids, test_ids))
            audits.append({
                "context": key,
                "scope": "repeated",
                "label": version,
                "outer_group": outer,
                "outer_train_wells": len(train_ids),
                "outer_test_wells": len(test_ids),
                "train_test_overlap": len(set(train_ids) & set(test_ids)),
                "pass": bool(train_ids) and bool(test_ids) and not (set(train_ids) & set(test_ids)),
            })
    for scope, assignments, groups in (
        ("spatial", spatial_assignments, int(config["stress"]["spatial_bins"])),
        ("typewell", typewell_assignments, int(config["stress"]["typewell_clusters"])),
    ):
        for outer in range(groups):
            test_ids = tuple(sorted(well_id for well_id in well_ids if int(assignments[well_id]) == outer))
            train_ids = tuple(sorted(set(well_ids) - set(test_ids)))
            key = f"{scope}:{outer}"
            contexts.append(PlacementContext(key, scope, scope, outer, train_ids, test_ids))
            audits.append({
                "context": key,
                "scope": scope,
                "label": scope,
                "outer_group": outer,
                "outer_train_wells": len(train_ids),
                "outer_test_wells": len(test_ids),
                "train_test_overlap": len(set(train_ids) & set(test_ids)),
                "pass": bool(train_ids) and bool(test_ids) and not (set(train_ids) & set(test_ids)),
            })
    if not all(bool(row["pass"]) for row in audits):
        raise DataValidationError("E007 placement membership audit failed")
    return contexts, audits


def _residual_correlation(stats: Mapping[str, BlendSufficient], weights: Mapping[str, float]) -> float | None:
    count = 0
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    for well_id, sufficient in stats.items():
        weight = float(weights[well_id])
        count += sufficient.rows
        x = sufficient.sum_base + weight * sufficient.sum_delta
        y = sufficient.sum_base
        x2 = sufficient.sum_base_sq + 2.0 * weight * sufficient.sum_base_delta + weight * weight * sufficient.sum_delta_sq
        y2 = sufficient.sum_base_sq
        xy = sufficient.sum_base_sq + weight * sufficient.sum_base_delta
        sum_x += x
        sum_y += y
        sum_x2 += x2
        sum_y2 += y2
        sum_xy += xy
    if count <= 1:
        return None
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_x2 - sum_x * sum_x / count
    variance_y = sum_y2 - sum_y * sum_y / count
    denominator = math.sqrt(max(0.0, variance_x) * max(0.0, variance_y))
    return None if denominator <= 0.0 else covariance / denominator


def _write_gzip_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: format(value, ".8f") if isinstance(value, float) else value for key, value in row.items()})


def _mark_candidate_rows(rows: Sequence[dict[str, Any]], passing: Sequence[str]) -> None:
    allowed = set(passing)
    for row in rows:
        row["passed_all_gates"] = row["candidate"] in allowed


def run_e007(
    *,
    root: Path,
    train_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: Mapping[str, Any],
    code_sha: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = root.resolve()
    train_dir = train_dir.resolve()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    validate_e007_config(config)
    parent_path = root / str(config["parents"]["e006_oof_path"])
    if not parent_path.exists():
        raise DataValidationError(f"missing E006 parent OOF {parent_path}")
    parent_sha = _sha256(parent_path)
    if parent_sha != str(config["parents"]["e006_oof_sha256"]):
        raise DataValidationError("E007 parent OOF hash mismatch")
    e006_summary = json.loads((root / str(config["parents"]["e006_summary"])).read_text(encoding="utf-8"))
    if str(e006_summary["data"]["data_signature"]) != str(config["data_signature"]):
        raise DataValidationError("E007/E006 data signature mismatch")
    if e006_summary.get("status") != "deployment_ready" or not bool(e006_summary.get("deployment", {}).get("deployment_ready")):
        raise DataValidationError("E007 requires the audited deployment-ready E006 parent")

    horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
    if len(horizontal_files) != int(config["expected_wells"]):
        raise DataValidationError("E007 horizontal well count mismatch")
    horizontal_by_id = {path.name.split("__", 1)[0]: path for path in horizontal_files}
    well_ids = sorted(horizontal_by_id)

    stats: dict[str, BlendSufficient] = {}
    shuffled_stats: dict[str, BlendSufficient] = {}
    comparator_metrics: dict[str, dict[str, WellMetric]] = {candidate: {} for candidate in COMPARATORS}
    oracle_metrics: dict[str, WellMetric] = {}
    diagnostics: dict[str, SelfCorrDiagnostic] = {}
    delta_paths: dict[str, array] = {}
    templates: dict[str, tuple[float, ...]] = {}
    spatial_values: dict[str, tuple[float, ...]] = {}
    typewell_values: dict[str, tuple[float, ...]] = {}
    hidden_rows: dict[str, int] = {}
    pseudo_aggregate = {
        float(fraction): {"rows": 0, "base_sse": 0.0, "fixed_sse": 0.0, "fallback_wells": 0}
        for fraction in config["visible_only_design"]["pseudo_boundary_fractions"]
    }
    parent_rows = 0
    maximum_correction = 0.0

    for chunk in _parent_wells(parent_path):
        well_id = str(chunk["well_id"])
        if well_id not in horizontal_by_id:
            raise DataValidationError(f"E007 parent OOF has unknown well {well_id}")
        well = read_horizontal_selfcorr(horizontal_by_id[well_id])
        if len(chunk["target"]) != well.hidden_rows:
            raise DataValidationError(f"{well_id}: parent OOF length differs from horizontal suffix")
        if chunk["row_index"] != list(range(well.known_rows, len(well.md))):
            raise DataValidationError(f"{well_id}: parent OOF row indices differ from horizontal suffix")
        if chunk["ids"] != [f"{well_id}_{index}" for index in range(well.known_rows, len(well.md))]:
            raise DataValidationError(f"{well_id}: parent OOF IDs differ from horizontal suffix")
        pseudo_by_fraction = {
            fraction: _pseudo_metrics(well, fraction, config["self_correlation"])
            for fraction in pseudo_aggregate
        }
        for fraction, detail in pseudo_by_fraction.items():
            aggregate = pseudo_aggregate[fraction]
            aggregate["rows"] += int(detail["rows"])
            aggregate["base_sse"] += float(detail["base_sse"])
            aggregate["fixed_sse"] += float(detail["fixed_sse"])
            aggregate["fallback_wells"] += int(bool(detail["fallback_reason"]))
        primary = pseudo_by_fraction.get(0.75) or next(iter(pseudo_by_fraction.values()))
        selfcorr, detail = selfcorr_path(well, chunk["e006"], config["self_correlation"], primary)
        delta = array("d", (float(value) - float(anchor) for anchor, value in zip(chunk["e006"], selfcorr)))
        delta_paths[well_id] = delta
        templates[well_id] = _correction_template(chunk["e006"], selfcorr, TEMPLATE_POINTS)
        diagnostics[well_id] = detail
        stats[well_id] = BlendSufficient.from_paths(well_id, chunk["target"], chunk["e006"], selfcorr)
        comparator_metrics["last_known_tvt"][well_id] = _metric_for_path(well_id, chunk["target"], chunk["last"])
        comparator_metrics["e004_geometry_prefix"][well_id] = _metric_for_path(well_id, chunk["target"], chunk["e004"])
        comparator_metrics["e006_nested_fusion"][well_id] = stats[well_id].metric(0.0)
        oracle_metrics[well_id] = _oracle_metric(well_id, chunk["target"], chunk["e006"], selfcorr)
        spatial_values[well_id] = (0.5 * (well.x[0] + well.x[-1]), 0.5 * (well.y[0] + well.y[-1]))
        curve = TypewellCurve.read(train_dir / f"{well_id}__typewell.csv")
        typewell_values[well_id] = (curve.gr_mean, curve.gr_std, curve.maximum_tvt - curve.minimum_tvt)
        hidden_rows[well_id] = well.hidden_rows
        parent_rows += well.hidden_rows
        maximum_correction = max(maximum_correction, detail.maximum_absolute_correction)
    if set(stats) != set(well_ids):
        raise DataValidationError("E007 parent OOF did not cover all wells")

    global_shuffle = _context_shuffle(well_ids, "e007-global-template-shuffle")
    for chunk in _parent_wells(parent_path):
        well_id = str(chunk["well_id"])
        source = global_shuffle[well_id]
        template = templates[source]
        shuffled = [
            float(anchor) + _template_value(template, index, len(chunk["e006"]))
            for index, anchor in enumerate(chunk["e006"])
        ]
        shuffled_stats[well_id] = BlendSufficient.from_paths(well_id, chunk["target"], chunk["e006"], shuffled)

    folds = _load_fold_maps(root, config["fold_files"], str(config["data_signature"]))
    spatial_assignments = _group_assignments(spatial_values, int(config["stress"]["spatial_bins"]))
    typewell_assignments = _group_assignments(typewell_values, int(config["stress"]["typewell_clusters"]))
    contexts, membership_rows = _build_contexts(root, well_ids, folds, spatial_assignments, typewell_assignments, config)
    selections: dict[str, dict[str, Any]] = {}
    inner_grid_rows: list[dict[str, Any]] = []
    context_weights: dict[str, dict[str, dict[str, float]]] = {
        context.key: {candidate: {} for candidate in ELIGIBLE} for context in contexts
    }
    for context in contexts:
        train_stats = {well_id: stats[well_id] for well_id in context.train_ids}
        rmse_weight, conservative_weight, rows = _select_weights(train_stats, config)
        shuffled_rmse_weight, shuffled_weight, _ = _select_weights(
            {well_id: shuffled_stats[well_id] for well_id in context.train_ids},
            config,
        )
        reference = _make_reference({well_id: diagnostics[well_id] for well_id in context.train_ids})
        selections[context.key] = {
            "rmse_weight": rmse_weight,
            "conservative_weight": conservative_weight,
            "shuffled_rmse_weight": shuffled_rmse_weight,
            "shuffled_weight": shuffled_weight,
            "reference": reference,
        }
        for row in rows:
            inner_grid_rows.append({
                "context": context.key,
                "scope": context.scope,
                "label": context.label,
                "outer_group": context.outer_group,
                **row,
            })
        for well_id in context.test_ids:
            context_weights[context.key]["visible_fixed_0p10"][well_id] = float(config["placement"]["visible_fixed_weight"])
            context_weights[context.key]["crossfit_conservative"][well_id] = conservative_weight
            context_weights[context.key]["crossfit_reliability_shrink"][well_id] = _effective_weight(
                "crossfit_reliability_shrink", conservative_weight, diagnostics[well_id], reference, well_id, config
            )
            context_weights[context.key]["crossfit_positive_gate"][well_id] = _effective_weight(
                "crossfit_positive_gate", conservative_weight, diagnostics[well_id], reference, well_id, config
            )

    outer_metrics: dict[str, dict[str, dict[str, WellMetric]]] = {
        context.key: {candidate: {} for candidate in ALL_CANDIDATES} for context in contexts
    }
    outer_cell_rows: list[dict[str, Any]] = []
    for context in contexts:
        selection = selections[context.key]
        for well_id in context.test_ids:
            outer_metrics[context.key]["last_known_tvt"][well_id] = comparator_metrics["last_known_tvt"][well_id]
            outer_metrics[context.key]["e004_geometry_prefix"][well_id] = comparator_metrics["e004_geometry_prefix"][well_id]
            outer_metrics[context.key]["e006_nested_fusion"][well_id] = comparator_metrics["e006_nested_fusion"][well_id]
            outer_metrics[context.key]["selfcorr_raw_bounded"][well_id] = stats[well_id].metric(1.0)
            outer_metrics[context.key]["visible_fixed_0p10"][well_id] = stats[well_id].metric(float(config["placement"]["visible_fixed_weight"]))
            outer_metrics[context.key]["crossfit_rmse_grid"][well_id] = stats[well_id].metric(float(selection["rmse_weight"]))
            for candidate in ELIGIBLE[1:]:
                outer_metrics[context.key][candidate][well_id] = stats[well_id].metric(context_weights[context.key][candidate][well_id])
            outer_metrics[context.key]["duplicate_conservative"][well_id] = stats[well_id].metric(float(selection["conservative_weight"]))
            outer_metrics[context.key]["shuffled_selfcorr_conservative"][well_id] = shuffled_stats[well_id].metric(float(selection["shuffled_weight"]))
            outer_metrics[context.key]["oracle_rowwise_best"][well_id] = oracle_metrics[well_id]
        for candidate in ALL_CANDIDATES:
            outer_cell_rows.append({
                "context": context.key,
                "scope": context.scope,
                "label": context.label,
                "outer_group": context.outer_group,
                "candidate": candidate,
                "training_rmse_weight": selection["rmse_weight"],
                "training_conservative_weight": selection["conservative_weight"],
                "training_shuffled_weight": selection["shuffled_weight"],
                **_summarize([outer_metrics[context.key][candidate][well_id] for well_id in context.test_ids]),
            })
    cell_lookup = {(row["context"], row["candidate"]): row for row in outer_cell_rows}

    map_rows: list[dict[str, Any]] = []
    for fold_map in folds:
        version = str(fold_map["version"])
        keys = [f"repeated:{version}:{fold}" for fold in range(int(fold_map["n_folds"]))]
        for candidate in ALL_CANDIDATES:
            metrics: list[WellMetric] = []
            for key in keys:
                metrics.extend(outer_metrics[key][candidate].values())
            map_rows.append({"map": version, "candidate": candidate, **_summarize(metrics)})
    map_lookup = {(row["map"], row["candidate"]): row for row in map_rows}

    stress_rows: list[dict[str, Any]] = []
    for context in contexts:
        if context.scope == "repeated":
            continue
        for candidate in ALL_CANDIDATES:
            stress_rows.append({
                "stress": context.scope,
                "group": context.outer_group,
                "candidate": candidate,
                **_summarize(list(outer_metrics[context.key][candidate].values())),
            })
    stress_lookup = {(row["stress"], int(row["group"]), row["candidate"]): row for row in stress_rows}

    repeated_contexts = [context for context in contexts if context.scope == "repeated"]
    final_weights: dict[str, dict[str, float]] = {candidate: {} for candidate in ELIGIBLE}
    final_rmse_weights: dict[str, float] = {}
    final_duplicate_weights: dict[str, float] = {}
    final_shuffled_weights: dict[str, float] = {}
    for well_id in well_ids:
        keys = [
            f"repeated:{fold['version']}:{int(fold['assignments'][well_id])}"
            for fold in folds
        ]
        for candidate in ELIGIBLE:
            final_weights[candidate][well_id] = _mean([context_weights[key][candidate][well_id] for key in keys])
        final_rmse_weights[well_id] = _mean([float(selections[key]["rmse_weight"]) for key in keys])
        final_duplicate_weights[well_id] = _mean([float(selections[key]["conservative_weight"]) for key in keys])
        final_shuffled_weights[well_id] = _mean([float(selections[key]["shuffled_weight"]) for key in keys])

    averaged_metrics: dict[str, dict[str, WellMetric]] = {candidate: {} for candidate in ALL_CANDIDATES}
    for well_id in well_ids:
        averaged_metrics["last_known_tvt"][well_id] = comparator_metrics["last_known_tvt"][well_id]
        averaged_metrics["e004_geometry_prefix"][well_id] = comparator_metrics["e004_geometry_prefix"][well_id]
        averaged_metrics["e006_nested_fusion"][well_id] = comparator_metrics["e006_nested_fusion"][well_id]
        averaged_metrics["selfcorr_raw_bounded"][well_id] = stats[well_id].metric(1.0)
        averaged_metrics["visible_fixed_0p10"][well_id] = stats[well_id].metric(final_weights["visible_fixed_0p10"][well_id])
        averaged_metrics["crossfit_rmse_grid"][well_id] = stats[well_id].metric(final_rmse_weights[well_id])
        for candidate in ELIGIBLE[1:]:
            averaged_metrics[candidate][well_id] = stats[well_id].metric(final_weights[candidate][well_id])
        averaged_metrics["duplicate_conservative"][well_id] = stats[well_id].metric(final_duplicate_weights[well_id])
        averaged_metrics["shuffled_selfcorr_conservative"][well_id] = shuffled_stats[well_id].metric(final_shuffled_weights[well_id])
        averaged_metrics["oracle_rowwise_best"][well_id] = oracle_metrics[well_id]
    summaries = {
        candidate: _summarize([averaged_metrics[candidate][well_id] for well_id in well_ids])
        for candidate in ALL_CANDIDATES
    }
    correlations = {
        "selfcorr_raw_bounded": _residual_correlation(stats, {well_id: 1.0 for well_id in well_ids}),
        **{candidate: _residual_correlation(stats, final_weights[candidate]) for candidate in ELIGIBLE},
    }

    long_threshold = float(_quantile([float(value) for value in hidden_rows.values()], float(config["stress"]["long_suffix_quantile"])))
    missing_values = {well_id: 1.0 - diagnostics[well_id].hidden_gr_coverage for well_id in well_ids}
    missing_threshold = float(_quantile(list(missing_values.values()), float(config["stress"]["high_missing_gr_quantile"])))
    pseudo_threshold = float(_quantile([diagnostics[well_id].pseudo_gain for well_id in well_ids], float(config["stress"]["poor_pseudo_gain_quantile"])))
    margin_threshold = float(_quantile([diagnostics[well_id].median_margin for well_id in well_ids], float(config["stress"]["low_match_margin_quantile"])))
    special_sets = {
        "long_suffix": [well_id for well_id in well_ids if hidden_rows[well_id] >= long_threshold],
        "high_gr_missingness": [well_id for well_id in well_ids if missing_values[well_id] >= missing_threshold],
        "poor_visible_pseudo_gain": [well_id for well_id in well_ids if diagnostics[well_id].pseudo_gain <= pseudo_threshold],
        "low_match_margin": [well_id for well_id in well_ids if diagnostics[well_id].median_margin <= margin_threshold],
    }
    special_rows: list[dict[str, Any]] = []
    for name, ids in special_sets.items():
        for candidate in ALL_CANDIDATES:
            special_rows.append({
                "slice": name,
                "candidate": candidate,
                **_summarize([averaged_metrics[candidate][well_id] for well_id in ids]),
            })
    special_lookup = {(row["slice"], row["candidate"]): row for row in special_rows}

    weight_rows: list[dict[str, Any]] = []
    weight_stability: dict[str, dict[str, Any]] = {}
    for candidate in ELIGIBLE:
        medians: list[float] = []
        by_map: dict[str, list[float]] = defaultdict(list)
        for context in repeated_contexts:
            values = [context_weights[context.key][candidate][well_id] for well_id in context.test_ids]
            median_weight = _median(values)
            medians.append(median_weight)
            by_map[context.label].append(median_weight)
            weight_rows.append({
                "context": context.key,
                "map": context.label,
                "outer_group": context.outer_group,
                "candidate": candidate,
                "training_base_weight": selections[context.key]["conservative_weight"],
                "median_effective_weight": median_weight,
                "minimum_effective_weight": min(values) if values else 0.0,
                "maximum_effective_weight": max(values) if values else 0.0,
                "nonzero_wells": sum(value > 0.0 for value in values),
                "test_wells": len(values),
            })
        weight_stability[candidate] = {
            "nonzero_outer_cells": sum(value > 0.0 for value in medians),
            "maps_with_nonzero_median_weight": sum(_median(values) > 0.0 for values in by_map.values()),
            "outer_weight_range": max(medians) - min(medians) if medians else 0.0,
            "median_outer_weight": _median(medians),
        }

    e006_rmse = float(summaries["e006_nested_fusion"]["rmse"])
    last_rmse = float(summaries["last_known_tvt"]["rmse"])
    map_wins: dict[str, int] = {}
    outer_cell_wins: dict[str, int] = {}
    mean_map_rmse: dict[str, float] = {}
    for candidate in ELIGIBLE:
        gains = [
            float(map_lookup[(str(fold["version"]), "e006_nested_fusion")]["rmse"])
            - float(map_lookup[(str(fold["version"]), candidate)]["rmse"])
            for fold in folds
        ]
        map_wins[candidate] = sum(gain >= float(config["promotion"]["minimum_map_gain"]) for gain in gains)
        outer_cell_wins[candidate] = sum(
            float(cell_lookup[(context.key, candidate)]["rmse"])
            < float(cell_lookup[(context.key, "e006_nested_fusion")]["rmse"])
            for context in repeated_contexts
        )
        mean_map_rmse[candidate] = _mean([
            float(map_lookup[(str(fold["version"]), candidate)]["rmse"])
            for fold in folds
        ])

    visible_design_rows: list[dict[str, Any]] = []
    visible_design_pass = True
    for fraction, detail in sorted(pseudo_aggregate.items()):
        rows = int(detail["rows"])
        base_rmse = math.sqrt(float(detail["base_sse"]) / max(1, rows))
        fixed_rmse = math.sqrt(float(detail["fixed_sse"]) / max(1, rows))
        gain = base_rmse - fixed_rmse
        passed = gain > 0.0
        visible_design_pass = visible_design_pass and passed
        visible_design_rows.append({
            "pseudo_boundary_fraction": fraction,
            "rows_scored": rows,
            "base_rmse": base_rmse,
            "fixed_0p10_rmse": fixed_rmse,
            "gain": gain,
            "fallback_wells": detail["fallback_wells"],
            "pass": passed,
        })

    zero_prediction_delta = max(
        abs(0.0 * float(delta))
        for well_id in well_ids
        for delta in delta_paths[well_id]
    )
    duplicate_prediction_delta = max(
        abs(final_duplicate_weights[well_id] - final_weights["crossfit_conservative"][well_id])
        * max(abs(float(delta)) for delta in delta_paths[well_id])
        for well_id in well_ids
    )

    parent_control = {
        "pass": parent_sha == str(config["parents"]["e006_oof_sha256"])
        and abs(e006_rmse - float(config["parents"]["e006_expected_rmse"])) <= float(config["controls"]["parent_oof_rmse_tolerance"])
        and abs(float(summaries["e004_geometry_prefix"]["rmse"]) - float(config["parents"]["e004_expected_rmse"])) <= float(config["controls"]["parent_oof_rmse_tolerance"])
        and abs(last_rmse - float(config["parents"]["last_known_expected_rmse"])) <= float(config["controls"]["parent_oof_rmse_tolerance"]),
        "sha256": parent_sha,
        "expected_sha256": config["parents"]["e006_oof_sha256"],
        "rows": parent_rows,
        "e006_rmse": e006_rmse,
        "expected_e006_rmse": config["parents"]["e006_expected_rmse"],
        "e004_rmse": summaries["e004_geometry_prefix"]["rmse"],
        "last_known_rmse": last_rmse,
    }
    raw_correlation = correlations["selfcorr_raw_bounded"]
    controls: dict[str, dict[str, Any]] = {
        "data_integrity": {"pass": len(well_ids) == int(config["expected_wells"]) and str(e006_summary["data"]["data_signature"]) == str(config["data_signature"]), "wells": len(well_ids), "data_signature": e006_summary["data"]["data_signature"]},
        "parent_oof": parent_control,
        "placement_membership": {"pass": all(bool(row["pass"]) for row in membership_rows), "contexts": len(membership_rows)},
        "visible_only_design": {"pass": visible_design_pass, "fractions": len(visible_design_rows)},
        "zero_weight_noop": {
            "pass": zero_prediction_delta <= float(config["controls"]["zero_weight_maximum_prediction_delta"]),
            "maximum_prediction_delta": zero_prediction_delta,
        },
        "duplicate_placement": {
            "pass": duplicate_prediction_delta <= float(config["controls"]["duplicate_maximum_prediction_delta"]),
            "maximum_prediction_delta": duplicate_prediction_delta,
        },
        "correction_soft_cap": {"pass": maximum_correction <= float(config["self_correlation"]["correction_soft_cap_ft"]) + 1e-12, "maximum_absolute_correction": maximum_correction, "cap": config["self_correlation"]["correction_soft_cap_ft"]},
        "raw_residual_diversity": {"pass": raw_correlation is not None and float(raw_correlation) <= float(config["controls"]["maximum_raw_residual_correlation_to_e006"]), "residual_correlation": raw_correlation, "maximum": config["controls"]["maximum_raw_residual_correlation_to_e006"]},
        "shuffled_selfcorr": {
            "pass": float(summaries["shuffled_selfcorr_conservative"]["rmse"]) - float(summaries["crossfit_conservative"]["rmse"]) >= float(config["controls"]["shuffled_minimum_loss_vs_unshuffled"])
            and e006_rmse - float(summaries["shuffled_selfcorr_conservative"]["rmse"]) <= float(config["controls"]["shuffled_maximum_gain_vs_e006"]),
            "shuffled_rmse": summaries["shuffled_selfcorr_conservative"]["rmse"],
            "unshuffled_rmse": summaries["crossfit_conservative"]["rmse"],
            "gain_vs_e006": e006_rmse - float(summaries["shuffled_selfcorr_conservative"]["rmse"]),
        },
        "oracle_positive": {
            "pass": min(float(summaries[candidate]["rmse"]) for candidate in ELIGIBLE) - float(summaries["oracle_rowwise_best"]["rmse"]) >= float(config["controls"]["oracle_minimum_gain_vs_best_legal"]),
            "oracle_rmse": summaries["oracle_rowwise_best"]["rmse"],
            "best_legal_rmse": min(float(summaries[candidate]["rmse"]) for candidate in ELIGIBLE),
            "eligible": False,
        },
        "oof_identity": {"pass": parent_rows == int(e006_summary["data"]["hidden_rows"]), "rows": parent_rows},
    }

    promotion = config["promotion"]
    gates_by_candidate: dict[str, dict[str, bool]] = {}
    for candidate in ELIGIBLE:
        selected = summaries[candidate]
        spatial_min_gain = min(
            float(stress_lookup[("spatial", group, "e006_nested_fusion")]["rmse"])
            - float(stress_lookup[("spatial", group, candidate)]["rmse"])
            for group in range(int(config["stress"]["spatial_bins"]))
        )
        typewell_min_gain = min(
            float(stress_lookup[("typewell", group, "e006_nested_fusion")]["rmse"])
            - float(stress_lookup[("typewell", group, candidate)]["rmse"])
            for group in range(int(config["stress"]["typewell_clusters"]))
        )
        special_max_deterioration = max(
            float(special_lookup[(name, candidate)]["rmse"])
            - float(special_lookup[(name, "e006_nested_fusion")]["rmse"])
            for name in special_sets
        )
        stability = weight_stability[candidate]
        gates_by_candidate[candidate] = {
            "controls": all(bool(detail["pass"]) for detail in controls.values()),
            "gain_vs_last_known": last_rmse - float(selected["rmse"]) >= float(promotion["minimum_gain_vs_last_known"]),
            "gain_vs_e006": e006_rmse - float(selected["rmse"]) >= float(promotion["minimum_gain_vs_e006"]),
            "repeated_maps": map_wins[candidate] >= int(promotion["minimum_map_wins"]),
            "outer_cells": outer_cell_wins[candidate] >= int(promotion["minimum_outer_cell_wins"]),
            "p90_vs_e006": float(selected["p90_well_rmse"]) - float(summaries["e006_nested_fusion"]["p90_well_rmse"]) <= float(promotion["maximum_p90_deterioration_vs_e006"]),
            "p90_vs_last_known": float(selected["p90_well_rmse"]) - float(summaries["last_known_tvt"]["p90_well_rmse"]) <= float(promotion["maximum_p90_deterioration_vs_last_known"]),
            "worst5_vs_e006": float(selected["worst_5pct_sse_share"]) - float(summaries["e006_nested_fusion"]["worst_5pct_sse_share"]) <= float(promotion["maximum_worst5_sse_share_increase_vs_e006"]),
            "worst5_vs_last_known": float(selected["worst_5pct_sse_share"]) - float(summaries["last_known_tvt"]["worst_5pct_sse_share"]) <= float(promotion["maximum_worst5_sse_share_increase_vs_last_known"]),
            "spatial_stress": spatial_min_gain > 0.0 if bool(promotion["require_positive_spatial_group_gain_vs_e006"]) else True,
            "typewell_stress": typewell_min_gain > 0.0 if bool(promotion["require_positive_typewell_group_gain_vs_e006"]) else True,
            "special_slices": special_max_deterioration <= float(config["stress"]["maximum_special_slice_deterioration_vs_e006"]),
            "nonzero_outer_cells": int(stability["nonzero_outer_cells"]) >= int(config["controls"]["minimum_nonzero_outer_cells"]),
            "nonzero_maps": int(stability["maps_with_nonzero_median_weight"]) >= int(config["controls"]["minimum_maps_with_nonzero_median_weight"]),
            "weight_range": float(stability["outer_weight_range"]) <= float(config["controls"]["maximum_outer_weight_range"]),
            "raw_diversity": bool(controls["raw_residual_diversity"]["pass"]),
        }

    candidate_rows: list[dict[str, Any]] = []
    for candidate in ALL_CANDIDATES:
        summary = summaries[candidate]
        candidate_rows.append({
            "candidate": candidate,
            "eligible": candidate in ELIGIBLE,
            "passed_all_gates": False,
            "rmse": summary["rmse"],
            "gain_vs_last_known": last_rmse - float(summary["rmse"]),
            "gain_vs_e006": e006_rmse - float(summary["rmse"]),
            "median_well_rmse": summary["median_well_rmse"],
            "p90_well_rmse": summary["p90_well_rmse"],
            "p95_well_rmse": summary["p95_well_rmse"],
            "max_well_rmse": summary["max_well_rmse"],
            "worst_5pct_sse_share": summary["worst_5pct_sse_share"],
            "worst_10pct_sse_share": summary["worst_10pct_sse_share"],
            "map_wins": map_wins.get(candidate, ""),
            "outer_cell_wins": outer_cell_wins.get(candidate, ""),
            "mean_map_rmse": mean_map_rmse.get(candidate, ""),
            "residual_correlation_to_e006": correlations.get(candidate, ""),
        })

    best_eligible = min(ELIGIBLE, key=lambda candidate: (mean_map_rmse[candidate], list(ELIGIBLE).index(candidate)))
    direct_sse = {candidate: 0.0 for candidate in ALL_CANDIDATES}
    oof_path = artifact_dir / "oof_predictions.csv.gz"
    oof_fields = ["id", "well_id", "row_index", "hidden_index", "target", *ALL_CANDIDATES]

    def oof_rows() -> Iterable[Mapping[str, Any]]:
        for chunk in _parent_wells(parent_path):
            well_id = str(chunk["well_id"])
            delta = delta_paths[well_id]
            source_template = templates[global_shuffle[well_id]]
            count = len(chunk["target"])
            for index in range(count):
                target = float(chunk["target"][index])
                base = float(chunk["e006"][index])
                raw = base + float(delta[index])
                shuffled_delta = _template_value(source_template, index, count)
                predictions = {
                    "last_known_tvt": float(chunk["last"][index]),
                    "e004_geometry_prefix": float(chunk["e004"][index]),
                    "e006_nested_fusion": base,
                    "selfcorr_raw_bounded": raw,
                    "visible_fixed_0p10": base + final_weights["visible_fixed_0p10"][well_id] * float(delta[index]),
                    "crossfit_rmse_grid": base + final_rmse_weights[well_id] * float(delta[index]),
                    "crossfit_conservative": base + final_weights["crossfit_conservative"][well_id] * float(delta[index]),
                    "crossfit_reliability_shrink": base + final_weights["crossfit_reliability_shrink"][well_id] * float(delta[index]),
                    "crossfit_positive_gate": base + final_weights["crossfit_positive_gate"][well_id] * float(delta[index]),
                    "duplicate_conservative": base + final_duplicate_weights[well_id] * float(delta[index]),
                    "shuffled_selfcorr_conservative": base + final_shuffled_weights[well_id] * shuffled_delta,
                    "oracle_rowwise_best": base if abs(base - target) <= abs(raw - target) else raw,
                }
                for candidate, prediction in predictions.items():
                    direct_sse[candidate] += (prediction - target) ** 2
                yield {
                    "id": chunk["ids"][index],
                    "well_id": well_id,
                    "row_index": chunk["row_index"][index],
                    "hidden_index": index,
                    "target": target,
                    **predictions,
                }

    _write_gzip_csv(oof_path, oof_fields, oof_rows())
    pooled_difference = max(
        abs(direct_sse[candidate] - float(summaries[candidate]["sse"]))
        / max(1.0, direct_sse[candidate], float(summaries[candidate]["sse"]))
        for candidate in ALL_CANDIDATES
    )
    controls["pooled_sse_consistency"] = {
        "pass": pooled_difference <= float(config["controls"]["pooled_sse_relative_tolerance"]),
        "maximum_relative_difference": pooled_difference,
    }

    wall_seconds = time.perf_counter() - started
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    controls["runtime"] = {
        "pass": wall_seconds <= 60.0 * float(promotion["maximum_runtime_minutes"]),
        "wall_seconds": wall_seconds,
        "budget_minutes": promotion["maximum_runtime_minutes"],
    }
    controls["memory"] = {
        "pass": max_rss_kb <= 1024.0 * float(promotion["maximum_rss_mb"]),
        "max_rss_kb": max_rss_kb,
        "budget_mb": promotion["maximum_rss_mb"],
    }
    for candidate in gates_by_candidate:
        gates_by_candidate[candidate]["controls"] = all(bool(detail["pass"]) for detail in controls.values())
        gates_by_candidate[candidate]["runtime"] = bool(controls["runtime"]["pass"])
        gates_by_candidate[candidate]["memory"] = bool(controls["memory"]["pass"])
    passing = [candidate for candidate in ELIGIBLE if all(bool(value) for value in gates_by_candidate[candidate].values())]
    promoted_candidate = min(passing, key=lambda candidate: (mean_map_rmse[candidate], list(ELIGIBLE).index(candidate))) if passing else None
    _mark_candidate_rows(candidate_rows, passing)
    status = "promoted" if promoted_candidate else "rejected"

    selected_for_metrics = promoted_candidate or best_eligible
    selected_well_rows = []
    for well_id in well_ids:
        metric = averaged_metrics[selected_for_metrics][well_id]
        detail = diagnostics[well_id]
        selected_well_rows.append({
            "well_id": well_id,
            "split": "oof",
            "candidate": selected_for_metrics,
            "promoted": selected_for_metrics == promoted_candidate,
            "rows_scored": metric.rows_scored,
            "rmse": metric.rmse,
            "mean_error": metric.mean_error,
            "sse": metric.sse,
            "regime": "fallback" if detail.fallback_reason else "selfcorr_active",
            "uncertainty": detail.median_best_distance,
            "datum_sse": metric.datum_sse,
            "trend_sse": metric.trend_sse,
            "shape_sse": metric.shape_sse,
            "trend_per_row": metric.trend_per_row,
        })

    diagnostics_rows = [
        {
            "well_id": well_id,
            "visible_gr_coverage": detail.visible_gr_coverage,
            "hidden_gr_coverage": detail.hidden_gr_coverage,
            "template_states": detail.template_states,
            "query_anchors": detail.query_anchors,
            "median_best_fingerprint_distance": detail.median_best_distance,
            "median_fingerprint_margin": detail.median_margin,
            "visible_pseudo_holdout_gain": detail.pseudo_gain,
            "pseudo_base_rmse": detail.pseudo_base_rmse,
            "pseudo_fixed_rmse": detail.pseudo_fixed_rmse,
            "fallback_reason": detail.fallback_reason,
            "maximum_absolute_correction": detail.maximum_absolute_correction,
        }
        for well_id, detail in sorted(diagnostics.items())
    ]
    control_rows = [
        {
            "control": name,
            "status": "pass" if detail["pass"] else "fail",
            "detail_json": _canonical_json({key: value for key, value in detail.items() if key != "pass"}),
        }
        for name, detail in sorted(controls.items())
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "experiment_id": "E007",
        "status": status,
        "code_sha": code_sha,
        "data": e006_summary["data"],
        "best_eligible_candidate": best_eligible,
        "selected_candidate": promoted_candidate,
        "reported_candidate": selected_for_metrics,
        "eligible_candidates": passing,
        "baseline_metrics": summaries["last_known_tvt"],
        "e004_metrics": summaries["e004_geometry_prefix"],
        "e006_metrics": summaries["e006_nested_fusion"],
        "candidate_metrics": summaries,
        "mean_map_rmse": mean_map_rmse,
        "map_wins": map_wins,
        "outer_cell_wins": outer_cell_wins,
        "weight_stability": weight_stability,
        "residual_correlations": correlations,
        "gates_by_candidate": gates_by_candidate,
        "controls": controls,
        "visible_only_design": {"fractions": visible_design_rows},
        "stress": {
            "long_suffix_threshold": long_threshold,
            "high_gr_missingness_threshold": missing_threshold,
            "poor_pseudo_gain_threshold": pseudo_threshold,
            "low_match_margin_threshold": margin_threshold,
            "special_slice_wells": {name: len(ids) for name, ids in special_sets.items()},
        },
        "deployment": {
            "statistically_authorized": promoted_candidate is not None,
            "local_package_built": False,
            "local_notebook_parity": False,
            "private_internet_disabled_kaggle_parity": False,
            "deployment_ready": False,
            "submission_created": False,
            "submission_made": False,
            "reason": "Build deployment package only after statistical promotion." if promoted_candidate else "No E007 placement passed every frozen gate; retain E006.",
        },
        "runtime": {"wall_seconds": wall_seconds, "max_rss_kb": max_rss_kb},
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "candidate_metrics.csv", list(candidate_rows[0]), candidate_rows)
    _write_csv(output_dir / "visible_design_audit.csv", list(visible_design_rows[0]), visible_design_rows)
    _write_csv(output_dir / "selfcorr_diagnostics.csv", list(diagnostics_rows[0]), diagnostics_rows)
    _write_csv(output_dir / "inner_weight_grid.csv", list(inner_grid_rows[0]), inner_grid_rows)
    _write_csv(output_dir / "outer_cell_metrics.csv", list(outer_cell_rows[0]), outer_cell_rows)
    _write_csv(output_dir / "map_metrics.csv", list(map_rows[0]), map_rows)
    _write_csv(output_dir / "stress_metrics.csv", list(stress_rows[0]), stress_rows)
    _write_csv(output_dir / "special_slice_metrics.csv", list(special_rows[0]), special_rows)
    _write_csv(output_dir / "weight_metrics.csv", list(weight_rows[0]), weight_rows)
    _write_csv(output_dir / "membership_audit.csv", list(membership_rows[0]), membership_rows)
    _write_csv(output_dir / "selected_well_metrics.csv", list(selected_well_rows[0]), selected_well_rows)
    _write_csv(output_dir / "control_metrics.csv", list(control_rows[0]), control_rows)
    result_files = [path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    artifact_manifest = {
        "schema_version": 1,
        "code_sha": code_sha,
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(result_files, key=lambda item: item.name)
        ],
        "external_artifacts": [
            {"path": str(oof_path.relative_to(root)), "sha256": _sha256(oof_path), "bytes": oof_path.stat().st_size, "tracked": False}
        ],
        "parent_artifacts": [
            {"path": str(config["parents"]["e006_oof_path"]), "sha256": parent_sha, "expected_sha256": config["parents"]["e006_oof_sha256"]},
            {"path": str(config["parents"]["e006_model"]), "sha256": _sha256(root / str(config["parents"]["e006_model"]))},
        ],
        "config": {
            "path": "experiments/E007/config.json",
            "sha256": _sha256(root / "experiments/E007/config.json"),
            "bytes": (root / "experiments/E007/config.json").stat().st_size,
        },
        "fold_files": [
            {"path": relative, "sha256": _sha256(root / relative), "bytes": (root / relative).stat().st_size}
            for relative in config["fold_files"]
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return summary
