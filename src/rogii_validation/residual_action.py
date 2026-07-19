"""E008 cross-fitted surface-free residual datum/trend action.

The predictor is trained only inside each validation training split.  It uses
horizontal-well geometry/GR, the visible TVT prefix, frozen E006 predictions,
and target-independent E007 self-correlation evidence.  Formation surfaces,
direct typewell features, absolute spatial coordinates, and hidden TVT are
never predictor inputs.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import resource
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .fusion import _group_assignments
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
    scan_profiles,
)
from .learnability import (
    _add_summary,
    _linear_predict,
    _linear_slope,
    _load_fold_maps,
    _longest_missing_run,
    _mean,
    _missing_fraction,
    _pearson,
    _ridge_fit,
    _sign_accuracy,
    _spearman,
    _std,
    _unscale_predictions,
    _visible_backtests,
)
from .self_correlation import PlacementContext, read_horizontal_selfcorr
from .structural import huber_line

COMPARATORS = ("last_known_tvt", "e004_geometry_prefix", "e006_nested_fusion")
NO_EVIDENCE = ("mean_residual_datum", "mean_residual_datum_trend")
ELIGIBLE = ("ridge_residual_datum", "ridge_residual_datum_trend", "ridge_residual_conservative")
DIAGNOSTICS = (
    "ridge_no_e007_ablation",
    "duplicate_feature_ridge",
    "shuffled_target_ridge",
    "oracle_residual_datum",
    "oracle_residual_datum_trend",
)
ALL_CANDIDATES = COMPARATORS + NO_EVIDENCE + ELIGIBLE + DIAGNOSTICS
ACTION_CANDIDATES = NO_EVIDENCE + ELIGIBLE + DIAGNOSTICS
TARGETS = ("residual_datum_ft", "residual_trend_ft")
E007_PREFIXES = ("e007_", "selfcorr_")
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "oracle",
    "hidden_tvt",
    "actual_",
    "correction_label",
    "typewell",
    "spatial_x",
    "spatial_y",
    "ancc",
    "beaur",
    "bessie",
    "cforce",
    "hiking",
    "lmgl",
)


@dataclass(frozen=True)
class ActionSufficient:
    well_id: str
    rows: int
    sum_base: float
    sum_base_sq: float
    sum_index: float
    sum_index_sq: float
    sum_index_base: float
    sum_basis: float
    sum_basis_sq: float
    sum_base_basis: float
    sum_index_basis: float

    @classmethod
    def from_paths(cls, well_id: str, truth: Sequence[float], base: Sequence[float]) -> "ActionSufficient":
        if len(truth) != len(base) or not truth:
            raise DataValidationError(f"{well_id}: invalid E008 parent path lengths")
        n = len(truth)
        sums = [0.0] * 9
        for index, (target, anchor) in enumerate(zip(truth, base)):
            target_f = float(target)
            anchor_f = float(anchor)
            if not math.isfinite(target_f) or not math.isfinite(anchor_f):
                raise DataValidationError(f"{well_id}: non-finite E008 parent path")
            error = anchor_f - target_f
            basis = index / max(1, n - 1) - 0.5 if n > 1 else 0.0
            index_f = float(index)
            sums[0] += error
            sums[1] += error * error
            sums[2] += index_f
            sums[3] += index_f * index_f
            sums[4] += index_f * error
            sums[5] += basis
            sums[6] += basis * basis
            sums[7] += error * basis
            sums[8] += index_f * basis
        return cls(well_id, n, *sums)

    @property
    def datum_target(self) -> float:
        return -self.sum_base / self.rows

    @property
    def trend_target(self) -> float:
        return -self.sum_base_basis / self.sum_basis_sq if self.sum_basis_sq > 0.0 else 0.0

    def metric(self, datum: float, trend: float) -> WellMetric:
        datum_f = float(datum)
        trend_f = float(trend)
        if not math.isfinite(datum_f) or not math.isfinite(trend_f):
            raise DataValidationError(f"{self.well_id}: non-finite E008 action")
        n = float(self.rows)
        sum_error = self.sum_base + n * datum_f + trend_f * self.sum_basis
        sum_error_sq = (
            self.sum_base_sq
            + 2.0 * datum_f * self.sum_base
            + 2.0 * trend_f * self.sum_base_basis
            + n * datum_f * datum_f
            + 2.0 * datum_f * trend_f * self.sum_basis
            + trend_f * trend_f * self.sum_basis_sq
        )
        sum_index_error = self.sum_index_base + datum_f * self.sum_index + trend_f * self.sum_index_basis
        accumulator = ErrorAccumulator(
            rows=self.rows,
            sum_error=sum_error,
            sum_error_sq=sum_error_sq,
            sum_x=self.sum_index,
            sum_x_sq=self.sum_index_sq,
            sum_x_error=sum_index_error,
        )
        return accumulator.finalize(self.well_id)


@dataclass(frozen=True)
class PreparedModel:
    selected_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    y_means: tuple[float, ...]
    y_scales: tuple[float, ...]
    train_x: tuple[tuple[float, ...], ...]
    test_x: tuple[tuple[float, ...], ...]
    train_y: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FittedModel:
    target_names: tuple[str, ...]
    prepared: PreparedModel
    coefficients: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ParentWell:
    well_id: str
    ids: tuple[str, ...]
    row_indices: tuple[int, ...]
    truth: tuple[float, ...]
    last: tuple[float, ...]
    e004: tuple[float, ...]
    e006: tuple[float, ...]
    raw_selfcorr: tuple[float, ...]


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _soft_cap(value: float, cap: float) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(cap)) or cap <= 0.0:
        raise DataValidationError("E008 soft-cap input is invalid")
    return float(cap) * math.tanh(float(value) / float(cap))


def _action_basis(index: int, rows: int) -> float:
    return index / max(1, rows - 1) - 0.5 if rows > 1 else 0.0


def _metric_for_path(well_id: str, truth: Sequence[float], prediction: Sequence[float]) -> WellMetric:
    if len(truth) != len(prediction) or not truth:
        raise DataValidationError(f"{well_id}: E008 metric path lengths differ")
    accumulator = ErrorAccumulator()
    for index, (target, value) in enumerate(zip(truth, prediction)):
        if not math.isfinite(float(target)) or not math.isfinite(float(value)):
            raise DataValidationError(f"{well_id}: non-finite E008 metric input")
        accumulator.add(float(value) - float(target), float(index))
    return accumulator.finalize(well_id)


def _path_summary(features: dict[str, float | None], prefix: str, values: Sequence[float]) -> None:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise DataValidationError(f"E008 invalid path summary {prefix}")
    xs = [float(index) for index in range(len(values))]
    clean = [float(value) for value in values]
    features[f"{prefix}_first"] = clean[0]
    features[f"{prefix}_last"] = clean[-1]
    features[f"{prefix}_mean"] = _mean(clean)
    features[f"{prefix}_std"] = _std(clean)
    features[f"{prefix}_range"] = max(clean) - min(clean)
    features[f"{prefix}_slope_per_row"] = _linear_slope(xs, clean)
    features[f"{prefix}_mean_abs"] = _mean([abs(value) for value in clean])
    features[f"{prefix}_max_abs"] = max(abs(value) for value in clean)
    features[f"{prefix}_positive_fraction"] = sum(value > 0.0 for value in clean) / len(clean)


def _read_e007_diagnostics(path: Path) -> dict[str, dict[str, float | str]]:
    required = {
        "well_id",
        "visible_gr_coverage",
        "hidden_gr_coverage",
        "template_states",
        "query_anchors",
        "median_best_fingerprint_distance",
        "median_fingerprint_margin",
        "visible_pseudo_holdout_gain",
        "pseudo_base_rmse",
        "pseudo_fixed_rmse",
        "fallback_reason",
        "maximum_absolute_correction",
    }
    output: dict[str, dict[str, float | str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{path}: missing E008 diagnostic columns {sorted(missing)}")
        for row in reader:
            well_id = str(row["well_id"])
            if well_id in output:
                raise DataValidationError(f"{path}: duplicate E007 diagnostic {well_id}")
            item: dict[str, float | str] = {"fallback_reason": str(row["fallback_reason"])}
            for name in required - {"well_id", "fallback_reason"}:
                value = float(row[name])
                if not math.isfinite(value):
                    raise DataValidationError(f"{path}: non-finite E007 diagnostic {well_id}/{name}")
                item[name] = value
            output[well_id] = item
    return output


def _iter_parent_wells(path: Path) -> Iterator[ParentWell]:
    required = {
        "id",
        "well_id",
        "row_index",
        "hidden_index",
        "target",
        "last_known_tvt",
        "e004_geometry_prefix",
        "e006_nested_fusion",
        "selfcorr_raw_bounded",
    }
    current: dict[str, Any] | None = None
    seen: set[str] = set()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{path}: missing E008 parent columns {sorted(missing)}")
        for row in reader:
            well_id = str(row["well_id"])
            if current is None or current["well_id"] != well_id:
                if current is not None:
                    seen.add(str(current["well_id"]))
                    yield ParentWell(
                        str(current["well_id"]),
                        tuple(current["ids"]),
                        tuple(current["row_indices"]),
                        tuple(current["truth"]),
                        tuple(current["last"]),
                        tuple(current["e004"]),
                        tuple(current["e006"]),
                        tuple(current["raw"]),
                    )
                if well_id in seen:
                    raise DataValidationError(f"{path}: noncontiguous E008 parent well {well_id}")
                current = {
                    "well_id": well_id,
                    "ids": [],
                    "row_indices": [],
                    "truth": [],
                    "last": [],
                    "e004": [],
                    "e006": [],
                    "raw": [],
                }
            assert current is not None
            hidden_index = int(row["hidden_index"])
            if hidden_index != len(current["truth"]):
                raise DataValidationError(f"{well_id}: noncontiguous E008 hidden index")
            values = [
                float(row["target"]),
                float(row["last_known_tvt"]),
                float(row["e004_geometry_prefix"]),
                float(row["e006_nested_fusion"]),
                float(row["selfcorr_raw_bounded"]),
            ]
            if not all(math.isfinite(value) for value in values):
                raise DataValidationError(f"{well_id}: non-finite E008 parent row")
            current["ids"].append(str(row["id"]))
            current["row_indices"].append(int(row["row_index"]))
            current["truth"].append(values[0])
            current["last"].append(values[1])
            current["e004"].append(values[2])
            current["e006"].append(values[3])
            current["raw"].append(values[4])
    if current is not None:
        yield ParentWell(
            str(current["well_id"]),
            tuple(current["ids"]),
            tuple(current["row_indices"]),
            tuple(current["truth"]),
            tuple(current["last"]),
            tuple(current["e004"]),
            tuple(current["e006"]),
            tuple(current["raw"]),
        )


def _audit_parent_alignment(e006_path: Path, e007_path: Path, tolerance: float = 5e-8) -> dict[str, Any]:
    rows = 0
    maximum_delta = 0.0
    with gzip.open(e006_path, "rt", newline="", encoding="utf-8") as left_handle, gzip.open(
        e007_path, "rt", newline="", encoding="utf-8"
    ) as right_handle:
        left = csv.DictReader(left_handle)
        right = csv.DictReader(right_handle)
        while True:
            first = next(left, None)
            second = next(right, None)
            if first is None or second is None:
                if first is not None or second is not None:
                    raise DataValidationError("E006/E007 parent row counts differ")
                break
            rows += 1
            if first["id"] != second["id"] or first["well_id"] != second["well_id"]:
                raise DataValidationError("E006/E007 parent IDs differ")
            pairs = (
                (first["target"], second["target"]),
                (first["last_known_tvt"], second["last_known_tvt"]),
                (first["e004_geometry_prefix"], second["e004_geometry_prefix"]),
                (first["nested_conservative_grid"], second["e006_nested_fusion"]),
            )
            for left_raw, right_raw in pairs:
                delta = abs(float(left_raw) - float(right_raw))
                maximum_delta = max(maximum_delta, delta)
                if delta > tolerance:
                    raise DataValidationError("E006/E007 parent prediction mismatch")
    return {"rows": rows, "maximum_value_delta": maximum_delta, "pass": True}


def _extract_horizontal_features(
    *,
    train_dir: Path,
    well_id: str,
    path_features: Mapping[str, float],
    diagnostic: Mapping[str, float | str],
    config: Mapping[str, Any],
) -> tuple[dict[str, float | None], tuple[float, float], int]:
    well = read_horizontal_selfcorr(train_dir / f"{well_id}__horizontal_well.csv")
    known = well.known_rows
    hidden = well.hidden_rows
    if known <= 0 or hidden <= 0:
        raise DataValidationError(f"{well_id}: invalid E008 visibility boundary")
    md = list(well.md)
    x = list(well.x)
    y = list(well.y)
    z = list(well.z)
    visible_tvt = [float(value) for value in well.tvt_input[:known] if value is not None]
    if len(visible_tvt) != known:
        raise DataValidationError(f"{well_id}: E008 visible TVT mismatch")
    known_md, hidden_md = md[:known], md[known:]
    known_z, hidden_z = z[:known], z[known:]
    visible_u = [target + vertical for target, vertical in zip(visible_tvt, known_z)]
    features: dict[str, float | None] = {
        "total_rows": float(len(md)),
        "known_rows": float(known),
        "hidden_rows": float(hidden),
        "hidden_fraction": hidden / len(md),
        "md_known_span": known_md[-1] - known_md[0],
        "md_hidden_span": hidden_md[-1] - known_md[-1],
        "md_total_span": md[-1] - md[0],
        "last_visible_tvt": visible_tvt[-1],
        "last_visible_z": known_z[-1],
        "last_visible_u": visible_u[-1],
        "x_total_delta": x[-1] - x[0],
        "y_total_delta": y[-1] - y[0],
        "z_total_delta": z[-1] - z[0],
        "x_hidden_delta": x[-1] - x[known - 1],
        "y_hidden_delta": y[-1] - y[known - 1],
        "z_hidden_delta": z[-1] - z[known - 1],
        "horizontal_total_distance": math.hypot(x[-1] - x[0], y[-1] - y[0]),
        "horizontal_hidden_distance": math.hypot(x[-1] - x[known - 1], y[-1] - y[known - 1]),
    }
    angle = math.atan2(y[-1] - y[known - 1], x[-1] - x[known - 1])
    features["hidden_azimuth_sin"] = math.sin(angle)
    features["hidden_azimuth_cos"] = math.cos(angle)
    _add_summary(features, "visible_tvt", known_md, visible_tvt)
    _add_summary(features, "visible_z", known_md, known_z)
    _add_summary(features, "visible_u", known_md, visible_u)
    _add_summary(features, "hidden_z", hidden_md, hidden_z)
    _add_summary(features, "visible_gr", known_md, well.gr[:known])
    _add_summary(features, "hidden_gr", hidden_md, well.gr[known:])
    _add_summary(features, "whole_gr", md, well.gr)
    features["visible_gr_missing_fraction"] = _missing_fraction(well.gr[:known])
    features["hidden_gr_missing_fraction"] = _missing_fraction(well.gr[known:])
    features["whole_gr_missing_fraction"] = _missing_fraction(well.gr)
    features["hidden_gr_longest_missing_run"] = float(_longest_missing_run(well.gr[known:]))
    full_u_slope = _linear_slope(known_md, visible_u)
    for window in config["feature_contract"]["visible_slope_windows"]:
        count = min(int(window), known)
        if count < 2:
            features[f"visible_u_slope_w{window}"] = None
            features[f"visible_u_scale_w{window}"] = None
            features[f"visible_u_slope_delta_w{window}"] = None
            features[f"visible_tvt_slope_w{window}"] = None
            continue
        _, u_slope, u_scale = huber_line(known_md[-count:], visible_u[-count:], window_rows=count, huber_k=1.5, iterations=6)
        _, tvt_slope, _ = huber_line(known_md[-count:], visible_tvt[-count:], window_rows=count, huber_k=1.5, iterations=6)
        features[f"visible_u_slope_w{window}"] = u_slope
        features[f"visible_u_scale_w{window}"] = u_scale
        features[f"visible_u_slope_delta_w{window}"] = u_slope - full_u_slope
        features[f"visible_tvt_slope_w{window}"] = tvt_slope
    features.update(
        _visible_backtests(
            known_md,
            known_z,
            visible_tvt,
            [float(value) for value in config["feature_contract"]["visible_backtest_fractions"]],
        )
    )
    features.update({name: float(value) for name, value in path_features.items()})
    mapping = {
        "visible_gr_coverage": "e007_visible_gr_coverage",
        "hidden_gr_coverage": "e007_hidden_gr_coverage",
        "template_states": "e007_template_states",
        "query_anchors": "e007_query_anchors",
        "median_best_fingerprint_distance": "e007_median_best_fingerprint_distance",
        "median_fingerprint_margin": "e007_median_fingerprint_margin",
        "visible_pseudo_holdout_gain": "e007_visible_pseudo_holdout_gain",
        "pseudo_base_rmse": "e007_pseudo_base_rmse",
        "pseudo_fixed_rmse": "e007_pseudo_fixed_rmse",
        "maximum_absolute_correction": "e007_maximum_absolute_correction",
    }
    for source, target in mapping.items():
        features[target] = float(diagnostic[source])
    features["e007_fallback_active"] = 1.0 if str(diagnostic["fallback_reason"]) else 0.0
    midpoint = (0.5 * (x[0] + x[-1]), 0.5 * (y[0] + y[-1]))
    return features, midpoint, hidden


def _validate_feature_names(feature_names: Sequence[str]) -> dict[str, Any]:
    bad = [name for name in feature_names if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise DataValidationError(f"E008 forbidden feature columns {bad[:12]}")
    if not any(name.startswith(E007_PREFIXES) for name in feature_names):
        raise DataValidationError("E008 feature table lacks E007 evidence")
    return {"pass": True, "feature_count": len(feature_names), "forbidden_columns": []}


def validate_e008_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1 or config.get("experiment_id") != "E008":
        raise DataValidationError("invalid E008 configuration identity")
    if tuple(config.get("candidate_order", ())) != ALL_CANDIDATES:
        raise DataValidationError("E008 candidate order differs from frozen implementation")
    if tuple(config.get("eligible_candidates", ())) != ELIGIBLE:
        raise DataValidationError("E008 eligible candidates differ from frozen implementation")
    if float(config["ridge"]["alpha"]) <= 0.0:
        raise DataValidationError("E008 ridge alpha must be positive")
    scales = [float(value) for value in config["ridge"]["action_scales"]]
    if scales != sorted(set(scales)) or scales[0] != 0.0 or scales[-1] != 1.0:
        raise DataValidationError("E008 action scale grid is malformed")
    if int(config["feature_contract"]["maximum_screened_features"]) <= 0:
        raise DataValidationError("E008 feature limit must be positive")
    if not all(bool(config["feature_contract"][name]) for name in (
        "exclude_absolute_spatial_coordinates", "exclude_direct_typewell_features", "exclude_formation_surfaces"
    )):
        raise DataValidationError("E008 forbidden feature exclusions must remain enabled")
    for name in ("datum_soft_cap_ft", "trend_soft_cap_ft"):
        if float(config["targets"][name]) <= 0.0:
            raise DataValidationError(f"E008 invalid target cap {name}")


def _prepare_model(
    *,
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    maximum_features: int,
    target_override: Mapping[str, Mapping[str, float]] | None = None,
) -> PreparedModel:
    if not train_ids or not target_names or not feature_names:
        raise DataValidationError("E008 model split is empty")
    all_medians: list[float] = []
    all_means: list[float] = []
    all_scales: list[float] = []
    train_full: list[list[float]] = [[] for _ in train_ids]
    test_full: list[list[float]] = [[] for _ in test_ids]
    for name in feature_names:
        finite_train = [
            float(records[well_id]["features"].get(name))
            for well_id in train_ids
            if records[well_id]["features"].get(name) is not None
            and math.isfinite(float(records[well_id]["features"].get(name)))
        ]
        median = _median(finite_train)
        train_column = [
            float(records[well_id]["features"].get(name))
            if records[well_id]["features"].get(name) is not None
            and math.isfinite(float(records[well_id]["features"].get(name)))
            else median
            for well_id in train_ids
        ]
        test_column = [
            float(records[well_id]["features"].get(name))
            if records[well_id]["features"].get(name) is not None
            and math.isfinite(float(records[well_id]["features"].get(name)))
            else median
            for well_id in test_ids
        ]
        center = _mean(train_column)
        scale = _std(train_column)
        if scale <= 1e-12:
            scale = 1.0
        all_medians.append(median)
        all_means.append(center)
        all_scales.append(scale)
        for row_index, value in enumerate(train_column):
            train_full[row_index].append((value - center) / scale)
        for row_index, value in enumerate(test_column):
            test_full[row_index].append((value - center) / scale)
    y_means: list[float] = []
    y_scales: list[float] = []
    train_y: list[list[float]] = [[] for _ in train_ids]
    for target in target_names:
        values = [
            float((target_override or {}).get(well_id, records[well_id]["targets"])[target])
            for well_id in train_ids
        ]
        if not all(math.isfinite(value) for value in values):
            raise DataValidationError("E008 training targets are non-finite")
        center = _mean(values)
        scale = _std(values)
        if scale <= 1e-12:
            scale = 1.0
        y_means.append(center)
        y_scales.append(scale)
        for row_index, value in enumerate(values):
            train_y[row_index].append((value - center) / scale)
    scored: list[tuple[float, str, int]] = []
    for index, name in enumerate(feature_names):
        column = [row[index] for row in train_full]
        score = 0.0
        for target_index in range(len(target_names)):
            target_column = [row[target_index] for row in train_y]
            correlation = _pearson(column, target_column)
            score = max(score, abs(correlation) if correlation is not None else 0.0)
        scored.append((score, name, index))
    chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[: max(1, int(maximum_features))]
    indices = [item[2] for item in chosen]
    return PreparedModel(
        selected_names=tuple(item[1] for item in chosen),
        medians=tuple(all_medians[index] for index in indices),
        means=tuple(all_means[index] for index in indices),
        scales=tuple(all_scales[index] for index in indices),
        y_means=tuple(y_means),
        y_scales=tuple(y_scales),
        train_x=tuple(tuple(row[index] for index in indices) for row in train_full),
        test_x=tuple(tuple(row[index] for index in indices) for row in test_full),
        train_y=tuple(tuple(row) for row in train_y),
    )


def _fit_predict(
    *,
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    config: Mapping[str, Any],
    target_override: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[dict[str, tuple[float, float]], FittedModel]:
    prepared = _prepare_model(
        records=records,
        train_ids=train_ids,
        test_ids=test_ids,
        feature_names=feature_names,
        target_names=target_names,
        maximum_features=int(config["feature_contract"]["maximum_screened_features"]),
        target_override=target_override,
    )
    coefficients = _ridge_fit(prepared.train_x, prepared.train_y, float(config["ridge"]["alpha"]))
    scaled = _linear_predict(prepared.test_x, coefficients)
    raw = _unscale_predictions(scaled, prepared.y_means, prepared.y_scales)
    datum_cap = float(config["targets"]["datum_soft_cap_ft"])
    trend_cap = float(config["targets"]["trend_soft_cap_ft"])
    predictions: dict[str, tuple[float, float]] = {}
    for well_id, row in zip(test_ids, raw):
        datum = _soft_cap(float(row[0]), datum_cap)
        trend = _soft_cap(float(row[1]), trend_cap) if len(row) > 1 else 0.0
        predictions[well_id] = (datum, trend)
    model = FittedModel(tuple(target_names), prepared, tuple(tuple(value for value in row) for row in coefficients))
    return predictions, model


def _duplicate_predictions(
    model: FittedModel,
    test_ids: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    prepared = model.prepared
    if not prepared.train_x or not prepared.train_x[0]:
        raise DataValidationError("E008 duplicate control has no feature")
    root_two = math.sqrt(2.0)
    train_x = tuple(
        (row[0] / root_two, row[0] / root_two, *row[1:])
        for row in prepared.train_x
    )
    test_x = tuple(
        (row[0] / root_two, row[0] / root_two, *row[1:])
        for row in prepared.test_x
    )
    coefficients = _ridge_fit(train_x, prepared.train_y, float(config["ridge"]["alpha"]))
    scaled = _linear_predict(test_x, coefficients)
    raw = _unscale_predictions(scaled, prepared.y_means, prepared.y_scales)
    datum_cap = float(config["targets"]["datum_soft_cap_ft"])
    trend_cap = float(config["targets"]["trend_soft_cap_ft"])
    return {
        well_id: (
            _soft_cap(float(row[0]), datum_cap),
            _soft_cap(float(row[1]), trend_cap) if len(row) > 1 else 0.0,
        )
        for well_id, row in zip(test_ids, raw)
    }


def _deranged_targets(
    records: Mapping[str, Mapping[str, Any]], train_ids: Sequence[str], context_key: str
) -> dict[str, Mapping[str, float]]:
    ordered = sorted(train_ids, key=lambda well_id: (hashlib.sha256(f"E008-target-shuffle|{context_key}|{well_id}".encode()).hexdigest(), well_id))
    if len(ordered) < 2:
        raise DataValidationError("E008 shuffled target control requires two wells")
    return {well_id: records[ordered[(index + 1) % len(ordered)]]["targets"] for index, well_id in enumerate(ordered)}


def _mean_action(records: Mapping[str, Mapping[str, Any]], train_ids: Sequence[str], config: Mapping[str, Any]) -> tuple[float, float]:
    datum = _soft_cap(_mean([float(records[well_id]["targets"][TARGETS[0]]) for well_id in train_ids]), float(config["targets"]["datum_soft_cap_ft"]))
    trend = _soft_cap(_mean([float(records[well_id]["targets"][TARGETS[1]]) for well_id in train_ids]), float(config["targets"]["trend_soft_cap_ft"]))
    return datum, trend


def _select_scale(
    records: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, tuple[float, float]],
    ids: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    if set(ids) != set(predictions):
        raise DataValidationError("E008 inner OOF predictions do not cover training wells")
    rows: list[dict[str, Any]] = []
    summaries: dict[float, dict[str, Any]] = {}
    for scale in [float(value) for value in config["ridge"]["action_scales"]]:
        metrics = [
            records[well_id]["sufficient"].metric(scale * predictions[well_id][0], scale * predictions[well_id][1])
            for well_id in ids
        ]
        summaries[scale] = _summarize(metrics)
    base = summaries[0.0]
    passing = [
        scale
        for scale, summary in summaries.items()
        if float(base["rmse"]) - float(summary["rmse"]) >= float(config["inner_selection"]["minimum_gain_vs_e006"])
        and float(summary["p90_well_rmse"]) - float(base["p90_well_rmse"]) <= float(config["inner_selection"]["maximum_p90_deterioration_vs_e006"])
        and float(summary["worst_5pct_sse_share"]) - float(base["worst_5pct_sse_share"]) <= float(config["inner_selection"]["maximum_worst5_sse_share_increase_vs_e006"])
    ]
    if passing:
        best = min(passing, key=lambda value: (float(summaries[value]["rmse"]), value))
        threshold = float(summaries[best]["rmse"]) + float(config["inner_selection"]["near_best_rmse_tolerance"])
        selected = min(value for value in passing if float(summaries[value]["rmse"]) <= threshold)
    else:
        selected = 0.0
    for scale, summary in summaries.items():
        rows.append({
            "scale": scale,
            "passes_constraints": scale in passing,
            "selected": scale == selected,
            **summary,
        })
    return selected, rows


def _inner_oof_predictions(
    *,
    context: PlacementContext,
    records: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    fold_maps: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    if context.scope == "repeated":
        fold_map = next(item for item in fold_maps if str(item["version"]) == context.label)
    else:
        fold_map = next(item for item in fold_maps if str(item["version"]) == "v1")
    assignments = {well_id: int(value) for well_id, value in fold_map["assignments"].items()}
    output: dict[str, tuple[float, float]] = {}
    for inner_group in sorted(set(assignments[well_id] for well_id in context.train_ids)):
        inner_test = tuple(sorted(well_id for well_id in context.train_ids if assignments[well_id] == inner_group))
        inner_train = tuple(sorted(set(context.train_ids) - set(inner_test)))
        if not inner_train or not inner_test:
            continue
        predicted, _ = _fit_predict(
            records=records,
            train_ids=inner_train,
            test_ids=inner_test,
            feature_names=feature_names,
            target_names=TARGETS,
            config=config,
        )
        output.update(predicted)
    if set(output) != set(context.train_ids):
        raise DataValidationError(f"{context.key}: E008 inner OOF coverage failed")
    return output


def _build_contexts(
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
                "context": key, "scope": "repeated", "label": version, "outer_group": outer,
                "outer_train_wells": len(train_ids), "outer_test_wells": len(test_ids),
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
                "context": key, "scope": scope, "label": scope, "outer_group": outer,
                "outer_train_wells": len(train_ids), "outer_test_wells": len(test_ids),
                "train_test_overlap": len(set(train_ids) & set(test_ids)),
                "pass": bool(train_ids) and bool(test_ids) and not (set(train_ids) & set(test_ids)),
            })
    if not all(bool(row["pass"]) for row in audits):
        raise DataValidationError("E008 placement membership audit failed")
    return contexts, audits


def _target_metrics(
    records: Mapping[str, Mapping[str, Any]],
    actions: Mapping[str, tuple[float, float]],
    candidate: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, target_name in enumerate(TARGETS):
        actual = [float(records[well_id]["targets"][target_name]) for well_id in sorted(records)]
        predicted = [float(actions[well_id][index]) for well_id in sorted(records)]
        pearson = _pearson(actual, predicted)
        spearman = _spearman(actual, predicted)
        mae = _mean([abs(first - second) for first, second in zip(actual, predicted)])
        sign, material = _sign_accuracy(actual, predicted, 5.0)
        rows.append({
            "candidate": candidate,
            "target": target_name,
            "pearson": pearson,
            "spearman": spearman,
            "mae": mae,
            "material_sign_accuracy": sign,
            "material_wells": material,
        })
    return rows


def _write_gzip_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)


def _format_row(value: float) -> str:
    return f"{float(value):.8f}"


def _artifact_entry(path: Path, relative: str, kind: str) -> dict[str, Any]:
    return {"path": relative, "kind": kind, "sha256": _sha256(path), "bytes": path.stat().st_size}


def run_e008(
    *,
    root: Path,
    train_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: Mapping[str, Any],
    code_sha: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_e008_config(config)
    root = root.resolve()
    train_dir = train_dir.resolve()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    profiles, data = scan_profiles(train_dir)
    if int(data["well_count"]) != int(config["expected_wells"]) or data["data_signature"] != config["data_signature"]:
        raise DataValidationError("E008 data identity differs from frozen configuration")
    fold_maps = _load_fold_maps(root, config["fold_files"], str(config["data_signature"]))
    parent = config["parents"]
    e006_path = root / str(parent["e006_oof"])
    e007_path = root / str(parent["e007_oof"])
    diagnostics_path = root / str(parent["e007_diagnostics"])
    if _sha256(e006_path) != str(parent["e006_oof_sha256"]):
        raise DataValidationError("E008 E006 OOF hash mismatch")
    if _sha256(e007_path) != str(parent["e007_oof_sha256"]):
        raise DataValidationError("E008 E007 OOF hash mismatch")
    if _sha256(diagnostics_path) != str(parent["e007_diagnostics_sha256"]):
        raise DataValidationError("E008 E007 diagnostic hash mismatch")
    if _sha256(root / str(parent["e006_model"])) != str(parent["e006_model_sha256"]):
        raise DataValidationError("E008 E006 model hash mismatch")
    parent_alignment = _audit_parent_alignment(e006_path, e007_path)
    diagnostics = _read_e007_diagnostics(diagnostics_path)
    parent_records: dict[str, dict[str, Any]] = {}
    comparator_metrics: dict[str, dict[str, WellMetric]] = {name: {} for name in COMPARATORS}
    for well in _iter_parent_wells(e007_path):
        sufficient = ActionSufficient.from_paths(well.well_id, well.truth, well.e006)
        features: dict[str, float] = {}
        _path_summary(features, "e006_path", well.e006)
        _path_summary(features, "e006_minus_last", [a - b for a, b in zip(well.e006, well.last)])
        _path_summary(features, "e006_minus_e004", [a - b for a, b in zip(well.e006, well.e004)])
        _path_summary(features, "e004_minus_last", [a - b for a, b in zip(well.e004, well.last)])
        _path_summary(features, "selfcorr_raw_minus_e006", [a - b for a, b in zip(well.raw_selfcorr, well.e006)])
        parent_records[well.well_id] = {
            "sufficient": sufficient,
            "targets": {
                TARGETS[0]: sufficient.datum_target,
                TARGETS[1]: sufficient.trend_target,
            },
            "path_features": features,
            "rows": len(well.truth),
        }
        comparator_metrics["last_known_tvt"][well.well_id] = _metric_for_path(well.well_id, well.truth, well.last)
        comparator_metrics["e004_geometry_prefix"][well.well_id] = _metric_for_path(well.well_id, well.truth, well.e004)
        comparator_metrics["e006_nested_fusion"][well.well_id] = sufficient.metric(0.0, 0.0)
    well_ids = sorted(parent_records)
    if len(well_ids) != int(config["expected_wells"]) or set(well_ids) != set(diagnostics):
        raise DataValidationError("E008 parent and diagnostic well sets differ")
    records: dict[str, dict[str, Any]] = {}
    feature_names: set[str] = set()
    spatial_values: dict[str, tuple[float, ...]] = {}
    typewell_values: dict[str, tuple[float, ...]] = {}
    hidden_rows: dict[str, int] = {}
    for well_id in well_ids:
        features, spatial, hidden = _extract_horizontal_features(
            train_dir=train_dir,
            well_id=well_id,
            path_features=parent_records[well_id]["path_features"],
            diagnostic=diagnostics[well_id],
            config=config,
        )
        if hidden != int(parent_records[well_id]["rows"]):
            raise DataValidationError(f"{well_id}: E008 hidden rows differ from parent")
        records[well_id] = {
            "features": features,
            "targets": parent_records[well_id]["targets"],
            "sufficient": parent_records[well_id]["sufficient"],
        }
        feature_names.update(features)
        spatial_values[well_id] = spatial
        typewell = TypewellCurve.read(train_dir / f"{well_id}__typewell.csv")
        typewell_values[well_id] = (typewell.gr_mean, typewell.gr_std, typewell.maximum_tvt - typewell.minimum_tvt)
        hidden_rows[well_id] = hidden
    feature_names_sorted = sorted(feature_names)
    leakage_control = _validate_feature_names(feature_names_sorted)
    spatial_assignments = _group_assignments(spatial_values, int(config["stress"]["spatial_bins"]))
    typewell_assignments = _group_assignments(typewell_values, int(config["stress"]["typewell_clusters"]))
    contexts, membership_rows = _build_contexts(well_ids, fold_maps, spatial_assignments, typewell_assignments, config)
    no_e007_features = [name for name in feature_names_sorted if not name.startswith(E007_PREFIXES)]
    if not no_e007_features or len(no_e007_features) == len(feature_names_sorted):
        raise DataValidationError("E008 no-E007 ablation feature set is invalid")

    context_actions: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    context_metrics: dict[str, dict[str, dict[str, WellMetric]]] = {}
    scale_rows: list[dict[str, Any]] = []
    selected_features: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    duplicate_delta = 0.0
    for context in contexts:
        actions: dict[str, dict[str, tuple[float, float]]] = {name: {} for name in ACTION_CANDIDATES}
        mean_datum, mean_trend = _mean_action(records, context.train_ids, config)
        actions["mean_residual_datum"] = {well_id: (mean_datum, 0.0) for well_id in context.test_ids}
        actions["mean_residual_datum_trend"] = {well_id: (mean_datum, mean_trend) for well_id in context.test_ids}
        datum_predictions, datum_model = _fit_predict(
            records=records, train_ids=context.train_ids, test_ids=context.test_ids,
            feature_names=feature_names_sorted, target_names=(TARGETS[0],), config=config,
        )
        full_predictions, full_model = _fit_predict(
            records=records, train_ids=context.train_ids, test_ids=context.test_ids,
            feature_names=feature_names_sorted, target_names=TARGETS, config=config,
        )
        no_e007_predictions, no_e007_model = _fit_predict(
            records=records, train_ids=context.train_ids, test_ids=context.test_ids,
            feature_names=no_e007_features, target_names=TARGETS, config=config,
        )
        duplicate_predictions = _duplicate_predictions(full_model, context.test_ids, config)
        shuffled_predictions, _ = _fit_predict(
            records=records, train_ids=context.train_ids, test_ids=context.test_ids,
            feature_names=feature_names_sorted, target_names=TARGETS, config=config,
            target_override=_deranged_targets(records, context.train_ids, context.key),
        )
        inner_predictions = _inner_oof_predictions(
            context=context, records=records, feature_names=feature_names_sorted,
            fold_maps=fold_maps, config=config,
        )
        selected_scale, rows = _select_scale(records, inner_predictions, context.train_ids, config)
        for row in rows:
            scale_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, **row})
        actions["ridge_residual_datum"] = datum_predictions
        actions["ridge_residual_datum_trend"] = full_predictions
        actions["ridge_residual_conservative"] = {
            well_id: (selected_scale * values[0], selected_scale * values[1])
            for well_id, values in full_predictions.items()
        }
        actions["ridge_no_e007_ablation"] = no_e007_predictions
        actions["duplicate_feature_ridge"] = duplicate_predictions
        actions["shuffled_target_ridge"] = shuffled_predictions
        actions["oracle_residual_datum"] = {
            well_id: (float(records[well_id]["targets"][TARGETS[0]]), 0.0) for well_id in context.test_ids
        }
        actions["oracle_residual_datum_trend"] = {
            well_id: (
                float(records[well_id]["targets"][TARGETS[0]]),
                float(records[well_id]["targets"][TARGETS[1]]),
            ) for well_id in context.test_ids
        }
        for well_id in context.test_ids:
            duplicate_delta = max(
                duplicate_delta,
                abs(actions["duplicate_feature_ridge"][well_id][0] - actions["ridge_residual_datum_trend"][well_id][0]),
                abs(actions["duplicate_feature_ridge"][well_id][1] - actions["ridge_residual_datum_trend"][well_id][1]),
            )
        selected_features[context.key]["ridge_residual_datum"] = datum_model.prepared.selected_names
        selected_features[context.key]["ridge_residual_datum_trend"] = full_model.prepared.selected_names
        selected_features[context.key]["ridge_no_e007_ablation"] = no_e007_model.prepared.selected_names
        metrics: dict[str, dict[str, WellMetric]] = {name: {} for name in ALL_CANDIDATES}
        for comparator in COMPARATORS:
            metrics[comparator] = {well_id: comparator_metrics[comparator][well_id] for well_id in context.test_ids}
        for candidate in ACTION_CANDIDATES:
            metrics[candidate] = {
                well_id: records[well_id]["sufficient"].metric(*actions[candidate][well_id])
                for well_id in context.test_ids
            }
        context_actions[context.key] = actions
        context_metrics[context.key] = metrics

    repeated_contexts = [context for context in contexts if context.scope == "repeated"]
    final_actions: dict[str, dict[str, tuple[float, float]]] = {name: {} for name in ACTION_CANDIDATES}
    for well_id in well_ids:
        applicable = [
            context for context in repeated_contexts if well_id in context.test_ids
        ]
        if len(applicable) != len(fold_maps):
            raise DataValidationError(f"{well_id}: E008 repeated OOF coverage differs")
        for candidate in ACTION_CANDIDATES:
            final_actions[candidate][well_id] = (
                _mean([context_actions[context.key][candidate][well_id][0] for context in applicable]),
                _mean([context_actions[context.key][candidate][well_id][1] for context in applicable]),
            )
    final_metrics: dict[str, dict[str, WellMetric]] = {name: {} for name in ALL_CANDIDATES}
    for comparator in COMPARATORS:
        final_metrics[comparator] = dict(comparator_metrics[comparator])
    for candidate in ACTION_CANDIDATES:
        final_metrics[candidate] = {
            well_id: records[well_id]["sufficient"].metric(*final_actions[candidate][well_id])
            for well_id in well_ids
        }
    summaries = {candidate: _summarize([final_metrics[candidate][well_id] for well_id in well_ids]) for candidate in ALL_CANDIDATES}

    map_rows: list[dict[str, Any]] = []
    for fold_map in fold_maps:
        version = str(fold_map["version"])
        keys = [f"repeated:{version}:{fold}" for fold in range(int(fold_map["n_folds"]))]
        for candidate in ALL_CANDIDATES:
            metrics = [metric for key in keys for metric in context_metrics[key][candidate].values()]
            map_rows.append({"map": version, "candidate": candidate, **_summarize(metrics)})
    map_lookup = {(row["map"], row["candidate"]): row for row in map_rows}
    outer_rows: list[dict[str, Any]] = []
    for context in contexts:
        selected_scale = next(float(row["scale"]) for row in scale_rows if row["context"] == context.key and bool(row["selected"]))
        for candidate in ALL_CANDIDATES:
            outer_rows.append({
                "context": context.key, "scope": context.scope, "label": context.label,
                "outer_group": context.outer_group, "candidate": candidate,
                "selected_conservative_scale": selected_scale,
                **_summarize(list(context_metrics[context.key][candidate].values())),
            })
    outer_lookup = {(row["context"], row["candidate"]): row for row in outer_rows}
    stress_rows = [row for row in outer_rows if row["scope"] != "repeated"]
    stress_lookup = {(row["scope"], int(row["outer_group"]), row["candidate"]): row for row in stress_rows}

    long_threshold = float(_quantile(list(hidden_rows.values()), float(config["stress"]["long_suffix_quantile"])))
    missing_values = {well_id: 1.0 - float(diagnostics[well_id]["hidden_gr_coverage"]) for well_id in well_ids}
    missing_threshold = float(_quantile(list(missing_values.values()), float(config["stress"]["high_missing_gr_quantile"])))
    pseudo_values = {well_id: float(diagnostics[well_id]["visible_pseudo_holdout_gain"]) for well_id in well_ids}
    pseudo_threshold = float(_quantile(list(pseudo_values.values()), float(config["stress"]["poor_pseudo_gain_quantile"])))
    margin_values = {well_id: float(diagnostics[well_id]["median_fingerprint_margin"]) for well_id in well_ids}
    margin_threshold = float(_quantile(list(margin_values.values()), float(config["stress"]["low_match_margin_quantile"])))
    special_sets = {
        "long_suffix": [well_id for well_id in well_ids if hidden_rows[well_id] >= long_threshold],
        "high_gr_missingness": [well_id for well_id in well_ids if missing_values[well_id] >= missing_threshold],
        "poor_visible_pseudo_gain": [well_id for well_id in well_ids if pseudo_values[well_id] <= pseudo_threshold],
        "low_match_margin": [well_id for well_id in well_ids if margin_values[well_id] <= margin_threshold],
    }
    special_rows: list[dict[str, Any]] = []
    for name, ids in special_sets.items():
        for candidate in ALL_CANDIDATES:
            special_rows.append({"slice": name, "candidate": candidate, **_summarize([final_metrics[candidate][well_id] for well_id in ids])})
    special_lookup = {(row["slice"], row["candidate"]): row for row in special_rows}

    target_rows: list[dict[str, Any]] = []
    for candidate in ("ridge_residual_datum", "ridge_residual_datum_trend", "ridge_residual_conservative", "ridge_no_e007_ablation", "shuffled_target_ridge"):
        target_rows.extend(_target_metrics(records, final_actions[candidate], candidate))
    target_lookup = {(row["candidate"], row["target"]): row for row in target_rows}

    selected_frequency: Counter[tuple[str, str]] = Counter()
    maps_with_e007 = 0
    for fold_map in fold_maps:
        version = str(fold_map["version"])
        map_has = False
        for context in repeated_contexts:
            if context.label != version:
                continue
            for candidate, names in selected_features[context.key].items():
                for name in names:
                    selected_frequency[(candidate, name)] += 1
                    if candidate == "ridge_residual_datum_trend" and name.startswith(E007_PREFIXES):
                        map_has = True
        maps_with_e007 += int(map_has)
    feature_frequency_rows = [
        {"candidate": candidate, "feature": name, "outer_fit_count": count}
        for (candidate, name), count in sorted(selected_frequency.items())
    ]

    repeated_scale_values = [
        next(float(row["scale"]) for row in scale_rows if row["context"] == context.key and bool(row["selected"]))
        for context in repeated_contexts
    ]
    scale_by_map: dict[str, list[float]] = defaultdict(list)
    for context, value in zip(repeated_contexts, repeated_scale_values):
        scale_by_map[context.label].append(value)
    scale_stability = {
        "nonzero_outer_cells": sum(value > 0.0 for value in repeated_scale_values),
        "maps_with_nonzero_median_scale": sum(_median(values) > 0.0 for values in scale_by_map.values()),
        "outer_scale_range": max(repeated_scale_values) - min(repeated_scale_values),
        "median_outer_scale": _median(repeated_scale_values),
    }

    map_wins: dict[str, int] = {}
    outer_wins: dict[str, int] = {}
    mean_map_rmse: dict[str, float] = {}
    gates: dict[str, dict[str, bool]] = {}
    base_summary = summaries["e006_nested_fusion"]
    last_summary = summaries["last_known_tvt"]
    global_control_stub = True
    for candidate in ELIGIBLE:
        map_wins[candidate] = sum(
            float(map_lookup[(str(fold["version"]), "e006_nested_fusion")]["rmse"])
            - float(map_lookup[(str(fold["version"]), candidate)]["rmse"])
            >= float(config["promotion"]["minimum_map_gain"])
            for fold in fold_maps
        )
        outer_wins[candidate] = sum(
            float(outer_lookup[(context.key, candidate)]["rmse"])
            < float(outer_lookup[(context.key, "e006_nested_fusion")]["rmse"])
            for context in repeated_contexts
        )
        mean_map_rmse[candidate] = _mean([float(map_lookup[(str(fold["version"]), candidate)]["rmse"]) for fold in fold_maps])
        spatial_min = min(
            float(stress_lookup[("spatial", group, "e006_nested_fusion")]["rmse"])
            - float(stress_lookup[("spatial", group, candidate)]["rmse"])
            for group in range(int(config["stress"]["spatial_bins"]))
        )
        typewell_min = min(
            float(stress_lookup[("typewell", group, "e006_nested_fusion")]["rmse"])
            - float(stress_lookup[("typewell", group, candidate)]["rmse"])
            for group in range(int(config["stress"]["typewell_clusters"]))
        )
        special_min = min(
            float(special_lookup[(name, "e006_nested_fusion")]["rmse"])
            - float(special_lookup[(name, candidate)]["rmse"])
            for name in special_sets
        )
        gates[candidate] = {
            "gain_vs_e006": float(base_summary["rmse"]) - float(summaries[candidate]["rmse"]) >= float(config["promotion"]["minimum_gain_vs_e006"]),
            "gain_vs_last_known": float(last_summary["rmse"]) - float(summaries[candidate]["rmse"]) >= float(config["promotion"]["minimum_gain_vs_last_known"]),
            "repeated_maps": map_wins[candidate] >= int(config["promotion"]["minimum_map_wins"]),
            "outer_cells": outer_wins[candidate] >= int(config["promotion"]["minimum_outer_cell_wins"]),
            "p90_vs_e006": float(summaries[candidate]["p90_well_rmse"]) - float(base_summary["p90_well_rmse"]) <= float(config["promotion"]["maximum_p90_deterioration_vs_e006"]),
            "p90_vs_last_known": float(summaries[candidate]["p90_well_rmse"]) - float(last_summary["p90_well_rmse"]) <= float(config["promotion"]["maximum_p90_deterioration_vs_last_known"]),
            "worst5_vs_e006": float(summaries[candidate]["worst_5pct_sse_share"]) - float(base_summary["worst_5pct_sse_share"]) <= float(config["promotion"]["maximum_worst5_sse_share_increase_vs_e006"]),
            "worst5_vs_last_known": float(summaries[candidate]["worst_5pct_sse_share"]) - float(last_summary["worst_5pct_sse_share"]) <= float(config["promotion"]["maximum_worst5_sse_share_increase_vs_last_known"]),
            "spatial_stress": spatial_min > 0.0 if bool(config["promotion"]["require_positive_spatial_group_gain_vs_e006"]) else True,
            "typewell_stress": typewell_min > 0.0 if bool(config["promotion"]["require_positive_typewell_group_gain_vs_e006"]) else True,
            "special_slices": special_min >= -float(config["promotion"]["maximum_special_slice_deterioration_vs_e006"]),
            "scale_stability": (
                scale_stability["nonzero_outer_cells"] >= int(config["controls"]["minimum_nonzero_outer_cells"])
                and scale_stability["maps_with_nonzero_median_scale"] >= int(config["controls"]["minimum_maps_with_nonzero_median_scale"])
                and scale_stability["outer_scale_range"] <= float(config["controls"]["maximum_outer_scale_range"])
            ) if candidate == "ridge_residual_conservative" else True,
            "controls": global_control_stub,
        }

    reported = min(ELIGIBLE, key=lambda name: (float(summaries[name]["rmse"]), ELIGIBLE.index(name)))
    best_legal_rmse = float(summaries[reported]["rmse"])
    oracle_rmse = float(summaries["oracle_residual_datum_trend"]["rmse"])
    e006_rmse = float(summaries["e006_nested_fusion"]["rmse"])
    e004_rmse = float(summaries["e004_geometry_prefix"]["rmse"])
    last_known_rmse = float(summaries["last_known_tvt"]["rmse"])
    parent_tolerance = float(config["controls"]["parent_rmse_tolerance"])
    parent_pass = (
        abs(e006_rmse - float(parent["e006_expected_rmse"])) <= parent_tolerance
        and abs(e004_rmse - float(parent["e004_expected_rmse"])) <= parent_tolerance
        and abs(last_known_rmse - float(parent["last_known_expected_rmse"])) <= parent_tolerance
    )
    shuffled_datum_corr = target_lookup[("shuffled_target_ridge", TARGETS[0])]["pearson"]
    shuffled_trend_corr = target_lookup[("shuffled_target_ridge", TARGETS[1])]["pearson"]
    e007_ablation_advantage = float(summaries["ridge_no_e007_ablation"]["rmse"]) - float(summaries["ridge_residual_datum_trend"]["rmse"])
    controls: dict[str, Any] = {
        "parent_artifacts": {
            "pass": parent_pass and bool(parent_alignment["pass"]),
            "e006_sha256": _sha256(e006_path),
            "e007_sha256": _sha256(e007_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
            "e006_rmse": e006_rmse,
            "expected_e006_rmse": float(parent["e006_expected_rmse"]),
            "e004_rmse": e004_rmse,
            "expected_e004_rmse": float(parent["e004_expected_rmse"]),
            "last_known_rmse": last_known_rmse,
            "expected_last_known_rmse": float(parent["last_known_expected_rmse"]),
            **parent_alignment,
        },
        "data_integrity": {"pass": len(well_ids) == int(config["expected_wells"]), "wells": len(well_ids), "data_signature": data["data_signature"]},
        "feature_leakage": leakage_control,
        "zero_action": {"pass": True, "maximum_prediction_delta": 0.0},
        "duplicate_feature": {"pass": duplicate_delta <= float(config["controls"]["duplicate_feature_maximum_prediction_delta"]), "maximum_action_delta": duplicate_delta},
        "shuffled_target": {
            "pass": e006_rmse - float(summaries["shuffled_target_ridge"]["rmse"]) <= float(config["controls"]["maximum_shuffled_gain_vs_e006"])
            and (shuffled_datum_corr is None or abs(float(shuffled_datum_corr)) <= float(config["controls"]["maximum_shuffled_abs_datum_correlation"]))
            and (shuffled_trend_corr is None or abs(float(shuffled_trend_corr)) <= float(config["controls"]["maximum_shuffled_abs_trend_correlation"])),
            "gain_vs_e006": e006_rmse - float(summaries["shuffled_target_ridge"]["rmse"]),
            "datum_correlation": shuffled_datum_corr,
            "trend_correlation": shuffled_trend_corr,
        },
        "oracle_positive": {"pass": best_legal_rmse - oracle_rmse >= float(config["controls"]["oracle_minimum_gain_vs_best_legal"]), "eligible": False, "best_legal_rmse": best_legal_rmse, "oracle_rmse": oracle_rmse},
        "e007_evidence": {
            "pass": maps_with_e007 >= int(config["controls"]["minimum_e007_feature_selection_maps"])
            and e007_ablation_advantage >= -float(config["controls"]["maximum_e007_ablation_advantage_ft"]),
            "maps_with_e007_selected": maps_with_e007,
            "full_minus_ablation_gain": e007_ablation_advantage,
        },
        "placement_membership": {"pass": all(bool(row["pass"]) for row in membership_rows), "contexts": len(membership_rows)},
        "scale_stability": {
            "pass": scale_stability["nonzero_outer_cells"] >= int(config["controls"]["minimum_nonzero_outer_cells"])
            and scale_stability["maps_with_nonzero_median_scale"] >= int(config["controls"]["minimum_maps_with_nonzero_median_scale"])
            and scale_stability["outer_scale_range"] <= float(config["controls"]["maximum_outer_scale_range"]),
            **scale_stability,
        },
    }

    full_model_predictions, full_model = _fit_predict(
        records=records, train_ids=well_ids, test_ids=well_ids,
        feature_names=feature_names_sorted, target_names=TARGETS, config=config,
    )
    full_oof_scale, _ = _select_scale(records, final_actions["ridge_residual_datum_trend"], well_ids, config)
    model_rows: list[dict[str, Any]] = []
    for index, name in enumerate(full_model.prepared.selected_names):
        model_rows.append({
            "model": "ridge_residual_datum_trend_full_fit",
            "feature": name,
            "median": full_model.prepared.medians[index],
            "mean": full_model.prepared.means[index],
            "scale": full_model.prepared.scales[index],
            "coefficient_datum": full_model.coefficients[index][0],
            "coefficient_trend": full_model.coefficients[index][1],
        })
    model_meta = {
        "schema_version": 1,
        "experiment_id": "E008",
        "code_sha": code_sha,
        "feature_version": config["feature_version"],
        "target_names": list(TARGETS),
        "target_means": list(full_model.prepared.y_means),
        "target_scales": list(full_model.prepared.y_scales),
        "datum_soft_cap_ft": float(config["targets"]["datum_soft_cap_ft"]),
        "trend_soft_cap_ft": float(config["targets"]["trend_soft_cap_ft"]),
        "ridge_alpha": float(config["ridge"]["alpha"]),
        "full_oof_conservative_scale": full_oof_scale,
        "selected_features": list(full_model.prepared.selected_names),
        "diagnostic_only_until_promoted": True,
    }

    oof_path = artifact_dir / "oof_predictions.csv.gz"
    fieldnames = ["id", "well_id", "row_index", "hidden_index", "target", *ALL_CANDIDATES]
    direct_sse = {candidate: 0.0 for candidate in ALL_CANDIDATES}
    direct_rows = 0
    maximum_correction = 0.0
    seen_oof_wells: set[str] = set()
    current_oof_well = ""
    expected_hidden_index = 0
    def oof_rows() -> Iterator[dict[str, Any]]:
        nonlocal direct_rows, maximum_correction, current_oof_well, expected_hidden_index
        with gzip.open(e007_path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                well_id = str(row["well_id"])
                hidden_index = int(row["hidden_index"])
                if well_id != current_oof_well:
                    if well_id in seen_oof_wells:
                        raise DataValidationError("E008 OOF wells are not contiguous")
                    seen_oof_wells.add(well_id)
                    current_oof_well = well_id
                    expected_hidden_index = 0
                if hidden_index != expected_hidden_index:
                    raise DataValidationError("E008 OOF hidden indices are not contiguous")
                expected_hidden_index += 1
                rows = hidden_rows[well_id]
                basis = _action_basis(hidden_index, rows)
                target = float(row["target"])
                values: dict[str, float] = {
                    "last_known_tvt": float(row["last_known_tvt"]),
                    "e004_geometry_prefix": float(row["e004_geometry_prefix"]),
                    "e006_nested_fusion": float(row["e006_nested_fusion"]),
                }
                base = values["e006_nested_fusion"]
                for candidate in ACTION_CANDIDATES:
                    datum, trend = final_actions[candidate][well_id]
                    correction = datum + trend * basis
                    if candidate not in {"oracle_residual_datum", "oracle_residual_datum_trend"}:
                        maximum_correction = max(maximum_correction, abs(correction))
                    values[candidate] = base + correction
                identifier = str(row["id"])
                if identifier != f"{well_id}_{int(row['row_index'])}":
                    raise DataValidationError("E008 OOF ID is not canonical")
                direct_rows += 1
                for candidate, value in values.items():
                    if not math.isfinite(value):
                        raise DataValidationError("E008 non-finite OOF prediction")
                    direct_sse[candidate] += (value - target) ** 2
                yield {
                    "id": identifier,
                    "well_id": well_id,
                    "row_index": row["row_index"],
                    "hidden_index": hidden_index,
                    "target": _format_row(target),
                    **{candidate: _format_row(values[candidate]) for candidate in ALL_CANDIDATES},
                }
    _write_gzip_csv(oof_path, fieldnames, oof_rows())
    maximum_relative = 0.0
    for candidate in ALL_CANDIDATES:
        expected = float(summaries[candidate]["sse"])
        maximum_relative = max(maximum_relative, abs(direct_sse[candidate] - expected) / max(1.0, expected))
    controls["pooled_sse_consistency"] = {"pass": maximum_relative <= float(config["controls"]["pooled_sse_relative_tolerance"]), "maximum_relative_difference": maximum_relative}
    controls["oof_identity"] = {
        "pass": direct_rows == int(data["hidden_rows"]) and len(seen_oof_wells) == len(well_ids),
        "rows": direct_rows,
        "wells": len(seen_oof_wells),
    }
    controls["correction_bound"] = {"pass": maximum_correction <= float(config["controls"]["maximum_absolute_emitted_correction_ft"]) + 1e-9, "maximum_absolute_correction": maximum_correction, "cap": float(config["controls"]["maximum_absolute_emitted_correction_ft"])}
    wall_seconds = time.perf_counter() - started
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    controls["runtime"] = {"pass": wall_seconds <= 60.0 * float(config["promotion"]["maximum_runtime_minutes"]), "wall_seconds": wall_seconds, "budget_minutes": float(config["promotion"]["maximum_runtime_minutes"])}
    controls["memory"] = {"pass": max_rss_kb <= 1024.0 * float(config["promotion"]["maximum_rss_mb"]), "max_rss_kb": max_rss_kb, "budget_mb": float(config["promotion"]["maximum_rss_mb"])}
    all_controls_pass = all(bool(value["pass"]) for value in controls.values())
    for candidate in ELIGIBLE:
        gates[candidate]["controls"] = all_controls_pass
        gates[candidate]["runtime"] = bool(controls["runtime"]["pass"])
        gates[candidate]["memory"] = bool(controls["memory"]["pass"])
        gates[candidate]["correction_bound"] = bool(controls["correction_bound"]["pass"])
        gates[candidate]["e007_evidence"] = bool(controls["e007_evidence"]["pass"])
    passing = [candidate for candidate in ELIGIBLE if all(gates[candidate].values())]
    selected = min(passing, key=lambda name: (mean_map_rmse[name], ELIGIBLE.index(name))) if passing else None
    status = "promoted" if selected else "rejected"

    reported_metric = final_metrics[selected or reported]
    well_rows = []
    for well_id in well_ids:
        metric = reported_metric[well_id]
        well_rows.append({
            "well_id": well_id, "split": "oof", "candidate": selected or reported,
            "promoted": bool(selected), "rows_scored": metric.rows_scored, "rmse": metric.rmse,
            "mean_error": metric.mean_error, "sse": metric.sse, "regime": "e008_residual_action",
            "uncertainty": abs(final_actions[selected or reported][well_id][0]) if (selected or reported) in ACTION_CANDIDATES else 0.0,
            "datum_sse": metric.datum_sse, "trend_sse": metric.trend_sse,
            "shape_sse": metric.shape_sse, "trend_per_row": metric.trend_per_row,
        })
    feature_rows = [{"well_id": well_id, **{name: records[well_id]["features"].get(name) for name in feature_names_sorted}} for well_id in well_ids]
    residual_target_rows = [{"well_id": well_id, **records[well_id]["targets"]} for well_id in well_ids]
    candidate_rows = [{"candidate": candidate, "eligible": candidate in ELIGIBLE, "selected": candidate == selected, "reported": candidate == (selected or reported), **summaries[candidate]} for candidate in ALL_CANDIDATES]
    control_rows = [{"control": name, "pass": bool(value["pass"]), "details": _canonical_json(value)} for name, value in controls.items()]
    output_files = {
        "candidate_metrics.csv": (candidate_rows, None),
        "control_metrics.csv": (control_rows, None),
        "legal_features.csv": (feature_rows, None),
        "residual_targets.csv": (residual_target_rows, None),
        "target_metrics.csv": (target_rows, None),
        "selected_feature_frequency.csv": (feature_frequency_rows, None),
        "model_coefficients.csv": (model_rows, None),
        "map_metrics.csv": (map_rows, None),
        "outer_cell_metrics.csv": (outer_rows, None),
        "stress_metrics.csv": (stress_rows, None),
        "special_slice_metrics.csv": (special_rows, None),
        "scale_metrics.csv": (scale_rows, None),
        "membership_audit.csv": (membership_rows, None),
        "selected_well_metrics.csv": (well_rows, None),
    }
    for filename, (rows, _) in output_files.items():
        rows_list = list(rows)
        if not rows_list:
            raise DataValidationError(f"E008 output {filename} is empty")
        _write_csv(output_dir / filename, list(rows_list[0]), rows_list)
    _write_json(output_dir / "model.json", model_meta)
    summary = {
        "schema_version": 1,
        "experiment_id": "E008",
        "status": status,
        "code_sha": code_sha,
        "selected_candidate": selected,
        "reported_candidate": selected or reported,
        "eligible_candidates": passing,
        "data": data,
        "baseline_metrics": summaries["last_known_tvt"],
        "e006_metrics": summaries["e006_nested_fusion"],
        "candidate_metrics": {candidate: summaries[candidate] for candidate in ALL_CANDIDATES},
        "map_wins": map_wins,
        "outer_cell_wins": outer_wins,
        "mean_map_rmse": mean_map_rmse,
        "gates_by_candidate": gates,
        "controls": controls,
        "target_metrics": target_rows,
        "feature_count": len(feature_names_sorted),
        "maps_with_e007_selected": maps_with_e007,
        "scale_stability": scale_stability,
        "stress": {
            "long_suffix_threshold": long_threshold,
            "high_gr_missingness_threshold": missing_threshold,
            "poor_pseudo_gain_threshold": pseudo_threshold,
            "low_match_margin_threshold": margin_threshold,
            "special_slice_wells": {name: len(ids) for name, ids in special_sets.items()},
        },
        "runtime": {"wall_seconds": wall_seconds, "max_rss_kb": max_rss_kb},
        "deployment": {
            "statistically_authorized": bool(selected),
            "local_package_built": False,
            "local_notebook_parity": False,
            "private_internet_disabled_kaggle_parity": False,
            "deployment_ready": False,
            "submission_created": False,
            "submission_made": False,
            "reason": "Statistical promotion passed; deployment package still required." if selected else "No E008 candidate passed every frozen gate; retain E006.",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    result_files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E008",
        "code_sha": code_sha,
        "config": {"path": str((root / "experiments/E008/config.json").relative_to(root)), "sha256": _sha256(root / "experiments/E008/config.json"), "bytes": (root / "experiments/E008/config.json").stat().st_size},
        "parent_artifacts": [
            {"path": str(parent["e006_oof"]), "sha256": _sha256(e006_path), "expected_sha256": str(parent["e006_oof_sha256"])},
            {"path": str(parent["e007_oof"]), "sha256": _sha256(e007_path), "expected_sha256": str(parent["e007_oof_sha256"])},
            {"path": str(parent["e007_diagnostics"]), "sha256": _sha256(diagnostics_path), "expected_sha256": str(parent["e007_diagnostics_sha256"])},
            {"path": str(parent["e006_model"]), "sha256": _sha256(root / str(parent["e006_model"])), "expected_sha256": str(parent["e006_model_sha256"])},
        ],
        "fold_files": [{"path": path, "sha256": _sha256(root / path), "bytes": (root / path).stat().st_size} for path in config["fold_files"]],
        "files": [{"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size} for path in result_files],
        "external_artifacts": [{"path": str(oof_path.relative_to(root)), "kind": "oof_predictions_gzip", "sha256": _sha256(oof_path), "bytes": oof_path.stat().st_size, "tracked": False}],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return summary
