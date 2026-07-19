"""E003 datum, trend, and risk learnability study.

All predictive features in this module are available from the supplied test-time
well and typewell files. Hidden TVT values are used only to construct oracle
analysis targets and to evaluate strictly cross-fitted predictions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .harness import (
    DataValidationError,
    ErrorAccumulator,
    WellMetric,
    WellProfile,
    _canonical_json,
    _quantile,
    _sha256,
    _summarize,
    _write_csv,
    _write_json,
    scan_profiles,
)
from .structural import MISSING_VALUES, huber_line

SURFACES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
MODEL_TARGETS = (
    "datum_correction_ft",
    "trend_correction_ft",
    "u_slope_delta_ft_per_md",
    "log1p_baseline_rmse",
)
MODEL_NAMES = ("ridge", "forest")
ACTION_CANDIDATES = (
    "ridge_datum",
    "ridge_datum_trend",
    "forest_datum",
    "forest_datum_trend",
)
NO_EVIDENCE_ACTIONS = ("mean_datum", "mean_datum_trend")
ALL_ACTIONS = NO_EVIDENCE_ACTIONS + ACTION_CANDIDATES
PREDICTION_MODELS = ("mean", "ridge", "forest")


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _optional_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() in MISSING_VALUES:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _paired(xs: Sequence[float], values: Sequence[float | None]) -> tuple[list[float], list[float]]:
    clean_x: list[float] = []
    clean_y: list[float] = []
    for x, value in zip(xs, values):
        if value is not None and math.isfinite(value):
            clean_x.append(float(x))
            clean_y.append(float(value))
    return clean_x, clean_y


def _series_summary(xs: Sequence[float], values: Sequence[float | None]) -> dict[str, float | None]:
    clean_x, clean_y = _paired(xs, values)
    if not clean_y:
        return {"mean": None, "std": None, "range": None, "delta": None, "slope": None}
    return {
        "mean": _mean(clean_y),
        "std": _std(clean_y),
        "range": max(clean_y) - min(clean_y),
        "delta": clean_y[-1] - clean_y[0],
        "slope": _linear_slope(clean_x, clean_y),
    }


def _add_summary(features: dict[str, float | None], prefix: str, xs: Sequence[float], values: Sequence[float | None]) -> None:
    for name, value in _series_summary(xs, values).items():
        features[f"{prefix}_{name}"] = value


def _missing_fraction(values: Sequence[float | None]) -> float:
    return sum(value is None for value in values) / max(1, len(values))


def _longest_missing_run(values: Sequence[float | None]) -> int:
    longest = current = 0
    for value in values:
        if value is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = sum((x - mean_x) ** 2 for x in xs)
    denominator_y = sum((y - mean_y) ** 2 for y in ys)
    if denominator_x <= 0.0 or denominator_y <= 0.0:
        return None
    return numerator / math.sqrt(denominator_x * denominator_y)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_values = _ranks(scores)
    rank_sum = sum(rank for rank, label in zip(rank_values, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _sign_accuracy(actual: Sequence[float], predicted: Sequence[float], threshold: float) -> tuple[float | None, int]:
    selected = [
        (a, p)
        for a, p in zip(actual, predicted)
        if abs(a) >= threshold
    ]
    if not selected:
        return None, 0
    correct = sum((a > 0.0) == (p > 0.0) for a, p in selected)
    return correct / len(selected), len(selected)


def _load_fold_maps(root: Path, paths: Sequence[str], signature: str) -> list[dict[str, Any]]:
    fold_maps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative in paths:
        data = json.loads((root / relative).read_text(encoding="utf-8"))
        if data.get("data_signature") != signature:
            raise DataValidationError(f"{relative}: E003 fold data signature mismatch")
        fingerprint = str(data.get("fingerprint") or "")
        if not fingerprint or fingerprint in seen:
            raise DataValidationError(f"{relative}: missing or duplicate fold fingerprint")
        seen.add(fingerprint)
        fold_maps.append(data)
    if not fold_maps:
        raise DataValidationError("E003 requires at least one frozen fold map")
    return fold_maps


def _read_horizontal(path: Path) -> dict[str, list[float | None]]:
    columns: dict[str, list[float | None]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"MD", "X", "Y", "Z", "TVT", "TVT_input", "GR", *SURFACES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{path}: missing E003 columns {sorted(missing)}")
        for row in reader:
            for name in reader.fieldnames or []:
                columns[name].append(_optional_float(row.get(name)))
    return dict(columns)


def _typewell_features(path: Path, duplicate_count: int) -> dict[str, float | None]:
    tvt: list[float | None] = []
    gr: list[float | None] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tvt.append(_optional_float(row.get("TVT")))
            gr.append(_optional_float(row.get("GR")))
    clean_tvt = [value for value in tvt if value is not None]
    clean_gr = [value for value in gr if value is not None]
    paired_tvt, paired_gr = _paired([float(index) for index in range(len(gr))], gr)
    return {
        "typewell_rows": float(len(tvt)),
        "typewell_tvt_span": (max(clean_tvt) - min(clean_tvt)) if clean_tvt else None,
        "typewell_gr_mean": _mean(clean_gr) if clean_gr else None,
        "typewell_gr_std": _std(clean_gr) if clean_gr else None,
        "typewell_gr_range": (max(clean_gr) - min(clean_gr)) if clean_gr else None,
        "typewell_gr_index_slope": _linear_slope(paired_tvt, paired_gr) if paired_gr else None,
        "typewell_gr_missing_fraction": _missing_fraction(gr),
        "typewell_duplicate_count": float(duplicate_count),
    }


def _visible_backtests(
    md: Sequence[float],
    z: Sequence[float],
    tvt: Sequence[float],
    fractions: Sequence[float],
) -> dict[str, float | None]:
    features: dict[str, float | None] = {}
    count = len(tvt)
    for fraction in fractions:
        label = str(fraction).replace(".", "p")
        cut = min(count - 2, max(2, int(round(count * fraction))))
        if cut < 2 or count - cut < 2:
            for suffix in ("mean", "rmse", "trend", "toe"):
                features[f"backtest_{label}_{suffix}"] = None
            continue
        baseline = tvt[cut - 1]
        residuals = [value - baseline for value in tvt[cut:]]
        centered = [index / max(1, len(residuals) - 1) - 0.5 for index in range(len(residuals))]
        features[f"backtest_{label}_mean"] = _mean(residuals)
        features[f"backtest_{label}_rmse"] = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        features[f"backtest_{label}_trend"] = _linear_slope(centered, residuals)
        features[f"backtest_{label}_toe"] = residuals[-1]
        u_before = [target + vertical for target, vertical in zip(tvt[:cut], z[:cut])]
        u_after = [target + vertical for target, vertical in zip(tvt[cut:], z[cut:])]
        features[f"backtest_{label}_u_slope_delta"] = (
            _linear_slope(md[cut:], u_after) - _linear_slope(md[:cut], u_before)
        )
    return features


def _metric_from_sufficient(well_id: str, sufficient: Mapping[str, float], datum: float, trend: float) -> WellMetric:
    n = int(sufficient["rows"])
    sum_i = float(sufficient["sum_i"])
    sum_i_sq = float(sufficient["sum_i_sq"])
    sum_r = float(sufficient["sum_r"])
    sum_r_sq = float(sufficient["sum_r_sq"])
    sum_i_r = float(sufficient["sum_i_r"])
    if n <= 1:
        scale = 0.0
        offset = 0.0
    else:
        scale = 1.0 / (n - 1)
        offset = -0.5
    intercept = datum + trend * offset
    slope = trend * scale
    sum_error = n * intercept + slope * sum_i - sum_r
    sum_error_sq = (
        n * intercept * intercept
        + 2.0 * intercept * slope * sum_i
        + slope * slope * sum_i_sq
        - 2.0 * intercept * sum_r
        - 2.0 * slope * sum_i_r
        + sum_r_sq
    )
    sum_i_error = intercept * sum_i + slope * sum_i_sq - sum_i_r
    tolerance = max(1e-9, abs(sum_error_sq) * 1e-12)
    if sum_error_sq < 0.0 and abs(sum_error_sq) <= tolerance:
        sum_error_sq = 0.0
    if sum_error_sq < 0.0:
        raise ArithmeticError(f"negative E003 SSE for {well_id}: {sum_error_sq}")
    count = float(n)
    mean_error = sum_error / count
    centered_i_sq = sum_i_sq - sum_i * sum_i / count
    centered_i_error = sum_i_error - sum_i * sum_error / count
    row_slope = centered_i_error / centered_i_sq if centered_i_sq > 0.0 else 0.0
    datum_sse = sum_error * sum_error / count
    trend_sse = row_slope * row_slope * centered_i_sq
    shape_sse = sum_error_sq - datum_sse - trend_sse
    if shape_sse < 0.0 and abs(shape_sse) <= tolerance:
        shape_sse = 0.0
    if shape_sse < 0.0:
        raise ArithmeticError(f"negative E003 shape SSE for {well_id}: {shape_sse}")
    return WellMetric(
        well_id=well_id,
        rows_scored=n,
        rmse=math.sqrt(sum_error_sq / count),
        mean_error=mean_error,
        sse=sum_error_sq,
        datum_sse=datum_sse,
        trend_sse=trend_sse,
        shape_sse=shape_sse,
        trend_per_row=row_slope,
    )


def _extract_records(
    *,
    train_dir: Path,
    profiles: Sequence[WellProfile],
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    type_hashes: dict[str, str] = {}
    duplicate_counts: Counter[str] = Counter()
    for profile in profiles:
        path = train_dir / f"{profile.well_id}__typewell.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        type_hashes[profile.well_id] = digest
        duplicate_counts[digest] += 1

    records: dict[str, dict[str, Any]] = {}
    feature_names: set[str] = set()
    windows = [int(value) for value in config["visible_slope_windows"]]
    fractions = [float(value) for value in config["visible_backtest_fractions"]]

    for profile in sorted(profiles, key=lambda item: item.well_id):
        path = train_dir / f"{profile.well_id}__horizontal_well.csv"
        columns = _read_horizontal(path)
        total_rows = len(columns["MD"])
        known_rows = profile.known_rows
        hidden_rows = profile.hidden_rows
        if total_rows != known_rows + hidden_rows:
            raise DataValidationError(f"{path}: E003 row accounting mismatch")
        required_numeric = ("MD", "X", "Y", "Z", "TVT")
        for name in required_numeric:
            if any(value is None for value in columns[name]):
                raise DataValidationError(f"{path}: missing required E003 numeric column {name}")
        md = [float(value) for value in columns["MD"] if value is not None]
        x = [float(value) for value in columns["X"] if value is not None]
        y = [float(value) for value in columns["Y"] if value is not None]
        z = [float(value) for value in columns["Z"] if value is not None]
        tvt = [float(value) for value in columns["TVT"] if value is not None]
        visible_tvt = [
            float(value) for value in columns["TVT_input"][:known_rows] if value is not None
        ]
        if len(visible_tvt) != known_rows:
            raise DataValidationError(f"{path}: E003 visible target count mismatch")
        known_md, hidden_md = md[:known_rows], md[known_rows:]
        known_x, hidden_x = x[:known_rows], x[known_rows:]
        known_y, hidden_y = y[:known_rows], y[known_rows:]
        known_z, hidden_z = z[:known_rows], z[known_rows:]
        hidden_tvt = tvt[known_rows:]
        visible_u = [target + vertical for target, vertical in zip(visible_tvt, known_z)]
        last_tvt = visible_tvt[-1]
        residuals = [target - last_tvt for target in hidden_tvt]
        accumulator = ErrorAccumulator()
        sum_i = sum_i_sq = sum_r = sum_r_sq = sum_i_r = 0.0
        centered_x: list[float] = []
        for index, residual in enumerate(residuals):
            accumulator.add(-residual, float(index))
            value_i = float(index)
            sum_i += value_i
            sum_i_sq += value_i * value_i
            sum_r += residual
            sum_r_sq += residual * residual
            sum_i_r += value_i * residual
            centered_x.append(index / max(1, hidden_rows - 1) - 0.5)
        baseline_metric = accumulator.finalize(profile.well_id)
        sum_centered_sq = sum(value * value for value in centered_x)
        sum_centered_residual = sum(value * residual for value, residual in zip(centered_x, residuals))
        datum_target = sum_r / hidden_rows
        trend_target = sum_centered_residual / sum_centered_sq if sum_centered_sq > 0.0 else 0.0
        hidden_u = [target + vertical for target, vertical in zip(hidden_tvt, hidden_z)]
        visible_u_slope = _linear_slope(known_md, visible_u)
        hidden_u_slope = _linear_slope(hidden_md, hidden_u)
        hidden_tvt_slope = _linear_slope(hidden_md, hidden_tvt)
        features: dict[str, float | None] = {
            "total_rows": float(total_rows),
            "known_rows": float(known_rows),
            "hidden_rows": float(hidden_rows),
            "hidden_fraction": hidden_rows / total_rows,
            "md_known_span": known_md[-1] - known_md[0],
            "md_hidden_span": hidden_md[-1] - known_md[-1],
            "md_total_span": md[-1] - md[0],
            "last_visible_tvt": last_tvt,
            "last_visible_z": known_z[-1],
            "last_visible_u": visible_u[-1],
            "last_visible_x": known_x[-1],
            "last_visible_y": known_y[-1],
            "x_total_delta": x[-1] - x[0],
            "y_total_delta": y[-1] - y[0],
            "z_total_delta": z[-1] - z[0],
            "x_hidden_delta": hidden_x[-1] - known_x[-1],
            "y_hidden_delta": hidden_y[-1] - known_y[-1],
            "z_hidden_delta": hidden_z[-1] - known_z[-1],
            "horizontal_total_distance": math.hypot(x[-1] - x[0], y[-1] - y[0]),
            "horizontal_hidden_distance": math.hypot(hidden_x[-1] - known_x[-1], hidden_y[-1] - known_y[-1]),
        }
        angle = math.atan2(hidden_y[-1] - known_y[-1], hidden_x[-1] - known_x[-1])
        features["hidden_azimuth_sin"] = math.sin(angle)
        features["hidden_azimuth_cos"] = math.cos(angle)
        _add_summary(features, "visible_tvt", known_md, visible_tvt)
        _add_summary(features, "visible_z", known_md, known_z)
        _add_summary(features, "visible_u", known_md, visible_u)
        _add_summary(features, "hidden_z", hidden_md, hidden_z)
        _add_summary(features, "visible_gr", known_md, columns["GR"][:known_rows])
        _add_summary(features, "hidden_gr", hidden_md, columns["GR"][known_rows:])
        _add_summary(features, "whole_gr", md, columns["GR"])
        features["visible_gr_missing_fraction"] = _missing_fraction(columns["GR"][:known_rows])
        features["hidden_gr_missing_fraction"] = _missing_fraction(columns["GR"][known_rows:])
        features["whole_gr_missing_fraction"] = _missing_fraction(columns["GR"])
        features["hidden_gr_longest_missing_run"] = float(_longest_missing_run(columns["GR"][known_rows:]))

        full_u_slope = _linear_slope(known_md, visible_u)
        for window in windows:
            count = min(window, known_rows)
            if count < 2:
                features[f"visible_u_slope_w{window}"] = None
                features[f"visible_u_scale_w{window}"] = None
                features[f"visible_u_slope_delta_w{window}"] = None
                features[f"visible_tvt_slope_w{window}"] = None
                continue
            _, u_slope, u_scale = huber_line(
                known_md[-count:],
                visible_u[-count:],
                window_rows=count,
                huber_k=1.5,
                iterations=6,
            )
            _, tvt_slope, _ = huber_line(
                known_md[-count:],
                visible_tvt[-count:],
                window_rows=count,
                huber_k=1.5,
                iterations=6,
            )
            features[f"visible_u_slope_w{window}"] = u_slope
            features[f"visible_u_scale_w{window}"] = u_scale
            features[f"visible_u_slope_delta_w{window}"] = u_slope - full_u_slope
            features[f"visible_tvt_slope_w{window}"] = tvt_slope
        features.update(_visible_backtests(known_md, known_z, visible_tvt, fractions))

        for surface in SURFACES:
            values = columns[surface]
            visible_values = values[:known_rows]
            hidden_values = values[known_rows:]
            visible_xs, visible_clean = _paired(known_md, visible_values)
            hidden_xs, hidden_clean = _paired(hidden_md, hidden_values)
            prefix = surface.lower()
            features[f"{prefix}_visible_last"] = visible_clean[-1] if visible_clean else None
            features[f"{prefix}_visible_slope"] = _linear_slope(visible_xs, visible_clean) if visible_clean else None
            features[f"{prefix}_hidden_delta"] = (hidden_clean[-1] - visible_clean[-1]) if visible_clean and hidden_clean else None
            features[f"{prefix}_hidden_slope"] = _linear_slope(hidden_xs, hidden_clean) if hidden_clean else None
            features[f"{prefix}_hidden_range"] = (max(hidden_clean) - min(hidden_clean)) if hidden_clean else None
            features[f"{prefix}_last_offset_to_z"] = (visible_clean[-1] - known_z[-1]) if visible_clean else None
            features[f"{prefix}_hidden_offset_change_to_z"] = (
                (hidden_clean[-1] - hidden_z[-1]) - (visible_clean[-1] - known_z[-1])
                if visible_clean and hidden_clean else None
            )
            features[f"{prefix}_missing_fraction"] = _missing_fraction(values)

        type_path = train_dir / f"{profile.well_id}__typewell.csv"
        features.update(_typewell_features(type_path, duplicate_counts[type_hashes[profile.well_id]]))
        features["spatial_x_mid"] = 0.5 * (x[0] + x[-1])
        features["spatial_y_mid"] = 0.5 * (y[0] + y[-1])
        synthetic = (
            1.7 * math.log1p(hidden_rows)
            + 0.9 * float(features["z_hidden_delta"])
            + 40.0 * full_u_slope
        )
        oracle_datum_metric = _metric_from_sufficient(
            profile.well_id,
            {
                "rows": float(hidden_rows),
                "sum_i": sum_i,
                "sum_i_sq": sum_i_sq,
                "sum_r": sum_r,
                "sum_r_sq": sum_r_sq,
                "sum_i_r": sum_i_r,
            },
            datum_target,
            0.0,
        )
        oracle_trend_metric = _metric_from_sufficient(
            profile.well_id,
            {
                "rows": float(hidden_rows),
                "sum_i": sum_i,
                "sum_i_sq": sum_i_sq,
                "sum_r": sum_r,
                "sum_r_sq": sum_r_sq,
                "sum_i_r": sum_i_r,
            },
            datum_target,
            trend_target,
        )
        records[profile.well_id] = {
            "features": features,
            "targets": {
                "datum_correction_ft": datum_target,
                "trend_correction_ft": trend_target,
                "u_slope_delta_ft_per_md": hidden_u_slope - visible_u_slope,
                "log1p_baseline_rmse": math.log1p(baseline_metric.rmse),
                "hidden_tvt_slope_ft_per_md": hidden_tvt_slope,
                "hidden_u_slope_ft_per_md": hidden_u_slope,
                "visible_u_slope_ft_per_md": visible_u_slope,
                "toe_drift_ft": residuals[-1],
                "synthetic_positive": synthetic,
            },
            "sufficient": {
                "rows": float(hidden_rows),
                "sum_i": sum_i,
                "sum_i_sq": sum_i_sq,
                "sum_r": sum_r,
                "sum_r_sq": sum_r_sq,
                "sum_i_r": sum_i_r,
            },
            "baseline_metric": baseline_metric,
            "oracle_datum_metric": oracle_datum_metric,
            "oracle_trend_metric": oracle_trend_metric,
            "spatial_x": float(features["spatial_x_mid"]),
            "spatial_y": float(features["spatial_y_mid"]),
        }
        feature_names.update(features)
    return records, sorted(feature_names)


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _prepare_split(
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    maximum_features: int,
) -> dict[str, Any]:
    medians: list[float] = []
    means: list[float] = []
    scales: list[float] = []
    train_full: list[list[float]] = [[] for _ in train_ids]
    test_full: list[list[float]] = [[] for _ in test_ids]
    for name in feature_names:
        finite_train = [
            float(records[well_id]["features"][name])
            for well_id in train_ids
            if records[well_id]["features"].get(name) is not None
            and math.isfinite(float(records[well_id]["features"][name]))
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
        medians.append(median)
        means.append(center)
        scales.append(scale)
        for row_index, value in enumerate(train_column):
            train_full[row_index].append((value - center) / scale)
        for row_index, value in enumerate(test_column):
            test_full[row_index].append((value - center) / scale)

    y_means: list[float] = []
    y_scales: list[float] = []
    train_y: list[list[float]] = [[] for _ in train_ids]
    for target in target_names:
        values = [float(records[well_id]["targets"][target]) for well_id in train_ids]
        center = _mean(values)
        scale = _std(values)
        if scale <= 1e-12:
            scale = 1.0
        y_means.append(center)
        y_scales.append(scale)
        for row_index, value in enumerate(values):
            train_y[row_index].append((value - center) / scale)

    feature_scores: list[tuple[float, str, int]] = []
    for feature_index, name in enumerate(feature_names):
        column = [row[feature_index] for row in train_full]
        score = 0.0
        for target_index in range(len(target_names)):
            target_column = [row[target_index] for row in train_y]
            correlation = _pearson(column, target_column)
            score = max(score, abs(correlation) if correlation is not None else 0.0)
        feature_scores.append((score, name, feature_index))
    chosen = sorted(feature_scores, key=lambda item: (-item[0], item[1]))[: max(1, int(maximum_features))]
    chosen_indices = [item[2] for item in chosen]
    selected_names = [item[1] for item in chosen]
    train_x = [[row[index] for index in chosen_indices] for row in train_full]
    test_x = [[row[index] for index in chosen_indices] for row in test_full]
    return {
        "train_x": train_x,
        "test_x": test_x,
        "train_y": train_y,
        "y_means": y_means,
        "y_scales": y_scales,
        "selected_names": selected_names,
    }


def _solve_multi(matrix: Sequence[Sequence[float]], rhs: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    outputs = len(rhs[0]) if rhs else 0
    augmented = [list(matrix[row]) + list(rhs[row]) for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            continue
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + outputs):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) <= 1e-18:
                continue
            for index in range(column, size + outputs):
                augmented[row][index] -= factor * augmented[column][index]
    return [row[size:] for row in augmented]


def _ridge_fit(train_x: Sequence[Sequence[float]], train_y: Sequence[Sequence[float]], alpha: float) -> list[list[float]]:
    if not train_x or not train_y:
        raise ValueError("ridge requires non-empty training data")
    feature_count = len(train_x[0])
    output_count = len(train_y[0])
    matrix = [[0.0] * feature_count for _ in range(feature_count)]
    rhs = [[0.0] * output_count for _ in range(feature_count)]
    for index in range(feature_count):
        matrix[index][index] = float(alpha)
    for x_row, y_row in zip(train_x, train_y):
        for first in range(feature_count):
            x_first = x_row[first]
            if x_first == 0.0:
                continue
            for second in range(first, feature_count):
                matrix[first][second] += x_first * x_row[second]
            for output in range(output_count):
                rhs[first][output] += x_first * y_row[output]
    for first in range(feature_count):
        for second in range(first):
            matrix[first][second] = matrix[second][first]
    return _solve_multi(matrix, rhs)


def _linear_predict(x_rows: Sequence[Sequence[float]], coefficients: Sequence[Sequence[float]]) -> list[list[float]]:
    if not coefficients:
        return [[] for _ in x_rows]
    output_count = len(coefficients[0])
    return [
        [
            sum(value * coefficients[feature][output] for feature, value in enumerate(row))
            for output in range(output_count)
        ]
        for row in x_rows
    ]


def _unscale_predictions(
    predictions: Sequence[Sequence[float]],
    means: Sequence[float],
    scales: Sequence[float],
) -> list[list[float]]:
    return [
        [means[index] + scales[index] * value for index, value in enumerate(row)]
        for row in predictions
    ]


def _node_mean(indices: Sequence[int], targets: Sequence[Sequence[float]]) -> list[float]:
    outputs = len(targets[0])
    return [sum(targets[index][output] for index in indices) / len(indices) for output in range(outputs)]


def _node_sse(indices: Sequence[int], targets: Sequence[Sequence[float]]) -> float:
    means = _node_mean(indices, targets)
    return sum(
        sum((targets[index][output] - means[output]) ** 2 for output in range(len(means)))
        for index in indices
    )


def _build_tree(
    x_rows: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    indices: Sequence[int],
    *,
    depth: int,
    max_depth: int,
    min_leaf: int,
    feature_subsample: int,
    quantiles: Sequence[float],
    rng: random.Random,
) -> dict[str, Any]:
    value = _node_mean(indices, targets)
    if depth >= max_depth or len(indices) < 2 * min_leaf:
        return {"value": value}
    feature_count = len(x_rows[0])
    candidates = list(range(feature_count))
    rng.shuffle(candidates)
    candidates = candidates[: min(feature_count, feature_subsample)]
    parent_sse = _node_sse(indices, targets)
    best: tuple[float, int, float, list[int], list[int]] | None = None
    for feature in candidates:
        ordered = sorted(indices, key=lambda index: (x_rows[index][feature], index))
        for quantile in quantiles:
            position = int(round(quantile * (len(ordered) - 1))) + 1
            position = max(min_leaf, min(len(ordered) - min_leaf, position))
            if position <= 0 or position >= len(ordered):
                continue
            left_value = x_rows[ordered[position - 1]][feature]
            right_value = x_rows[ordered[position]][feature]
            if left_value == right_value:
                continue
            left = ordered[:position]
            right = ordered[position:]
            loss = _node_sse(left, targets) + _node_sse(right, targets)
            threshold = 0.5 * (left_value + right_value)
            candidate = (loss, feature, threshold, left, right)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None or best[0] >= parent_sse - 1e-12:
        return {"value": value}
    _, feature, threshold, left, right = best
    return {
        "value": value,
        "feature": feature,
        "threshold": threshold,
        "left": _build_tree(
            x_rows,
            targets,
            left,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
            feature_subsample=feature_subsample,
            quantiles=quantiles,
            rng=rng,
        ),
        "right": _build_tree(
            x_rows,
            targets,
            right,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
            feature_subsample=feature_subsample,
            quantiles=quantiles,
            rng=rng,
        ),
    }


def _tree_predict(node: Mapping[str, Any], row: Sequence[float]) -> list[float]:
    current = node
    while "feature" in current:
        current = current["left"] if row[int(current["feature"])] <= float(current["threshold"]) else current["right"]
    return [float(value) for value in current["value"]]


def _forest_predict(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[Sequence[float]],
    test_x: Sequence[Sequence[float]],
    config: Mapping[str, Any],
    seed: int,
) -> list[list[float]]:
    rng = random.Random(seed)
    tree_count = int(config["trees"])
    min_leaf = int(config["min_leaf"])
    sample_size = max(2 * min_leaf, int(round(float(config["row_fraction"]) * len(train_x))))
    sample_size = min(len(train_x), sample_size)
    sums = [[0.0] * len(train_y[0]) for _ in test_x]
    for tree_index in range(tree_count):
        tree_rng = random.Random(rng.randrange(1 << 63) ^ tree_index)
        indices = tree_rng.sample(range(len(train_x)), sample_size)
        tree = _build_tree(
            train_x,
            train_y,
            indices,
            depth=0,
            max_depth=int(config["max_depth"]),
            min_leaf=min_leaf,
            feature_subsample=int(config["feature_subsample"]),
            quantiles=[float(value) for value in config["threshold_quantiles"]],
            rng=tree_rng,
        )
        for row_index, row in enumerate(test_x):
            prediction = _tree_predict(tree, row)
            for output, value in enumerate(prediction):
                sums[row_index][output] += value
    return [[value / tree_count for value in row] for row in sums]


def _fit_predict_split(
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    prepared = _prepare_split(
        records,
        train_ids,
        test_ids,
        feature_names,
        target_names,
        int(config["feature_screening"]["maximum_features"]),
    )
    ridge_coefficients = _ridge_fit(
        prepared["train_x"],
        prepared["train_y"],
        float(config["ridge"]["alpha"]),
    )
    ridge_scaled = _linear_predict(prepared["test_x"], ridge_coefficients)
    forest_scaled = _forest_predict(
        prepared["train_x"],
        prepared["train_y"],
        prepared["test_x"],
        config["forest"],
        seed,
    )
    ridge = _unscale_predictions(ridge_scaled, prepared["y_means"], prepared["y_scales"])
    forest = _unscale_predictions(forest_scaled, prepared["y_means"], prepared["y_scales"])
    return {
        well_id: {
            "mean": [float(value) for value in prepared["y_means"]],
            "ridge": ridge[index],
            "forest": forest[index],
        }
        for index, well_id in enumerate(test_ids)
    }, {**prepared, "ridge_coefficients": ridge_coefficients}


def _crossfit(
    records: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    assignments: Mapping[str, int],
    n_folds: int,
    config: Mapping[str, Any],
    seed_prefix: object,
    *,
    collect_controls: bool = False,
) -> tuple[dict[str, dict[str, list[float]]], Counter[str], dict[str, Any]]:
    predictions: dict[str, dict[str, list[float]]] = {}
    feature_frequency: Counter[str] = Counter()
    control_data: dict[str, Any] = {"duplicate_max_delta": 0.0, "shuffled": {}}
    all_ids = sorted(records)
    for fold in range(n_folds):
        test_ids = [well_id for well_id in all_ids if int(assignments[well_id]) == fold]
        train_ids = [well_id for well_id in all_ids if int(assignments[well_id]) != fold]
        fold_predictions, prepared = _fit_predict_split(
            records,
            train_ids,
            test_ids,
            feature_names,
            target_names,
            config,
            _stable_int("e003-forest", seed_prefix, fold),
        )
        predictions.update(fold_predictions)
        feature_frequency.update(prepared["selected_names"])
        if collect_controls:
            root = math.sqrt(2.0)
            train_x = prepared["train_x"]
            test_x = prepared["test_x"]
            duplicated_train = [
                [row[0] / root, row[0] / root, *row[1:]] for row in train_x
            ]
            duplicated_test = [
                [row[0] / root, row[0] / root, *row[1:]] for row in test_x
            ]
            duplicate_coefficients = _ridge_fit(
                duplicated_train,
                prepared["train_y"],
                float(config["ridge"]["alpha"]),
            )
            duplicate_scaled = _linear_predict(duplicated_test, duplicate_coefficients)
            duplicate = _unscale_predictions(
                duplicate_scaled,
                prepared["y_means"],
                prepared["y_scales"],
            )
            base = [fold_predictions[well_id]["ridge"] for well_id in test_ids]
            for first, second in zip(base, duplicate):
                control_data["duplicate_max_delta"] = max(
                    float(control_data["duplicate_max_delta"]),
                    max(abs(a - b) for a, b in zip(first, second)),
                )
            shuffled_y = [list(row) for row in prepared["train_y"]]
            order = list(range(len(shuffled_y)))
            random.Random(_stable_int("e003-shuffle", seed_prefix, fold)).shuffle(order)
            shuffled_y = [shuffled_y[index] for index in order]
            shuffled_coefficients = _ridge_fit(
                prepared["train_x"],
                shuffled_y,
                float(config["ridge"]["alpha"]),
            )
            shuffled_scaled = _linear_predict(prepared["test_x"], shuffled_coefficients)
            shuffled = _unscale_predictions(
                shuffled_scaled,
                prepared["y_means"],
                prepared["y_scales"],
            )
            for index, well_id in enumerate(test_ids):
                control_data["shuffled"][well_id] = shuffled[index]
    if set(predictions) != set(records):
        raise DataValidationError("E003 crossfit did not predict every well exactly once")
    return predictions, feature_frequency, control_data


def _action_parameters(candidate: str, prediction: Mapping[str, Sequence[float]]) -> tuple[float, float]:
    model, mode = candidate.split("_", 1)
    values = prediction[model]
    datum = float(values[0])
    trend = float(values[1]) if mode == "datum_trend" else 0.0
    return datum, trend


def _score_action(
    records: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    candidate: str,
) -> tuple[dict[str, Any], dict[str, WellMetric]]:
    metrics: dict[str, WellMetric] = {}
    for well_id in sorted(records):
        datum, trend = _action_parameters(candidate, predictions[well_id])
        metrics[well_id] = _metric_from_sufficient(
            well_id,
            records[well_id]["sufficient"],
            datum,
            trend,
        )
    return _summarize([metrics[well_id] for well_id in sorted(metrics)]), metrics


def _target_metrics(
    records: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    model: str,
    materiality: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = sorted(records)
    for target_index, target in enumerate(MODEL_TARGETS[:3]):
        actual = [float(records[well_id]["targets"][target]) for well_id in ids]
        predicted = [float(predictions[well_id][model][target_index]) for well_id in ids]
        sign_accuracy, sign_rows = _sign_accuracy(actual, predicted, materiality if target_index < 2 else 0.0)
        rows.append(
            {
                "model": model,
                "target": target,
                "pearson": _pearson(actual, predicted),
                "spearman": _spearman(actual, predicted),
                "mae": _mean([abs(a - p) for a, p in zip(actual, predicted)]),
                "sign_accuracy": sign_accuracy,
                "sign_rows": sign_rows,
            }
        )
    return rows


def _risk_metrics(
    records: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    model: str,
    fraction: float,
) -> dict[str, Any]:
    ids = sorted(records)
    actual_rmse = [float(records[well_id]["baseline_metric"].rmse) for well_id in ids]
    predicted = [float(predictions[well_id][model][3]) for well_id in ids]
    count = max(1, int(math.ceil(len(ids) * fraction)))
    actual_order = sorted(range(len(ids)), key=lambda index: (-actual_rmse[index], ids[index]))
    predicted_order = sorted(range(len(ids)), key=lambda index: (-predicted[index], ids[index]))
    actual_top = set(actual_order[:count])
    predicted_top = set(predicted_order[:count])
    labels = [1 if index in actual_top else 0 for index in range(len(ids))]
    total_sse = sum(float(records[well_id]["baseline_metric"].sse) for well_id in ids)
    predicted_sse = sum(float(records[ids[index]]["baseline_metric"].sse) for index in predicted_top)
    actual_sse = sum(float(records[ids[index]]["baseline_metric"].sse) for index in actual_top)
    return {
        "model": model,
        "spearman": _spearman(actual_rmse, predicted),
        "pearson_log_rmse": _pearson(
            [math.log1p(value) for value in actual_rmse],
            predicted,
        ),
        "auc_top_fraction": _auc(labels, predicted),
        "top_fraction_overlap": len(actual_top & predicted_top) / count,
        "predicted_top_sse_capture": predicted_sse / total_sse,
        "oracle_top_sse_capture": actual_sse / total_sse,
        "top_count": count,
    }


def _average_predictions(
    map_predictions: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]]
) -> dict[str, dict[str, list[float]]]:
    versions = sorted(map_predictions)
    well_ids = sorted(next(iter(map_predictions.values())))
    result: dict[str, dict[str, list[float]]] = {}
    for well_id in well_ids:
        result[well_id] = {}
        for model in PREDICTION_MODELS:
            output_count = len(map_predictions[versions[0]][well_id][model])
            result[well_id][model] = [
                sum(float(map_predictions[version][well_id][model][output]) for version in versions) / len(versions)
                for output in range(output_count)
            ]
    return result


def _spatial_assignments(records: Mapping[str, Mapping[str, Any]], bins: int) -> dict[str, int]:
    ordered = sorted(
        records,
        key=lambda well_id: (
            float(records[well_id]["spatial_x"]),
            float(records[well_id]["spatial_y"]),
            well_id,
        ),
    )
    return {
        well_id: min(bins - 1, int(index * bins / len(ordered)))
        for index, well_id in enumerate(ordered)
    }


def _artifact_rows(
    records: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows = []
    target_rows = []
    for well_id in sorted(records):
        record = records[well_id]
        feature_rows.append({"well_id": well_id, **{name: record["features"].get(name) for name in feature_names}})
        target_rows.append(
            {
                "well_id": well_id,
                **record["targets"],
                "baseline_rmse": record["baseline_metric"].rmse,
                "baseline_sse": record["baseline_metric"].sse,
                "oracle_datum_rmse": record["oracle_datum_metric"].rmse,
                "oracle_datum_trend_rmse": record["oracle_trend_metric"].rmse,
            }
        )
    return feature_rows, target_rows


def run_e003(
    *,
    root: Path,
    train_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the frozen E003 learnability study and write deterministic artifacts."""
    started = time.perf_counter()
    root = root.resolve()
    train_dir = train_dir.resolve()
    output_dir = output_dir.resolve()
    profiles, data_profile = scan_profiles(train_dir)
    signature = str(config["data_signature"])
    if data_profile["data_signature"] != signature:
        raise DataValidationError("E003 data signature differs from frozen E001 signature")
    if len(profiles) != int(config["expected_wells"]):
        raise DataValidationError("E003 well count differs from pre-registration")
    fold_maps = _load_fold_maps(root, config["fold_files"], signature)
    records, feature_names = _extract_records(train_dir=train_dir, profiles=profiles, config=config)
    if set(records) != {profile.well_id for profile in profiles}:
        raise DataValidationError("E003 feature extraction lost wells")

    baseline_metrics = {well_id: records[well_id]["baseline_metric"] for well_id in records}
    oracle_datum_metrics = {well_id: records[well_id]["oracle_datum_metric"] for well_id in records}
    oracle_trend_metrics = {well_id: records[well_id]["oracle_trend_metric"] for well_id in records}
    baseline_summary = _summarize([baseline_metrics[well_id] for well_id in sorted(records)])
    oracle_datum_summary = _summarize([oracle_datum_metrics[well_id] for well_id in sorted(records)])
    oracle_trend_summary = _summarize([oracle_trend_metrics[well_id] for well_id in sorted(records)])

    map_predictions: dict[str, dict[str, dict[str, list[float]]]] = {}
    map_action_rows: list[dict[str, Any]] = []
    map_target_rows: list[dict[str, Any]] = []
    map_risk_rows: list[dict[str, Any]] = []
    feature_frequency: Counter[str] = Counter()
    first_controls: dict[str, Any] | None = None
    oof_rows: list[dict[str, Any]] = []
    materiality = float(config["sign_materiality_ft"])
    for fold_map in fold_maps:
        version = str(fold_map["version"])
        predictions, frequencies, controls = _crossfit(
            records,
            feature_names,
            MODEL_TARGETS,
            fold_map["assignments"],
            int(fold_map["n_folds"]),
            config,
            version,
            collect_controls=version == "v1",
        )
        map_predictions[version] = predictions
        feature_frequency.update(frequencies)
        if version == "v1":
            first_controls = controls
        for candidate in ALL_ACTIONS:
            summary, _ = _score_action(records, predictions, candidate)
            map_action_rows.append(
                {
                    "map": version,
                    "candidate": candidate,
                    "gain_vs_baseline": float(baseline_summary["rmse"]) - float(summary["rmse"]),
                    **summary,
                }
            )
        for model in MODEL_NAMES:
            for row in _target_metrics(records, predictions, model, materiality):
                map_target_rows.append({"map": version, **row})
            map_risk_rows.append({"map": version, **_risk_metrics(records, predictions, model, float(config["risk_fraction"]))})
        for well_id in sorted(records):
            row: dict[str, Any] = {"map": version, "well_id": well_id}
            row.update({f"actual_{name}": records[well_id]["targets"][name] for name in MODEL_TARGETS})
            for model in MODEL_NAMES:
                for index, target in enumerate(MODEL_TARGETS):
                    row[f"{model}_{target}"] = predictions[well_id][model][index]
            oof_rows.append(row)

    averaged = _average_predictions(map_predictions)
    aggregate_action_rows: list[dict[str, Any]] = []
    aggregate_metrics_by_candidate: dict[str, dict[str, Any]] = {}
    aggregate_well_by_candidate: dict[str, dict[str, WellMetric]] = {}
    for candidate in ALL_ACTIONS:
        summary, well_metrics = _score_action(records, averaged, candidate)
        aggregate_metrics_by_candidate[candidate] = summary
        aggregate_well_by_candidate[candidate] = well_metrics
        aggregate_action_rows.append(
            {
                "candidate": candidate,
                "gain_vs_baseline": float(baseline_summary["rmse"]) - float(summary["rmse"]),
                **summary,
            }
        )
    selected_action = min(
        ACTION_CANDIDATES,
        key=lambda candidate: (
            _mean([
                float(row["rmse"])
                for row in map_action_rows
                if row["candidate"] == candidate
            ]),
            candidate,
        ),
    )
    action_model = selected_action.split("_", 1)[0]
    aggregate_target_rows = _target_metrics(records, averaged, action_model, materiality)
    datum_row = next(row for row in aggregate_target_rows if row["target"] == "datum_correction_ft")
    map_gains = [
        float(row["gain_vs_baseline"])
        for row in map_action_rows
        if row["candidate"] == selected_action
    ]
    action_map_wins = sum(
        gain >= float(config["promotion"]["minimum_action_rmse_gain"])
        for gain in map_gains
    )

    risk_model = max(
        MODEL_NAMES,
        key=lambda model: (
            _mean([
                float(row["spearman"] or 0.0)
                for row in map_risk_rows
                if row["model"] == model
            ]),
            model,
        ),
    )
    risk_map_passes = sum(
        float(row["spearman"] or -1.0) >= float(config["promotion"]["minimum_risk_spearman"])
        and float(row["auc_top_fraction"] or -1.0) >= float(config["promotion"]["minimum_risk_auc"])
        for row in map_risk_rows
        if row["model"] == risk_model
    )
    aggregate_risk = _risk_metrics(records, averaged, risk_model, float(config["risk_fraction"]))

    spatial_assignments = _spatial_assignments(records, int(config["spatial_stress_bins"]))
    spatial_predictions, spatial_frequency, _ = _crossfit(
        records,
        feature_names,
        MODEL_TARGETS,
        spatial_assignments,
        int(config["spatial_stress_bins"]),
        config,
        "spatial-x-blocks",
    )
    feature_frequency.update(spatial_frequency)
    spatial_rows: list[dict[str, Any]] = []
    spatial_action_metrics: dict[str, dict[str, Any]] = {}
    for candidate in ALL_ACTIONS:
        summary, _ = _score_action(records, spatial_predictions, candidate)
        spatial_action_metrics[candidate] = summary
        spatial_rows.append(
            {
                "kind": "action",
                "name": candidate,
                "gain_vs_baseline": float(baseline_summary["rmse"]) - float(summary["rmse"]),
                "rmse": summary["rmse"],
                "spearman": "",
                "auc": "",
            }
        )
    spatial_risk_metrics: dict[str, dict[str, Any]] = {}
    for model in MODEL_NAMES:
        metric = _risk_metrics(records, spatial_predictions, model, float(config["risk_fraction"]))
        spatial_risk_metrics[model] = metric
        spatial_rows.append(
            {
                "kind": "risk",
                "name": model,
                "gain_vs_baseline": "",
                "rmse": "",
                "spearman": metric["spearman"],
                "auc": metric["auc_top_fraction"],
            }
        )

    if first_controls is None:
        raise RuntimeError("E003 did not collect v1 controls")
    synthetic_predictions, _, _ = _crossfit(
        records,
        feature_names,
        ("synthetic_positive",),
        fold_maps[0]["assignments"],
        int(fold_maps[0]["n_folds"]),
        config,
        "positive-control",
    )
    ids = sorted(records)
    synthetic_actual = [float(records[well_id]["targets"]["synthetic_positive"]) for well_id in ids]
    synthetic_predicted = [float(synthetic_predictions[well_id]["ridge"][0]) for well_id in ids]
    positive_correlation = _pearson(synthetic_actual, synthetic_predicted)

    shuffled_predictions = {
        well_id: {"ridge": first_controls["shuffled"][well_id], "forest": first_controls["shuffled"][well_id]}
        for well_id in records
    }
    shuffled_summary, _ = _score_action(records, shuffled_predictions, "ridge_datum_trend")
    mean_v1_summary, _ = _score_action(records, map_predictions["v1"], "mean_datum_trend")
    shuffled_datum_actual = [float(records[well_id]["targets"]["datum_correction_ft"]) for well_id in ids]
    shuffled_datum_predicted = [float(shuffled_predictions[well_id]["ridge"][0]) for well_id in ids]
    shuffled_datum_correlation = _pearson(shuffled_datum_actual, shuffled_datum_predicted)

    selected_summary = aggregate_metrics_by_candidate[selected_action]
    selected_spatial_gain = float(baseline_summary["rmse"]) - float(spatial_action_metrics[selected_action]["rmse"])
    selected_p90_delta = float(selected_summary["p90_well_rmse"]) - float(baseline_summary["p90_well_rmse"])
    selected_worst5_delta = float(selected_summary["worst_5pct_sse_share"]) - float(baseline_summary["worst_5pct_sse_share"])
    elapsed = time.perf_counter() - started
    control_config = config["controls"]
    reference = config.get("reference_metrics", {})
    expected_baseline = reference.get("baseline_rmse")
    expected_oracle_datum = reference.get("oracle_datum_rmse")
    expected_oracle_trend = reference.get("oracle_datum_trend_rmse")
    oracle_reference_pass = (
        (expected_oracle_datum is None or abs(float(oracle_datum_summary["rmse"]) - float(expected_oracle_datum))
         <= float(control_config["oracle_datum_rmse_tolerance"]))
        and
        (expected_oracle_trend is None or abs(float(oracle_trend_summary["rmse"]) - float(expected_oracle_trend))
         <= float(control_config["oracle_datum_trend_rmse_tolerance"]))
    )
    baseline_reference_pass = (
        expected_baseline is None
        or abs(float(baseline_summary["rmse"]) - float(expected_baseline)) <= 0.02
    )
    controls = {
        "data_integrity": {
            "pass": data_profile["data_signature"] == signature and len(records) == int(config["expected_wells"]),
            "data_signature": data_profile["data_signature"],
            "wells": len(records),
            "features": len(feature_names),
        },
        "fold_integrity": {
            "pass": len(fold_maps) == 5
            and len({str(item["fingerprint"]) for item in fold_maps}) == len(fold_maps)
            and all(set(item["assignments"]) == set(records) for item in fold_maps),
            "maps": len(fold_maps),
        },
        "oracle_algebra": {
            "pass": oracle_reference_pass,
            "baseline_rmse": baseline_summary["rmse"],
            "oracle_datum_rmse": oracle_datum_summary["rmse"],
            "oracle_datum_trend_rmse": oracle_trend_summary["rmse"],
        },
        "noop": {
            "pass": baseline_reference_pass,
            "rmse": baseline_summary["rmse"],
        },
        "synthetic_positive": {
            "pass": positive_correlation is not None
            and positive_correlation >= float(control_config["minimum_positive_control_correlation"]),
            "correlation": positive_correlation,
        },
        "duplicate_feature": {
            "pass": float(first_controls["duplicate_max_delta"])
            <= float(control_config["maximum_duplicate_prediction_delta"]),
            "maximum_prediction_delta": first_controls["duplicate_max_delta"],
            "construction": "replace one standardized feature by two copies scaled by 1/sqrt(2), preserving the ridge penalty",
        },
        "shuffled_evidence": {
            "pass": abs(float(shuffled_datum_correlation or 0.0))
            <= float(control_config["maximum_shuffled_abs_datum_correlation"])
            and float(mean_v1_summary["rmse"]) - float(shuffled_summary["rmse"])
            <= float(control_config["maximum_shuffled_action_gain"]),
            "datum_correlation": shuffled_datum_correlation,
            "action_rmse": shuffled_summary["rmse"],
            "gain_over_zero_action": float(baseline_summary["rmse"]) - float(shuffled_summary["rmse"]),
            "gain_over_mean_action": float(mean_v1_summary["rmse"]) - float(shuffled_summary["rmse"]),
            "mean_action_rmse": mean_v1_summary["rmse"],
        },
        "leakage_sentinel": {
            "pass": float(oracle_trend_summary["rmse"]) < float(baseline_summary["rmse"]) - 5.0,
            "rmse": oracle_trend_summary["rmse"],
            "uses_hidden_target": True,
            "eligible_for_modeling": False,
        },
        "runtime": {
            "pass": elapsed <= 60.0 * float(config["promotion"]["maximum_runtime_minutes"]),
            "budget_minutes": config["promotion"]["maximum_runtime_minutes"],
        },
    }
    controls_pass = all(bool(item["pass"]) for item in controls.values())
    promotion = config["promotion"]
    best_mean_action = min(
        NO_EVIDENCE_ACTIONS,
        key=lambda candidate: (float(aggregate_metrics_by_candidate[candidate]["rmse"]), candidate),
    )
    best_mean_summary = aggregate_metrics_by_candidate[best_mean_action]
    feature_gain_over_mean = float(best_mean_summary["rmse"]) - float(selected_summary["rmse"])
    signed_action_authorized = (
        controls_pass
        and float(baseline_summary["rmse"]) - float(selected_summary["rmse"])
        >= float(promotion["minimum_action_rmse_gain"])
        and feature_gain_over_mean >= float(promotion["minimum_feature_gain_over_mean_action"])
        and action_map_wins >= int(promotion["minimum_action_map_wins"])
        and selected_p90_delta <= float(promotion["maximum_p90_well_rmse_deterioration"])
        and selected_worst5_delta <= float(promotion["maximum_worst_5pct_sse_share_increase"])
        and float(datum_row["pearson"] or -1.0) >= float(promotion["minimum_datum_correlation"])
        and float(datum_row["sign_accuracy"] or 0.0) >= float(promotion["minimum_material_datum_sign_accuracy"])
        and (
            selected_spatial_gain > 0.0
            if bool(promotion["require_positive_spatial_action_gain"])
            else True
        )
    )
    risk_detection_authorized = (
        controls_pass
        and risk_map_passes >= int(promotion["minimum_risk_map_passes"])
        and (
            float(spatial_risk_metrics[risk_model]["spearman"] or -1.0) > 0.0
            if bool(promotion["require_positive_spatial_risk_spearman"])
            else True
        )
    )
    status = "promoted" if (signed_action_authorized or risk_detection_authorized) else "rejected"

    selected_well_rows: list[dict[str, Any]] = []
    selected_well_metrics = aggregate_well_by_candidate[selected_action]
    for well_id in sorted(records):
        datum, trend = _action_parameters(selected_action, averaged[well_id])
        metric = selected_well_metrics[well_id]
        selected_well_rows.append(
            {
                "well_id": well_id,
                "candidate": selected_action,
                "predicted_datum": datum,
                "predicted_trend": trend,
                "actual_datum": records[well_id]["targets"]["datum_correction_ft"],
                "actual_trend": records[well_id]["targets"]["trend_correction_ft"],
                "baseline_rmse": records[well_id]["baseline_metric"].rmse,
                "corrected_rmse": metric.rmse,
                "corrected_sse": metric.sse,
            }
        )

    summary = {
        "schema_version": 1,
        "experiment_id": "E003",
        "status": status,
        "data": data_profile,
        "feature_count": len(feature_names),
        "baseline_metrics": baseline_summary,
        "oracle_headroom": {
            "datum_only": oracle_datum_summary,
            "datum_and_trend": oracle_trend_summary,
        },
        "selected_action": selected_action,
        "selected_action_metrics": selected_summary,
        "selected_action_gain": float(baseline_summary["rmse"]) - float(selected_summary["rmse"]),
        "selected_action_map_wins": action_map_wins,
        "selected_action_spatial_gain": selected_spatial_gain,
        "best_no_evidence_action": best_mean_action,
        "best_no_evidence_action_metrics": best_mean_summary,
        "selected_action_gain_over_mean": feature_gain_over_mean,
        "selected_action_target_metrics": aggregate_target_rows,
        "selected_risk_model": risk_model,
        "selected_risk_metrics": aggregate_risk,
        "selected_risk_map_passes": risk_map_passes,
        "selected_risk_spatial_metrics": spatial_risk_metrics[risk_model],
        "aggregate_action_metrics": aggregate_metrics_by_candidate,
        "map_action_metrics": map_action_rows,
        "map_target_metrics": map_target_rows,
        "map_risk_metrics": map_risk_rows,
        "spatial_action_metrics": spatial_action_metrics,
        "spatial_risk_metrics": spatial_risk_metrics,
        "controls": controls,
        "decision": {
            "risk_detection_authorized": risk_detection_authorized,
            "signed_action_authorized": signed_action_authorized,
            "risk_does_not_authorize_action": True,
        },
        "predictor_policy": {
            "legal_features": {"uses_hidden_target": False, "eligible_for_modeling": True},
            "cross_fitted_ridge": {"uses_hidden_target_at_inference": False, "eligible_for_modeling": signed_action_authorized},
            "cross_fitted_forest": {"uses_hidden_target_at_inference": False, "eligible_for_modeling": signed_action_authorized},
            "mean_action_baselines": {"uses_hidden_target_at_inference": False, "eligible_for_modeling": True},
            "risk_probe": {"uses_hidden_target_at_inference": False, "eligible_for_risk_features": risk_detection_authorized, "eligible_for_signed_routing": False},
            "oracle_targets": {"uses_hidden_target": True, "eligible_for_modeling": False},
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    feature_rows, target_rows = _artifact_rows(records, feature_names)
    _write_csv(output_dir / "legal_features.csv", ["well_id", *feature_names], feature_rows)
    target_fields = list(target_rows[0])
    _write_csv(output_dir / "oracle_targets.csv", target_fields, target_rows)
    _write_csv(output_dir / "oof_predictions.csv", list(oof_rows[0]), oof_rows)
    _write_csv(output_dir / "map_action_metrics.csv", list(map_action_rows[0]), map_action_rows)
    _write_csv(output_dir / "map_target_metrics.csv", list(map_target_rows[0]), map_target_rows)
    _write_csv(output_dir / "map_risk_metrics.csv", list(map_risk_rows[0]), map_risk_rows)
    _write_csv(output_dir / "aggregate_action_metrics.csv", list(aggregate_action_rows[0]), aggregate_action_rows)
    _write_csv(output_dir / "spatial_stress_metrics.csv", list(spatial_rows[0]), spatial_rows)
    _write_csv(output_dir / "selected_well_metrics.csv", list(selected_well_rows[0]), selected_well_rows)
    feature_rows_frequency = [
        {"feature": name, "selection_count": count}
        for name, count in sorted(feature_frequency.items(), key=lambda item: (-item[1], item[0]))
    ]
    _write_csv(output_dir / "selected_feature_frequency.csv", ["feature", "selection_count"], feature_rows_frequency)
    control_rows = [
        {
            "control": name,
            "status": "pass" if detail["pass"] else "fail",
            "detail_json": _canonical_json({key: value for key, value in detail.items() if key != "pass"}),
        }
        for name, detail in sorted(controls.items())
    ]
    _write_csv(output_dir / "control_metrics.csv", ["control", "status", "detail_json"], control_rows)

    result_files = [
        summary_path,
        output_dir / "legal_features.csv",
        output_dir / "oracle_targets.csv",
        output_dir / "oof_predictions.csv",
        output_dir / "map_action_metrics.csv",
        output_dir / "map_target_metrics.csv",
        output_dir / "map_risk_metrics.csv",
        output_dir / "aggregate_action_metrics.csv",
        output_dir / "spatial_stress_metrics.csv",
        output_dir / "selected_well_metrics.csv",
        output_dir / "selected_feature_frequency.csv",
        output_dir / "control_metrics.csv",
    ]
    artifact_manifest = {
        "schema_version": 1,
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(result_files, key=lambda item: item.name)
        ],
        "fold_files": [
            {"path": relative, "sha256": _sha256(root / relative), "bytes": (root / relative).stat().st_size}
            for relative in config["fold_files"]
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return {**summary, "runtime_seconds": elapsed}
