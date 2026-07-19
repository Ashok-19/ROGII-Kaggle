"""Pure-standard-library E004 inference runtime.

This module intentionally has no project-internal imports. The exact source can be
embedded into a Kaggle notebook and executed with competition data plus a frozen
JSON model artifact only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

MISSING_VALUES = {"", "nan", "NaN", "NA", "N/A", "null", "None"}
REQUIRED_HORIZONTAL = ("MD", "X", "Y", "Z", "GR", "TVT_input")


class DeploymentDataError(ValueError):
    """Raised when inference inputs violate the frozen deployment contract."""


def optional_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() in MISSING_VALUES:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    center_x = mean(xs)
    center_y = mean(ys)
    denominator = sum((value - center_x) ** 2 for value in xs)
    if denominator <= 0.0:
        return 0.0
    return sum((x - center_x) * (y - center_y) for x, y in zip(xs, ys)) / denominator


def paired(xs: Sequence[float], values: Sequence[float | None]) -> tuple[list[float], list[float]]:
    clean_x: list[float] = []
    clean_y: list[float] = []
    for x, value in zip(xs, values):
        if value is not None and math.isfinite(value):
            clean_x.append(float(x))
            clean_y.append(float(value))
    return clean_x, clean_y


def series_summary(xs: Sequence[float], values: Sequence[float | None]) -> dict[str, float | None]:
    clean_x, clean_y = paired(xs, values)
    if not clean_y:
        return {"mean": None, "std": None, "range": None, "delta": None, "slope": None}
    return {
        "mean": mean(clean_y),
        "std": std(clean_y),
        "range": max(clean_y) - min(clean_y),
        "delta": clean_y[-1] - clean_y[0],
        "slope": linear_slope(clean_x, clean_y),
    }


def add_summary(features: dict[str, float | None], prefix: str, xs: Sequence[float], values: Sequence[float | None]) -> None:
    for name, value in series_summary(xs, values).items():
        features[f"{prefix}_{name}"] = value


def missing_fraction(values: Sequence[float | None]) -> float:
    return sum(value is None for value in values) / max(1, len(values))


def longest_missing_run(values: Sequence[float | None]) -> int:
    longest = current = 0
    for value in values:
        if value is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def weighted_line(xs: Sequence[float], ys: Sequence[float], weights: Sequence[float]) -> tuple[float, float]:
    total = sum(weights)
    if len(xs) < 2 or total <= 0.0:
        return (ys[-1] if ys else 0.0), 0.0
    center_x = sum(w * x for w, x in zip(weights, xs)) / total
    center_y = sum(w * y for w, y in zip(weights, ys)) / total
    denominator = sum(w * (x - center_x) ** 2 for w, x in zip(weights, xs))
    slope = 0.0 if denominator <= 0.0 else sum(
        w * (x - center_x) * (y - center_y)
        for w, x, y in zip(weights, xs, ys)
    ) / denominator
    return center_y - slope * center_x, slope


def huber_line(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    window_rows: int,
    huber_k: float = 1.5,
    iterations: int = 6,
) -> tuple[float, float, float]:
    count = min(len(xs), len(ys), max(2, int(window_rows)))
    x = [float(value) for value in xs[-count:]]
    y = [float(value) for value in ys[-count:]]
    weights = [1.0] * count
    intercept, slope = weighted_line(x, y, weights)
    scale = 0.0
    for _ in range(max(1, int(iterations))):
        residuals = [target - (intercept + slope * value) for value, target in zip(x, y)]
        median_residual = statistics.median(residuals)
        absolute = [abs(value - median_residual) for value in residuals]
        scale = 1.4826 * statistics.median(absolute) if absolute else 0.0
        if scale <= 1e-12:
            break
        cutoff = float(huber_k) * scale
        weights = [1.0 if abs(value) <= cutoff else cutoff / abs(value) for value in residuals]
        intercept, slope = weighted_line(x, y, weights)
    return intercept, slope, scale


def visible_backtests(md: Sequence[float], z: Sequence[float], tvt: Sequence[float], fractions: Sequence[float]) -> dict[str, float | None]:
    features: dict[str, float | None] = {}
    count = len(tvt)
    for fraction in fractions:
        label = str(float(fraction)).replace(".", "p")
        cut = min(count - 2, max(2, int(round(count * float(fraction)))))
        if cut < 2 or count - cut < 2:
            for suffix in ("mean", "rmse", "trend", "toe", "u_slope_delta"):
                features[f"backtest_{label}_{suffix}"] = None
            continue
        baseline = tvt[cut - 1]
        residuals = [value - baseline for value in tvt[cut:]]
        centered = [index / max(1, len(residuals) - 1) - 0.5 for index in range(len(residuals))]
        features[f"backtest_{label}_mean"] = mean(residuals)
        features[f"backtest_{label}_rmse"] = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        features[f"backtest_{label}_trend"] = linear_slope(centered, residuals)
        features[f"backtest_{label}_toe"] = residuals[-1]
        before_u = [target + vertical for target, vertical in zip(tvt[:cut], z[:cut])]
        after_u = [target + vertical for target, vertical in zip(tvt[cut:], z[cut:])]
        features[f"backtest_{label}_u_slope_delta"] = linear_slope(md[cut:], after_u) - linear_slope(md[:cut], before_u)
    return features


def read_horizontal(path: Path, *, require_truth: bool = False) -> dict[str, Any]:
    columns: dict[str, list[float | None]] = {}
    fieldnames: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = set(REQUIRED_HORIZONTAL) | ({"TVT"} if require_truth else set())
        missing = required - set(fieldnames)
        if missing:
            raise DeploymentDataError(f"{path}: missing required columns {sorted(missing)}")
        columns = {name: [] for name in fieldnames}
        for row in reader:
            for name in fieldnames:
                columns[name].append(optional_float(row.get(name)))
    row_count = len(columns.get("MD", []))
    if row_count < 4:
        raise DeploymentDataError(f"{path}: requires at least four rows")
    for name in ("MD", "X", "Y", "Z"):
        if any(value is None for value in columns[name]):
            raise DeploymentDataError(f"{path}: missing or non-finite {name}")
    md = [float(value) for value in columns["MD"] if value is not None]
    if any(second <= first for first, second in zip(md, md[1:])):
        raise DeploymentDataError(f"{path}: MD must be strictly increasing")
    known_rows = 0
    hidden_seen = False
    for value in columns["TVT_input"]:
        visible = value is not None
        if visible and hidden_seen:
            raise DeploymentDataError(f"{path}: TVT_input visibility is not a contiguous prefix")
        if visible:
            known_rows += 1
        else:
            hidden_seen = True
    if known_rows < 3:
        raise DeploymentDataError(f"{path}: requires at least three visible TVT_input rows")
    if known_rows >= row_count:
        raise DeploymentDataError(f"{path}: requires a non-empty hidden suffix")
    if require_truth:
        if any(value is None for value in columns["TVT"]):
            raise DeploymentDataError(f"{path}: training TVT must be complete")
        max_delta = max(
            abs(float(columns["TVT"][index]) - float(columns["TVT_input"][index]))
            for index in range(known_rows)
        )
        if max_delta > 1e-9:
            raise DeploymentDataError(f"{path}: TVT_input differs from TVT on visible rows")
    return {"columns": columns, "fieldnames": fieldnames, "row_count": row_count, "known_rows": known_rows}


def read_typewell(path: Path) -> dict[str, float | None]:
    tvt: list[float | None] = []
    gr: list[float | None] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"TVT", "GR"} - set(reader.fieldnames or [])
        if missing:
            raise DeploymentDataError(f"{path}: missing typewell columns {sorted(missing)}")
        for row in reader:
            tvt.append(optional_float(row.get("TVT")))
            gr.append(optional_float(row.get("GR")))
    if not tvt:
        raise DeploymentDataError(f"{path}: empty typewell")
    clean_tvt = [value for value in tvt if value is not None]
    clean_gr = [value for value in gr if value is not None]
    if not clean_tvt:
        raise DeploymentDataError(f"{path}: typewell TVT is entirely missing")
    index_x, paired_gr = paired([float(index) for index in range(len(gr))], gr)
    return {
        "typewell_rows": float(len(tvt)),
        "typewell_tvt_span": max(clean_tvt) - min(clean_tvt),
        "typewell_gr_mean": mean(clean_gr) if clean_gr else None,
        "typewell_gr_std": std(clean_gr) if clean_gr else None,
        "typewell_gr_range": max(clean_gr) - min(clean_gr) if clean_gr else None,
        "typewell_gr_index_slope": linear_slope(index_x, paired_gr) if paired_gr else None,
        "typewell_gr_missing_fraction": missing_fraction(gr),
    }


def extract_well_features(
    horizontal_path: Path,
    typewell_path: Path,
    *,
    visible_slope_windows: Sequence[int],
    visible_backtest_fractions: Sequence[float],
    require_truth: bool = False,
) -> dict[str, Any]:
    horizontal = read_horizontal(horizontal_path, require_truth=require_truth)
    columns = horizontal["columns"]
    total_rows = int(horizontal["row_count"])
    known_rows = int(horizontal["known_rows"])
    hidden_rows = total_rows - known_rows
    md = [float(value) for value in columns["MD"] if value is not None]
    x = [float(value) for value in columns["X"] if value is not None]
    y = [float(value) for value in columns["Y"] if value is not None]
    z = [float(value) for value in columns["Z"] if value is not None]
    visible_tvt = [float(value) for value in columns["TVT_input"][:known_rows] if value is not None]
    known_md, hidden_md = md[:known_rows], md[known_rows:]
    known_x, hidden_x = x[:known_rows], x[known_rows:]
    known_y, hidden_y = y[:known_rows], y[known_rows:]
    known_z, hidden_z = z[:known_rows], z[known_rows:]
    visible_u = [target + vertical for target, vertical in zip(visible_tvt, known_z)]
    features: dict[str, float | None] = {
        "total_rows": float(total_rows),
        "known_rows": float(known_rows),
        "hidden_rows": float(hidden_rows),
        "hidden_fraction": hidden_rows / total_rows,
        "md_known_span": known_md[-1] - known_md[0],
        "md_hidden_span": hidden_md[-1] - known_md[-1],
        "md_total_span": md[-1] - md[0],
        "last_visible_tvt": visible_tvt[-1],
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
        "spatial_x_mid": 0.5 * (x[0] + x[-1]),
        "spatial_y_mid": 0.5 * (y[0] + y[-1]),
    }
    angle = math.atan2(hidden_y[-1] - known_y[-1], hidden_x[-1] - known_x[-1])
    features["hidden_azimuth_sin"] = math.sin(angle)
    features["hidden_azimuth_cos"] = math.cos(angle)
    add_summary(features, "visible_tvt", known_md, visible_tvt)
    add_summary(features, "visible_z", known_md, known_z)
    add_summary(features, "visible_u", known_md, visible_u)
    add_summary(features, "hidden_z", hidden_md, hidden_z)
    add_summary(features, "visible_gr", known_md, columns["GR"][:known_rows])
    add_summary(features, "hidden_gr", hidden_md, columns["GR"][known_rows:])
    add_summary(features, "whole_gr", md, columns["GR"])
    features["visible_gr_missing_fraction"] = missing_fraction(columns["GR"][:known_rows])
    features["hidden_gr_missing_fraction"] = missing_fraction(columns["GR"][known_rows:])
    features["whole_gr_missing_fraction"] = missing_fraction(columns["GR"])
    features["hidden_gr_longest_missing_run"] = float(longest_missing_run(columns["GR"][known_rows:]))
    full_u_slope = linear_slope(known_md, visible_u)
    for window in [int(value) for value in visible_slope_windows]:
        count = min(window, known_rows)
        if count < 2:
            features[f"visible_u_slope_w{window}"] = None
            features[f"visible_u_scale_w{window}"] = None
            features[f"visible_u_slope_delta_w{window}"] = None
            features[f"visible_tvt_slope_w{window}"] = None
            continue
        _, u_slope, u_scale = huber_line(known_md[-count:], visible_u[-count:], window_rows=count)
        _, tvt_slope, _ = huber_line(known_md[-count:], visible_tvt[-count:], window_rows=count)
        features[f"visible_u_slope_w{window}"] = u_slope
        features[f"visible_u_scale_w{window}"] = u_scale
        features[f"visible_u_slope_delta_w{window}"] = u_slope - full_u_slope
        features[f"visible_tvt_slope_w{window}"] = tvt_slope
    features.update(visible_backtests(known_md, known_z, visible_tvt, visible_backtest_fractions))
    features.update(read_typewell(typewell_path))
    truth = None
    if require_truth:
        truth = [float(value) for value in columns["TVT"] if value is not None]
    return {
        "features": features,
        "metadata": {
            "well_id": horizontal_path.name.split("__", 1)[0],
            "total_rows": total_rows,
            "known_rows": known_rows,
            "hidden_rows": hidden_rows,
            "last_visible_tvt": visible_tvt[-1],
            "typewell_sha256": hashlib.sha256(typewell_path.read_bytes()).hexdigest(),
        },
        "truth": truth,
    }


def predict_coefficients(model: Mapping[str, Any], features: Mapping[str, float | None]) -> tuple[float, float]:
    selected = list(model["selected_features"])
    medians = list(model["feature_medians"])
    means = list(model["feature_means"])
    scales = list(model["feature_scales"])
    coefficients = list(model["ridge_coefficients"])
    if not (len(selected) == len(medians) == len(means) == len(scales) == len(coefficients)):
        raise DeploymentDataError("model feature arrays have inconsistent lengths")
    row: list[float] = []
    for name, median_value, center, scale in zip(selected, medians, means, scales):
        raw = features.get(name)
        value = float(raw) if raw is not None and math.isfinite(float(raw)) else float(median_value)
        divisor = float(scale)
        if not math.isfinite(divisor) or abs(divisor) <= 1e-12:
            divisor = 1.0
        row.append((value - float(center)) / divisor)
    scaled = [
        sum(row[index] * float(coefficients[index][output]) for index in range(len(row)))
        for output in range(2)
    ]
    targets = [
        float(model["target_means"][output]) + float(model["target_scales"][output]) * scaled[output]
        for output in range(2)
    ]
    if not all(math.isfinite(value) for value in targets):
        raise DeploymentDataError("model produced non-finite coefficient")
    return targets[0], targets[1]


def reconstruct_hidden(last_visible_tvt: float, hidden_rows: int, datum: float, trend: float) -> list[float]:
    if hidden_rows <= 0:
        raise DeploymentDataError("hidden row count must be positive")
    predictions: list[float] = []
    for index in range(hidden_rows):
        centered = index / max(1, hidden_rows - 1) - 0.5 if hidden_rows > 1 else 0.0
        value = float(last_visible_tvt) + float(datum) + float(trend) * centered
        if not math.isfinite(value):
            raise DeploymentDataError("reconstructed non-finite prediction")
        predictions.append(value)
    return predictions


def load_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("schema_version") != 1 or model.get("experiment_id") != "E004":
        raise DeploymentDataError(f"{path}: unsupported E004 model schema")
    return model


def build_prediction_map(model: Mapping[str, Any], test_dir: Path) -> tuple[dict[str, float], list[dict[str, Any]]]:
    horizontal_files = sorted(test_dir.glob("*__horizontal_well.csv"))
    if not horizontal_files:
        raise DeploymentDataError(f"{test_dir}: no horizontal well files")
    predictions: dict[str, float] = {}
    well_rows: list[dict[str, Any]] = []
    for horizontal_path in horizontal_files:
        well_id = horizontal_path.name.split("__", 1)[0]
        typewell_path = test_dir / f"{well_id}__typewell.csv"
        if not typewell_path.exists():
            raise DeploymentDataError(f"missing typewell for {well_id}")
        extracted = extract_well_features(
            horizontal_path,
            typewell_path,
            visible_slope_windows=model["visible_slope_windows"],
            visible_backtest_fractions=model["visible_backtest_fractions"],
            require_truth=False,
        )
        datum, trend = predict_coefficients(model, extracted["features"])
        metadata = extracted["metadata"]
        values = reconstruct_hidden(metadata["last_visible_tvt"], metadata["hidden_rows"], datum, trend)
        for offset, value in enumerate(values, start=metadata["known_rows"]):
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
        })
    return predictions, well_rows


def write_submission(model: Mapping[str, Any], test_dir: Path, sample_submission: Path, output_path: Path) -> dict[str, Any]:
    prediction_map, well_rows = build_prediction_map(model, test_dir)
    sample_ids: list[str] = []
    seen_ids: set[str] = set()
    with sample_submission.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != ["id", "tvt"]:
            raise DeploymentDataError(f"{sample_submission}: expected columns id,tvt")
        for row in reader:
            key = str(row.get("id") or "")
            if not key:
                raise DeploymentDataError(f"{sample_submission}: empty id")
            if key in seen_ids:
                raise DeploymentDataError(f"{sample_submission}: duplicate id {key}")
            seen_ids.add(key)
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
            writer.writerow({"id": key, "tvt": format(prediction_map[key], ".15g")})
    return {
        "rows": len(sample_ids),
        "wells": len(well_rows),
        "minimum_prediction": min(prediction_map.values()),
        "maximum_prediction": max(prediction_map.values()),
        "well_predictions": well_rows,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "bytes": output_path.stat().st_size,
    }
