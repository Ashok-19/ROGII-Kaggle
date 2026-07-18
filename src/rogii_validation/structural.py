"""E002 structural baseline ladder for the ROGII competition.

The implementations are deliberately standard-library only so they can run in
an offline Kaggle notebook. All fitted quantities use the visible TVT_input
prefix; hidden TVT values are read only by the evaluator.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .harness import (
    DataValidationError,
    ErrorAccumulator,
    OnlineCorrelation,
    WellMetric,
    WellProfile,
    _canonical_json,
    _fold_metrics,
    _regime,
    _regime_thresholds,
    _sha256,
    _summarize,
    _write_csv,
    _write_json,
    scan_profiles,
)

MISSING_VALUES = {"", "nan", "NaN", "NA", "null", "None"}
LEGAL_CANDIDATES = (
    "last_known_tvt",
    "constant_u",
    "robust_linear_u",
    "quadratic_u",
    "constrained_spline_u",
)
DIAGNOSTIC_CANDIDATES = (
    "wrong_sign_constant",
    "noop",
    "duplicate_constant_u",
    "shuffled_linear_u",
    "oracle_target",
)
ALL_CANDIDATES = LEGAL_CANDIDATES + DIAGNOSTIC_CANDIDATES


@dataclass(frozen=True)
class StructuralParameters:
    well_id: str
    known_rows: int
    hidden_rows: int
    last_visible_md: float
    last_visible_z: float
    last_visible_tvt: float
    u_last: float
    v_last: float
    robust_slope: float
    raw_robust_slope: float
    robust_scale: float
    quadratic_curvature: float
    raw_quadratic_curvature: float
    hidden_md_span: float
    spline_horizon: float
    spline_displacement: float


@dataclass
class TrendAccumulator:
    rows: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x_sq: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        self.rows += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x_sq += x * x
        self.sum_xy += x * y

    def finalize(self) -> tuple[float, float]:
        if self.rows <= 0:
            raise ValueError("trend accumulator has no rows")
        count = float(self.rows)
        denominator = self.sum_x_sq - self.sum_x * self.sum_x / count
        numerator = self.sum_xy - self.sum_x * self.sum_y / count
        slope = numerator / denominator if denominator > 0.0 else 0.0
        return self.sum_y / count, slope


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _finite_float(raw: str | None, *, field: str, path: Path, row_number: int) -> float:
    if raw is None or raw.strip() in MISSING_VALUES:
        raise DataValidationError(f"{path}:{row_number}: missing required numeric field {field}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataValidationError(f"{path}:{row_number}: invalid {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise DataValidationError(f"{path}:{row_number}: non-finite {field}={raw!r}")
    return value


def _weighted_line(xs: Sequence[float], ys: Sequence[float], weights: Sequence[float]) -> tuple[float, float]:
    if not xs or len(xs) != len(ys) or len(xs) != len(weights):
        raise ValueError("weighted line requires equal non-empty inputs")
    sum_w = sum(weights)
    if sum_w <= 0.0:
        raise ValueError("weighted line requires positive total weight")
    mean_x = sum(weight * x for weight, x in zip(weights, xs)) / sum_w
    mean_y = sum(weight * y for weight, y in zip(weights, ys)) / sum_w
    denominator = sum(weight * (x - mean_x) ** 2 for weight, x in zip(weights, xs))
    if denominator <= 0.0:
        return mean_y, 0.0
    slope = sum(
        weight * (x - mean_x) * (y - mean_y)
        for weight, x, y in zip(weights, xs, ys)
    ) / denominator
    return mean_y - slope * mean_x, slope


def huber_line(
    md: Sequence[float],
    values: Sequence[float],
    *,
    window_rows: int,
    huber_k: float,
    iterations: int,
) -> tuple[float, float, float]:
    """Return robust intercept, slope, and final robust scale."""
    if len(md) != len(values) or len(md) < 2:
        raise ValueError("huber_line requires at least two paired observations")
    count = min(len(md), int(window_rows))
    md_window = list(md[-count:])
    value_window = list(values[-count:])
    origin = md_window[-1]
    xs = [value - origin for value in md_window]
    weights = [1.0] * count
    intercept, slope = _weighted_line(xs, value_window, weights)
    scale = 0.0
    for _ in range(max(0, int(iterations))):
        residuals = [value - (intercept + slope * x) for x, value in zip(xs, value_window)]
        center = statistics.median(residuals)
        mad = statistics.median(abs(value - center) for value in residuals)
        scale = max(1e-9, 1.4826 * mad)
        threshold = max(1e-12, float(huber_k) * scale)
        weights = [
            1.0 if abs(value - center) <= threshold else threshold / abs(value - center)
            for value in residuals
        ]
        intercept, slope = _weighted_line(xs, value_window, weights)
    return intercept, slope, scale


def _quadratic_curvature(
    md: Sequence[float],
    u_values: Sequence[float],
    *,
    window_rows: int,
    huber_k: float,
    iterations: int,
) -> float:
    count = min(len(md), int(window_rows))
    if count < 8:
        return 0.0
    x = list(md[-count:])
    y = list(u_values[-count:])
    middle = count // 2
    first_x, second_x = x[:middle], x[middle:]
    first_y, second_y = y[:middle], y[middle:]
    _, first_slope, _ = huber_line(
        first_x,
        first_y,
        window_rows=len(first_x),
        huber_k=huber_k,
        iterations=iterations,
    )
    _, second_slope, _ = huber_line(
        second_x,
        second_y,
        window_rows=len(second_x),
        huber_k=huber_k,
        iterations=iterations,
    )
    center_distance = statistics.mean(second_x) - statistics.mean(first_x)
    return (second_slope - first_slope) / center_distance if center_distance > 0.0 else 0.0


def _hermite_value(
    u_start: float,
    start_slope: float,
    end_value: float,
    end_slope: float,
    delta_md: float,
    horizon: float,
) -> float:
    if horizon <= 0.0 or delta_md >= horizon:
        return end_value + end_slope * max(0.0, delta_md - horizon)
    t = max(0.0, delta_md / horizon)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00 * u_start
        + h10 * horizon * start_slope
        + h01 * end_value
        + h11 * horizon * end_slope
    )


def _load_fold_maps(root: Path, fold_files: Sequence[str], expected_signature: str) -> list[dict[str, Any]]:
    maps: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for relative in fold_files:
        path = root / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("data_signature") != expected_signature:
            raise DataValidationError(f"{relative}: data signature mismatch")
        fingerprint = str(data.get("fingerprint") or "")
        if not fingerprint or fingerprint in fingerprints:
            raise DataValidationError(f"{relative}: missing or duplicate fingerprint")
        fingerprints.add(fingerprint)
        maps.append(data)
    if not maps:
        raise DataValidationError("no fold maps configured")
    return maps


def _read_visible_and_span(path: Path) -> tuple[list[float], list[float], list[float], float, int]:
    md: list[float] = []
    z: list[float] = []
    tvt: list[float] = []
    hidden_rows = 0
    hidden_first_md: float | None = None
    hidden_last_md: float | None = None
    hidden_started = False
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, 2):
            current_md = _finite_float(row.get("MD"), field="MD", path=path, row_number=row_number)
            current_z = _finite_float(row.get("Z"), field="Z", path=path, row_number=row_number)
            raw_input = (row.get("TVT_input") or "").strip()
            if raw_input in MISSING_VALUES:
                hidden_started = True
                hidden_rows += 1
                if hidden_first_md is None:
                    hidden_first_md = current_md
                hidden_last_md = current_md
                continue
            if hidden_started:
                raise DataValidationError(f"{path}:{row_number}: TVT_input reappears after hidden suffix")
            md.append(current_md)
            z.append(current_z)
            tvt.append(_finite_float(raw_input, field="TVT_input", path=path, row_number=row_number))
    if len(md) < 2 or hidden_rows <= 0 or hidden_first_md is None or hidden_last_md is None:
        raise DataValidationError(f"{path}: invalid visible-prefix/hidden-suffix structure")
    hidden_span = hidden_last_md - md[-1]
    return md, z, tvt, hidden_span, hidden_rows


def _fit_parameters(
    profile: WellProfile,
    path: Path,
    config: Mapping[str, Any],
) -> tuple[StructuralParameters, dict[str, float | int]]:
    md, z, tvt, hidden_span, hidden_rows = _read_visible_and_span(path)
    u_values = [target + vertical for target, vertical in zip(tvt, z)]
    v_values = [target - vertical for target, vertical in zip(tvt, z)]
    plus_sse = minus_sse = 0.0
    diff_count = 0
    for index in range(1, len(md)):
        plus_delta = u_values[index] - u_values[index - 1]
        minus_delta = v_values[index] - v_values[index - 1]
        plus_sse += plus_delta * plus_delta
        minus_sse += minus_delta * minus_delta
        diff_count += 1

    linear = config["candidates"]["robust_linear_u"]
    _, raw_slope, robust_scale = huber_line(
        md,
        u_values,
        window_rows=int(linear["window_rows"]),
        huber_k=float(linear["huber_k"]),
        iterations=int(linear["iterations"]),
    )
    slope = _clip(raw_slope, float(linear["slope_min"]), float(linear["slope_max"]))

    quadratic = config["candidates"]["quadratic_u"]
    raw_curvature = _quadratic_curvature(
        md,
        u_values,
        window_rows=int(quadratic["window_rows"]),
        huber_k=float(quadratic["huber_k"]),
        iterations=int(quadratic["iterations"]),
    )
    curvature = _clip(
        raw_curvature,
        float(quadratic["curvature_min"]),
        float(quadratic["curvature_max"]),
    )

    spline = config["candidates"]["constrained_spline_u"]
    horizon = _clip(
        float(spline["transition_fraction"]) * hidden_span,
        float(spline["transition_min_rows"]),
        float(spline["transition_max_rows"]),
    )
    displacement = _clip(
        float(spline["endpoint_displacement_fraction"]) * slope * horizon,
        -float(spline["max_abs_u_drift"]),
        float(spline["max_abs_u_drift"]),
    )
    parameters = StructuralParameters(
        well_id=profile.well_id,
        known_rows=len(md),
        hidden_rows=hidden_rows,
        last_visible_md=md[-1],
        last_visible_z=z[-1],
        last_visible_tvt=tvt[-1],
        u_last=u_values[-1],
        v_last=v_values[-1],
        robust_slope=slope,
        raw_robust_slope=raw_slope,
        robust_scale=robust_scale,
        quadratic_curvature=curvature,
        raw_quadratic_curvature=raw_curvature,
        hidden_md_span=hidden_span,
        spline_horizon=horizon,
        spline_displacement=displacement,
    )
    return parameters, {
        "plus_diff_sse": plus_sse,
        "minus_diff_sse": minus_sse,
        "diff_rows": diff_count,
    }


def _prediction_values(
    parameters: StructuralParameters,
    *,
    md: float,
    z: float,
    target: float,
    shuffled_slope: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    delta_md = md - parameters.last_visible_md
    quadratic = config["candidates"]["quadratic_u"]
    quadratic_drift = _clip(
        parameters.robust_slope * delta_md
        + 0.5 * parameters.quadratic_curvature * delta_md * delta_md,
        -float(quadratic["max_abs_u_drift"]),
        float(quadratic["max_abs_u_drift"]),
    )
    spline = config["candidates"]["constrained_spline_u"]
    end_u = parameters.u_last + parameters.spline_displacement
    end_slope = float(spline["endpoint_slope_fraction"]) * parameters.robust_slope
    spline_u = _hermite_value(
        parameters.u_last,
        parameters.robust_slope,
        end_u,
        end_slope,
        delta_md,
        parameters.spline_horizon,
    )
    return {
        "last_known_tvt": parameters.last_visible_tvt,
        "constant_u": parameters.u_last - z,
        "robust_linear_u": parameters.u_last + parameters.robust_slope * delta_md - z,
        "quadratic_u": parameters.u_last + quadratic_drift - z,
        "constrained_spline_u": spline_u - z,
        "wrong_sign_constant": parameters.v_last + z,
        "noop": parameters.last_visible_tvt + 0.0,
        "duplicate_constant_u": ((parameters.u_last - z) + (parameters.u_last - z)) / 2.0,
        "shuffled_linear_u": parameters.u_last + shuffled_slope * delta_md - z,
        "oracle_target": target,
    }


def _synthetic_coefficient(profile: WellProfile) -> float:
    return 0.02 * math.sin(profile.last_visible_z / 37.0) + 0.005 * math.cos(profile.last_visible_md / 53.0)


def _write_gzip_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    normalized: dict[str, Any] = {}
                    for key in fieldnames:
                        value = row.get(key, "")
                        normalized[key] = format(value, ".12g") if isinstance(value, float) else value
                    writer.writerow(normalized)


def _regime_metrics(
    metrics: Mapping[str, Mapping[str, WellMetric]],
    profiles: Mapping[str, WellProfile],
    thresholds: tuple[float, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in ALL_CANDIDATES:
        by_regime: dict[str, list[WellMetric]] = {"short_hidden": [], "medium_hidden": [], "long_hidden": []}
        for well_id, metric in metrics[candidate].items():
            by_regime[_regime(profiles[well_id], thresholds)].append(metric)
        for regime, selected in by_regime.items():
            summary = _summarize(selected)
            rows.append({"candidate": candidate, "regime": regime, **summary})
    return rows


def _correlation_rows(correlations: Mapping[tuple[str, str], OnlineCorrelation]) -> list[dict[str, Any]]:
    return [
        {"candidate_a": pair[0], "candidate_b": pair[1], "residual_correlation": accumulator.value()}
        for pair, accumulator in sorted(correlations.items())
    ]


def run_e002(
    *,
    root: Path,
    train_dir: Path,
    fold_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen E002 ladder and write deterministic artifacts."""
    started = time.perf_counter()
    root = root.resolve()
    train_dir = train_dir.resolve()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    profiles, data_profile = scan_profiles(train_dir)
    expected_signature = str(config["data_signature"])
    if data_profile["data_signature"] != expected_signature:
        raise DataValidationError("E002 data signature differs from frozen E001 signature")
    profile_by_id = {profile.well_id: profile for profile in profiles}
    fold_maps = _load_fold_maps(root, config["fold_files"], expected_signature)

    parameters: dict[str, StructuralParameters] = {}
    plus_sse = minus_sse = 0.0
    visible_diff_rows = 0
    for profile in sorted(profiles, key=lambda item: item.well_id):
        path = train_dir / f"{profile.well_id}__horizontal_well.csv"
        fitted, sign = _fit_parameters(profile, path, config)
        parameters[profile.well_id] = fitted
        plus_sse += float(sign["plus_diff_sse"])
        minus_sse += float(sign["minus_diff_sse"])
        visible_diff_rows += int(sign["diff_rows"])

    sign_plus_rms = math.sqrt(plus_sse / visible_diff_rows)
    sign_minus_rms = math.sqrt(minus_sse / visible_diff_rows)
    sign_ratio = sign_plus_rms / sign_minus_rms

    shuffled_order = sorted(parameters, key=lambda well_id: _stable_int("e002-slope-shuffle", well_id))
    shuffled_slopes = {
        well_id: parameters[shuffled_order[(index + 1) % len(shuffled_order)]].robust_slope
        for index, well_id in enumerate(shuffled_order)
    }
    synthetic_coefficients = {profile.well_id: _synthetic_coefficient(profile) for profile in profiles}
    synthetic_shuffled = {
        well_id: synthetic_coefficients[shuffled_order[(index + 1) % len(shuffled_order)]]
        for index, well_id in enumerate(shuffled_order)
    }

    model_metrics: dict[str, dict[str, WellMetric]] = {candidate: {} for candidate in ALL_CANDIDATES}
    correlations: dict[tuple[str, str], OnlineCorrelation] = {}
    for index, first in enumerate(LEGAL_CANDIDATES):
        for second in LEGAL_CANDIDATES[index + 1 :]:
            correlations[(first, second)] = OnlineCorrelation()
    synthetic_sse = {"null": 0.0, "positive": 0.0, "shuffled": 0.0}
    synthetic_rows = 0
    direct_sse = {candidate: 0.0 for candidate in ALL_CANDIDATES}
    hidden_trends: dict[str, dict[str, float | int | bool]] = {}
    oof_path = artifact_dir / "oof_predictions.csv.gz"
    oof_fields = ["well_id", "row_index", "hidden_index", "md", "z", "target", *LEGAL_CANDIDATES]

    def prediction_rows() -> Iterable[Mapping[str, Any]]:
        nonlocal synthetic_rows
        for profile in sorted(profiles, key=lambda item: item.well_id):
            fitted = parameters[profile.well_id]
            path = train_dir / f"{profile.well_id}__horizontal_well.csv"
            accumulators = {candidate: ErrorAccumulator() for candidate in ALL_CANDIDATES}
            hidden_u_trend = TrendAccumulator()
            hidden_tvt_trend = TrendAccumulator()
            hidden_z_trend = TrendAccumulator()
            hidden_index = 0
            row_index = 0
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, 2):
                    raw_input = (row.get("TVT_input") or "").strip()
                    if raw_input not in MISSING_VALUES:
                        row_index += 1
                        continue
                    md = _finite_float(row.get("MD"), field="MD", path=path, row_number=row_number)
                    z = _finite_float(row.get("Z"), field="Z", path=path, row_number=row_number)
                    target = _finite_float(row.get("TVT"), field="TVT", path=path, row_number=row_number)
                    predictions = _prediction_values(
                        fitted,
                        md=md,
                        z=z,
                        target=target,
                        shuffled_slope=shuffled_slopes[profile.well_id],
                        config=config,
                    )
                    errors: dict[str, float] = {}
                    for candidate, prediction in predictions.items():
                        error = prediction - target
                        errors[candidate] = error
                        direct_sse[candidate] += error * error
                        accumulators[candidate].add(error, float(hidden_index))
                    for pair, accumulator in correlations.items():
                        accumulator.add(errors[pair[0]], errors[pair[1]])
                    hidden_u_trend.add(float(hidden_index), target + z)
                    hidden_tvt_trend.add(float(hidden_index), target)
                    hidden_z_trend.add(float(hidden_index), z)

                    delta_md = md - fitted.last_visible_md
                    synthetic_target = fitted.u_last + synthetic_coefficients[profile.well_id] * delta_md - z
                    synthetic_predictions = {
                        "null": fitted.u_last - z,
                        "positive": fitted.u_last + synthetic_coefficients[profile.well_id] * delta_md - z,
                        "shuffled": fitted.u_last + synthetic_shuffled[profile.well_id] * delta_md - z,
                    }
                    for name, prediction in synthetic_predictions.items():
                        error = prediction - synthetic_target
                        synthetic_sse[name] += error * error
                    synthetic_rows += 1
                    yield {
                        "well_id": profile.well_id,
                        "row_index": row_index,
                        "hidden_index": hidden_index,
                        "md": md,
                        "z": z,
                        "target": target,
                        **{candidate: predictions[candidate] for candidate in LEGAL_CANDIDATES},
                    }
                    hidden_index += 1
                    row_index += 1
            if hidden_index != profile.hidden_rows:
                raise DataValidationError(
                    f"{path}: scored hidden rows {hidden_index}, expected {profile.hidden_rows}"
                )
            _, hidden_u_slope = hidden_u_trend.finalize()
            _, hidden_tvt_slope = hidden_tvt_trend.finalize()
            _, hidden_z_slope = hidden_z_trend.finalize()
            hidden_trends[profile.well_id] = {
                "well_id": profile.well_id,
                "visible_u_slope": fitted.robust_slope,
                "hidden_u_slope_oracle": hidden_u_slope,
                "hidden_tvt_slope_oracle": hidden_tvt_slope,
                "hidden_z_slope": hidden_z_slope,
                "u_slope_delta_oracle": hidden_u_slope - fitted.robust_slope,
                "visible_hidden_u_sign_match_oracle": (
                    fitted.robust_slope == 0.0
                    or hidden_u_slope == 0.0
                    or math.copysign(1.0, fitted.robust_slope) == math.copysign(1.0, hidden_u_slope)
                ),
            }
            for candidate, accumulator in accumulators.items():
                model_metrics[candidate][profile.well_id] = accumulator.finalize(profile.well_id)

    _write_gzip_csv(oof_path, oof_fields, prediction_rows())

    summaries = {
        candidate: _summarize([model_metrics[candidate][well_id] for well_id in sorted(model_metrics[candidate])])
        for candidate in ALL_CANDIDATES
    }
    synthetic_summary = {
        name: {"rows_scored": synthetic_rows, "sse": value, "rmse": math.sqrt(value / synthetic_rows)}
        for name, value in synthetic_sse.items()
    }
    baseline = summaries["last_known_tvt"]
    fold_rows: list[dict[str, Any]] = []
    fold_lookup: dict[tuple[str, str, int], float] = {}
    for candidate in ALL_CANDIDATES:
        for fold_map in fold_maps:
            for row in _fold_metrics(model_metrics[candidate], fold_map):
                enriched = {"candidate": candidate, **row}
                fold_rows.append(enriched)
                fold_lookup[(candidate, str(row["map"]), int(row["fold"]))] = float(row["rmse"])

    improved_cells: dict[str, int] = {}
    for candidate in LEGAL_CANDIDATES[1:]:
        improved_cells[candidate] = sum(
            1
            for fold_map in fold_maps
            for fold in range(int(fold_map["n_folds"]))
            if fold_lookup[(candidate, str(fold_map["version"]), fold)]
            < fold_lookup[("last_known_tvt", str(fold_map["version"]), fold)]
        )

    thresholds = _regime_thresholds(profiles)
    regime_rows = _regime_metrics(model_metrics, profile_by_id, thresholds)
    regime_lookup = {
        (str(row["candidate"]), str(row["regime"])): row for row in regime_rows
    }
    challenger_ranked = sorted(
        LEGAL_CANDIDATES[1:],
        key=lambda candidate: (float(summaries[candidate]["rmse"]), candidate),
    )
    best_challenger = challenger_ranked[0]
    retained_candidate = min(
        LEGAL_CANDIDATES,
        key=lambda candidate: (float(summaries[candidate]["rmse"]), candidate),
    )
    promotion = config["promotion"]
    challenger_summary = summaries[best_challenger]
    pooled_gain = float(baseline["rmse"]) - float(challenger_summary["rmse"])
    p90_delta = float(challenger_summary["p90_well_rmse"]) - float(baseline["p90_well_rmse"])
    worst5_delta = float(challenger_summary["worst_5pct_sse_share"]) - float(baseline["worst_5pct_sse_share"])
    long_gain = (
        float(regime_lookup[("last_known_tvt", "long_hidden")]["rmse"])
        - float(regime_lookup[(best_challenger, "long_hidden")]["rmse"])
    )

    pooled_consistency = max(
        abs(direct_sse[candidate] - float(summaries[candidate]["sse"]))
        / max(1.0, direct_sse[candidate], float(summaries[candidate]["sse"]))
        for candidate in ALL_CANDIDATES
    )
    duplicate_correlation = OnlineCorrelation()
    for well_id in sorted(model_metrics["constant_u"]):
        duplicate_correlation.add(
            model_metrics["constant_u"][well_id].mean_error,
            model_metrics["duplicate_constant_u"][well_id].mean_error,
        )

    elapsed_seconds = time.perf_counter() - started
    controls = {
        "data_integrity": {
            "pass": data_profile["data_signature"] == expected_signature
            and len(profiles) == int(config.get("expected_wells", len(profiles))),
            "data_signature": data_profile["data_signature"],
            "expected_wells": int(config.get("expected_wells", len(profiles))),
            "well_count": len(profiles),
        },
        "visible_sign_verification": {
            "pass": sign_ratio <= float(config["sign_verification"]["required_plus_to_minus_rms_ratio_max"]),
            "plus_rms": sign_plus_rms,
            "minus_rms": sign_minus_rms,
            "plus_to_minus_ratio": sign_ratio,
        },
        "algebraic_roundtrip": {
            "pass": max(
                abs((parameters[well_id].u_last - parameters[well_id].last_visible_z) - parameters[well_id].last_visible_tvt)
                for well_id in parameters
            ) <= 1e-12,
        },
        "pooled_rmse_consistency": {
            "pass": pooled_consistency <= 1e-12,
            "maximum_relative_difference": pooled_consistency,
        },
        "noop": {
            "pass": abs(float(summaries["noop"]["rmse"]) - float(baseline["rmse"])) <= 1e-15,
            "rmse": summaries["noop"]["rmse"],
        },
        "duplicate": {
            "pass": abs(float(summaries["duplicate_constant_u"]["rmse"]) - float(summaries["constant_u"]["rmse"])) <= 1e-15
            and duplicate_correlation.value() is not None
            and abs(float(duplicate_correlation.value()) - 1.0) <= 1e-12,
            "rmse": summaries["duplicate_constant_u"]["rmse"],
            "well_mean_error_correlation": duplicate_correlation.value(),
        },
        "wrong_sign": {
            "pass": float(summaries["wrong_sign_constant"]["rmse"])
            > float(summaries["constant_u"]["rmse"]) + 0.1,
            "wrong_sign_rmse": summaries["wrong_sign_constant"]["rmse"],
            "correct_sign_rmse": summaries["constant_u"]["rmse"],
        },
        "synthetic_positive": {
            "pass": float(synthetic_summary["positive"]["rmse"]) <= 1e-12
            and float(synthetic_summary["null"]["rmse"]) > 1e-6,
            "null_rmse": synthetic_summary["null"]["rmse"],
            "positive_rmse": synthetic_summary["positive"]["rmse"],
        },
        "shuffled_evidence": {
            "pass": float(summaries["shuffled_linear_u"]["rmse"])
            > float(summaries["robust_linear_u"]["rmse"]) + 1e-6
            and float(synthetic_summary["shuffled"]["rmse"])
            > float(synthetic_summary["positive"]["rmse"]) + 1e-6,
            "actual_linear_rmse": summaries["robust_linear_u"]["rmse"],
            "actual_shuffled_rmse": summaries["shuffled_linear_u"]["rmse"],
            "synthetic_positive_rmse": synthetic_summary["positive"]["rmse"],
            "synthetic_shuffled_rmse": synthetic_summary["shuffled"]["rmse"],
        },
        "leakage_sentinel": {
            "pass": float(summaries["oracle_target"]["rmse"]) <= 1e-12,
            "rmse": summaries["oracle_target"]["rmse"],
            "uses_hidden_target": True,
            "eligible_for_modeling": False,
        },
        "runtime": {
            "pass": elapsed_seconds <= 60.0 * float(promotion["maximum_runtime_minutes"]),
            "budget_minutes": promotion["maximum_runtime_minutes"],
        },
    }
    control_pass = all(bool(item["pass"]) for item in controls.values())
    gates = {
        "minimum_pooled_gain": pooled_gain >= float(promotion["minimum_pooled_rmse_gain"]),
        "fold_cell_stability": improved_cells[best_challenger] >= int(promotion["minimum_improved_fold_cells"]),
        "p90_tail": p90_delta <= float(promotion["maximum_p90_well_rmse_deterioration"]),
        "worst_5pct_share": worst5_delta <= float(promotion["maximum_worst_5pct_sse_share_increase"]),
        "long_hidden_stress": long_gain > 0.0 if bool(promotion["require_long_hidden_improvement"]) else True,
        "controls": control_pass,
        "runtime": bool(controls["runtime"]["pass"]),
    }
    promoted = all(gates.values())

    trend_correlation = OnlineCorrelation()
    trend_rows = [hidden_trends[well_id] for well_id in sorted(hidden_trends)]
    for row in trend_rows:
        trend_correlation.add(float(row["visible_u_slope"]), float(row["hidden_u_slope_oracle"]))
    trend_transfer = {
        "visible_to_hidden_u_slope_correlation_oracle": trend_correlation.value(),
        "visible_hidden_u_sign_agreement_oracle": sum(
            1 for row in trend_rows if bool(row["visible_hidden_u_sign_match_oracle"])
        ) / len(trend_rows),
        "median_visible_u_slope": statistics.median(float(row["visible_u_slope"]) for row in trend_rows),
        "median_hidden_u_slope_oracle": statistics.median(float(row["hidden_u_slope_oracle"]) for row in trend_rows),
        "median_hidden_tvt_slope_oracle": statistics.median(float(row["hidden_tvt_slope_oracle"]) for row in trend_rows),
        "median_hidden_z_slope": statistics.median(float(row["hidden_z_slope"]) for row in trend_rows),
        "median_absolute_u_slope_delta_oracle": statistics.median(
            abs(float(row["u_slope_delta_oracle"])) for row in trend_rows
        ),
        "uses_hidden_target": True,
        "eligible_for_modeling": False,
    }

    candidate_rows = []
    for candidate in ALL_CANDIDATES:
        summary = summaries[candidate]
        candidate_rows.append(
            {
                "candidate": candidate,
                "eligible": candidate in LEGAL_CANDIDATES,
                "selected": candidate == retained_candidate,
                "best_challenger": candidate == best_challenger,
                "rmse": summary["rmse"],
                "gain_vs_last_known": float(baseline["rmse"]) - float(summary["rmse"]),
                "median_well_rmse": summary["median_well_rmse"],
                "p90_well_rmse": summary["p90_well_rmse"],
                "p95_well_rmse": summary["p95_well_rmse"],
                "max_well_rmse": summary["max_well_rmse"],
                "worst_5pct_sse_share": summary["worst_5pct_sse_share"],
                "worst_10pct_sse_share": summary["worst_10pct_sse_share"],
                "mean_error_sse_share": summary["mean_error_sse_share"],
                "linear_trend_sse_share": summary["linear_trend_sse_share"],
                "shape_sse_share": summary["shape_sse_share"],
                "improved_fold_cells": improved_cells.get(candidate, ""),
            }
        )

    parameter_rows = [asdict(parameters[well_id]) for well_id in sorted(parameters)]
    all_well_rows: list[dict[str, Any]] = []
    retained_well_rows: list[dict[str, Any]] = []
    challenger_well_rows: list[dict[str, Any]] = []
    for candidate in ALL_CANDIDATES:
        for well_id in sorted(model_metrics[candidate]):
            metric = model_metrics[candidate][well_id]
            folds = {
                f"fold_{fold_map['version']}": fold_map["assignments"][well_id]
                for fold_map in fold_maps
            }
            row = {
                "candidate": candidate,
                "well_id": well_id,
                "split": "cv",
                "rows_scored": metric.rows_scored,
                "rmse": metric.rmse,
                "mean_error": metric.mean_error,
                "sse": metric.sse,
                "regime": _regime(profile_by_id[well_id], thresholds),
                "uncertainty": parameters[well_id].robust_scale,
                "datum_sse": metric.datum_sse,
                "trend_sse": metric.trend_sse,
                "shape_sse": metric.shape_sse,
                "trend_per_row": metric.trend_per_row,
                **folds,
            }
            all_well_rows.append(row)
            if candidate == retained_candidate:
                retained_well_rows.append({key: value for key, value in row.items() if key != "candidate"})
            if candidate == best_challenger:
                challenger_well_rows.append({key: value for key, value in row.items() if key != "candidate"})

    summary = {
        "schema_version": 1,
        "experiment_id": "E002",
        "status": "promoted" if promoted else "rejected",
        "selected_candidate": retained_candidate,
        "best_challenger": best_challenger,
        "data": data_profile,
        "sign_verification": {
            "visible_difference_rows": visible_diff_rows,
            "u_equals_tvt_plus_z_rms": sign_plus_rms,
            "wrong_sign_tvt_minus_z_rms": sign_minus_rms,
            "plus_to_minus_ratio": sign_ratio,
        },
        "candidate_metrics": summaries,
        "synthetic_control_metrics": synthetic_summary,
        "improved_fold_cells": improved_cells,
        "regime_metrics": regime_rows,
        "residual_correlations": _correlation_rows(correlations),
        "trend_transfer_diagnostic": trend_transfer,
        "selection": {
            "candidate": best_challenger,
            "retained_candidate": retained_candidate,
            "pooled_rmse_gain": pooled_gain,
            "p90_well_rmse_delta": p90_delta,
            "worst_5pct_sse_share_delta": worst5_delta,
            "long_hidden_rmse_gain": long_gain,
            "gates": gates,
        },
        "controls": controls,
        "predictor_policy": {
            candidate: {
                "uses_hidden_target": candidate == "oracle_target",
                "eligible_for_modeling": candidate in LEGAL_CANDIDATES,
            }
            for candidate in ALL_CANDIDATES
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_csv(
        output_dir / "candidate_metrics.csv",
        list(candidate_rows[0]),
        candidate_rows,
    )
    _write_csv(
        output_dir / "fold_metrics.csv",
        ["candidate", "map", "fold", "wells", "rows_scored", "sse", "rmse"],
        fold_rows,
    )
    regime_fields = ["candidate", "regime", *[key for key in regime_rows[0] if key not in {"candidate", "regime"}]]
    _write_csv(output_dir / "regime_metrics.csv", regime_fields, regime_rows)
    _write_csv(output_dir / "parameter_diagnostics.csv", list(parameter_rows[0]), parameter_rows)
    correlation_rows = _correlation_rows(correlations)
    _write_csv(
        output_dir / "residual_correlations.csv",
        ["candidate_a", "candidate_b", "residual_correlation"],
        correlation_rows,
    )
    all_well_fields = list(all_well_rows[0])
    retained_well_fields = list(retained_well_rows[0])
    challenger_well_fields = list(challenger_well_rows[0])
    _write_csv(output_dir / "all_candidate_well_metrics.csv", all_well_fields, all_well_rows)
    _write_csv(output_dir / "selected_well_metrics.csv", retained_well_fields, retained_well_rows)
    _write_csv(output_dir / "challenger_well_metrics.csv", challenger_well_fields, challenger_well_rows)
    _write_csv(
        output_dir / "trend_transfer_diagnostics.csv",
        list(trend_rows[0]),
        trend_rows,
    )
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
        output_dir / "candidate_metrics.csv",
        output_dir / "fold_metrics.csv",
        output_dir / "regime_metrics.csv",
        output_dir / "parameter_diagnostics.csv",
        output_dir / "residual_correlations.csv",
        output_dir / "all_candidate_well_metrics.csv",
        output_dir / "selected_well_metrics.csv",
        output_dir / "challenger_well_metrics.csv",
        output_dir / "trend_transfer_diagnostics.csv",
        output_dir / "control_metrics.csv",
    ]
    artifact_manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(result_files, key=lambda item: str(item))
        ],
        "external_artifacts": [
            {
                "path": oof_path.name,
                "sha256": _sha256(oof_path),
                "bytes": oof_path.stat().st_size,
                "tracked": False,
            }
        ],
        "fold_files": [
            {
                "path": relative,
                "sha256": _sha256(root / relative),
                "bytes": (root / relative).stat().st_size,
            }
            for relative in config["fold_files"]
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return {**summary, "runtime_seconds": elapsed_seconds}
