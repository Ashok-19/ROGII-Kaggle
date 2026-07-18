"""E001: deterministic whole-well suffix folds, metrics, and controls.

This module intentionally uses only the Python standard library so the same
implementation can run locally and in an offline Kaggle notebook.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_FOLD_SEEDS = (104729, 130363, 155921, 181081, 209759)
REQUIRED_COLUMNS = {"MD", "Z", "TVT", "TVT_input"}
MISSING_VALUES = {"", "nan", "NaN", "NA", "null", "None"}


class DataValidationError(ValueError):
    """Raised when competition data violates the E001 invariants."""


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(raw: str | None, *, field: str, path: Path, row_number: int) -> float:
    if raw is None or raw.strip() in MISSING_VALUES:
        raise DataValidationError(f"{path}:{row_number}: missing required numeric field {field}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataValidationError(f"{path}:{row_number}: invalid {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise DataValidationError(f"{path}:{row_number}: non-finite {field}={raw!r}")
    return value


@dataclass(frozen=True)
class WellProfile:
    well_id: str
    path: str
    total_rows: int
    known_rows: int
    hidden_rows: int
    last_visible_md: float
    last_visible_z: float
    last_visible_tvt: float
    max_visible_target_delta: float


@dataclass
class ErrorAccumulator:
    rows: int = 0
    sum_error: float = 0.0
    sum_error_sq: float = 0.0
    sum_x: float = 0.0
    sum_x_sq: float = 0.0
    sum_x_error: float = 0.0

    def add(self, error: float, x: float) -> None:
        self.rows += 1
        self.sum_error += error
        self.sum_error_sq += error * error
        self.sum_x += x
        self.sum_x_sq += x * x
        self.sum_x_error += x * error

    def finalize(self, well_id: str) -> "WellMetric":
        if self.rows <= 0:
            raise DataValidationError(f"well {well_id} has no scored rows")
        n = float(self.rows)
        mean_error = self.sum_error / n
        centered_x_sq = self.sum_x_sq - (self.sum_x * self.sum_x) / n
        centered_x_error = self.sum_x_error - (self.sum_x * self.sum_error) / n
        slope = centered_x_error / centered_x_sq if centered_x_sq > 0.0 else 0.0
        datum_sse = (self.sum_error * self.sum_error) / n
        trend_sse = slope * slope * centered_x_sq
        shape_sse = self.sum_error_sq - datum_sse - trend_sse
        tolerance = max(1e-9, self.sum_error_sq * 1e-12)
        if shape_sse < 0.0 and abs(shape_sse) <= tolerance:
            shape_sse = 0.0
        if shape_sse < 0.0:
            raise ArithmeticError(f"negative shape SSE for {well_id}: {shape_sse}")
        return WellMetric(
            well_id=well_id,
            rows_scored=self.rows,
            rmse=math.sqrt(self.sum_error_sq / n),
            mean_error=mean_error,
            sse=self.sum_error_sq,
            datum_sse=datum_sse,
            trend_sse=trend_sse,
            shape_sse=shape_sse,
            trend_per_row=slope,
        )


@dataclass(frozen=True)
class WellMetric:
    well_id: str
    rows_scored: int
    rmse: float
    mean_error: float
    sse: float
    datum_sse: float
    trend_sse: float
    shape_sse: float
    trend_per_row: float


@dataclass
class OnlineCorrelation:
    rows: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x_sq: float = 0.0
    sum_y_sq: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        self.rows += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x_sq += x * x
        self.sum_y_sq += y * y
        self.sum_xy += x * y

    def value(self) -> float | None:
        if self.rows < 2:
            return None
        n = float(self.rows)
        covariance = self.sum_xy - self.sum_x * self.sum_y / n
        variance_x = self.sum_x_sq - self.sum_x * self.sum_x / n
        variance_y = self.sum_y_sq - self.sum_y * self.sum_y / n
        if variance_x <= 0.0 or variance_y <= 0.0:
            return None
        return covariance / math.sqrt(variance_x * variance_y)


def scan_profiles(train_dir: Path) -> tuple[list[WellProfile], dict[str, Any]]:
    """Validate all horizontal-well files and return deterministic profiles."""
    train_dir = train_dir.resolve()
    files = sorted(train_dir.glob("*__horizontal_well.csv"))
    if not files:
        raise FileNotFoundError(f"no horizontal-well CSVs found under {train_dir}")
    profiles: list[WellProfile] = []
    seen_ids: set[str] = set()
    total_rows = known_rows = hidden_rows = 0
    global_max_visible_delta = 0.0

    for path in files:
        well_id = path.name.split("__", 1)[0]
        if well_id in seen_ids:
            raise DataValidationError(f"duplicate well ID {well_id}")
        seen_ids.add(well_id)
        rows = known = hidden = 0
        missing_started = False
        previous_md: float | None = None
        last_visible_md = last_visible_z = last_visible_tvt = None
        max_visible_delta = 0.0
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing_columns = REQUIRED_COLUMNS - fields
            if missing_columns:
                raise DataValidationError(f"{path}: missing columns {sorted(missing_columns)}")
            for row_number, row in enumerate(reader, 2):
                rows += 1
                md = _float(row.get("MD"), field="MD", path=path, row_number=row_number)
                z = _float(row.get("Z"), field="Z", path=path, row_number=row_number)
                target = _float(row.get("TVT"), field="TVT", path=path, row_number=row_number)
                if previous_md is not None and md < previous_md:
                    raise DataValidationError(f"{path}:{row_number}: MD decreased from {previous_md} to {md}")
                previous_md = md
                raw_input = (row.get("TVT_input") or "").strip()
                is_missing = raw_input in MISSING_VALUES
                if is_missing:
                    missing_started = True
                    hidden += 1
                else:
                    if missing_started:
                        raise DataValidationError(f"{path}:{row_number}: TVT_input reappears after hidden suffix begins")
                    visible = _float(raw_input, field="TVT_input", path=path, row_number=row_number)
                    known += 1
                    last_visible_md = md
                    last_visible_z = z
                    last_visible_tvt = visible
                    max_visible_delta = max(max_visible_delta, abs(visible - target))
        if known == 0 or hidden == 0:
            raise DataValidationError(f"{path}: expected non-empty visible prefix and hidden suffix, got {known}/{hidden}")
        if rows != known + hidden:
            raise DataValidationError(f"{path}: row accounting mismatch")
        assert last_visible_md is not None and last_visible_z is not None and last_visible_tvt is not None
        profiles.append(
            WellProfile(
                well_id=well_id,
                path=str(path.relative_to(train_dir.parent.parent)),
                total_rows=rows,
                known_rows=known,
                hidden_rows=hidden,
                last_visible_md=last_visible_md,
                last_visible_z=last_visible_z,
                last_visible_tvt=last_visible_tvt,
                max_visible_target_delta=max_visible_delta,
            )
        )
        total_rows += rows
        known_rows += known
        hidden_rows += hidden
        global_max_visible_delta = max(global_max_visible_delta, max_visible_delta)

    signature_payload = [asdict(profile) for profile in sorted(profiles, key=lambda item: item.well_id)]
    data_signature = hashlib.sha256(_canonical_json(signature_payload).encode("utf-8")).hexdigest()
    return profiles, {
        "well_count": len(profiles),
        "total_rows": total_rows,
        "known_rows": known_rows,
        "hidden_rows": hidden_rows,
        "max_visible_target_delta": global_max_visible_delta,
        "data_signature": data_signature,
    }


def _fold_summary(assignments: Mapping[str, int], profiles: Sequence[WellProfile], n_folds: int) -> list[dict[str, int]]:
    summary = [{"fold": fold, "wells": 0, "known_rows": 0, "hidden_rows": 0, "total_rows": 0} for fold in range(n_folds)]
    for profile in profiles:
        fold = assignments[profile.well_id]
        item = summary[fold]
        item["wells"] += 1
        item["known_rows"] += profile.known_rows
        item["hidden_rows"] += profile.hidden_rows
        item["total_rows"] += profile.total_rows
    return summary


def generate_fold_maps(
    profiles: Sequence[WellProfile],
    data_signature: str,
    *,
    seeds: Sequence[int] = DEFAULT_FOLD_SEEDS,
    n_folds: int = 5,
) -> list[dict[str, Any]]:
    """Generate repeated deterministic whole-well maps balanced by scored rows."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(profiles) < n_folds:
        raise ValueError("number of wells must be at least n_folds")
    maps: list[dict[str, Any]] = []
    for version_index, seed in enumerate(seeds, 1):
        loads = [{"hidden": 0, "known": 0, "wells": 0} for _ in range(n_folds)]
        assignments: dict[str, int] = {}
        ordered = sorted(profiles, key=lambda profile: _stable_int("order", seed, profile.well_id))
        for profile in ordered:
            fold = min(
                range(n_folds),
                key=lambda candidate: (
                    loads[candidate]["hidden"],
                    loads[candidate]["wells"],
                    loads[candidate]["known"],
                    _stable_int("tie", seed, profile.well_id, candidate),
                ),
            )
            assignments[profile.well_id] = fold
            loads[fold]["hidden"] += profile.hidden_rows
            loads[fold]["known"] += profile.known_rows
            loads[fold]["wells"] += 1
        canonical_assignments = {well_id: assignments[well_id] for well_id in sorted(assignments)}
        fingerprint = hashlib.sha256(_canonical_json(canonical_assignments).encode("utf-8")).hexdigest()
        maps.append(
            {
                "schema_version": 1,
                "version": f"v{version_index}",
                "seed": int(seed),
                "n_folds": n_folds,
                "strategy": "stable-hash-order greedy hidden-row balance",
                "data_signature": data_signature,
                "well_count": len(profiles),
                "fingerprint": fingerprint,
                "assignments": canonical_assignments,
                "fold_summary": _fold_summary(canonical_assignments, profiles, n_folds),
            }
        )
    return maps


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize(metrics: Sequence[WellMetric]) -> dict[str, float | int | None]:
    rows = sum(metric.rows_scored for metric in metrics)
    total_sse = sum(metric.sse for metric in metrics)
    datum_sse = sum(metric.datum_sse for metric in metrics)
    trend_sse = sum(metric.trend_sse for metric in metrics)
    shape_sse = sum(metric.shape_sse for metric in metrics)
    by_sse = sorted(metrics, key=lambda metric: (-metric.sse, metric.well_id))
    worst_5_count = max(1, math.ceil(len(metrics) * 0.05))
    worst_10_count = max(1, math.ceil(len(metrics) * 0.10))
    rmse_values = [metric.rmse for metric in metrics]
    if total_sse == 0.0:
        datum_share = trend_share = shape_share = None
        worst_5_share = worst_10_share = None
    else:
        datum_share = datum_sse / total_sse
        trend_share = trend_sse / total_sse
        shape_share = shape_sse / total_sse
        worst_5_share = sum(metric.sse for metric in by_sse[:worst_5_count]) / total_sse
        worst_10_share = sum(metric.sse for metric in by_sse[:worst_10_count]) / total_sse
    return {
        "rows_scored": rows,
        "wells_scored": len(metrics),
        "sse": total_sse,
        "rmse": math.sqrt(total_sse / rows),
        "median_well_rmse": _quantile(rmse_values, 0.5),
        "p90_well_rmse": _quantile(rmse_values, 0.9),
        "p95_well_rmse": _quantile(rmse_values, 0.95),
        "max_well_rmse": max(rmse_values),
        "worst_5pct_sse_share": worst_5_share,
        "worst_10pct_sse_share": worst_10_share,
        "mean_error_sse_share": datum_share,
        "linear_trend_sse_share": trend_share,
        "shape_sse_share": shape_share,
        "datum_removed_rmse": math.sqrt(max(0.0, total_sse - datum_sse) / rows),
        "datum_trend_removed_rmse": math.sqrt(max(0.0, shape_sse) / rows),
    }


def _coefficient(profile: WellProfile) -> float:
    """Legal visible-prefix coefficient for the synthetic positive control."""
    return 0.02 * math.sin(profile.last_visible_z / 37.0) + 0.005 * math.cos(profile.last_visible_md / 53.0)


def _regime_thresholds(profiles: Sequence[WellProfile]) -> tuple[float, float]:
    values = [float(profile.hidden_rows) for profile in profiles]
    return _quantile(values, 1.0 / 3.0), _quantile(values, 2.0 / 3.0)


def _regime(profile: WellProfile, thresholds: tuple[float, float]) -> str:
    if profile.hidden_rows <= thresholds[0]:
        return "short_hidden"
    if profile.hidden_rows <= thresholds[1]:
        return "medium_hidden"
    return "long_hidden"


def _fold_metrics(
    metrics: Mapping[str, WellMetric], fold_map: Mapping[str, Any]
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    assignments = fold_map["assignments"]
    for fold in range(int(fold_map["n_folds"])):
        selected = [metric for well_id, metric in metrics.items() if assignments[well_id] == fold]
        scored_rows = sum(metric.rows_scored for metric in selected)
        sse = sum(metric.sse for metric in selected)
        rows.append(
            {
                "map": str(fold_map["version"]),
                "fold": fold,
                "wells": len(selected),
                "rows_scored": scored_rows,
                "sse": sse,
                "rmse": math.sqrt(sse / scored_rows),
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    normalized[key] = format(value, ".12g")
                elif value is None:
                    normalized[key] = ""
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def run_e001(
    *,
    train_dir: Path,
    fold_dir: Path,
    output_dir: Path,
    seeds: Sequence[int] = DEFAULT_FOLD_SEEDS,
    n_folds: int = 5,
    expected_wells: int | None = 773,
    expected_baseline_rmse: float | None = 15.91,
    baseline_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Run E001 and write deterministic fold and result artifacts."""
    profiles, data_profile = scan_profiles(train_dir)
    profile_by_id = {profile.well_id: profile for profile in profiles}
    fold_maps = generate_fold_maps(
        profiles,
        data_profile["data_signature"],
        seeds=seeds,
        n_folds=n_folds,
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold_map in fold_maps:
        _write_json(fold_dir / f"{fold_map['version']}.json", fold_map)

    coefficients = {profile.well_id: _coefficient(profile) for profile in profiles}
    shuffle_order = sorted(profile_by_id, key=lambda well_id: _stable_int("synthetic-shuffle", well_id))
    shuffled_coefficients = {
        well_id: coefficients[shuffle_order[(index + 1) % len(shuffle_order)]]
        for index, well_id in enumerate(shuffle_order)
    }
    actual_models = ("last_known_tvt", "noop", "duplicate", "oracle_target")
    model_well_metrics: dict[str, dict[str, WellMetric]] = {model: {} for model in actual_models}
    synthetic_sse = {"null": 0.0, "positive": 0.0, "shuffled": 0.0}
    synthetic_rows = 0
    direct_baseline_sse = 0.0
    duplicate_correlation = OnlineCorrelation()

    for profile in sorted(profiles, key=lambda item: item.well_id):
        path = train_dir / f"{profile.well_id}__horizontal_well.csv"
        accumulators = {model: ErrorAccumulator() for model in actual_models}
        hidden_index = 0
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, 2):
                if (row.get("TVT_input") or "").strip() not in MISSING_VALUES:
                    continue
                target = _float(row.get("TVT"), field="TVT", path=path, row_number=row_number)
                md = _float(row.get("MD"), field="MD", path=path, row_number=row_number)
                baseline = profile.last_visible_tvt
                predictions = {
                    "last_known_tvt": baseline,
                    "noop": baseline + 0.0,
                    "duplicate": (baseline + baseline) / 2.0,
                    "oracle_target": target,
                }
                errors: dict[str, float] = {}
                for model, prediction in predictions.items():
                    error = prediction - target
                    errors[model] = error
                    accumulators[model].add(error, float(hidden_index))
                direct_baseline_sse += errors["last_known_tvt"] ** 2
                duplicate_correlation.add(errors["last_known_tvt"], errors["duplicate"])

                delta_md = md - profile.last_visible_md
                synthetic_target = baseline + coefficients[profile.well_id] * delta_md
                synthetic_predictions = {
                    "null": baseline,
                    "positive": baseline + coefficients[profile.well_id] * delta_md,
                    "shuffled": baseline + shuffled_coefficients[profile.well_id] * delta_md,
                }
                for name, prediction in synthetic_predictions.items():
                    error = prediction - synthetic_target
                    synthetic_sse[name] += error * error
                synthetic_rows += 1
                hidden_index += 1
        if hidden_index != profile.hidden_rows:
            raise DataValidationError(
                f"{path}: second-pass hidden count {hidden_index}, expected {profile.hidden_rows}"
            )
        for model, accumulator in accumulators.items():
            model_well_metrics[model][profile.well_id] = accumulator.finalize(profile.well_id)

    actual_summary = {
        model: _summarize([metrics[well_id] for well_id in sorted(metrics)])
        for model, metrics in model_well_metrics.items()
    }
    synthetic_summary = {
        name: {"rows_scored": synthetic_rows, "sse": sse, "rmse": math.sqrt(sse / synthetic_rows)}
        for name, sse in synthetic_sse.items()
    }
    fold_metric_rows: list[dict[str, float | int | str]] = []
    for fold_map in fold_maps:
        fold_metric_rows.extend(_fold_metrics(model_well_metrics["last_known_tvt"], fold_map))

    hidden_loads = [item["hidden_rows"] for fold_map in fold_maps for item in fold_map["fold_summary"]]
    fingerprints = {fold_map["fingerprint"] for fold_map in fold_maps}
    baseline_rmse = float(actual_summary["last_known_tvt"]["rmse"])
    reference_pass = (
        expected_baseline_rmse is None
        or abs(baseline_rmse - expected_baseline_rmse) <= baseline_tolerance
    )
    controls = {
        "data_integrity": {
            "pass": (expected_wells is None or len(profiles) == expected_wells)
            and data_profile["known_rows"] > 0
            and data_profile["hidden_rows"] > 0
            and data_profile["max_visible_target_delta"] <= 1e-9,
            "expected_wells": expected_wells,
            "observed_wells": len(profiles),
            "max_visible_target_delta": data_profile["max_visible_target_delta"],
        },
        "reference_baseline": {
            "pass": reference_pass,
            "expected_rmse": expected_baseline_rmse,
            "observed_rmse": baseline_rmse,
            "tolerance": baseline_tolerance,
        },
        "pooled_rmse_consistency": {
            "pass": abs(direct_baseline_sse - float(actual_summary["last_known_tvt"]["sse"]))
            / max(direct_baseline_sse, float(actual_summary["last_known_tvt"]["sse"]), 1.0)
            <= 1e-12,
            "direct_sse": direct_baseline_sse,
            "well_aggregated_sse": actual_summary["last_known_tvt"]["sse"],
            "relative_difference": abs(
                direct_baseline_sse - float(actual_summary["last_known_tvt"]["sse"])
            ) / max(direct_baseline_sse, float(actual_summary["last_known_tvt"]["sse"]), 1.0),
            "relative_tolerance": 1e-12,
        },
        "noop": {
            "pass": abs(float(actual_summary["noop"]["rmse"]) - baseline_rmse) <= 1e-15,
            "rmse": actual_summary["noop"]["rmse"],
        },
        "duplicate": {
            "pass": abs(float(actual_summary["duplicate"]["rmse"]) - baseline_rmse) <= 1e-15
            and duplicate_correlation.value() is not None
            and abs(float(duplicate_correlation.value()) - 1.0) <= 1e-12,
            "rmse": actual_summary["duplicate"]["rmse"],
            "residual_correlation": duplicate_correlation.value(),
        },
        "known_signal_positive": {
            "pass": float(synthetic_summary["positive"]["rmse"]) <= 1e-12
            and float(synthetic_summary["null"]["rmse"]) > 1e-6,
            "synthetic_null_rmse": synthetic_summary["null"]["rmse"],
            "synthetic_positive_rmse": synthetic_summary["positive"]["rmse"],
        },
        "shuffled_evidence": {
            "pass": float(synthetic_summary["shuffled"]["rmse"])
            > float(synthetic_summary["positive"]["rmse"]) + 1e-6
            and float(synthetic_summary["shuffled"]["rmse"])
            >= 0.01 * float(synthetic_summary["null"]["rmse"]),
            "synthetic_shuffled_rmse": synthetic_summary["shuffled"]["rmse"],
            "synthetic_positive_rmse": synthetic_summary["positive"]["rmse"],
            "synthetic_null_rmse": synthetic_summary["null"]["rmse"],
        },
        "leakage_sentinel": {
            "pass": float(actual_summary["oracle_target"]["rmse"]) <= 1e-12,
            "rmse": actual_summary["oracle_target"]["rmse"],
            "uses_target": True,
            "eligible_for_modeling": False,
            "policy": "target-reading predictors are measured only as an explicit sentinel and are rejected",
        },
        "repeated_fold_maps": {
            "pass": len(fold_maps) == len(seeds)
            and len(fingerprints) == len(fold_maps)
            and max(hidden_loads) / min(hidden_loads) <= 1.02,
            "map_count": len(fold_maps),
            "unique_fingerprints": len(fingerprints),
            "max_to_min_hidden_rows": max(hidden_loads) / min(hidden_loads),
        },
    }
    overall_status = "pass" if all(bool(item["pass"]) for item in controls.values()) else "fail"
    summary = {
        "schema_version": 1,
        "experiment_id": "E001",
        "status": overall_status,
        "data": data_profile,
        "folds": [
            {
                "version": fold_map["version"],
                "seed": fold_map["seed"],
                "fingerprint": fold_map["fingerprint"],
                "fold_summary": fold_map["fold_summary"],
            }
            for fold_map in fold_maps
        ],
        "actual_target_metrics": actual_summary,
        "synthetic_control_metrics": synthetic_summary,
        "controls": controls,
        "predictor_policy": {
            "last_known_tvt": {"uses_target": False, "eligible_for_modeling": True},
            "noop": {"uses_target": False, "eligible_for_modeling": False},
            "duplicate": {"uses_target": False, "eligible_for_modeling": False},
            "oracle_target": {"uses_target": True, "eligible_for_modeling": False},
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    thresholds = _regime_thresholds(profiles)
    baseline_metrics = model_well_metrics["last_known_tvt"]
    profile_rows = []
    well_rows = []
    for profile in sorted(profiles, key=lambda item: item.well_id):
        folds = {f"fold_{fold_map['version']}": fold_map["assignments"][profile.well_id] for fold_map in fold_maps}
        profile_rows.append({**asdict(profile), **folds, "synthetic_coefficient": coefficients[profile.well_id]})
        metric = baseline_metrics[profile.well_id]
        well_rows.append(
            {
                "well_id": profile.well_id,
                "split": "cv",
                "rows_scored": metric.rows_scored,
                "rmse": metric.rmse,
                "mean_error": metric.mean_error,
                "sse": metric.sse,
                "regime": _regime(profile, thresholds),
                "uncertainty": "",
                "datum_sse": metric.datum_sse,
                "trend_sse": metric.trend_sse,
                "shape_sse": metric.shape_sse,
                "trend_per_row": metric.trend_per_row,
                **folds,
            }
        )
    fold_fields = ["map", "fold", "wells", "rows_scored", "sse", "rmse"]
    profile_fields = [
        "well_id", "path", "total_rows", "known_rows", "hidden_rows", "last_visible_md",
        "last_visible_z", "last_visible_tvt", "max_visible_target_delta", "synthetic_coefficient",
        *[f"fold_v{index}" for index in range(1, len(fold_maps) + 1)],
    ]
    well_fields = [
        "well_id", "split", "rows_scored", "rmse", "mean_error", "sse", "regime", "uncertainty",
        "datum_sse", "trend_sse", "shape_sse", "trend_per_row",
        *[f"fold_v{index}" for index in range(1, len(fold_maps) + 1)],
    ]
    _write_csv(output_dir / "fold_metrics.csv", fold_fields, fold_metric_rows)
    _write_csv(output_dir / "data_profile.csv", profile_fields, profile_rows)
    _write_csv(output_dir / "well_metrics.csv", well_fields, well_rows)
    control_rows = []
    for name, detail in sorted(controls.items()):
        control_rows.append(
            {
                "control": name,
                "status": "pass" if detail["pass"] else "fail",
                "detail_json": _canonical_json({key: value for key, value in detail.items() if key != "pass"}),
            }
        )
    _write_csv(output_dir / "control_metrics.csv", ["control", "status", "detail_json"], control_rows)

    artifact_files = [summary_path, output_dir / "fold_metrics.csv", output_dir / "data_profile.csv", output_dir / "well_metrics.csv", output_dir / "control_metrics.csv"]
    artifact_manifest = {
        "schema_version": 1,
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(artifact_files, key=lambda item: item.name)
        ],
        "fold_files": [
            {
                "path": f"{fold_map['version']}.json",
                "sha256": _sha256(fold_dir / f"{fold_map['version']}.json"),
                "bytes": (fold_dir / f"{fold_map['version']}.json").stat().st_size,
            }
            for fold_map in fold_maps
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return summary
