"""E004 deployment packaging and surface-free feature-family ablations."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .e004_inference import DeploymentDataError, extract_well_features, write_submission
from .harness import (
    DataValidationError,
    ErrorAccumulator,
    WellMetric,
    _canonical_json,
    _sha256,
    _summarize,
    _write_csv,
    _write_json,
    scan_profiles,
)
from .learnability import (
    MODEL_TARGETS,
    _load_fold_maps,
    _mean,
    _metric_from_sufficient,
    _pearson,
    _ridge_fit,
    _sign_accuracy,
    _spearman,
    _std,
)

DEPLOYABLE_FAMILIES = ("geometry", "prefix", "gr", "typewell", "spatial")


def feature_family(name: str) -> str:
    if name.startswith("typewell_"):
        return "typewell"
    if name.startswith(("visible_gr_", "hidden_gr_", "whole_gr_")):
        return "gr"
    if name.startswith(("backtest_", "visible_tvt_", "visible_u_")) or name in {"last_visible_tvt", "last_visible_u"}:
        return "prefix"
    if name in {"last_visible_x", "last_visible_y", "spatial_x_mid", "spatial_y_mid"}:
        return "spatial"
    return "geometry"


def _extract_training_records(
    train_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    profiles, data_profile = scan_profiles(train_dir)
    if data_profile["data_signature"] != str(config["data_signature"]):
        raise DataValidationError("E004 data signature differs from frozen E001 signature")
    if len(profiles) != int(config["expected_wells"]):
        raise DataValidationError("E004 train well count differs from pre-registration")
    profile_by_id = {profile.well_id: profile for profile in profiles}
    records: dict[str, dict[str, Any]] = {}
    feature_names: set[str] = set()
    for well_id in sorted(profile_by_id):
        profile = profile_by_id[well_id]
        horizontal = train_dir / f"{well_id}__horizontal_well.csv"
        typewell = train_dir / f"{well_id}__typewell.csv"
        try:
            extracted = extract_well_features(
                horizontal,
                typewell,
                visible_slope_windows=config["visible_slope_windows"],
                visible_backtest_fractions=config["visible_backtest_fractions"],
                require_truth=True,
            )
        except DeploymentDataError as exc:
            raise DataValidationError(str(exc)) from exc
        metadata = extracted["metadata"]
        if metadata["known_rows"] != profile.known_rows or metadata["hidden_rows"] != profile.hidden_rows:
            raise DataValidationError(f"{well_id}: E004 profile/extractor boundary mismatch")
        truth = list(extracted["truth"] or [])
        hidden_truth = truth[profile.known_rows:]
        last_visible = float(metadata["last_visible_tvt"])
        residuals = [float(value) - last_visible for value in hidden_truth]
        hidden_rows = len(residuals)
        centered = [index / max(1, hidden_rows - 1) - 0.5 if hidden_rows > 1 else 0.0 for index in range(hidden_rows)]
        sum_centered_sq = sum(value * value for value in centered)
        datum = sum(residuals) / hidden_rows
        trend = (
            sum(value * residual for value, residual in zip(centered, residuals)) / sum_centered_sq
            if sum_centered_sq > 0.0 else 0.0
        )
        sum_i = sum(float(index) for index in range(hidden_rows))
        sum_i_sq = sum(float(index * index) for index in range(hidden_rows))
        sum_r = sum(residuals)
        sum_r_sq = sum(value * value for value in residuals)
        sum_i_r = sum(float(index) * value for index, value in enumerate(residuals))
        sufficient = {
            "rows": float(hidden_rows),
            "sum_i": sum_i,
            "sum_i_sq": sum_i_sq,
            "sum_r": sum_r,
            "sum_r_sq": sum_r_sq,
            "sum_i_r": sum_i_r,
        }
        accumulator = ErrorAccumulator()
        for index, residual in enumerate(residuals):
            accumulator.add(-residual, float(index))
        baseline_metric = accumulator.finalize(well_id)
        features = dict(extracted["features"])
        visible_u_slope = float(features.get("visible_u_slope") or 0.0)
        hidden_md = []
        hidden_z = []
        with horizontal.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                if index >= profile.known_rows:
                    hidden_md.append(float(row["MD"]))
                    hidden_z.append(float(row["Z"]))
        hidden_u = [target + vertical for target, vertical in zip(hidden_truth, hidden_z)]
        if len(hidden_md) > 1:
            center_md = sum(hidden_md) / len(hidden_md)
            center_u = sum(hidden_u) / len(hidden_u)
            denominator = sum((value - center_md) ** 2 for value in hidden_md)
            hidden_u_slope = (
                sum((value - center_md) * (target - center_u) for value, target in zip(hidden_md, hidden_u)) / denominator
                if denominator > 0.0 else 0.0
            )
        else:
            hidden_u_slope = 0.0
        records[well_id] = {
            "features": features,
            "targets": {
                "datum_correction_ft": datum,
                "trend_correction_ft": trend,
                "u_slope_delta_ft_per_md": hidden_u_slope - visible_u_slope,
                "log1p_baseline_rmse": math.log1p(baseline_metric.rmse),
            },
            "sufficient": sufficient,
            "baseline_metric": baseline_metric,
            "spatial_x": float(features["spatial_x_mid"]),
            "spatial_y": float(features["spatial_y_mid"]),
        }
        feature_names.update(features)
    names = sorted(feature_names)
    unexpected = sorted({feature_family(name) for name in names} - set(DEPLOYABLE_FAMILIES))
    if unexpected:
        raise DataValidationError(f"E004 unknown feature families: {unexpected}")
    return records, names, data_profile


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


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _prepare_ridge(
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    maximum_features: int,
) -> dict[str, Any]:
    train_full: list[list[float]] = [[] for _ in train_ids]
    test_full: list[list[float]] = [[] for _ in test_ids]
    medians: list[float] = []
    means: list[float] = []
    scales: list[float] = []
    raw_minimums: list[float] = []
    raw_maximums: list[float] = []
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
        raw_minimums.append(min(train_column))
        raw_maximums.append(max(train_column))
        for row_index, value in enumerate(train_column):
            train_full[row_index].append((value - center) / scale)
        for row_index, value in enumerate(test_column):
            test_full[row_index].append((value - center) / scale)
    target_means: list[float] = []
    target_scales: list[float] = []
    train_y: list[list[float]] = [[] for _ in train_ids]
    for target in target_names:
        values = [float(records[well_id]["targets"][target]) for well_id in train_ids]
        center = _mean(values)
        scale = _std(values)
        if scale <= 1e-12:
            scale = 1.0
        target_means.append(center)
        target_scales.append(scale)
        for row_index, value in enumerate(values):
            train_y[row_index].append((value - center) / scale)
    target_columns = [[row[index] for row in train_y] for index in range(len(target_names))]
    scored: list[tuple[float, str, int]] = []
    for feature_index, name in enumerate(feature_names):
        column = [row[feature_index] for row in train_full]
        score = 0.0
        for target_column in target_columns:
            correlation = _pearson(column, target_column)
            score = max(score, abs(float(correlation or 0.0)))
        scored.append((score, name, feature_index))
    chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[: max(1, int(maximum_features))]
    indices = [item[2] for item in chosen]
    return {
        "train_x": [[row[index] for index in indices] for row in train_full],
        "test_x": [[row[index] for index in indices] for row in test_full],
        "train_y": train_y,
        "selected_features": [item[1] for item in chosen],
        "feature_medians": [medians[index] for index in indices],
        "feature_means": [means[index] for index in indices],
        "feature_scales": [scales[index] for index in indices],
        "feature_minimums": [raw_minimums[index] for index in indices],
        "feature_maximums": [raw_maximums[index] for index in indices],
        "target_means": target_means,
        "target_scales": target_scales,
    }


def _linear_predict(x_rows: Sequence[Sequence[float]], coefficients: Sequence[Sequence[float]]) -> list[list[float]]:
    outputs = len(coefficients[0]) if coefficients else 0
    return [
        [sum(value * float(coefficients[index][output]) for index, value in enumerate(row)) for output in range(outputs)]
        for row in x_rows
    ]


def _unscale(values: Sequence[Sequence[float]], means: Sequence[float], scales: Sequence[float]) -> list[list[float]]:
    return [[float(means[index]) + float(scales[index]) * value for index, value in enumerate(row)] for row in values]


def _fit_predict(
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    prepared = _prepare_ridge(
        records,
        train_ids,
        test_ids,
        feature_names,
        MODEL_TARGETS,
        int(config["feature_screening"]["maximum_features"]),
    )
    coefficients = _ridge_fit(prepared["train_x"], prepared["train_y"], float(config["ridge"]["alpha"]))
    scaled = _linear_predict(prepared["test_x"], coefficients)
    predictions = _unscale(scaled, prepared["target_means"], prepared["target_scales"])
    return {well_id: predictions[index] for index, well_id in enumerate(test_ids)}, {**prepared, "ridge_coefficients": coefficients}


def _crossfit(
    records: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, int],
    n_folds: int,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[float]], Counter[str], list[dict[str, Any]]]:
    predictions: dict[str, list[float]] = {}
    frequency: Counter[str] = Counter()
    drift_rows: list[dict[str, Any]] = []
    all_ids = sorted(records)
    for fold in range(n_folds):
        test_ids = [well_id for well_id in all_ids if int(assignments[well_id]) == fold]
        train_ids = [well_id for well_id in all_ids if int(assignments[well_id]) != fold]
        fold_predictions, fitted = _fit_predict(records, train_ids, test_ids, feature_names, config)
        predictions.update(fold_predictions)
        frequency.update(fitted["selected_features"])
        outside = 0
        cells = 0
        maximum_abs_z = 0.0
        for well_id, row in zip(test_ids, fitted["test_x"]):
            for index, value in enumerate(row):
                cells += 1
                maximum_abs_z = max(maximum_abs_z, abs(value))
                raw = value * fitted["feature_scales"][index] + fitted["feature_means"][index]
                if raw < fitted["feature_minimums"][index] or raw > fitted["feature_maximums"][index]:
                    outside += 1
        drift_rows.append({
            "fold": fold,
            "test_wells": len(test_ids),
            "feature_cells": cells,
            "outside_train_range_fraction": outside / max(1, cells),
            "maximum_absolute_standardized_value": maximum_abs_z,
        })
    if set(predictions) != set(records):
        raise DataValidationError("E004 crossfit did not predict every well exactly once")
    return predictions, frequency, drift_rows


def _score(
    records: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Sequence[float]],
) -> tuple[dict[str, Any], dict[str, WellMetric]]:
    metrics: dict[str, WellMetric] = {}
    for well_id in sorted(records):
        values = predictions[well_id]
        metrics[well_id] = _metric_from_sufficient(
            well_id,
            records[well_id]["sufficient"],
            float(values[0]),
            float(values[1]),
        )
    return _summarize([metrics[well_id] for well_id in sorted(metrics)]), metrics


def _target_metrics(records: Mapping[str, Mapping[str, Any]], predictions: Mapping[str, Sequence[float]], materiality: float) -> list[dict[str, Any]]:
    ids = sorted(records)
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(MODEL_TARGETS[:3]):
        actual = [float(records[well_id]["targets"][target]) for well_id in ids]
        predicted = [float(predictions[well_id][index]) for well_id in ids]
        sign, count = _sign_accuracy(actual, predicted, materiality if index < 2 else 0.0)
        rows.append({
            "target": target,
            "pearson": _pearson(actual, predicted),
            "spearman": _spearman(actual, predicted),
            "mae": _mean([abs(a - p) for a, p in zip(actual, predicted)]),
            "sign_accuracy": sign,
            "sign_rows": count,
        })
    return rows


def _average_predictions(map_predictions: Mapping[str, Mapping[str, Sequence[float]]]) -> dict[str, list[float]]:
    versions = sorted(map_predictions)
    ids = sorted(next(iter(map_predictions.values())))
    return {
        well_id: [
            sum(float(map_predictions[version][well_id][index]) for version in versions) / len(versions)
            for index in range(len(MODEL_TARGETS))
        ]
        for well_id in ids
    }


def _spatial_assignments(records: Mapping[str, Mapping[str, Any]], bins: int) -> dict[str, int]:
    ordered = sorted(records, key=lambda well_id: (records[well_id]["spatial_x"], records[well_id]["spatial_y"], well_id))
    return {well_id: min(bins - 1, int(index * bins / len(ordered))) for index, well_id in enumerate(ordered)}


def _load_test_records(test_dir: Path, config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    schemas: set[tuple[str, ...]] = set()
    for horizontal in sorted(test_dir.glob("*__horizontal_well.csv")):
        well_id = horizontal.name.split("__", 1)[0]
        typewell = test_dir / f"{well_id}__typewell.csv"
        if not typewell.exists():
            raise DataValidationError(f"missing test typewell for {well_id}")
        try:
            extracted = extract_well_features(
                horizontal,
                typewell,
                visible_slope_windows=config["visible_slope_windows"],
                visible_backtest_fractions=config["visible_backtest_fractions"],
                require_truth=False,
            )
        except DeploymentDataError as exc:
            raise DataValidationError(str(exc)) from exc
        with horizontal.open(newline="", encoding="utf-8-sig") as handle:
            schemas.add(tuple(csv.DictReader(handle).fieldnames or []))
        records[well_id] = {"features": extracted["features"], "metadata": extracted["metadata"]}
    if not records:
        raise DataValidationError("E004 test directory has no wells")
    return records, ["|".join(schema) for schema in sorted(schemas)]


def _fit_full_model(
    records: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    selected_candidate: str,
    families: Sequence[str],
    code_sha: str,
) -> dict[str, Any]:
    ids = sorted(records)
    prepared = _prepare_ridge(
        records,
        ids,
        [],
        feature_names,
        MODEL_TARGETS,
        int(config["feature_screening"]["maximum_features"]),
    )
    coefficients = _ridge_fit(prepared["train_x"], prepared["train_y"], float(config["ridge"]["alpha"]))
    return {
        "schema_version": 1,
        "experiment_id": "E004",
        "model_name": selected_candidate,
        "code_sha": code_sha,
        "data_signature": config["data_signature"],
        "training_wells": len(ids),
        "feature_families": list(families),
        "selected_features": prepared["selected_features"],
        "feature_medians": prepared["feature_medians"],
        "feature_means": prepared["feature_means"],
        "feature_scales": prepared["feature_scales"],
        "feature_minimums": prepared["feature_minimums"],
        "feature_maximums": prepared["feature_maximums"],
        "target_names": list(MODEL_TARGETS),
        "target_means": prepared["target_means"],
        "target_scales": prepared["target_scales"],
        "ridge_alpha": config["ridge"]["alpha"],
        "ridge_coefficients": coefficients,
        "visible_slope_windows": list(config["visible_slope_windows"]),
        "visible_backtest_fractions": list(config["visible_backtest_fractions"]),
        "surfaces_required": False,
        "external_artifacts_required": False,
        "internet_required": False,
    }


def _test_drift_rows(model: Mapping[str, Any], test_records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(model["selected_features"]):
        values: list[float] = []
        missing = 0
        for record in test_records.values():
            raw = record["features"].get(name)
            if raw is None or not math.isfinite(float(raw)):
                missing += 1
                value = float(model["feature_medians"][index])
            else:
                value = float(raw)
            values.append(value)
        center = float(model["feature_means"][index])
        scale = float(model["feature_scales"][index]) or 1.0
        minimum = float(model["feature_minimums"][index])
        maximum = float(model["feature_maximums"][index])
        rows.append({
            "feature": name,
            "family": feature_family(name),
            "test_missing_fraction": missing / max(1, len(test_records)),
            "test_minimum": min(values),
            "test_maximum": max(values),
            "outside_train_range_count": sum(value < minimum or value > maximum for value in values),
            "maximum_absolute_standardized_value": max(abs((value - center) / scale) for value in values),
        })
    return rows


def run_e004(
    *,
    root: Path,
    train_dir: Path,
    test_dir: Path,
    sample_submission: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: Mapping[str, Any],
    code_sha: str,
) -> dict[str, Any]:
    """Execute frozen E004 ablations and write deterministic deployment artifacts."""
    started = time.perf_counter()
    root = root.resolve()
    train_dir = train_dir.resolve()
    test_dir = test_dir.resolve()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    records, all_feature_names, data_profile = _extract_training_records(train_dir, config)
    fold_maps = _load_fold_maps(root, config["fold_files"], str(config["data_signature"]))
    family_features = {
        family: [name for name in all_feature_names if feature_family(name) == family]
        for family in DEPLOYABLE_FAMILIES
    }
    if any(not names for names in family_features.values()):
        raise DataValidationError(f"E004 empty feature family: {[key for key, value in family_features.items() if not value]}")
    baseline_summary = _summarize([records[well_id]["baseline_metric"] for well_id in sorted(records)])
    map_predictions: dict[str, dict[str, dict[str, list[float]]]] = {}
    map_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    candidate_features: dict[str, list[str]] = {}
    candidate_frequency: dict[str, Counter[str]] = {}
    for candidate, families in config["ablation_candidates"].items():
        unknown = sorted(set(families) - set(DEPLOYABLE_FAMILIES))
        if unknown:
            raise DataValidationError(f"{candidate}: unknown feature families {unknown}")
        features = sorted(name for family in families for name in family_features[family])
        candidate_features[candidate] = features
        map_predictions[candidate] = {}
        frequency: Counter[str] = Counter()
        for fold_map in fold_maps:
            version = str(fold_map["version"])
            predictions, fold_frequency, fold_drift = _crossfit(
                records,
                fold_map["assignments"],
                int(fold_map["n_folds"]),
                features,
                config,
            )
            map_predictions[candidate][version] = predictions
            frequency.update(fold_frequency)
            summary, _ = _score(records, predictions)
            map_rows.append({
                "candidate": candidate,
                "map": version,
                "families": "+".join(families),
                "gain_vs_baseline": float(baseline_summary["rmse"]) - float(summary["rmse"]),
                **summary,
            })
            for row in fold_drift:
                drift_rows.append({"scope": "repeated_fold", "candidate": candidate, "map": version, **row})
        candidate_frequency[candidate] = frequency
        averaged = _average_predictions(map_predictions[candidate])
        summary, _ = _score(records, averaged)
        metrics = _target_metrics(records, averaged, float(config["sign_materiality_ft"]))
        target_rows.extend({"candidate": candidate, **row} for row in metrics)
        gains = [
            float(row["gain_vs_baseline"])
            for row in map_rows
            if row["candidate"] == candidate
        ]
        candidate_rows.append({
            "candidate": candidate,
            "families": "+".join(families),
            "feature_count": len(features),
            "selected_feature_count": min(len(features), int(config["feature_screening"]["maximum_features"])),
            "map_wins": sum(gain >= float(config["promotion"]["minimum_rmse_gain"]) for gain in gains),
            "mean_map_rmse": _mean([
                float(row["rmse"]) for row in map_rows if row["candidate"] == candidate
            ]),
            "gain_vs_baseline": float(baseline_summary["rmse"]) - float(summary["rmse"]),
            **summary,
        })
        spatial_predictions, _, spatial_drift = _crossfit(
            records,
            _spatial_assignments(records, int(config["spatial_stress_bins"])),
            int(config["spatial_stress_bins"]),
            features,
            config,
        )
        spatial_summary, _ = _score(records, spatial_predictions)
        spatial_rows.append({
            "candidate": candidate,
            "families": "+".join(families),
            "gain_vs_baseline": float(baseline_summary["rmse"]) - float(spatial_summary["rmse"]),
            **spatial_summary,
        })
        for row in spatial_drift:
            drift_rows.append({"scope": "spatial_fold", "candidate": candidate, "map": "spatial", **row})
    promotion = config["promotion"]
    eligible: list[str] = []
    for row in candidate_rows:
        spatial = next(item for item in spatial_rows if item["candidate"] == row["candidate"])
        datum = next(item for item in target_rows if item["candidate"] == row["candidate"] and item["target"] == "datum_correction_ft")
        if (
            float(row["gain_vs_baseline"]) >= float(promotion["minimum_rmse_gain"])
            and int(row["map_wins"]) >= int(promotion["minimum_map_wins"])
            and float(row["p90_well_rmse"]) - float(baseline_summary["p90_well_rmse"]) <= float(promotion["maximum_p90_deterioration"])
            and float(row["worst_5pct_sse_share"]) - float(baseline_summary["worst_5pct_sse_share"]) <= float(promotion["maximum_worst5_increase"])
            and float(spatial["gain_vs_baseline"]) > 0.0
            and float(datum["pearson"] or -1.0) >= float(promotion["minimum_datum_correlation"])
            and float(datum["sign_accuracy"] or 0.0) >= float(promotion["minimum_datum_sign_accuracy"])
        ):
            eligible.append(str(row["candidate"]))
    selected_candidate = min(
        eligible or [str(row["candidate"]) for row in candidate_rows],
        key=lambda candidate: (
            next(float(row["mean_map_rmse"]) for row in candidate_rows if row["candidate"] == candidate),
            len(config["ablation_candidates"][candidate]),
            candidate,
        ),
    )
    selected_families = list(config["ablation_candidates"][selected_candidate])
    selected_features = candidate_features[selected_candidate]
    selected_row = next(row for row in candidate_rows if row["candidate"] == selected_candidate)
    selected_spatial = next(row for row in spatial_rows if row["candidate"] == selected_candidate)
    full_model = _fit_full_model(records, selected_features, config, selected_candidate, selected_families, code_sha)
    test_records, test_schemas = _load_test_records(test_dir, config)
    test_drift = _test_drift_rows(full_model, test_records)
    maximum_test_z = max(float(row["maximum_absolute_standardized_value"]) for row in test_drift)
    test_outside = sum(int(row["outside_train_range_count"]) for row in test_drift)
    model_controls = {
        "surfaces_excluded": not any("ancc" in name.lower() or "astn" in name.lower() or "egfd" in name.lower() or "buda" in name.lower() for name in full_model["selected_features"]),
        "required_test_schema": all("ANCC" not in schema and "TVT" not in schema for schema in test_schemas),
        "finite_model": all(
            math.isfinite(float(value))
            for key in ("feature_medians", "feature_means", "feature_scales", "target_means", "target_scales")
            for value in full_model[key]
        ) and all(math.isfinite(float(value)) for row in full_model["ridge_coefficients"] for value in row),
        "selected_candidate_eligible": selected_candidate in eligible,
        "visible_test_drift_bounded": maximum_test_z <= float(config["deployment"]["maximum_visible_test_abs_z"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    _write_json(model_path, full_model)
    submission_path = artifact_dir / "submission.csv"
    submission = write_submission(full_model, test_dir, sample_submission, submission_path)
    prediction_rows = list(submission.pop("well_predictions"))
    _write_csv(output_dir / "visible_test_predictions.csv", list(prediction_rows[0]), prediction_rows)
    feature_rows = [
        {
            "feature": name,
            "family": feature_family(name),
            "selection_count": candidate_frequency[selected_candidate][name],
            "selected_in_full_model": name in full_model["selected_features"],
        }
        for name in selected_features
    ]
    _write_csv(output_dir / "selected_features.csv", list(feature_rows[0]), feature_rows)
    _write_csv(output_dir / "candidate_metrics.csv", list(candidate_rows[0]), candidate_rows)
    _write_csv(output_dir / "map_metrics.csv", list(map_rows[0]), map_rows)
    _write_csv(output_dir / "target_metrics.csv", list(target_rows[0]), target_rows)
    _write_csv(output_dir / "spatial_metrics.csv", list(spatial_rows[0]), spatial_rows)
    _write_csv(output_dir / "fold_drift_metrics.csv", list(drift_rows[0]), drift_rows)
    _write_csv(output_dir / "visible_test_drift.csv", list(test_drift[0]), test_drift)
    controls = {
        "data_integrity": {"pass": data_profile["data_signature"] == config["data_signature"] and len(records) == int(config["expected_wells"])},
        "fold_integrity": {"pass": len(fold_maps) == 5 and len({item["fingerprint"] for item in fold_maps}) == 5},
        "surface_unavailability_detected": {"pass": all("ANCC" not in schema for schema in test_schemas), "schemas": test_schemas},
        "surface_features_excluded": {"pass": model_controls["surfaces_excluded"]},
        "finite_model": {"pass": model_controls["finite_model"]},
        "selected_candidate_eligible": {"pass": model_controls["selected_candidate_eligible"], "eligible_candidates": eligible},
        "visible_test_drift": {"pass": model_controls["visible_test_drift_bounded"], "maximum_abs_z": maximum_test_z, "outside_range_cells": test_outside, "authoring_examples_only": True},
        "exact_submission_contract": {"pass": submission["rows"] > 0 and submission["wells"] == len(test_records), **submission},
        "runtime": {"pass": time.perf_counter() - started <= 60.0 * float(config["promotion"]["maximum_runtime_minutes"]), "budget_minutes": config["promotion"]["maximum_runtime_minutes"]},
        "remote_kaggle_mcp_parity": {"pass": False, "status": "blocked_unavailable_tool", "required_for_deployment_ready": True},
    }
    local_model_ready = all(detail["pass"] for name, detail in controls.items() if name != "remote_kaggle_mcp_parity")
    deployment_ready = local_model_ready and controls["remote_kaggle_mcp_parity"]["pass"]
    status = "blocked" if local_model_ready and not deployment_ready else ("promoted" if deployment_ready else "rejected")
    summary = {
        "schema_version": 1,
        "experiment_id": "E004",
        "status": status,
        "data": data_profile,
        "surface_contract": {
            "train_has_surfaces": True,
            "test_has_surfaces": False,
            "surface_features_deployable": False,
        },
        "baseline_metrics": baseline_summary,
        "selected_candidate": selected_candidate,
        "selected_families": selected_families,
        "selected_candidate_metrics": selected_row,
        "selected_spatial_metrics": selected_spatial,
        "eligible_candidates": eligible,
        "candidate_metrics": {row["candidate"]: row for row in candidate_rows},
        "controls": controls,
        "deployment": {
            "local_model_ready": local_model_ready,
            "local_notebook_parity": False,
            "remote_kaggle_mcp_parity": False,
            "deployment_ready": deployment_ready,
            "model_path": "experiments/E004/results/model.json",
            "notebook_path": "notebooks/e004_deployment.ipynb",
            "submission_path": "artifacts/E004/submission.csv",
            "submission_sha256": submission["sha256"],
        },
        "visible_test_warning": "The three local test wells are train-derived authoring examples and are not hidden-test generalization evidence.",
    }
    _write_json(output_dir / "summary.json", summary)
    control_rows = [
        {"control": name, "status": "pass" if detail["pass"] else "blocked" if name == "remote_kaggle_mcp_parity" else "fail", "detail_json": _canonical_json({key: value for key, value in detail.items() if key != "pass"})}
        for name, detail in sorted(controls.items())
    ]
    _write_csv(output_dir / "control_metrics.csv", list(control_rows[0]), control_rows)
    result_files = [path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    artifact_manifest = {
        "schema_version": 1,
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(result_files, key=lambda item: item.name)
        ],
        "runtime_artifacts": [
            {"path": "artifacts/E004/submission.csv", "sha256": _sha256(submission_path), "bytes": submission_path.stat().st_size}
        ],
        "fold_files": [
            {"path": relative, "sha256": _sha256(root / relative), "bytes": (root / relative).stat().st_size}
            for relative in config["fold_files"]
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return {**summary, "runtime_seconds": time.perf_counter() - started}
