"""E012: nested confirmation of regime-specialized E011 coefficient experts."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .coefficient_learning import (
    CoefficientWell,
    Context,
    _build_contexts,
    _load_folds,
    _read_records,
    _ridge_selection,
    bound_coefficients,
)
from .harness import DataValidationError, WellMetric, _quantile, _summarize, _write_csv, _write_json

E012_RESULT_FILES = (
    "candidate_metrics.csv",
    "context_metrics.csv",
    "stress_metrics.csv",
    "support_metrics.csv",
    "special_slice_metrics.csv",
    "controls.csv",
    "summary.json",
)


@dataclass(frozen=True)
class BranchSpec:
    name: str
    family: str
    shrinkage: float
    eligible: bool
    negative_control: bool = False
    alpha: float = 10.0
    min_samples_leaf: int = 20
    max_depth: int = 2
    neighbors: int = 25
    weights: str = "distance"


def branch_specs(config: Mapping[str, Any]) -> tuple[BranchSpec, ...]:
    output = []
    for row in config.get("branches", []):
        output.append(
            BranchSpec(
                name=str(row["name"]),
                family=str(row["family"]),
                shrinkage=float(row["shrinkage"]),
                eligible=bool(row.get("eligible", True)),
                negative_control=bool(row.get("negative_control", False)),
                alpha=float(row.get("alpha", 10.0)),
                min_samples_leaf=int(row.get("min_samples_leaf", 20)),
                max_depth=int(row.get("max_depth", 2)),
                neighbors=int(row.get("neighbors", 25)),
                weights=str(row.get("weights", "distance")),
            )
        )
    names = [item.name for item in output]
    if len(output) != 9 or len(names) != len(set(names)):
        raise DataValidationError("E012 branch contract differs")
    if sum(item.negative_control for item in output) != 1:
        raise DataValidationError("E012 requires one negative routing control")
    return tuple(output)


def validate_e012_config(config: Mapping[str, Any]) -> None:
    if str(config.get("experiment_id")) != "E012" or str(config.get("hypothesis_id")) != "H015":
        raise DataValidationError("E012 identity differs")
    if str(config.get("representation")) != "spline4" or abs(float(config.get("e011_shrinkage", 0.0)) - 0.75) > 1e-12:
        raise DataValidationError("E012 parent placement differs")
    if int(config.get("minimum_regime_support", 0)) < 50:
        raise DataValidationError("E012 support floor is too small")
    if bool(config.get("submission_authorized")):
        raise DataValidationError("E012 submission must remain unauthorized")
    branch_specs(config)


def _feature_arrays(
    records: Mapping[str, CoefficientWell],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not train_ids or not test_ids or len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise DataValidationError("E012 invalid train/test membership")
    if set(train_ids) & set(test_ids):
        raise DataValidationError("E012 train/test overlap")
    def raw(ids: Sequence[str]) -> np.ndarray:
        matrix = np.empty((len(ids), len(feature_names)), dtype=np.float64)
        for row, well_id in enumerate(ids):
            if well_id not in records:
                raise DataValidationError("E012 unknown well ID")
            for column, name in enumerate(feature_names):
                value = records[well_id].features.get(name)
                matrix[row, column] = np.nan if value is None else float(value)
        return matrix
    train = raw(train_ids)
    test = raw(test_ids)
    medians = np.zeros(train.shape[1], dtype=np.float64)
    for column in range(train.shape[1]):
        finite = train[np.isfinite(train[:, column]), column]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    train_z = (train - means) / scales
    test_z = (test - means) / scales
    if not np.all(np.isfinite(train_z)) or not np.all(np.isfinite(test_z)):
        raise DataValidationError("E012 feature preprocessing emitted non-finite values")
    return train, test, train_z, test_z


def quantile_regimes(train_values: np.ndarray, test_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_values, dtype=np.float64)
    test = np.asarray(test_values, dtype=np.float64)
    if train.ndim != 1 or test.ndim != 1 or not np.all(np.isfinite(train)) or not np.all(np.isfinite(test)):
        raise DataValidationError("E012 quantile regime input is invalid")
    cuts = np.quantile(train, [1.0 / 3.0, 2.0 / 3.0])
    return np.digitize(train, cuts, right=False).astype(np.int32), np.digitize(test, cuts, right=False).astype(np.int32)


def _regime_mean_delta(
    residuals: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    residual = np.asarray(residuals, dtype=np.float64)
    train = np.asarray(train_labels)
    test = np.asarray(test_labels)
    if residual.ndim != 2 or residual.shape[0] != train.size or test.ndim != 1 or not np.all(np.isfinite(residual)):
        raise DataValidationError("E012 regime mean input is invalid")
    output = np.zeros((test.size, residual.shape[1]), dtype=np.float64)
    fallback = np.ones(test.size, dtype=bool)
    supports = []
    for label in sorted(set(train.tolist())):
        train_mask = train == label
        test_mask = test == label
        count = int(np.sum(train_mask))
        supports.append(count)
        if count < minimum_support or not np.any(test_mask):
            continue
        output[test_mask] = residual[train_mask].mean(axis=0)
        fallback[test_mask] = False
    minimum = min(supports) if supports else 0
    return output, fallback, minimum, len(supports)


def _base_coefficients(
    context: Context,
    records: Mapping[str, CoefficientWell],
    feature_names: Sequence[str],
    e011_config: Mapping[str, Any],
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selection = _ridge_selection(context, records, feature_names, "spline4", e011_config, "ridge_equal")
    train = np.vstack([shrinkage * selection.inner_predictions[well_id] for well_id in context.train_ids])
    test = np.vstack([shrinkage * selection.outer_predictions[well_id] for well_id in context.test_ids])
    target = np.vstack([records[well_id].statistics["spline4"].target_coefficients for well_id in context.train_ids])
    if train.shape != target.shape or test.shape != (len(context.test_ids), 4):
        raise DataValidationError("E012 E011 coefficient reconstruction differs")
    return train, test, target - train


def fit_branch(
    spec: BranchSpec,
    *,
    context: Context,
    records: Mapping[str, CoefficientWell],
    feature_names: Sequence[str],
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    train_z: np.ndarray,
    test_z: np.ndarray,
    train_residual: np.ndarray,
    test_base: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    minimum_support = int(config["minimum_regime_support"])
    family = spec.family
    residual_matrix = np.asarray(train_residual, dtype=np.float64)
    base_matrix = np.asarray(test_base, dtype=np.float64)
    if residual_matrix.shape != (len(context.train_ids), 4) or base_matrix.shape != (len(context.test_ids), 4):
        raise DataValidationError("E012 branch coefficient shapes differ")
    if not np.all(np.isfinite(residual_matrix)) or not np.all(np.isfinite(base_matrix)):
        raise DataValidationError("E012 branch coefficients contain non-finite values")
    train_residual = residual_matrix
    test_base = base_matrix
    fallback = np.zeros(len(context.test_ids), dtype=bool)
    minimum = len(context.train_ids)
    regimes = 1
    if family in {"length3_mean", "pseudo3_mean", "shuffled_length_mean", "partial_pseudo"}:
        feature_key = "length" if family in {"length3_mean", "shuffled_length_mean"} else "pseudo"
        feature = str(config["features"][feature_key])
        if feature not in feature_names:
            raise DataValidationError(f"E012 missing regime feature {feature}")
        column = feature_names.index(feature)
        train_labels, test_labels = quantile_regimes(train_raw[:, column], test_raw[:, column])
        if family == "shuffled_length_mean":
            order = sorted(
                range(len(context.train_ids)),
                key=lambda index: (
                    hashlib.sha256(f"E012-shuffle|{context.key}|{context.train_ids[index]}".encode()).hexdigest(),
                    context.train_ids[index],
                ),
            )
            train_labels = train_labels[np.asarray(order, dtype=int)]
        if family == "partial_pseudo":
            global_model = Ridge(alpha=spec.alpha).fit(train_z, train_residual)
            global_train = np.asarray(global_model.predict(train_z), dtype=np.float64)
            global_test = np.asarray(global_model.predict(test_z), dtype=np.float64)
            remainder, fallback, minimum, regimes = _regime_mean_delta(
                train_residual - global_train, train_labels, test_labels, minimum_support
            )
            delta = global_test + remainder
        else:
            delta, fallback, minimum, regimes = _regime_mean_delta(
                train_residual, train_labels, test_labels, minimum_support
            )
    elif family == "kmeans2_mean":
        model = KMeans(n_clusters=2, random_state=int(config["models"]["random_state"]), n_init=20).fit(train_z)
        delta, fallback, minimum, regimes = _regime_mean_delta(
            train_residual, model.labels_.astype(np.int32), model.predict(test_z).astype(np.int32), minimum_support
        )
    elif family == "latent3":
        distinct = np.unique(np.round(train_residual, 10), axis=0)
        if distinct.shape[0] < 3:
            delta = np.zeros_like(test_base)
            fallback = np.ones(len(context.test_ids), dtype=bool)
            minimum = 0
            regimes = int(distinct.shape[0])
        else:
            cluster = KMeans(n_clusters=3, random_state=int(config["models"]["random_state"]), n_init=20).fit(train_residual)
            classifier = ExtraTreesClassifier(
                n_estimators=int(config["models"]["latent_classifier_estimators"]),
                max_depth=int(config["models"]["latent_classifier_depth"]),
                min_samples_leaf=int(config["models"]["latent_classifier_leaf"]),
                max_features=0.5,
                random_state=int(config["models"]["random_state"]),
                n_jobs=1,
            ).fit(train_z, cluster.labels_)
            delta, fallback, minimum, regimes = _regime_mean_delta(
                train_residual,
                cluster.labels_.astype(np.int32),
                classifier.predict(test_z).astype(np.int32),
                minimum_support,
            )
    elif family == "global_ridge":
        delta = np.asarray(Ridge(alpha=spec.alpha).fit(train_z, train_residual).predict(test_z), dtype=np.float64)
    elif family == "extra_trees":
        delta = np.asarray(
            ExtraTreesRegressor(
                n_estimators=int(config["models"]["extra_trees_estimators"]),
                max_depth=spec.max_depth,
                min_samples_leaf=spec.min_samples_leaf,
                max_features=0.5,
                random_state=int(config["models"]["random_state"]),
                n_jobs=1,
            ).fit(train_z, train_residual).predict(test_z),
            dtype=np.float64,
        )
    elif family == "knn":
        if len(context.train_ids) < spec.neighbors:
            delta = np.zeros_like(test_base)
            fallback = np.ones(len(context.test_ids), dtype=bool)
            minimum = len(context.train_ids)
        else:
            delta = np.asarray(
                KNeighborsRegressor(n_neighbors=spec.neighbors, weights=spec.weights).fit(train_z, train_residual).predict(test_z),
                dtype=np.float64,
            )
    else:
        raise DataValidationError(f"E012 unknown branch family {family}")
    if delta.shape != test_base.shape or not np.all(np.isfinite(delta)):
        raise DataValidationError(f"E012 {spec.name} emitted invalid residual coefficients")
    coefficients: dict[str, np.ndarray] = {}
    fallback_delta = 0.0
    maximum_coefficient = 0.0
    for index, well_id in enumerate(context.test_ids):
        raw = test_base[index] + spec.shrinkage * delta[index]
        values = bound_coefficients(raw, "spline4", config["e011_config"])
        coefficients[well_id] = values
        maximum_coefficient = max(maximum_coefficient, float(np.max(np.abs(values))))
        if fallback[index]:
            fallback_delta = max(fallback_delta, float(np.max(np.abs(values - bound_coefficients(test_base[index], "spline4", config["e011_config"])))))
    return coefficients, {
        "minimum_train_support": int(minimum),
        "regimes": int(regimes),
        "fallback_wells": int(np.sum(fallback)),
        "fallback_rate": float(np.mean(fallback)) if fallback.size else 0.0,
        "maximum_fallback_delta": fallback_delta,
        "maximum_absolute_coefficient": maximum_coefficient,
    }


def _metrics(records: Mapping[str, CoefficientWell], coefficients: Mapping[str, np.ndarray]) -> list[WellMetric]:
    return [records[well_id].statistics["spline4"].metric(well_id, coefficients[well_id]) for well_id in sorted(coefficients)]


def run_e012(
    *,
    root: Path,
    train_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    e011_config: Mapping[str, Any],
    code_sha: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_e012_config(config)
    local_config = dict(config)
    local_config["e011_config"] = e011_config
    records, feature_names, audit = _read_records(root, train_dir, e011_config)
    if len(records) != int(config["expected_wells"]) or int(audit["rows"]) != int(config["expected_hidden_rows"]):
        raise DataValidationError("E012 parent data dimensions differ")
    folds = _load_folds(root, e011_config, sorted(records))
    contexts, membership, spatial_assignments, typewell_assignments = _build_contexts(records, folds, e011_config)
    repeated = [context for context in contexts if context.scope == "repeated"]
    stress = [context for context in contexts if context.scope != "repeated"]
    specs = branch_specs(config)
    eligible = [spec for spec in specs if spec.eligible]
    well_ids = tuple(sorted(records))
    well_index = {well_id: index for index, well_id in enumerate(well_ids)}
    all_names = ("e011", *(spec.name for spec in specs))
    sums = {name: np.zeros((len(well_ids), 4), dtype=np.float64) for name in all_names}
    counts = {name: np.zeros(len(well_ids), dtype=np.int16) for name in all_names}
    context_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    context_metrics: dict[str, dict[str, Mapping[str, Any]]] = {}
    deterministic_delta = 0.0
    maximum_fallback_delta = 0.0
    maximum_coefficient = 0.0

    for context_index, context in enumerate((*repeated, *stress)):
        train_base, test_base, train_residual = _base_coefficients(
            context, records, feature_names, e011_config, float(config["e011_shrinkage"])
        )
        train_raw, test_raw, train_z, test_z = _feature_arrays(
            records, context.train_ids, context.test_ids, feature_names
        )
        base = {
            well_id: bound_coefficients(test_base[index], "spline4", e011_config)
            for index, well_id in enumerate(context.test_ids)
        }
        predictions: dict[str, dict[str, np.ndarray]] = {"e011": base}
        details: dict[str, Mapping[str, Any]] = {}
        for spec in specs:
            branch, detail = fit_branch(
                spec,
                context=context,
                records=records,
                feature_names=feature_names,
                train_raw=train_raw,
                test_raw=test_raw,
                train_z=train_z,
                test_z=test_z,
                train_residual=train_residual,
                test_base=test_base,
                config=local_config,
            )
            predictions[spec.name] = branch
            details[spec.name] = detail
            maximum_fallback_delta = max(maximum_fallback_delta, float(detail["maximum_fallback_delta"]))
            maximum_coefficient = max(maximum_coefficient, float(detail["maximum_absolute_coefficient"]))
            support_rows.append({"context": context.key, "scope": context.scope, "branch": spec.name, **detail})
        if context_index == 0:
            for spec in specs:
                duplicate, _detail = fit_branch(
                    spec,
                    context=context,
                    records=records,
                    feature_names=feature_names,
                    train_raw=train_raw,
                    test_raw=test_raw,
                    train_z=train_z,
                    test_z=test_z,
                    train_residual=train_residual,
                    test_base=test_base,
                    config=local_config,
                )
                deterministic_delta = max(
                    deterministic_delta,
                    max(float(np.max(np.abs(predictions[spec.name][well_id] - duplicate[well_id]))) for well_id in context.test_ids),
                )
        by_candidate = {}
        for candidate, coefficients in predictions.items():
            summary = _summarize(_metrics(records, coefficients))
            by_candidate[candidate] = summary
            row = {
                "context": context.key,
                "scope": context.scope,
                "label": context.label,
                "outer_group": context.outer_group,
                "candidate": candidate,
                **summary,
            }
            (context_rows if context.scope == "repeated" else stress_rows).append(row)
            if context.scope == "repeated":
                for well_id, values in coefficients.items():
                    index = well_index[well_id]
                    sums[candidate][index] += values
                    counts[candidate][index] += 1
        context_metrics[context.key] = by_candidate

    final_coefficients: dict[str, dict[str, np.ndarray]] = {name: {} for name in all_names}
    final_metrics: dict[str, dict[str, WellMetric]] = {name: {} for name in all_names}
    for candidate in all_names:
        if np.any(counts[candidate] != 5):
            raise DataValidationError(f"E012 repeated coverage differs for {candidate}")
        averaged = sums[candidate] / counts[candidate][:, None]
        for index, well_id in enumerate(well_ids):
            values = bound_coefficients(averaged[index], "spline4", e011_config)
            final_coefficients[candidate][well_id] = values
            final_metrics[candidate][well_id] = records[well_id].statistics["spline4"].metric(well_id, values)
    summaries = {candidate: _summarize([final_metrics[candidate][well_id] for well_id in well_ids]) for candidate in all_names}
    e011_rmse = float(summaries["e011"]["rmse"])

    map_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for fold in folds:
        version = str(fold["version"])
        keys = [f"repeated:{version}:{group}" for group in range(int(fold["n_folds"]))]
        for candidate in all_names:
            metrics = [
                metric
                for key in keys
                for metric in [
                    final_metrics[candidate][well_id]
                    for well_id in next(context.test_ids for context in repeated if context.key == key)
                ]
            ]
            map_rows[(version, candidate)] = _summarize(metrics)

    outer_wins = {
        candidate: sum(
            float(context_metrics[context.key][candidate]["rmse"]) < float(context_metrics[context.key]["e011"]["rmse"])
            for context in repeated
        )
        for candidate in all_names
    }
    map_wins = {
        candidate: sum(
            float(map_rows[(str(fold["version"]), candidate)]["rmse"]) < float(map_rows[(str(fold["version"]), "e011")]["rmse"])
            for fold in folds
        )
        for candidate in all_names
    }

    long_threshold = float(_quantile([record.rows for record in records.values()], 0.8))
    missing_threshold = float(_quantile([record.hidden_gr_missing_fraction for record in records.values()], 0.8))
    special_sets = {
        "long_suffix": [well_id for well_id, record in records.items() if record.rows >= long_threshold],
        "high_gr_missingness": [well_id for well_id, record in records.items() if record.hidden_gr_missing_fraction >= missing_threshold],
        "e011_catastrophe": [well_id for well_id in well_ids if final_metrics["e011"][well_id].rmse >= 12.0],
    }
    special_rows: list[dict[str, Any]] = []
    special_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for slice_name, slice_ids in special_sets.items():
        for candidate in all_names:
            summary = _summarize([final_metrics[candidate][well_id] for well_id in slice_ids])
            row = {"slice": slice_name, "candidate": candidate, "wells": len(slice_ids), **summary}
            special_rows.append(row)
            special_lookup[(slice_name, candidate)] = summary

    support_minimum = {
        spec.name: min(int(row["minimum_train_support"]) for row in support_rows if row["branch"] == spec.name)
        for spec in specs
    }
    stress_lookup = {(row["scope"], int(row["outer_group"]), row["candidate"]): row for row in stress_rows}
    promotion = config["promotion"]
    gates: dict[str, dict[str, bool]] = {}
    for spec in eligible:
        candidate = spec.name
        gates[candidate] = {
            "gain": e011_rmse - float(summaries[candidate]["rmse"]) >= float(promotion["minimum_gain_vs_e011"]),
            "maps": map_wins[candidate] >= int(promotion["minimum_map_wins"]),
            "outer_cells": outer_wins[candidate] >= int(promotion["minimum_outer_cell_wins"]),
            "p90": float(summaries[candidate]["p90_well_rmse"]) - float(summaries["e011"]["p90_well_rmse"]) <= float(promotion["maximum_p90_deterioration"]),
            "worst5": float(summaries[candidate]["worst_5pct_sse_share"]) - float(summaries["e011"]["worst_5pct_sse_share"]) <= float(promotion["maximum_worst5_increase"]),
            "support": support_minimum[candidate] >= int(config["minimum_regime_support"]),
            "spatial": all(float(stress_lookup[("spatial", group, "e011")]["rmse"]) - float(stress_lookup[("spatial", group, candidate)]["rmse"]) > 0.0 for group in range(5)),
            "typewell": all(float(stress_lookup[("typewell", group, "e011")]["rmse"]) - float(stress_lookup[("typewell", group, candidate)]["rmse"]) > 0.0 for group in range(5)),
            "special_slices": all(float(special_lookup[(name, "e011")]["rmse"]) - float(special_lookup[(name, candidate)]["rmse"]) > 0.0 for name in special_sets),
        }

    negative = next(spec.name for spec in specs if spec.negative_control)
    negative_gain = e011_rmse - float(summaries[negative]["rmse"])
    controls = {
        "parent_e011_reconstruction": {
            "pass": abs(e011_rmse - 12.550756295689666) <= 2e-4,
            "rmse": e011_rmse,
            "expected": 12.550756295689666,
        },
        "membership": {"pass": len(repeated) == 25 and len(stress) == 10 and all(bool(row["pass"]) for row in membership), "contexts": len(contexts)},
        "branch_completion": {"pass": len(context_rows) == 25 * len(all_names) and len(stress_rows) == 10 * len(all_names), "repeated_rows": len(context_rows), "stress_rows": len(stress_rows)},
        "finite_predictions": {"pass": all(math.isfinite(float(summary["rmse"])) for summary in summaries.values())},
        "coefficient_bounds": {"pass": maximum_coefficient <= 80.0 + 1e-8, "maximum_absolute_coefficient": maximum_coefficient},
        "exact_fallback": {"pass": maximum_fallback_delta <= 1e-12, "maximum_delta": maximum_fallback_delta},
        "deterministic_rerun": {"pass": deterministic_delta <= 1e-12, "maximum_delta": deterministic_delta},
        "shuffled_routing": {"pass": negative_gain <= float(promotion["shuffled_routing_maximum_gain"]), "gain_vs_e011": negative_gain},
        "oof_coverage": {"pass": all(np.all(counts[name] == 5) for name in all_names), "wells": len(well_ids)},
        "submission_unauthorized": {"pass": not bool(config["submission_authorized"])},
    }
    all_controls = all(bool(item["pass"]) for item in controls.values())
    passing = [spec.name for spec in eligible if all(gates[spec.name].values()) and all_controls]
    selected = min(passing, key=lambda name: float(summaries[name]["rmse"])) if passing else None
    status = "promoted" if selected else "rejected"

    candidate_rows = []
    for candidate in all_names:
        spec = next((item for item in specs if item.name == candidate), None)
        candidate_rows.append({
            "candidate": candidate,
            "family": "parent" if spec is None else spec.family,
            "eligible": False if spec is None else spec.eligible,
            "negative_control": False if spec is None else spec.negative_control,
            "selected": candidate == selected,
            "gain_vs_e011": e011_rmse - float(summaries[candidate]["rmse"]),
            "map_wins": map_wins[candidate],
            "outer_cell_wins": outer_wins[candidate],
            "minimum_train_support": len(well_ids) if spec is None else support_minimum[spec.name],
            **summaries[candidate],
        })
    control_rows = [{"control": name, "pass": bool(detail["pass"]), "details": json.dumps(detail, sort_keys=True)} for name, detail in sorted(controls.items())]
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("candidate_metrics.csv", candidate_rows),
        ("context_metrics.csv", context_rows),
        ("stress_metrics.csv", stress_rows),
        ("support_metrics.csv", support_rows),
        ("special_slice_metrics.csv", special_rows),
        ("controls.csv", control_rows),
    ):
        _write_csv(output_dir / filename, list(rows[0]), rows)
    summary = {
        "schema_version": 1,
        "experiment_id": "E012",
        "hypothesis_id": "H015",
        "status": status,
        "selected_candidate": selected,
        "code_sha": code_sha,
        "e011_metrics": summaries["e011"],
        "candidate_metrics": summaries,
        "map_wins": map_wins,
        "outer_cell_wins": outer_wins,
        "minimum_support": support_minimum,
        "gates_by_candidate": gates,
        "controls": controls,
        "all_controls_pass": all_controls,
        "passing_candidates": passing,
        "special_slice_wells": {name: len(values) for name, values in special_sets.items()},
        "spatial_assignments": spatial_assignments,
        "typewell_assignments": typewell_assignments,
        "runtime_seconds": time.perf_counter() - started,
        "decision": "retain_e011" if selected is None else "promote_specialist",
        "submission_made": False,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
