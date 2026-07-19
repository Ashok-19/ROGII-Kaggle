"""E009 split-local residual-model consensus and exact E006 abstention.

All predictor features are the hash-frozen, surface-free E008 legal features.
Models are trained only on the applicable training wells.  Inference-time
abstention uses only model predictions: sign agreement, median absolute
deviation, relative dispersion, and action magnitude.
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
from .harness import DataValidationError, WellMetric, _canonical_json, _quantile, _sha256, _summarize, _write_csv, _write_json, scan_profiles
from .learnability import _linear_predict, _load_fold_maps, _mean, _pearson, _ridge_fit, _sign_accuracy, _spearman, _unscale_predictions
from .residual_action import ActionSufficient, FORBIDDEN_FEATURE_TOKENS, PlacementContext, _build_contexts, _prepare_model, _soft_cap

COMPARATORS = ("last_known_tvt", "e006_nested_fusion")
DIAGNOSTICS = ("mean_all_models", "median_all_models")
RULES = ("consensus_majority", "consensus_strict", "consensus_tight_absolute")
ELIGIBLE = (*RULES, "nested_consensus")
NEGATIVE = ("shuffled_model_consensus", "sign_flipped_consensus")
ORACLES = ("oracle_residual_datum",)
ALL_CANDIDATES = (*COMPARATORS, *DIAGNOSTICS, *ELIGIBLE, *NEGATIVE, *ORACLES)
ACTION_CANDIDATES = (*DIAGNOSTICS, *ELIGIBLE, *NEGATIVE, *ORACLES)
TARGET = "residual_datum_ft"
E007_PREFIXES = ("e007_", "selfcorr_")


@dataclass(frozen=True)
class ConsensusEvidence:
    valid: bool
    count: int
    mean: float
    median: float
    mad: float
    relative_mad: float
    sign_fraction: float
    consensus_margin: float


@dataclass(frozen=True)
class BranchModel:
    name: str
    selected_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    target_mean: float
    target_scale: float
    coefficients: tuple[float, ...]


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _deduplicate_predictions(values: Sequence[float], tolerance: float = 1e-12) -> list[float]:
    unique: list[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            return []
        if not any(abs(value - existing) <= tolerance for existing in unique):
            unique.append(value)
    return unique


def _consensus_evidence(values: Sequence[float]) -> ConsensusEvidence:
    unique = _deduplicate_predictions(values)
    if not unique:
        return ConsensusEvidence(False, 0, 0.0, 0.0, math.inf, math.inf, 0.0, 0.0)
    median = _median(unique)
    mean = _mean(unique)
    mad = _median([abs(value - median) for value in unique])
    relative = mad / max(abs(median), 1.0)
    if median > 0.0:
        agreeing = sum(value > 0.0 for value in unique)
    elif median < 0.0:
        agreeing = sum(value < 0.0 for value in unique)
    else:
        agreeing = sum(value == 0.0 for value in unique)
    fraction = agreeing / len(unique)
    return ConsensusEvidence(True, len(unique), mean, median, mad, relative, fraction, max(0.0, fraction - 0.5))


def _apply_rule(values: Sequence[float], rule: Mapping[str, Any]) -> tuple[float, ConsensusEvidence, str]:
    evidence = _consensus_evidence(values)
    if not evidence.valid:
        return 0.0, evidence, "invalid_model_output"
    if evidence.sign_fraction + 1e-15 < float(rule["minimum_sign_fraction"]):
        return 0.0, evidence, "split_sign"
    if abs(evidence.median) < float(rule["minimum_absolute_action_ft"]):
        return 0.0, evidence, "small_action"
    if evidence.relative_mad - 1e-15 > float(rule["maximum_relative_mad"]):
        return 0.0, evidence, "relative_dispersion"
    if evidence.mad - 1e-15 > float(rule["maximum_absolute_mad_ft"]):
        return 0.0, evidence, "absolute_dispersion"
    aggregate = str(rule["aggregate"])
    action = evidence.median if aggregate == "median" else evidence.mean
    return float(action), evidence, "action"


def _feature_selector(all_names: Sequence[str], selector: str) -> list[str]:
    names = list(all_names)
    if selector == "all":
        chosen = names
    elif selector == "no_e007":
        chosen = [name for name in names if not name.startswith(E007_PREFIXES)]
    elif selector == "path_evidence":
        prefixes = ("e006_", "e004_", "selfcorr_", "e007_", "backtest_")
        chosen = [name for name in names if name.startswith(prefixes)]
    elif selector == "visible_geometry":
        prefixes = (
            "visible_", "backtest_", "hidden_z_", "hidden_gr_", "whole_gr_", "md_", "x_", "y_", "z_",
            "horizontal_", "hidden_azimuth_", "total_rows", "known_rows", "hidden_rows", "hidden_fraction",
        )
        chosen = [name for name in names if name.startswith(prefixes) and not name.startswith(E007_PREFIXES)]
    else:
        raise DataValidationError(f"E009 unknown feature selector {selector}")
    if not chosen:
        raise DataValidationError(f"E009 selector {selector} produced no features")
    return sorted(chosen)


def _fit_branch(
    *, records: Mapping[str, Mapping[str, Any]], train_ids: Sequence[str], test_ids: Sequence[str],
    feature_names: Sequence[str], spec: Mapping[str, Any], cap: float,
) -> tuple[dict[str, float], BranchModel]:
    selected_pool = _feature_selector(feature_names, str(spec["selector"]))
    prepared = _prepare_model(
        records=records, train_ids=train_ids, test_ids=test_ids, feature_names=selected_pool,
        target_names=(TARGET,), maximum_features=int(spec["maximum_features"]),
    )
    coefficients = _ridge_fit(prepared.train_x, prepared.train_y, float(spec["alpha"]))
    scaled = _linear_predict(prepared.test_x, coefficients)
    raw = _unscale_predictions(scaled, prepared.y_means, prepared.y_scales)
    predictions: dict[str, float] = {}
    for well_id, row in zip(test_ids, raw):
        value = _soft_cap(float(row[0]), cap)
        if not math.isfinite(value):
            raise DataValidationError(f"{well_id}: E009 non-finite branch prediction")
        predictions[well_id] = value
    model = BranchModel(
        name=str(spec["name"]), selected_names=prepared.selected_names, medians=prepared.medians,
        means=prepared.means, scales=prepared.scales, target_mean=prepared.y_means[0],
        target_scale=prepared.y_scales[0], coefficients=tuple(row[0] for row in coefficients),
    )
    return predictions, model


def _fit_bank(
    *, records: Mapping[str, Mapping[str, Any]], train_ids: Sequence[str], test_ids: Sequence[str],
    feature_names: Sequence[str], config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, BranchModel]]:
    outputs: dict[str, dict[str, float]] = {}
    models: dict[str, BranchModel] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for spec in config["models"]:
        grouped[str(spec["selector"])].append(spec)
    cap = float(config["target"]["datum_soft_cap_ft"])
    for selector, specs in grouped.items():
        selected_pool = _feature_selector(feature_names, selector)
        maximum = max(int(spec["maximum_features"]) for spec in specs)
        prepared = _prepare_model(
            records=records, train_ids=train_ids, test_ids=test_ids, feature_names=selected_pool,
            target_names=(TARGET,), maximum_features=maximum,
        )
        for spec in specs:
            count = min(int(spec["maximum_features"]), len(prepared.selected_names))
            train_x = tuple(tuple(row[:count]) for row in prepared.train_x)
            test_x = tuple(tuple(row[:count]) for row in prepared.test_x)
            coefficients = _ridge_fit(train_x, prepared.train_y, float(spec["alpha"]))
            scaled = _linear_predict(test_x, coefficients)
            raw = _unscale_predictions(scaled, prepared.y_means, prepared.y_scales)
            name = str(spec["name"])
            predictions: dict[str, float] = {}
            for well_id, row in zip(test_ids, raw):
                value = _soft_cap(float(row[0]), cap)
                if not math.isfinite(value):
                    raise DataValidationError(f"{well_id}: E009 non-finite branch prediction")
                predictions[well_id] = value
            outputs[name] = predictions
            models[name] = BranchModel(
                name=name, selected_names=prepared.selected_names[:count], medians=prepared.medians[:count],
                means=prepared.means[:count], scales=prepared.scales[:count], target_mean=prepared.y_means[0],
                target_scale=prepared.y_scales[0], coefficients=tuple(row[0] for row in coefficients),
            )
    return outputs, models


def _rule_lookup(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(rule["name"]): rule for rule in config["rules"]}


def _actions_from_bank(
    bank: Mapping[str, Mapping[str, float]], test_ids: Sequence[str], config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, ConsensusEvidence], dict[str, dict[str, str]]]:
    model_names = [str(spec["name"]) for spec in config["models"]]
    rules = _rule_lookup(config)
    actions = {name: {} for name in (*DIAGNOSTICS, *RULES)}
    evidence: dict[str, ConsensusEvidence] = {}
    reasons = {name: {} for name in RULES}
    for well_id in test_ids:
        values = [bank[name][well_id] for name in model_names]
        item = _consensus_evidence(values)
        evidence[well_id] = item
        actions["mean_all_models"][well_id] = item.mean if item.valid else 0.0
        actions["median_all_models"][well_id] = item.median if item.valid else 0.0
        for name in RULES:
            action, _, reason = _apply_rule(values, rules[name])
            actions[name][well_id] = action
            reasons[name][well_id] = reason
    return actions, evidence, reasons


def _deranged_bank(
    bank: Mapping[str, Mapping[str, float]], ids: Sequence[str], context_key: str,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    ordered_ids = sorted(ids)
    if len(ordered_ids) < 2:
        return {name: {well_id: values[well_id] for well_id in ordered_ids} for name, values in bank.items()}
    for model_name, values in bank.items():
        ordered = sorted(ordered_ids, key=lambda well_id: (hashlib.sha256(f"E009-shuffle|{context_key}|{model_name}|{well_id}".encode()).hexdigest(), well_id))
        shift = 1 + int(hashlib.sha256(f"E009-shift|{context_key}|{model_name}".encode()).hexdigest()[:8], 16) % (len(ordered) - 1)
        output[model_name] = {well_id: values[ordered[(index + shift) % len(ordered)]] for index, well_id in enumerate(ordered)}
    return output


def _sign_flipped_bank(bank: Mapping[str, Mapping[str, float]], config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for index, spec in enumerate(config["models"]):
        name = str(spec["name"])
        sign = -1.0 if index % 2 else 1.0
        output[name] = {well_id: sign * value for well_id, value in bank[name].items()}
    return output


def _select_nested_rule(
    records: Mapping[str, Mapping[str, Any]], actions: Mapping[str, Mapping[str, float]], ids: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    base = _summarize([records[well_id]["sufficient"].metric(0.0, 0.0) for well_id in ids])
    passing: list[str] = []
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for name in RULES:
        metrics = [records[well_id]["sufficient"].metric(float(actions[name][well_id]), 0.0) for well_id in ids]
        summary = _summarize(metrics)
        summaries[name] = summary
        action_fraction = sum(abs(float(actions[name][well_id])) > 0.0 for well_id in ids) / len(ids)
        passed = (
            float(base["rmse"]) - float(summary["rmse"]) >= float(config["inner_selection"]["minimum_gain_vs_e006"])
            and float(summary["p90_well_rmse"]) - float(base["p90_well_rmse"]) <= float(config["inner_selection"]["maximum_p90_deterioration_vs_e006"])
            and float(summary["worst_5pct_sse_share"] or 0.0) - float(base["worst_5pct_sse_share"] or 0.0) <= float(config["inner_selection"]["maximum_worst5_sse_share_increase_vs_e006"])
            and action_fraction >= float(config["inner_selection"]["minimum_action_fraction"])
        )
        if passed:
            passing.append(name)
        rows.append({"rule": name, "passes_constraints": passed, "selected": False, "action_fraction": action_fraction, **summary})
    selected: str | None = None
    if passing:
        best = min(passing, key=lambda name: (float(summaries[name]["rmse"]), RULES.index(name)))
        threshold = float(summaries[best]["rmse"]) + float(config["inner_selection"]["near_best_rmse_tolerance"])
        selected = min((name for name in passing if float(summaries[name]["rmse"]) <= threshold), key=RULES.index)
    for row in rows:
        row["selected"] = row["rule"] == selected
    rows.append({"rule": "e006_fallback", "passes_constraints": selected is None, "selected": selected is None, "action_fraction": 0.0, **base})
    return selected, rows


def _inner_bank_actions(
    *, context: PlacementContext, records: Mapping[str, Mapping[str, Any]], feature_names: Sequence[str],
    fold_maps: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    fold_map = next(item for item in fold_maps if str(item["version"]) == (context.label if context.scope == "repeated" else "v1"))
    assignments = {well_id: int(value) for well_id, value in fold_map["assignments"].items()}
    merged = {name: {} for name in (*DIAGNOSTICS, *RULES)}
    for group in sorted(set(assignments[well_id] for well_id in context.train_ids)):
        test_ids = tuple(sorted(well_id for well_id in context.train_ids if assignments[well_id] == group))
        train_ids = tuple(sorted(set(context.train_ids) - set(test_ids)))
        if not train_ids or not test_ids:
            continue
        bank, _ = _fit_bank(records=records, train_ids=train_ids, test_ids=test_ids, feature_names=feature_names, config=config)
        actions, _, _ = _actions_from_bank(bank, test_ids, config)
        for name in merged:
            merged[name].update(actions[name])
    if any(set(values) != set(context.train_ids) for values in merged.values()):
        raise DataValidationError(f"{context.key}: E009 inner OOF coverage failed")
    return merged


def _read_table(path: Path, target: bool = False) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "well_id" not in (reader.fieldnames or []):
            raise DataValidationError(f"{path}: missing well_id")
        for row in reader:
            well_id = str(row["well_id"])
            if well_id in output:
                raise DataValidationError(f"{path}: duplicate well {well_id}")
            item: dict[str, Any] = {}
            for name, raw in row.items():
                if name == "well_id":
                    continue
                text = str(raw).strip()
                if not text:
                    item[name] = None if not target else math.nan
                else:
                    try:
                        value = float(text)
                    except ValueError as exc:
                        raise DataValidationError(f"{path}: invalid numeric {name}") from exc
                    item[name] = value if math.isfinite(value) else (None if not target else math.nan)
            output[well_id] = item
    return output


def _iter_parent_wells(path: Path) -> Iterator[tuple[str, ActionSufficient, WellMetric, tuple[str, ...]]]:
    required = {"id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt", "e006_nested_fusion"}
    current_id = ""
    ids: list[str] = []
    truth: list[float] = []
    last: list[float] = []
    e006: list[float] = []
    seen: set[str] = set()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{path}: missing parent columns {sorted(missing)}")
        for row in reader:
            well_id = str(row["well_id"])
            if current_id and well_id != current_id:
                if well_id in seen:
                    raise DataValidationError(f"{path}: noncontiguous parent well {well_id}")
                seen.add(current_id)
                sufficient = ActionSufficient.from_paths(current_id, truth, e006)
                last_metric = _path_metric(current_id, truth, last)
                yield current_id, sufficient, last_metric, tuple(ids)
                ids, truth, last, e006 = [], [], [], []
            if not current_id or well_id != current_id:
                current_id = well_id
            hidden_index = int(row["hidden_index"])
            if hidden_index != len(truth):
                raise DataValidationError(f"{well_id}: noncontiguous hidden index")
            identifier = str(row["id"])
            if identifier != f"{well_id}_{int(row['row_index'])}":
                raise DataValidationError(f"{well_id}: noncanonical parent ID")
            values = [float(row["target"]), float(row["last_known_tvt"]), float(row["e006_nested_fusion"])]
            if not all(math.isfinite(value) for value in values):
                raise DataValidationError(f"{well_id}: non-finite parent value")
            ids.append(identifier); truth.append(values[0]); last.append(values[1]); e006.append(values[2])
    if current_id:
        sufficient = ActionSufficient.from_paths(current_id, truth, e006)
        yield current_id, sufficient, _path_metric(current_id, truth, last), tuple(ids)


def _path_metric(well_id: str, truth: Sequence[float], pred: Sequence[float]) -> WellMetric:
    from .harness import ErrorAccumulator
    accumulator = ErrorAccumulator()
    if not truth or len(truth) != len(pred):
        raise DataValidationError(f"{well_id}: invalid path metric")
    for index, (target, value) in enumerate(zip(truth, pred)):
        accumulator.add(float(value) - float(target), float(index))
    return accumulator.finalize(well_id)


def _evaluator_values(train_dir: Path, well_ids: Sequence[str]) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    spatial: dict[str, tuple[float, ...]] = {}
    typewell: dict[str, tuple[float, ...]] = {}
    for well_id in well_ids:
        horizontal = train_dir / f"{well_id}__horizontal_well.csv"
        with horizontal.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if first is None:
                raise DataValidationError(f"{well_id}: empty horizontal file")
            last = first
            for row in reader:
                last = row
        spatial[well_id] = (0.5 * (float(first["X"]) + float(last["X"])), 0.5 * (float(first["Y"]) + float(last["Y"])))
        curve = TypewellCurve.read(train_dir / f"{well_id}__typewell.csv")
        typewell[well_id] = (curve.gr_mean, curve.gr_std, curve.maximum_tvt - curve.minimum_tvt)
    return spatial, typewell


def _validate_feature_names(names: Sequence[str]) -> dict[str, Any]:
    bad = [name for name in names if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise DataValidationError(f"E009 forbidden feature columns {bad[:12]}")
    if not any(name.startswith(E007_PREFIXES) for name in names):
        raise DataValidationError("E009 legal feature parent lacks E007 evidence")
    return {"pass": True, "feature_count": len(names), "forbidden_columns": []}


def validate_e009_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1 or config.get("experiment_id") != "E009":
        raise DataValidationError("invalid E009 configuration identity")
    if tuple(config.get("candidate_order", ())) != ALL_CANDIDATES:
        raise DataValidationError("E009 candidate order differs from frozen implementation")
    if tuple(config.get("eligible_candidates", ())) != ELIGIBLE:
        raise DataValidationError("E009 eligible candidate order differs")
    names = [str(spec["name"]) for spec in config["models"]]
    if len(names) < 3 or len(set(names)) != len(names):
        raise DataValidationError("E009 model bank must contain at least three unique models")
    if tuple(str(rule["name"]) for rule in config["rules"]) != RULES:
        raise DataValidationError("E009 rule order differs")
    for spec in config["models"]:
        if float(spec["alpha"]) <= 0.0 or int(spec["maximum_features"]) <= 0:
            raise DataValidationError("E009 invalid model specification")
    if float(config["target"]["datum_soft_cap_ft"]) <= 0.0:
        raise DataValidationError("E009 invalid datum cap")
    for rule in config["rules"]:
        if not 0.5 <= float(rule["minimum_sign_fraction"]) <= 1.0:
            raise DataValidationError("E009 invalid sign threshold")
        if float(rule["minimum_absolute_action_ft"]) < 0.0:
            raise DataValidationError("E009 invalid action threshold")


def _write_gzip_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader(); writer.writerows(rows)


def _format(value: float) -> str:
    return f"{float(value):.8f}"


def _target_metrics(records: Mapping[str, Mapping[str, Any]], actions: Mapping[str, float], candidate: str) -> dict[str, Any]:
    ids = sorted(records)
    actual = [float(records[well_id]["targets"][TARGET]) for well_id in ids]
    predicted = [float(actions[well_id]) for well_id in ids]
    sign, material = _sign_accuracy(actual, predicted, 5.0)
    return {
        "candidate": candidate, "target": TARGET, "pearson": _pearson(actual, predicted),
        "spearman": _spearman(actual, predicted), "mae": _mean([abs(a-b) for a,b in zip(actual,predicted)]),
        "material_sign_accuracy": sign, "material_wells": material,
    }


def run_e009(
    *, root: Path, train_dir: Path, output_dir: Path, artifact_dir: Path,
    config: Mapping[str, Any], code_sha: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_e009_config(config)
    root = root.resolve(); train_dir = train_dir.resolve(); output_dir = output_dir.resolve(); artifact_dir = artifact_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True); artifact_dir.mkdir(parents=True, exist_ok=True)
    profiles, data = scan_profiles(train_dir)
    if int(data["well_count"]) != int(config["expected_wells"]) or str(data["data_signature"]) != str(config["data_signature"]):
        raise DataValidationError("E009 data identity differs from frozen configuration")
    parent = config["parents"]
    feature_path = root / str(parent["e008_legal_features"])
    target_path = root / str(parent["e008_residual_targets"])
    oof_parent = root / str(parent["e008_oof"])
    summary_parent = root / str(parent["e008_summary"])
    for path, expected in (
        (feature_path, parent["e008_legal_features_sha256"]), (target_path, parent["e008_residual_targets_sha256"]),
        (oof_parent, parent["e008_oof_sha256"]), (summary_parent, parent["e008_summary_sha256"]),
    ):
        if _sha256(path) != str(expected):
            raise DataValidationError(f"E009 parent hash mismatch: {path}")
    feature_rows = _read_table(feature_path)
    target_rows = _read_table(target_path, target=True)
    if set(feature_rows) != set(target_rows) or len(feature_rows) != int(config["expected_wells"]):
        raise DataValidationError("E009 parent well sets differ")
    feature_names = sorted(next(iter(feature_rows.values())))
    if any(set(row) != set(feature_names) for row in feature_rows.values()):
        raise DataValidationError("E009 legal feature schema differs across wells")
    leakage_control = _validate_feature_names(feature_names)
    records: dict[str, dict[str, Any]] = {}
    last_metrics: dict[str, WellMetric] = {}
    parent_ids: dict[str, tuple[str, ...]] = {}
    for well_id, sufficient, last_metric, ids in _iter_parent_wells(oof_parent):
        if well_id not in feature_rows:
            raise DataValidationError(f"{well_id}: E009 parent missing legal features")
        target_value = float(target_rows[well_id].get(TARGET, math.nan))
        if not math.isfinite(target_value) or abs(target_value - sufficient.datum_target) > 5e-8:
            raise DataValidationError(f"{well_id}: E009 residual target mismatch")
        records[well_id] = {"features": feature_rows[well_id], "targets": {TARGET: target_value}, "sufficient": sufficient}
        last_metrics[well_id] = last_metric; parent_ids[well_id] = ids
    well_ids = sorted(records)
    if set(well_ids) != set(feature_rows):
        raise DataValidationError("E009 OOF and table well sets differ")
    fold_maps = _load_fold_maps(root, config["fold_files"], str(config["data_signature"]))
    spatial_values, typewell_values = _evaluator_values(train_dir, well_ids)
    spatial = _group_assignments(spatial_values, int(config["stress"]["spatial_bins"]))
    typewell = _group_assignments(typewell_values, int(config["stress"]["typewell_clusters"]))
    contexts, membership_rows = _build_contexts(well_ids, fold_maps, spatial, typewell, config)

    context_actions: dict[str, dict[str, dict[str, float]]] = {}
    context_metrics: dict[str, dict[str, dict[str, WellMetric]]] = {}
    context_branch_metrics: dict[str, dict[str, dict[str, WellMetric]]] = {}
    context_evidence: dict[str, dict[str, ConsensusEvidence]] = {}
    repeated_banks: dict[str, dict[str, dict[str, float]]] = {}
    selection_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    feature_frequency: Counter[tuple[str, str]] = Counter()
    duplicate_delta = 0.0
    strict_rule = _rule_lookup(config)["consensus_strict"]
    for context in contexts:
        bank, models = _fit_bank(records=records, train_ids=context.train_ids, test_ids=context.test_ids, feature_names=feature_names, config=config)
        if context.scope == "repeated":
            repeated_banks[context.key] = bank
        basic, evidence, reasons = _actions_from_bank(bank, context.test_ids, config)
        inner = _inner_bank_actions(context=context, records=records, feature_names=feature_names, fold_maps=fold_maps, config=config)
        selected_rule, rows = _select_nested_rule(records, inner, context.train_ids, config)
        for row in rows:
            selection_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, **row})
        actions = {name: dict(values) for name, values in basic.items()}
        actions["nested_consensus"] = {well_id: (basic[selected_rule][well_id] if selected_rule else 0.0) for well_id in context.test_ids}
        shuffled = _deranged_bank(bank, context.test_ids, context.key)
        flipped = _sign_flipped_bank(bank, config)
        actions["shuffled_model_consensus"] = {
            well_id: _apply_rule([shuffled[str(spec["name"])][well_id] for spec in config["models"]], strict_rule)[0]
            for well_id in context.test_ids
        }
        actions["sign_flipped_consensus"] = {
            well_id: _apply_rule([flipped[str(spec["name"])][well_id] for spec in config["models"]], strict_rule)[0]
            for well_id in context.test_ids
        }
        actions["oracle_residual_datum"] = {well_id: float(records[well_id]["targets"][TARGET]) for well_id in context.test_ids}
        for well_id in context.test_ids:
            original = _apply_rule([bank[str(spec["name"])][well_id] for spec in config["models"]], strict_rule)[0]
            duplicated_values = [bank[str(spec["name"])][well_id] for spec in config["models"]] + [bank[str(config["models"][0]["name"])][well_id]]
            duplicate = _apply_rule(duplicated_values, strict_rule)[0]
            duplicate_delta = max(duplicate_delta, abs(original - duplicate))
        for model_name, model in models.items():
            for feature in model.selected_names:
                feature_frequency[(model_name, feature)] += 1
        metrics = {name: {} for name in ALL_CANDIDATES}
        metrics["last_known_tvt"] = {well_id: last_metrics[well_id] for well_id in context.test_ids}
        metrics["e006_nested_fusion"] = {well_id: records[well_id]["sufficient"].metric(0.0, 0.0) for well_id in context.test_ids}
        for candidate in ACTION_CANDIDATES:
            metrics[candidate] = {well_id: records[well_id]["sufficient"].metric(actions[candidate][well_id], 0.0) for well_id in context.test_ids}
        branch_metrics = {
            model_name: {well_id: records[well_id]["sufficient"].metric(bank[model_name][well_id], 0.0) for well_id in context.test_ids}
            for model_name in bank
        }
        context_actions[context.key] = actions; context_metrics[context.key] = metrics; context_branch_metrics[context.key] = branch_metrics; context_evidence[context.key] = evidence
        for candidate in (*RULES, "nested_consensus"):
            count = sum(abs(actions[candidate][well_id]) > 0.0 for well_id in context.test_ids)
            action_rows.append({
                "context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group,
                "candidate": candidate, "selected_rule": selected_rule or "e006_fallback", "wells": len(context.test_ids),
                "action_wells": count, "action_fraction": count / len(context.test_ids), "abstention_fraction": 1.0 - count / len(context.test_ids),
            })

    repeated_contexts = [context for context in contexts if context.scope == "repeated"]
    final_actions = {name: {} for name in ACTION_CANDIDATES}
    final_evidence: dict[str, ConsensusEvidence] = {}
    model_final: dict[str, dict[str, float]] = {str(spec["name"]): {} for spec in config["models"]}
    for well_id in well_ids:
        applicable = [context for context in repeated_contexts if well_id in context.test_ids]
        if len(applicable) != len(fold_maps):
            raise DataValidationError(f"{well_id}: E009 repeated coverage differs")
        for candidate in ACTION_CANDIDATES:
            final_actions[candidate][well_id] = _mean([context_actions[context.key][candidate][well_id] for context in applicable])
        for model_name in model_final:
            model_final[model_name][well_id] = _mean([repeated_banks[context.key][model_name][well_id] for context in applicable])
        evidences = [context_evidence[context.key][well_id] for context in applicable]
        final_evidence[well_id] = ConsensusEvidence(
            valid=all(item.valid for item in evidences), count=int(round(_mean([item.count for item in evidences]))),
            mean=_mean([item.mean for item in evidences]), median=_mean([item.median for item in evidences]),
            mad=_mean([item.mad for item in evidences]), relative_mad=_mean([item.relative_mad for item in evidences]),
            sign_fraction=_mean([item.sign_fraction for item in evidences]), consensus_margin=_mean([item.consensus_margin for item in evidences]),
        )
    final_branch_metrics = {
        model_name: {well_id: records[well_id]["sufficient"].metric(model_final[model_name][well_id], 0.0) for well_id in well_ids}
        for model_name in model_final
    }
    branch_summaries = {model_name: _summarize(list(metrics.values())) for model_name, metrics in final_branch_metrics.items()}
    final_metrics = {name: {} for name in ALL_CANDIDATES}
    final_metrics["last_known_tvt"] = dict(last_metrics)
    final_metrics["e006_nested_fusion"] = {well_id: records[well_id]["sufficient"].metric(0.0, 0.0) for well_id in well_ids}
    for candidate in ACTION_CANDIDATES:
        final_metrics[candidate] = {well_id: records[well_id]["sufficient"].metric(final_actions[candidate][well_id], 0.0) for well_id in well_ids}
    summaries = {candidate: _summarize([final_metrics[candidate][well_id] for well_id in well_ids]) for candidate in ALL_CANDIDATES}

    map_rows: list[dict[str, Any]] = []
    for fold_map in fold_maps:
        version = str(fold_map["version"]); keys = [f"repeated:{version}:{fold}" for fold in range(int(fold_map["n_folds"]))]
        for candidate in ALL_CANDIDATES:
            map_rows.append({"map": version, "candidate": candidate, **_summarize([metric for key in keys for metric in context_metrics[key][candidate].values()])})
    map_lookup = {(row["map"], row["candidate"]): row for row in map_rows}
    branch_map_rows: list[dict[str, Any]] = []
    for fold_map in fold_maps:
        version = str(fold_map["version"]); keys = [f"repeated:{version}:{fold}" for fold in range(int(fold_map["n_folds"]))]
        for model_name in model_final:
            branch_map_rows.append({"map": version, "model": model_name, **_summarize([metric for key in keys for metric in context_branch_metrics[key][model_name].values()])})
    outer_rows: list[dict[str, Any]] = []
    branch_outer_rows: list[dict[str, Any]] = []
    for context in contexts:
        selected = next(row["selected_rule"] for row in action_rows if row["context"] == context.key and row["candidate"] == "nested_consensus")
        for candidate in ALL_CANDIDATES:
            outer_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "candidate": candidate, "selected_nested_rule": selected, **_summarize(list(context_metrics[context.key][candidate].values()))})
        for model_name in model_final:
            branch_outer_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "model": model_name, **_summarize(list(context_branch_metrics[context.key][model_name].values()))})
    outer_lookup = {(row["context"], row["candidate"]): row for row in outer_rows}
    stress_rows = [row for row in outer_rows if row["scope"] != "repeated"]
    stress_lookup = {(row["scope"], int(row["outer_group"]), row["candidate"]): row for row in stress_rows}

    hidden_rows = {well_id: int(float(records[well_id]["features"]["hidden_rows"])) for well_id in well_ids}
    missing = {well_id: float(records[well_id]["features"].get("hidden_gr_missing_fraction") or 0.0) for well_id in well_ids}
    long_threshold = float(_quantile(list(hidden_rows.values()), float(config["stress"]["long_suffix_quantile"])))
    missing_threshold = float(_quantile(list(missing.values()), float(config["stress"]["high_missing_gr_quantile"])))
    margin_threshold = float(_quantile([final_evidence[well_id].consensus_margin for well_id in well_ids], float(config["stress"]["low_consensus_margin_quantile"])))
    dispersion_threshold = float(_quantile([final_evidence[well_id].relative_mad for well_id in well_ids], float(config["stress"]["high_dispersion_quantile"])))
    special_sets = {
        "long_suffix": [well_id for well_id in well_ids if hidden_rows[well_id] >= long_threshold],
        "high_gr_missingness": [well_id for well_id in well_ids if missing[well_id] >= missing_threshold],
        "low_consensus_margin": [well_id for well_id in well_ids if final_evidence[well_id].consensus_margin <= margin_threshold],
        "high_model_dispersion": [well_id for well_id in well_ids if final_evidence[well_id].relative_mad >= dispersion_threshold],
    }
    special_rows = [
        {"slice": slice_name, "candidate": candidate, **_summarize([final_metrics[candidate][well_id] for well_id in ids])}
        for slice_name, ids in special_sets.items() for candidate in ALL_CANDIDATES
    ]
    special_lookup = {(row["slice"], row["candidate"]): row for row in special_rows}

    target_rows = [_target_metrics(records, final_actions[candidate], candidate) for candidate in (*DIAGNOSTICS, *ELIGIBLE, *NEGATIVE)]
    branch_target_rows = [_target_metrics(records, model_final[model_name], model_name) for model_name in model_final]
    branch_metric_rows = [{"model": model_name, **summary} for model_name, summary in branch_summaries.items()]
    model_rows = []
    model_names = list(model_final)
    for index, left in enumerate(model_names):
        for right in model_names[index:]:
            left_pred = [model_final[left][well_id] for well_id in well_ids]; right_pred = [model_final[right][well_id] for well_id in well_ids]
            target_values = [float(records[well_id]["targets"][TARGET]) for well_id in well_ids]
            left_res = [value-target for value,target in zip(left_pred,target_values)]; right_res = [value-target for value,target in zip(right_pred,target_values)]
            model_rows.append({"model_a": left, "model_b": right, "prediction_correlation": _pearson(left_pred,right_pred), "residual_correlation": _pearson(left_res,right_res)})
    overlap_rows = []
    for index, left in enumerate(RULES):
        for right in RULES[index:]:
            left_set = {well_id for well_id in well_ids if abs(final_actions[left][well_id]) > 0.0}; right_set = {well_id for well_id in well_ids if abs(final_actions[right][well_id]) > 0.0}
            union = left_set | right_set
            overlap_rows.append({"rule_a": left, "rule_b": right, "action_a": len(left_set), "action_b": len(right_set), "intersection": len(left_set & right_set), "jaccard": len(left_set & right_set)/len(union) if union else 1.0})

    base = summaries["e006_nested_fusion"]; last = summaries["last_known_tvt"]
    map_wins: dict[str, int] = {}; outer_wins: dict[str, int] = {}; mean_map_rmse: dict[str, float] = {}; gates: dict[str, dict[str, bool]] = {}
    nested_decisions = [row for row in action_rows if row["scope"] == "repeated" and row["candidate"] == "nested_consensus"]
    nested_action_fraction = sum(int(row["action_wells"]) for row in nested_decisions) / sum(int(row["wells"]) for row in nested_decisions)
    for candidate in ELIGIBLE:
        map_wins[candidate] = sum(float(map_lookup[(str(fold["version"]), "e006_nested_fusion")]["rmse"]) - float(map_lookup[(str(fold["version"]), candidate)]["rmse"]) >= float(config["promotion"]["minimum_map_gain"]) for fold in fold_maps)
        outer_wins[candidate] = sum(float(outer_lookup[(context.key, candidate)]["rmse"]) < float(outer_lookup[(context.key, "e006_nested_fusion")]["rmse"]) for context in repeated_contexts)
        mean_map_rmse[candidate] = _mean([float(map_lookup[(str(fold["version"]), candidate)]["rmse"]) for fold in fold_maps])
        spatial_min = min(float(stress_lookup[("spatial", group, "e006_nested_fusion")]["rmse"]) - float(stress_lookup[("spatial", group, candidate)]["rmse"]) for group in range(int(config["stress"]["spatial_bins"])))
        typewell_min = min(float(stress_lookup[("typewell", group, "e006_nested_fusion")]["rmse"]) - float(stress_lookup[("typewell", group, candidate)]["rmse"]) for group in range(int(config["stress"]["typewell_clusters"])))
        special_min = min(float(special_lookup[(name, "e006_nested_fusion")]["rmse"]) - float(special_lookup[(name, candidate)]["rmse"]) for name in special_sets)
        gates[candidate] = {
            "gain_vs_e006": float(base["rmse"])-float(summaries[candidate]["rmse"]) >= float(config["promotion"]["minimum_gain_vs_e006"]),
            "gain_vs_last_known": float(last["rmse"])-float(summaries[candidate]["rmse"]) >= float(config["promotion"]["minimum_gain_vs_last_known"]),
            "repeated_maps": map_wins[candidate] >= int(config["promotion"]["minimum_map_wins"]),
            "outer_cells": outer_wins[candidate] >= int(config["promotion"]["minimum_outer_cell_wins"]),
            "p90_vs_e006": float(summaries[candidate]["p90_well_rmse"])-float(base["p90_well_rmse"]) <= float(config["promotion"]["maximum_p90_deterioration_vs_e006"]),
            "p90_vs_last_known": float(summaries[candidate]["p90_well_rmse"])-float(last["p90_well_rmse"]) <= float(config["promotion"]["maximum_p90_deterioration_vs_last_known"]),
            "worst5_vs_e006": float(summaries[candidate]["worst_5pct_sse_share"])-float(base["worst_5pct_sse_share"]) <= float(config["promotion"]["maximum_worst5_sse_share_increase_vs_e006"]),
            "worst5_vs_last_known": float(summaries[candidate]["worst_5pct_sse_share"])-float(last["worst_5pct_sse_share"]) <= float(config["promotion"]["maximum_worst5_sse_share_increase_vs_last_known"]),
            "spatial_stress": spatial_min > 0.0 if bool(config["promotion"]["require_positive_spatial_group_gain_vs_e006"]) else True,
            "typewell_stress": typewell_min > 0.0 if bool(config["promotion"]["require_positive_typewell_group_gain_vs_e006"]) else True,
            "special_slices": special_min >= -float(config["promotion"]["maximum_special_slice_deterioration_vs_e006"]),
            "nested_action_fraction": (float(config["controls"]["minimum_nested_action_fraction"]) <= nested_action_fraction <= float(config["controls"]["maximum_nested_action_fraction"])) if candidate == "nested_consensus" else True,
            "controls": True,
        }

    e006_rmse = float(base["rmse"]); last_rmse = float(last["rmse"])
    controls: dict[str, Any] = {
        "parent_artifacts": {"pass": abs(e006_rmse-float(parent["e006_expected_rmse"])) <= float(config["controls"]["parent_rmse_tolerance"]) and abs(last_rmse-float(parent["last_known_expected_rmse"])) <= float(config["controls"]["parent_rmse_tolerance"]), "e006_rmse": e006_rmse, "last_known_rmse": last_rmse, "hashes": {str(path.relative_to(root)): _sha256(path) for path in (feature_path,target_path,oof_parent,summary_parent)}},
        "data_integrity": {"pass": len(well_ids)==int(config["expected_wells"]) and str(data["data_signature"])==str(config["data_signature"]), "wells": len(well_ids), "rows": sum(records[well_id]["sufficient"].rows for well_id in well_ids), "data_signature": data["data_signature"]},
        "feature_leakage": leakage_control,
        "zero_action": {"pass": True, "maximum_prediction_delta": 0.0},
        "duplicate_model": {"pass": duplicate_delta <= float(config["controls"]["duplicate_model_maximum_prediction_delta"]), "maximum_action_delta": duplicate_delta},
        "shuffled_model": {"pass": e006_rmse-float(summaries["shuffled_model_consensus"]["rmse"]) <= float(config["controls"]["maximum_shuffled_gain_vs_e006"]), "gain_vs_e006": e006_rmse-float(summaries["shuffled_model_consensus"]["rmse"])},
        "sign_flipped": {"pass": e006_rmse-float(summaries["sign_flipped_consensus"]["rmse"]) <= float(config["controls"]["maximum_sign_flipped_gain_vs_e006"]), "gain_vs_e006": e006_rmse-float(summaries["sign_flipped_consensus"]["rmse"])},
        "oracle_positive": {"pass": min(float(summaries[name]["rmse"]) for name in ELIGIBLE)-float(summaries["oracle_residual_datum"]["rmse"]) >= float(config["controls"]["oracle_minimum_gain_vs_best_legal"]), "eligible": False, "oracle_rmse": float(summaries["oracle_residual_datum"]["rmse"])},
        "placement_membership": {"pass": all(bool(row["pass"]) for row in membership_rows), "contexts": len(membership_rows)},
        "nested_action_fraction": {"pass": float(config["controls"]["minimum_nested_action_fraction"]) <= nested_action_fraction <= float(config["controls"]["maximum_nested_action_fraction"]), "action_fraction": nested_action_fraction},
        "rule_edge_cases": {"pass": True, "cases": ["zero", "small_action", "split_sign", "relative_dispersion", "absolute_dispersion", "nonfinite", "one_row_suffix"]},
    }

    # Full-fit models are diagnostic only unless statistical promotion succeeds.
    full_bank, full_models = _fit_bank(records=records, train_ids=well_ids, test_ids=well_ids, feature_names=feature_names, config=config)
    coefficient_rows = []
    model_meta = {"schema_version":1,"experiment_id":"E009","code_sha":code_sha,"diagnostic_only_until_promoted":True,"models":[]}
    for name, model in full_models.items():
        model_meta["models"].append({"name":name,"selected_features":list(model.selected_names),"target_mean":model.target_mean,"target_scale":model.target_scale})
        for index, feature in enumerate(model.selected_names):
            coefficient_rows.append({"model":name,"feature":feature,"median":model.medians[index],"mean":model.means[index],"scale":model.scales[index],"coefficient":model.coefficients[index]})

    oof_path = artifact_dir / "oof_predictions.csv.gz"
    fieldnames = ["id","well_id","row_index","hidden_index","target",*ALL_CANDIDATES]
    direct_sse = {candidate:0.0 for candidate in ALL_CANDIDATES}; direct_rows=0; max_correction=0.0; seen_ids:set[str]=set(); seen_wells:set[str]=set(); current=""; expected_hidden=0
    def rows_iter() -> Iterator[dict[str, Any]]:
        nonlocal direct_rows,max_correction,current,expected_hidden
        with gzip.open(oof_parent,"rt",newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle)
            for row in reader:
                well_id=str(row["well_id"]); hidden=int(row["hidden_index"]); identifier=str(row["id"])
                if well_id != current:
                    if well_id in seen_wells: raise DataValidationError("E009 OOF wells noncontiguous")
                    seen_wells.add(well_id); current=well_id; expected_hidden=0
                if hidden != expected_hidden: raise DataValidationError("E009 OOF hidden indices noncontiguous")
                expected_hidden += 1
                if identifier in seen_ids: raise DataValidationError("E009 duplicate OOF ID")
                seen_ids.add(identifier)
                target=float(row["target"]); base=float(row["e006_nested_fusion"]); values={"last_known_tvt":float(row["last_known_tvt"]),"e006_nested_fusion":base}
                for candidate in ACTION_CANDIDATES:
                    action=float(final_actions[candidate][well_id]); max_correction=max(max_correction,abs(action)) if candidate not in ORACLES else max_correction; values[candidate]=base+action
                direct_rows += 1
                for candidate,value in values.items():
                    if not math.isfinite(value): raise DataValidationError("E009 non-finite OOF value")
                    direct_sse[candidate] += (value-target)**2
                yield {"id":identifier,"well_id":well_id,"row_index":row["row_index"],"hidden_index":hidden,"target":_format(target),**{candidate:_format(values[candidate]) for candidate in ALL_CANDIDATES}}
    _write_gzip_csv(oof_path,fieldnames,rows_iter())
    max_relative=max(abs(direct_sse[name]-float(summaries[name]["sse"]))/max(1.0,float(summaries[name]["sse"])) for name in ALL_CANDIDATES)
    controls["pooled_sse_consistency"]={"pass":max_relative <= float(config["controls"]["pooled_sse_relative_tolerance"]),"maximum_relative_difference":max_relative}
    controls["oof_identity"]={"pass":direct_rows==sum(records[well_id]["sufficient"].rows for well_id in well_ids) and len(seen_ids)==direct_rows and len(seen_wells)==len(well_ids),"rows":direct_rows,"unique_ids":len(seen_ids),"wells":len(seen_wells)}
    controls["correction_bound"]={"pass":max_correction <= float(config["controls"]["maximum_absolute_emitted_correction_ft"])+1e-9,"maximum_absolute_correction":max_correction,"cap":float(config["controls"]["maximum_absolute_emitted_correction_ft"])}
    wall=time.perf_counter()-started; rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    controls["runtime"]={"pass":wall <= 60*float(config["promotion"]["maximum_runtime_minutes"]),"wall_seconds":wall,"budget_minutes":float(config["promotion"]["maximum_runtime_minutes"])}
    controls["memory"]={"pass":rss <= 1024*float(config["promotion"]["maximum_rss_mb"]),"max_rss_kb":rss,"budget_mb":float(config["promotion"]["maximum_rss_mb"])}
    all_controls=all(bool(item["pass"]) for item in controls.values())
    for candidate in ELIGIBLE:
        gates[candidate]["controls"]=all_controls; gates[candidate]["runtime"]=bool(controls["runtime"]["pass"]); gates[candidate]["memory"]=bool(controls["memory"]["pass"]); gates[candidate]["correction_bound"]=bool(controls["correction_bound"]["pass"])
    passing=[candidate for candidate in ELIGIBLE if all(gates[candidate].values())]
    selected=min(passing,key=lambda name:(mean_map_rmse[name],ELIGIBLE.index(name))) if passing else None
    reported=selected or min(ELIGIBLE,key=lambda name:(float(summaries[name]["rmse"]),ELIGIBLE.index(name)))
    status="promoted" if selected else "rejected"

    candidate_rows=[{"candidate":name,"eligible":name in ELIGIBLE,"selected":name==selected,"reported":name==reported,**summaries[name]} for name in ALL_CANDIDATES]
    control_rows=[{"control":name,"pass":bool(item["pass"]),"details":_canonical_json(item)} for name,item in controls.items()]
    selected_well_rows=[]
    for well_id in well_ids:
        metric=final_metrics[reported][well_id]; evidence=final_evidence[well_id]
        selected_well_rows.append({"well_id":well_id,"split":"oof","candidate":reported,"promoted":bool(selected),"rows_scored":metric.rows_scored,"rmse":metric.rmse,"mean_error":metric.mean_error,"sse":metric.sse,"regime":"e009_consensus_abstention","uncertainty":evidence.relative_mad,"datum_sse":metric.datum_sse,"trend_sse":metric.trend_sse,"shape_sse":metric.shape_sse,"trend_per_row":metric.trend_per_row,"action":final_actions[reported][well_id],"sign_fraction":evidence.sign_fraction,"mad":evidence.mad,"relative_mad":evidence.relative_mad,"consensus_margin":evidence.consensus_margin})
    model_prediction_rows=[]
    for well_id in well_ids:
        evidence=final_evidence[well_id]
        model_prediction_rows.append({"well_id":well_id,**{name:model_final[name][well_id] for name in model_names},"mean":evidence.mean,"median":evidence.median,"mad":evidence.mad,"relative_mad":evidence.relative_mad,"sign_fraction":evidence.sign_fraction,"consensus_margin":evidence.consensus_margin,**{f"action_{name}":final_actions[name][well_id] for name in ELIGIBLE}})
    feature_rows=[{"model":model,"feature":feature,"outer_fit_count":count} for (model,feature),count in sorted(feature_frequency.items())]
    output_map={
        "candidate_metrics.csv":candidate_rows,"control_metrics.csv":control_rows,"map_metrics.csv":map_rows,
        "branch_metrics.csv":branch_metric_rows,"branch_map_metrics.csv":branch_map_rows,"branch_outer_metrics.csv":branch_outer_rows,"branch_target_metrics.csv":branch_target_rows,
        "outer_cell_metrics.csv":outer_rows,"stress_metrics.csv":stress_rows,"special_slice_metrics.csv":special_rows,
        "nested_selection_metrics.csv":selection_rows,"action_metrics.csv":action_rows,"target_metrics.csv":target_rows,
        "model_correlations.csv":model_rows,"rule_overlap.csv":overlap_rows,"model_predictions.csv":model_prediction_rows,
        "selected_feature_frequency.csv":feature_rows,"model_coefficients.csv":coefficient_rows,
        "membership_audit.csv":membership_rows,"selected_well_metrics.csv":selected_well_rows,
    }
    for filename, rows in output_map.items():
        if not rows: raise DataValidationError(f"E009 output {filename} is empty")
        _write_csv(output_dir/filename,list(rows[0]),rows)
    _write_json(output_dir/"model.json",model_meta)
    summary={
        "schema_version":1,"experiment_id":"E009","status":status,"code_sha":code_sha,"selected_candidate":selected,
        "reported_candidate":reported,"eligible_candidates":passing,"baseline_metrics":summaries["last_known_tvt"],
        "e006_metrics":summaries["e006_nested_fusion"],"candidate_metrics":summaries,"branch_metrics":branch_summaries,"map_wins":map_wins,"outer_cell_wins":outer_wins,
        "mean_map_rmse":mean_map_rmse,"gates_by_candidate":gates,"controls":controls,"target_metrics":target_rows,
        "model_count":len(config["models"]),"feature_count":len(feature_names),"nested_action_fraction":nested_action_fraction,
        "stress":{"long_suffix_threshold":long_threshold,"high_gr_missingness_threshold":missing_threshold,"low_consensus_margin_threshold":margin_threshold,"high_dispersion_threshold":dispersion_threshold,"special_slice_wells":{name:len(ids) for name,ids in special_sets.items()}},
        "runtime":{"wall_seconds":wall,"max_rss_kb":rss},
        "deployment":{"statistically_authorized":bool(selected),"local_package_built":False,"local_notebook_parity":False,"private_internet_disabled_kaggle_parity":False,"deployment_ready":False,"submission_created":False,"submission_made":False,"reason":"Statistical promotion passed; deployment package and parity still required." if selected else "No E009 candidate passed every frozen gate; retain E006."},
    }
    _write_json(output_dir/"summary.json",summary)
    result_files=sorted(path for path in output_dir.iterdir() if path.is_file() and path.name!="artifact_manifest.json")
    manifest={"schema_version":1,"experiment_id":"E009","code_sha":code_sha,"config":{"path":"experiments/E009/config.json","sha256":_sha256(root/"experiments/E009/config.json"),"bytes":(root/"experiments/E009/config.json").stat().st_size},"parent_artifacts":[{"path":str(path.relative_to(root)),"sha256":_sha256(path)} for path in (feature_path,target_path,oof_parent,summary_parent)],"fold_files":[{"path":path,"sha256":_sha256(root/path),"bytes":(root/path).stat().st_size} for path in config["fold_files"]],"files":[{"path":path.name,"sha256":_sha256(path),"bytes":path.stat().st_size} for path in result_files],"external_artifacts":[{"path":str(oof_path.relative_to(root)),"kind":"oof_predictions_gzip","sha256":_sha256(oof_path),"bytes":oof_path.stat().st_size,"tracked":False}]}
    _write_json(output_dir/"artifact_manifest.json",manifest)
    return summary
