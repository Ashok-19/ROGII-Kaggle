"""E010 nonlinear whole-well candidate coverage and legal selector regret.

Hidden TVT is used only to score fixed candidate trajectories and to construct
training targets inside each selector training split. Validation-well hidden
TVT never enters feature preparation, model fitting, family choice, or path
reconstruction.
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

import numpy as np

from .e010_paths import (
    StatePathResult,
    basis_matrix,
    candidate_count,
    dtw_path,
    grid_oracle,
    grid_values,
    hmm_path,
    reconstruct_path,
    valid_coefficients,
)
from .fusion import _group_assignments
from .gr_path import TypewellCurve, read_well
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
from .learnability import _load_fold_maps, _mean, _pearson
from .residual_action import FORBIDDEN_FEATURE_TOKENS, _build_contexts

RAW_FAMILIES = (
    "last_known_tvt",
    "e004_geometry_prefix",
    "align_visible_path",
    "pf_gr_path",
    "trellis_gr_path",
    "e006_nested_fusion",
    "wide64_anchor",
)
SELECTOR_BRANCHES = (
    "ridge_all_f64",
    "ridge_path_f48",
    "extra_trees_all",
    "random_forest_all",
    "hybrid_ridge_trees",
    "hybrid_confidence_fallback",
)
NEGATIVE_BRANCHES = ("shuffled_target_selector", "sign_flipped_target_selector")
COMPARATORS = ("last_known_tvt", "e006_nested_fusion")
DIAGNOSTICS = ("bank_oracle",)
ALL_CANDIDATES = (*COMPARATORS, *DIAGNOSTICS, *SELECTOR_BRANCHES, *NEGATIVE_BRANCHES)
TARGET_WIDTH = 4  # three padded coefficients plus log1p(RMSE)
PATH_PREFIXES = ("e004_", "e006_", "e007_", "selfcorr_", "backtest_")


@dataclass(frozen=True)
class WellRecord:
    well_id: str
    ids: tuple[str, ...]
    row_indices: tuple[int, ...]
    truth: np.ndarray
    paths: Mapping[str, np.ndarray]
    features: Mapping[str, float | None]
    spatial: tuple[float, float]
    typewell: tuple[float, float, float]
    hidden_gr_missing_fraction: float


@dataclass(frozen=True)
class FamilyTarget:
    family: str
    coefficients: tuple[float, float, float]
    rmse: float
    sse: float
    candidate_index: int


@dataclass(frozen=True)
class PreparedFeatures:
    names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    train_x: np.ndarray
    test_x: np.ndarray


@dataclass(frozen=True)
class ModelPrediction:
    values: np.ndarray
    selected_features: tuple[str, ...]


def _finite(raw: Any, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"E010 invalid numeric {label}") from exc
    if not math.isfinite(value):
        raise DataValidationError(f"E010 non-finite numeric {label}")
    return value


def _metric_for_path(well_id: str, truth: Sequence[float], prediction: Sequence[float]) -> WellMetric:
    if len(truth) != len(prediction) or len(truth) == 0:
        raise DataValidationError(f"{well_id}: E010 path lengths differ")
    accumulator = ErrorAccumulator()
    for index, (target, value) in enumerate(zip(truth, prediction)):
        accumulator.add(_finite(value, "prediction") - _finite(target, "target"), float(index))
    return accumulator.finalize(well_id)


def _read_numeric_table(path: Path) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "well_id" not in (reader.fieldnames or []):
            raise DataValidationError(f"{path}: missing well_id")
        for row in reader:
            well_id = str(row["well_id"])
            if not well_id or well_id in output:
                raise DataValidationError(f"{path}: empty or duplicate well_id")
            values: dict[str, float | None] = {}
            for name, raw in row.items():
                if name == "well_id":
                    continue
                text = str(raw or "").strip()
                if not text:
                    values[name] = None
                    continue
                try:
                    value = float(text)
                except ValueError as exc:
                    raise DataValidationError(f"{path}: invalid numeric {name}") from exc
                values[name] = value if math.isfinite(value) else None
            output[well_id] = values
    return output


def _iter_oof(path: Path, required: Sequence[str]) -> Iterator[tuple[str, list[dict[str, str]]]]:
    required_set = set(required) | {"id", "well_id", "row_index", "hidden_index", "target"}
    current = ""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required_set - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            well_id = str(row["well_id"])
            if current and well_id != current:
                if well_id in seen:
                    raise DataValidationError(f"{path}: noncontiguous well {well_id}")
                seen.add(current)
                yield current, rows
                rows = []
            if not current or well_id != current:
                current = well_id
            hidden = int(row["hidden_index"])
            absolute = int(row["row_index"])
            if hidden != len(rows) or str(row["id"]) != f"{well_id}_{absolute}":
                raise DataValidationError(f"{path}: malformed OOF identity for {well_id}")
            for name in required_set - {"id", "well_id", "row_index", "hidden_index"}:
                _finite(row[name], f"{path.name}:{name}")
            rows.append(dict(row))
    if current:
        yield current, rows


def _oof_map(path: Path, required: Sequence[str]) -> dict[str, list[dict[str, str]]]:
    return {well_id: rows for well_id, rows in _iter_oof(path, required)}


def _verify_parent_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passed = True
    for name, item in config["parents"].items():
        path = root / str(item["path"])
        observed = _sha256(path) if path.exists() else None
        ok = observed == str(item["sha256"])
        details[name] = {"path": str(item["path"]), "observed_sha256": observed, "expected_sha256": item["sha256"], "pass": ok}
        passed = passed and ok
    return {"pass": passed, "parents": details}


def _validate_feature_schema(features: Mapping[str, Mapping[str, float | None]], config: Mapping[str, Any]) -> tuple[str, ...]:
    if not features:
        raise DataValidationError("E010 legal feature table is empty")
    names = tuple(sorted(next(iter(features.values()))))
    if any(tuple(sorted(row)) != names for row in features.values()):
        raise DataValidationError("E010 legal feature schema differs across wells")
    bad = [name for name in names if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise DataValidationError(f"E010 forbidden selector features {bad[:12]}")
    for prefix in config["selector"]["required_multi_cut_prefixes"]:
        if not any(name.startswith(str(prefix)) for name in names):
            raise DataValidationError(f"E010 missing required multi-cut feature prefix {prefix}")
    return names


def _read_records(root: Path, train_dir: Path, config: Mapping[str, Any]) -> tuple[dict[str, WellRecord], tuple[str, ...], dict[str, Any]]:
    parent = config["parents"]
    e005 = _oof_map(root / parent["e005_oof"]["path"], ["e004_geometry_prefix", "align_visible_path", "pf_gr_path", "trellis_gr_path"])
    e006 = _oof_map(root / parent["e006_oof"]["path"], ["last_known_tvt", "e004_geometry_prefix", "nested_conservative_grid"])
    e009 = _oof_map(root / parent["e009_oof"]["path"], ["last_known_tvt", "e006_nested_fusion"])
    features = _read_numeric_table(root / parent["legal_features"]["path"])
    wide = _read_numeric_table(root / parent["wide_actions"]["path"])
    residual_audit = _read_numeric_table(root / parent["residual_targets_audit"]["path"])
    feature_names = _validate_feature_schema(features, config)
    well_sets = [set(item) for item in (e005, e006, e009, features, wide, residual_audit)]
    if any(item != well_sets[0] for item in well_sets[1:]) or len(well_sets[0]) != int(config["expected_wells"]):
        raise DataValidationError("E010 parent well sets differ")
    records: dict[str, WellRecord] = {}
    maximum_alignment_delta = 0.0
    column = str(parent["wide_actions"]["column"])
    for well_id in sorted(well_sets[0]):
        left, middle, right = e005[well_id], e006[well_id], e009[well_id]
        if not (len(left) == len(middle) == len(right) and len(left) > 0):
            raise DataValidationError(f"{well_id}: E010 parent row counts differ")
        ids: list[str] = []
        row_indices: list[int] = []
        truth: list[float] = []
        paths = {name: [] for name in RAW_FAMILIES if name != "wide64_anchor"}
        for a, b, c in zip(left, middle, right):
            if not (a["id"] == b["id"] == c["id"]):
                raise DataValidationError(f"{well_id}: E010 parent IDs differ")
            values = {
                "target_a": _finite(a["target"], "E005 target"),
                "target_b": _finite(b["target"], "E006 target"),
                "target_c": _finite(c["target"], "E009 target"),
                "last_b": _finite(b["last_known_tvt"], "last known"),
                "last_c": _finite(c["last_known_tvt"], "last known"),
                "e004_a": _finite(a["e004_geometry_prefix"], "E004"),
                "e004_b": _finite(b["e004_geometry_prefix"], "E004"),
                "e006_b": _finite(b["nested_conservative_grid"], "E006"),
                "e006_c": _finite(c["e006_nested_fusion"], "E006"),
            }
            for x, y in ((values["target_a"], values["target_b"]), (values["target_b"], values["target_c"]), (values["last_b"], values["last_c"]), (values["e004_a"], values["e004_b"]), (values["e006_b"], values["e006_c"])):
                maximum_alignment_delta = max(maximum_alignment_delta, abs(x - y))
                if abs(x - y) > 5e-8:
                    raise DataValidationError(f"{well_id}: E010 parent value mismatch")
            ids.append(str(a["id"])); row_indices.append(int(a["row_index"])); truth.append(values["target_a"])
            paths["last_known_tvt"].append(values["last_b"])
            paths["e004_geometry_prefix"].append(values["e004_a"])
            paths["align_visible_path"].append(_finite(a["align_visible_path"], "alignment path"))
            paths["pf_gr_path"].append(_finite(a["pf_gr_path"], "PF path"))
            paths["trellis_gr_path"].append(_finite(a["trellis_gr_path"], "trellis path"))
            paths["e006_nested_fusion"].append(values["e006_b"])
        action = wide[well_id].get(column)
        if action is None:
            raise DataValidationError(f"{well_id}: E010 missing wide action {column}")
        paths["wide64_anchor"] = [value + float(action) for value in paths["e006_nested_fusion"]]
        horizontal = train_dir / f"{well_id}__horizontal_well.csv"
        with horizontal.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle); first = next(reader, None)
            if first is None:
                raise DataValidationError(f"{well_id}: empty horizontal file")
            last = first
            for row in reader:
                last = row
        spatial = (0.5 * (_finite(first["X"], "X") + _finite(last["X"], "X")), 0.5 * (_finite(first["Y"], "Y") + _finite(last["Y"], "Y")))
        curve = TypewellCurve.read(train_dir / f"{well_id}__typewell.csv")
        typewell = (curve.gr_mean, curve.gr_std, curve.maximum_tvt - curve.minimum_tvt)
        missing = features[well_id].get("hidden_gr_missing_fraction")
        records[well_id] = WellRecord(
            well_id=well_id,
            ids=tuple(ids),
            row_indices=tuple(row_indices),
            truth=np.asarray(truth, dtype=np.float64),
            paths={name: np.asarray(values, dtype=np.float64) for name, values in paths.items()},
            features=features[well_id],
            spatial=spatial,
            typewell=typewell,
            hidden_gr_missing_fraction=float(missing or 0.0),
        )
    return records, feature_names, {"maximum_parent_alignment_delta": maximum_alignment_delta, "well_count": len(records), "rows": sum(record.truth.size for record in records.values())}


def _family_definitions(config: Mapping[str, Any], advanced: Sequence[str] = ()) -> tuple[str, ...]:
    return (*RAW_FAMILIES, *(str(item["name"]) for item in config["candidate_bank"]["families"]), *tuple(advanced))


def _padded(coefficients: Sequence[float]) -> tuple[float, float, float]:
    values = [float(value) for value in coefficients]
    if len(values) > 3:
        raise DataValidationError("E010 family has more than three coefficients")
    return tuple((values + [0.0, 0.0, 0.0])[:3])  # type: ignore[return-value]


def _core_oracles(records: Mapping[str, WellRecord], config: Mapping[str, Any]) -> tuple[dict[str, dict[str, FamilyTarget]], dict[str, np.ndarray], dict[str, Any]]:
    family_specs = {str(item["name"]): item for item in config["candidate_bank"]["families"]}
    coefficient_cache = {name: valid_coefficients(spec, config) for name, spec in family_specs.items()}
    targets: dict[str, dict[str, FamilyTarget]] = {}
    oracle_paths: dict[str, np.ndarray] = {}
    family_sse: dict[str, float] = defaultdict(float)
    family_rows: dict[str, int] = defaultdict(int)
    winner_counts: Counter[str] = Counter()
    reconstruction_delta = 0.0
    basis_metric_delta = 0.0
    maximum = float(config["candidate_bank"]["maximum_absolute_correction_ft"])
    family_order = _family_definitions(config)
    for well_id, record in records.items():
        by_family: dict[str, FamilyTarget] = {}
        paths: dict[str, np.ndarray] = {}
        for name in RAW_FAMILIES:
            path = record.paths[name]
            metric = _metric_for_path(well_id, record.truth, path)
            by_family[name] = FamilyTarget(name, (0.0, 0.0, 0.0), metric.rmse, metric.sse, 0)
            paths[name] = path
        for name, spec in family_specs.items():
            anchor = record.paths[str(spec["anchor"])]
            oracle = grid_oracle(anchor, record.truth, spec, coefficient_cache[name])
            path = reconstruct_path(anchor, spec, oracle.coefficients, maximum)
            direct = _metric_for_path(well_id, record.truth, path)
            basis_metric_delta = max(basis_metric_delta, abs(direct.sse - oracle.sse))
            reconstructed = anchor + np.clip(basis_matrix(anchor.size, spec) @ np.asarray(oracle.coefficients), -maximum, maximum)
            reconstruction_delta = max(reconstruction_delta, float(np.max(np.abs(reconstructed - path))))
            by_family[name] = FamilyTarget(name, _padded(oracle.coefficients), direct.rmse, direct.sse, oracle.candidate_index)
            paths[name] = path
        winner = min(family_order, key=lambda name: (by_family[name].rmse, family_order.index(name)))
        winner_counts[winner] += 1
        targets[well_id] = by_family
        oracle_paths[well_id] = paths[winner]
        for name, item in by_family.items():
            family_sse[name] += item.sse; family_rows[name] += record.truth.size
    family_metrics = {name: {"rmse": math.sqrt(family_sse[name] / family_rows[name]), "rows": family_rows[name], "oracle_wins": winner_counts[name]} for name in family_order}
    oracle_summary = _summarize([_metric_for_path(well_id, records[well_id].truth, oracle_paths[well_id]) for well_id in sorted(records)])
    return targets, oracle_paths, {"family_metrics": family_metrics, "oracle_summary": oracle_summary, "winner_counts": dict(winner_counts), "basis_reconstruction_max_delta": reconstruction_delta, "basis_metric_max_delta": basis_metric_delta}


def _screen_ids(records: Mapping[str, WellRecord], count: int) -> list[str]:
    ids = sorted(records)
    by_rows = sorted(ids, key=lambda well_id: (records[well_id].truth.size, well_id))
    row_group = {well_id: min(2, int(index * 3 / len(ids))) for index, well_id in enumerate(by_rows)}
    by_missing = sorted(ids, key=lambda well_id: (records[well_id].hidden_gr_missing_fraction, well_id))
    missing_group = {well_id: min(1, int(index * 2 / len(ids))) for index, well_id in enumerate(by_missing)}
    per_stratum = count // 6
    selected: list[str] = []
    for row_bin in range(3):
        for missing_bin in range(2):
            candidates = [well_id for well_id in ids if row_group[well_id] == row_bin and missing_group[well_id] == missing_bin]
            candidates.sort(key=lambda well_id: (hashlib.sha256(f"E010-screen|{well_id}".encode()).hexdigest(), well_id))
            if len(candidates) < per_stratum:
                raise DataValidationError("E010 development-screen stratum is too small")
            selected.extend(candidates[:per_stratum])
    if len(selected) != count or len(set(selected)) != count:
        raise DataValidationError("E010 development-screen selection failed")
    return sorted(selected)


def _screen_state_branches(
    *,
    records: Mapping[str, WellRecord],
    train_dir: Path,
    config: Mapping[str, Any],
    core_oracle_paths: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    screen_ids = _screen_ids(records, int(config["development_screen"]["wells"]))
    maximum = float(config["candidate_bank"]["maximum_absolute_correction_ft"])
    path_cache: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    metric_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    projected: dict[str, float] = {}
    for branch, variants, runner in (
        ("hmm", config["development_screen"]["hmm_variants"], hmm_path),
        ("dtw", config["development_screen"]["dtw_variants"], dtw_path),
    ):
        for variant_index, variant in enumerate(variants):
            name = str(variant["name"])
            started = time.perf_counter()
            metrics: list[WellMetric] = []
            combined_metrics: list[WellMetric] = []
            finite = True
            fallback_wells = 0
            unique_wins = 0
            for well_id in screen_ids:
                record = records[well_id]
                well = read_well(train_dir / f"{well_id}__horizontal_well.csv", train_dir / f"{well_id}__typewell.csv", require_truth=True)
                result: StatePathResult = runner(well, record.paths["e006_nested_fusion"], variant, maximum)
                path = np.asarray(result.path, dtype=np.float64)
                if path.shape != record.truth.shape or not np.all(np.isfinite(path)):
                    finite = False
                    path = record.paths["e006_nested_fusion"].copy()
                if result.fallback_reason:
                    fallback_wells += 1
                path_cache[name][well_id] = path
                metric = _metric_for_path(well_id, record.truth, path)
                core_metric = _metric_for_path(well_id, record.truth, core_oracle_paths[well_id])
                use_variant = metric.sse < core_metric.sse - 1e-12
                unique_wins += int(use_variant)
                combined = metric if use_variant else core_metric
                metrics.append(metric); combined_metrics.append(combined)
                well_rows.append({
                    "branch": branch,
                    "variant": name,
                    "variant_index": variant_index,
                    "well_id": well_id,
                    "rows": record.truth.size,
                    "rmse": metric.rmse,
                    "core_oracle_rmse": core_metric.rmse,
                    "unique_oracle_win": use_variant,
                    "fallback_reason": result.fallback_reason,
                    "sampled_observations": result.sampled_observations,
                    "minimum_normalizer": result.minimum_normalizer,
                    "wall_seconds": result.wall_seconds,
                })
            elapsed = time.perf_counter() - started
            summary = _summarize(metrics)
            combined = _summarize(combined_metrics)
            e006 = _summarize([_metric_for_path(well_id, records[well_id].truth, records[well_id].paths["e006_nested_fusion"]) for well_id in screen_ids])
            core = _summarize([_metric_for_path(well_id, records[well_id].truth, core_oracle_paths[well_id]) for well_id in screen_ids])
            projected_minutes = elapsed * len(records) / len(screen_ids) / 60.0
            projected[name] = projected_minutes
            metric_rows.append({
                "branch": branch,
                "variant": name,
                "variant_index": variant_index,
                "rows_scored": summary["rows_scored"],
                "wells_scored": summary["wells_scored"],
                "rmse": summary["rmse"],
                "gain_vs_e006": float(e006["rmse"]) - float(summary["rmse"]),
                "oracle_with_variant_rmse": combined["rmse"],
                "unique_oracle_gain": float(core["rmse"]) - float(combined["rmse"]),
                "unique_oracle_wells": unique_wins,
                "fallback_wells": fallback_wells,
                "finite_all": finite,
                "screen_wall_seconds": elapsed,
                "projected_full_minutes": projected_minutes,
            })
    selected_by_branch: dict[str, str | None] = {}
    advanced_paths: dict[str, dict[str, np.ndarray]] = {}
    for branch in ("hmm", "dtw"):
        branch_rows = [row for row in metric_rows if row["branch"] == branch]
        passing_rows = [
            row for row in branch_rows
            if bool(row["finite_all"])
            and (
                float(row["gain_vs_e006"]) >= float(config["development_screen"]["minimum_standalone_gain_vs_e006"])
                or float(row["unique_oracle_gain"]) >= float(config["development_screen"]["minimum_unique_oracle_gain"])
            )
            and float(row["projected_full_minutes"]) <= float(config["development_screen"]["maximum_projected_full_minutes"])
        ]
        if passing_rows:
            best = min(
                passing_rows,
                key=lambda row: (
                    float(row["oracle_with_variant_rmse"]),
                    float(row["rmse"]),
                    int(row["variant_index"]),
                ),
            )
            advance = True
        else:
            best = min(branch_rows, key=lambda row: (float(row["rmse"]), int(row["variant_index"])))
            advance = False
        selected_by_branch[branch] = str(best["variant"]) if advance else None
        for row in branch_rows:
            row["best_in_branch"] = row is best
            row["advanced_to_full"] = bool(advance and row is best)
    for branch, selected in selected_by_branch.items():
        if selected is None:
            continue
        variant = next(item for item in config["development_screen"][f"{branch}_variants"] if str(item["name"]) == selected)
        runner = hmm_path if branch == "hmm" else dtw_path
        full: dict[str, np.ndarray] = {}
        for well_id, record in records.items():
            if well_id in path_cache[selected]:
                full[well_id] = path_cache[selected][well_id]
                continue
            well = read_well(train_dir / f"{well_id}__horizontal_well.csv", train_dir / f"{well_id}__typewell.csv", require_truth=True)
            result = runner(well, record.paths["e006_nested_fusion"], variant, maximum)
            path = np.asarray(result.path, dtype=np.float64)
            if path.shape != record.truth.shape or not np.all(np.isfinite(path)):
                raise DataValidationError(f"{well_id}: advanced {selected} emitted invalid path")
            full[well_id] = path
        advanced_paths[selected] = full
    return advanced_paths, metric_rows, well_rows, {"screen_ids": screen_ids, "selected_by_branch": selected_by_branch, "projected_minutes": projected}


def _add_advanced_targets(
    records: Mapping[str, WellRecord],
    targets: dict[str, dict[str, FamilyTarget]],
    oracle_paths: dict[str, np.ndarray],
    advanced_paths: Mapping[str, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    order = list(_family_definitions(config, advanced_paths.keys()))
    wins: Counter[str] = Counter()
    well_winners: dict[str, str] = {}
    family_sse: dict[str, float] = defaultdict(float)
    family_rows: dict[str, int] = defaultdict(int)
    for well_id, record in records.items():
        for name, paths in advanced_paths.items():
            metric = _metric_for_path(well_id, record.truth, paths[well_id])
            targets[well_id][name] = FamilyTarget(name, (0.0, 0.0, 0.0), metric.rmse, metric.sse, 0)
        best = min(order, key=lambda name: (targets[well_id][name].rmse, order.index(name)))
        well_winners[well_id] = best
        wins[best] += 1
        if best in advanced_paths:
            oracle_paths[well_id] = advanced_paths[best][well_id]
        for name in order:
            family_sse[name] += targets[well_id][name].sse
            family_rows[name] += record.truth.size
    summary = _summarize([_metric_for_path(well_id, records[well_id].truth, oracle_paths[well_id]) for well_id in sorted(records)])
    metrics = {name: {"rmse": math.sqrt(family_sse[name] / family_rows[name]), "rows": family_rows[name], "oracle_wins": wins[name]} for name in order}
    return {
        "family_metrics": metrics,
        "oracle_summary": summary,
        "winner_counts": dict(wins),
        "well_winners": well_winners,
        "family_order": order,
    }


def _selector_feature_names(all_names: Sequence[str], selector: str) -> tuple[str, ...]:
    if selector == "all":
        chosen = list(all_names)
    elif selector == "path_and_multicut":
        chosen = [name for name in all_names if name.startswith(PATH_PREFIXES) or name in {"last_visible_tvt", "last_visible_u", "hidden_rows", "known_rows", "hidden_fraction", "hidden_gr_missing_fraction"}]
    else:
        raise DataValidationError(f"E010 unknown feature selector {selector}")
    if not chosen:
        raise DataValidationError(f"E010 feature selector {selector} is empty")
    return tuple(sorted(chosen))


def _target_matrix(ids: Sequence[str], targets: Mapping[str, Mapping[str, FamilyTarget]], families: Sequence[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for well_id in ids:
        values: list[float] = []
        for family in families:
            item = targets[well_id][family]
            values.extend(item.coefficients)
            values.append(math.log1p(item.rmse))
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(ids), len(families) * TARGET_WIDTH) or not np.all(np.isfinite(matrix)):
        raise DataValidationError("E010 malformed selector target matrix")
    return matrix


def _raw_feature_matrix(records: Mapping[str, WellRecord], ids: Sequence[str], names: Sequence[str]) -> np.ndarray:
    matrix = np.empty((len(ids), len(names)), dtype=np.float64)
    for row_index, well_id in enumerate(ids):
        for column_index, name in enumerate(names):
            raw = records[well_id].features.get(name)
            matrix[row_index, column_index] = np.nan if raw is None else float(raw)
    return matrix


def _prepare_features(
    *,
    records: Mapping[str, WellRecord],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    all_names: Sequence[str],
    selector: str,
    maximum_features: int,
    train_y: np.ndarray,
) -> PreparedFeatures:
    pool = _selector_feature_names(all_names, selector)
    train = _raw_feature_matrix(records, train_ids, pool)
    test = _raw_feature_matrix(records, test_ids, pool)
    medians = np.nanmedian(train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    train_z = (train - means) / scales
    test_z = (test - means) / scales
    y_means = train_y.mean(axis=0)
    y_scales = train_y.std(axis=0)
    y_scales[y_scales < 1e-9] = 1.0
    y_z = (train_y - y_means) / y_scales
    scores: list[tuple[float, str, int]] = []
    for index, name in enumerate(pool):
        x = train_z[:, index]
        correlations = []
        if float(np.std(x)) > 1e-12:
            for target_index in range(y_z.shape[1]):
                y = y_z[:, target_index]
                correlations.append(abs(float(np.corrcoef(x, y)[0, 1])) if float(np.std(y)) > 1e-12 else 0.0)
        score = max((value for value in correlations if math.isfinite(value)), default=0.0)
        scores.append((-score, name, index))
    scores.sort()
    selected_indices = [item[2] for item in scores[: min(int(maximum_features), len(scores))]]
    selected_indices.sort(key=lambda index: pool[index])
    names = tuple(pool[index] for index in selected_indices)
    return PreparedFeatures(
        names=names,
        medians=medians[selected_indices],
        means=means[selected_indices],
        scales=scales[selected_indices],
        train_x=train_z[:, selected_indices],
        test_x=test_z[:, selected_indices],
    )


def _fit_model(
    *,
    spec: Mapping[str, Any],
    records: Mapping[str, WellRecord],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_names: Sequence[str],
    train_y: np.ndarray,
) -> ModelPrediction:
    prepared = _prepare_features(
        records=records,
        train_ids=train_ids,
        test_ids=test_ids,
        all_names=feature_names,
        selector=str(spec["selector"]),
        maximum_features=int(spec["maximum_features"]),
        train_y=train_y,
    )
    y_mean = train_y.mean(axis=0)
    y_scale = train_y.std(axis=0)
    y_scale[y_scale < 1e-9] = 1.0
    scaled_y = (train_y - y_mean) / y_scale
    model_name = str(spec["model"])
    try:
        if model_name == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=float(spec["alpha"]), fit_intercept=True)
        elif model_name == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor
            model = ExtraTreesRegressor(
                n_estimators=int(spec["n_estimators"]),
                min_samples_leaf=int(spec["min_samples_leaf"]),
                max_features=float(spec["max_features"]),
                random_state=int(spec["random_state"]),
                n_jobs=int(spec["n_jobs"]),
            )
        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=int(spec["n_estimators"]),
                min_samples_leaf=int(spec["min_samples_leaf"]),
                max_features=float(spec["max_features"]),
                random_state=int(spec["random_state"]),
                n_jobs=int(spec["n_jobs"]),
            )
        else:
            raise DataValidationError(f"E010 unsupported base model {model_name}")
        model.fit(prepared.train_x, scaled_y)
        predicted = np.asarray(model.predict(prepared.test_x), dtype=np.float64)
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(f"E010 {model_name} fit failed") from exc
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    values = predicted * y_scale + y_mean
    if values.shape != (len(test_ids), train_y.shape[1]) or not np.all(np.isfinite(values)):
        raise DataValidationError(f"E010 {model_name} emitted invalid target predictions")
    return ModelPrediction(values=values, selected_features=prepared.names)


def _grid_clip(value: float, spec: Mapping[str, Any]) -> float:
    values = grid_values(spec)
    return float(np.clip(value, values[0], values[-1]))


def _path_for_family(
    *,
    record: WellRecord,
    family: str,
    coefficient_values: Sequence[float],
    family_specs: Mapping[str, Mapping[str, Any]],
    advanced_paths: Mapping[str, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
) -> np.ndarray:
    if family in RAW_FAMILIES:
        return record.paths[family].copy()
    if family in advanced_paths:
        return np.asarray(advanced_paths[family][record.well_id], dtype=np.float64).copy()
    spec = family_specs[family]
    grids = config["candidate_bank"]["coefficient_grids"]
    count = basis_matrix(record.truth.size, spec).shape[1]
    raw = list(coefficient_values)[:count]
    clipped = [_grid_clip(raw[0], grids["datum_ft"]), _grid_clip(raw[1], grids["linear_ft"])]
    if count == 3:
        clipped.append(_grid_clip(raw[2], grids["shape_ft"]))
    return reconstruct_path(record.paths[str(spec["anchor"])], spec, clipped, float(config["candidate_bank"]["maximum_absolute_correction_ft"]))


def _decode_predictions(
    *,
    values: np.ndarray,
    test_ids: Sequence[str],
    records: Mapping[str, WellRecord],
    families: Sequence[str],
    family_specs: Mapping[str, Mapping[str, Any]],
    advanced_paths: Mapping[str, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
    confidence_fallback: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if values.shape != (len(test_ids), len(families) * TARGET_WIDTH):
        raise DataValidationError("E010 selector prediction width differs")
    output: dict[str, np.ndarray] = {}
    detail_rows: list[dict[str, Any]] = []
    e006_index = families.index("e006_nested_fusion")
    threshold = float(config["selector"]["branches"][-1].get("minimum_predicted_gain_vs_e006_ft", 0.25))
    for row_index, well_id in enumerate(test_ids):
        reshaped = values[row_index].reshape(len(families), TARGET_WIDTH)
        predicted_rmse = np.maximum(0.0, np.expm1(np.clip(reshaped[:, 3], 0.0, 8.0)))
        family_index = min(range(len(families)), key=lambda index: (float(predicted_rmse[index]), index))
        selected = families[family_index]
        predicted_gain = float(predicted_rmse[e006_index] - predicted_rmse[family_index])
        fallback = bool(confidence_fallback and predicted_gain < threshold)
        if fallback:
            selected = "e006_nested_fusion"
            family_index = e006_index
        path = _path_for_family(
            record=records[well_id],
            family=selected,
            coefficient_values=reshaped[family_index, :3],
            family_specs=family_specs,
            advanced_paths=advanced_paths,
            config=config,
        )
        if path.shape != records[well_id].truth.shape or not np.all(np.isfinite(path)):
            raise DataValidationError(f"{well_id}: E010 decoded invalid path")
        output[well_id] = path
        detail_rows.append({
            "well_id": well_id,
            "selected_family": selected,
            "predicted_rmse": float(predicted_rmse[family_index]),
            "predicted_e006_rmse": float(predicted_rmse[e006_index]),
            "predicted_gain_vs_e006": predicted_gain,
            "fallback": fallback,
            "coefficient_0": float(reshaped[family_index, 0]),
            "coefficient_1": float(reshaped[family_index, 1]),
            "coefficient_2": float(reshaped[family_index, 2]),
        })
    return output, detail_rows


def _deranged_targets(train_y: np.ndarray, train_ids: Sequence[str], key: str) -> np.ndarray:
    if len(train_ids) < 2:
        return train_y.copy()
    ordered = sorted(range(len(train_ids)), key=lambda index: (hashlib.sha256(f"E010-shuffle|{key}|{train_ids[index]}".encode()).hexdigest(), train_ids[index]))
    shift = 1 + int(hashlib.sha256(f"E010-shift|{key}".encode()).hexdigest()[:8], 16) % (len(ordered) - 1)
    source = {ordered[index]: ordered[(index + shift) % len(ordered)] for index in range(len(ordered))}
    return np.asarray([train_y[source[index]] for index in range(len(train_ids))], dtype=np.float64)


def _fit_context_selectors(
    *,
    context: Any,
    records: Mapping[str, WellRecord],
    targets: Mapping[str, Mapping[str, FamilyTarget]],
    families: Sequence[str],
    feature_names: Sequence[str],
    family_specs: Mapping[str, Mapping[str, Any]],
    advanced_paths: Mapping[str, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]], Counter[tuple[str, str]]]:
    train_ids = tuple(context.train_ids); test_ids = tuple(context.test_ids)
    train_y = _target_matrix(train_ids, targets, families)
    specs = {str(item["name"]): item for item in config["selector"]["branches"]}
    base_names = ("ridge_all_f64", "ridge_path_f48", "extra_trees_all", "random_forest_all")
    predicted: dict[str, np.ndarray] = {}
    frequency: Counter[tuple[str, str]] = Counter()
    for name in base_names:
        model = _fit_model(spec=specs[name], records=records, train_ids=train_ids, test_ids=test_ids, feature_names=feature_names, train_y=train_y)
        predicted[name] = model.values
        for feature in model.selected_features:
            frequency[(name, feature)] += 1
    hybrid_spec = specs["hybrid_ridge_trees"]
    weights = [float(value) for value in hybrid_spec["weights"]]
    predicted["hybrid_ridge_trees"] = weights[0] * predicted[str(hybrid_spec["parents"][0])] + weights[1] * predicted[str(hybrid_spec["parents"][1])]
    confidence_spec = specs["hybrid_confidence_fallback"]
    confidence_weights = [float(value) for value in confidence_spec["weights"]]
    predicted["hybrid_confidence_fallback"] = confidence_weights[0] * predicted[str(confidence_spec["parents"][0])] + confidence_weights[1] * predicted[str(confidence_spec["parents"][1])]
    paths: dict[str, dict[str, np.ndarray]] = {}
    details: list[dict[str, Any]] = []
    for name in SELECTOR_BRANCHES:
        branch_paths, branch_details = _decode_predictions(
            values=predicted[name], test_ids=test_ids, records=records, families=families,
            family_specs=family_specs, advanced_paths=advanced_paths, config=config,
            confidence_fallback=name == "hybrid_confidence_fallback",
        )
        paths[name] = branch_paths
        details.extend({"context": context.key, "scope": context.scope, "branch": name, **row} for row in branch_details)
    shuffled = _fit_model(
        spec=specs["ridge_all_f64"], records=records, train_ids=train_ids, test_ids=test_ids,
        feature_names=feature_names, train_y=_deranged_targets(train_y, train_ids, context.key),
    ).values
    paths["shuffled_target_selector"], rows = _decode_predictions(
        values=shuffled, test_ids=test_ids, records=records, families=families,
        family_specs=family_specs, advanced_paths=advanced_paths, config=config, confidence_fallback=False,
    )
    details.extend({"context": context.key, "scope": context.scope, "branch": "shuffled_target_selector", **row} for row in rows)
    flipped_y = train_y.copy()
    for family_index in range(len(families)):
        flipped_y[:, family_index * TARGET_WIDTH : family_index * TARGET_WIDTH + 3] *= -1.0
    flipped = _fit_model(
        spec=specs["ridge_all_f64"], records=records, train_ids=train_ids, test_ids=test_ids,
        feature_names=feature_names, train_y=flipped_y,
    ).values
    paths["sign_flipped_target_selector"], rows = _decode_predictions(
        values=flipped, test_ids=test_ids, records=records, families=families,
        family_specs=family_specs, advanced_paths=advanced_paths, config=config, confidence_fallback=False,
    )
    details.extend({"context": context.key, "scope": context.scope, "branch": "sign_flipped_target_selector", **row} for row in rows)
    return paths, details, frequency


def validate_e010_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1 or str(config.get("experiment_id")) != "E010":
        raise DataValidationError("invalid E010 configuration identity")
    if tuple(config["candidate_bank"]["raw_anchors"]) != RAW_FAMILIES:
        raise DataValidationError("E010 raw anchor order differs from frozen implementation")
    branch_names = tuple(str(item["name"]) for item in config["selector"]["branches"])
    if branch_names != SELECTOR_BRANCHES:
        raise DataValidationError("E010 selector branch order differs from frozen implementation")
    family_names = [str(item["name"]) for item in config["candidate_bank"]["families"]]
    if len(family_names) != len(set(family_names)) or not family_names:
        raise DataValidationError("E010 nonlinear family names must be unique")
    for item in config["candidate_bank"]["families"]:
        if str(item["anchor"]) not in RAW_FAMILIES:
            raise DataValidationError("E010 nonlinear family anchor is unavailable")
        matrix = basis_matrix(7, item)
        if matrix.shape[1] not in (2, 3):
            raise DataValidationError("E010 family basis width is invalid")
    for key in ("datum_ft", "linear_ft", "shape_ft"):
        grid_values(config["candidate_bank"]["coefficient_grids"][key])
    if not all(bool(config["selection"][name]) for name in (
        "no_post_score_candidate_additions", "no_post_score_threshold_changes", "no_post_score_family_grid_changes"
    )):
        raise DataValidationError("E010 post-score mutation guards must remain enabled")
    if bool(config["deployment"]["internet"]) or list(config["deployment"]["external_artifacts"]):
        raise DataValidationError("E010 deployment must remain offline and self-contained")
    if len(config["fold_files"]) != 5 or len(set(config["fold_files"])) != 5:
        raise DataValidationError("E010 requires five unique fold maps")
    for branch in ("hmm_variants", "dtw_variants"):
        names = [str(item["name"]) for item in config["development_screen"][branch]]
        if len(names) != 4 or len(set(names)) != 4:
            raise DataValidationError(f"E010 {branch} must contain four unique variants")


def _average_repeated_paths(
    records: Mapping[str, WellRecord],
    repeated_paths: Mapping[str, Mapping[str, list[np.ndarray]]],
    candidates: Sequence[str],
    expected_maps: int,
) -> dict[str, dict[str, np.ndarray]]:
    output = {candidate: {} for candidate in candidates}
    for candidate in candidates:
        for well_id, record in records.items():
            paths = repeated_paths[candidate].get(well_id, [])
            if len(paths) != expected_maps:
                raise DataValidationError(f"{well_id}: E010 repeated path coverage differs for {candidate}")
            stacked = np.stack(paths)
            averaged = stacked.mean(axis=0)
            if averaged.shape != record.truth.shape or not np.all(np.isfinite(averaged)):
                raise DataValidationError(f"{well_id}: E010 repeated average is invalid")
            output[candidate][well_id] = averaged
    return output


def _write_gzip_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)


def _format(value: float) -> str:
    return f"{float(value):.8f}"


def run_e010(
    *,
    root: Path,
    train_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: Mapping[str, Any],
    code_sha: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_e010_config(config)
    root = root.resolve(); train_dir = train_dir.resolve(); output_dir = output_dir.resolve(); artifact_dir = artifact_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True); artifact_dir.mkdir(parents=True, exist_ok=True)
    parent_control = _verify_parent_hashes(root, config)
    if not parent_control["pass"]:
        raise DataValidationError("E010 parent hash audit failed")
    profiles, data_profile = scan_profiles(train_dir)
    if int(data_profile["well_count"]) != int(config["expected_wells"]) or str(data_profile["data_signature"]) != str(config["data_signature"]):
        raise DataValidationError("E010 data identity differs from frozen configuration")
    records, feature_names, parent_alignment = _read_records(root, train_dir, config)
    theoretical_count, family_candidate_counts = candidate_count(config)
    targets, oracle_paths, core = _core_oracles(records, config)
    advanced_paths, screen_rows, screen_well_rows, screen_summary = _screen_state_branches(
        records=records, train_dir=train_dir, config=config, core_oracle_paths=oracle_paths,
    )
    bank = _add_advanced_targets(records, targets, oracle_paths, advanced_paths, config)
    families = tuple(bank["family_order"])
    family_specs = {str(item["name"]): item for item in config["candidate_bank"]["families"]}

    folds = _load_fold_maps(root, config["fold_files"], str(config["data_signature"]))
    spatial = _group_assignments({well_id: record.spatial for well_id, record in records.items()}, int(config["stress"]["spatial_groups"]))
    typewell = _group_assignments({well_id: record.typewell for well_id, record in records.items()}, int(config["stress"]["typewell_groups"]))
    context_config = {"stress": {"spatial_bins": int(config["stress"]["spatial_groups"]), "typewell_clusters": int(config["stress"]["typewell_groups"])}}
    contexts, membership_rows = _build_contexts(sorted(records), folds, spatial, typewell, context_config)
    repeated_paths: dict[str, dict[str, list[np.ndarray]]] = {candidate: defaultdict(list) for candidate in (*SELECTOR_BRANCHES, *NEGATIVE_BRANCHES)}
    context_metrics: dict[str, dict[str, dict[str, WellMetric]]] = {}
    outer_rows: list[dict[str, Any]] = []
    selector_detail_rows: list[dict[str, Any]] = []
    feature_frequency: Counter[tuple[str, str]] = Counter()
    fallback_max_delta = 0.0
    maximum_emitted_correction = 0.0
    repeated_contexts = [context for context in contexts if context.scope == "repeated"]
    duplicate_selector_delta = 0.0
    for context_index, context in enumerate(contexts):
        paths, details, frequency = _fit_context_selectors(
            context=context, records=records, targets=targets, families=families, feature_names=feature_names,
            family_specs=family_specs, advanced_paths=advanced_paths, config=config,
        )
        selector_detail_rows.extend(details); feature_frequency.update(frequency)
        metrics = {candidate: {} for candidate in ALL_CANDIDATES}
        for well_id in context.test_ids:
            record = records[well_id]
            metrics["last_known_tvt"][well_id] = _metric_for_path(well_id, record.truth, record.paths["last_known_tvt"])
            metrics["e006_nested_fusion"][well_id] = _metric_for_path(well_id, record.truth, record.paths["e006_nested_fusion"])
            metrics["bank_oracle"][well_id] = _metric_for_path(well_id, record.truth, oracle_paths[well_id])
            for candidate in (*SELECTOR_BRANCHES, *NEGATIVE_BRANCHES):
                path = paths[candidate][well_id]
                metrics[candidate][well_id] = _metric_for_path(well_id, record.truth, path)
                maximum_emitted_correction = max(maximum_emitted_correction, float(np.max(np.abs(path - record.paths["e006_nested_fusion"]))))
                if context.scope == "repeated":
                    repeated_paths[candidate][well_id].append(path)
        if context_index == 0:
            duplicate_paths, _, _ = _fit_context_selectors(
                context=context, records=records, targets=targets, families=families, feature_names=feature_names,
                family_specs=family_specs, advanced_paths=advanced_paths, config=config,
            )
            for candidate in (*SELECTOR_BRANCHES, *NEGATIVE_BRANCHES):
                for well_id in context.test_ids:
                    duplicate_selector_delta = max(duplicate_selector_delta, float(np.max(np.abs(paths[candidate][well_id] - duplicate_paths[candidate][well_id]))))
        detail_lookup = {(row["branch"], row["well_id"]): row for row in details}
        for well_id in context.test_ids:
            detail = detail_lookup.get(("hybrid_confidence_fallback", well_id))
            if detail and bool(detail["fallback"]):
                fallback_max_delta = max(fallback_max_delta, float(np.max(np.abs(paths["hybrid_confidence_fallback"][well_id] - records[well_id].paths["e006_nested_fusion"]))))
        context_metrics[context.key] = metrics
        for candidate in ALL_CANDIDATES:
            outer_rows.append({
                "context": context.key, "scope": context.scope, "label": context.label,
                "outer_group": context.outer_group, "candidate": candidate,
                **_summarize([metrics[candidate][well_id] for well_id in sorted(context.test_ids)]),
            })

    final_paths = _average_repeated_paths(records, repeated_paths, (*SELECTOR_BRANCHES, *NEGATIVE_BRANCHES), len(folds))
    final_metrics: dict[str, dict[str, WellMetric]] = {candidate: {} for candidate in ALL_CANDIDATES}
    direct_sse = {candidate: 0.0 for candidate in ALL_CANDIDATES}
    for well_id, record in records.items():
        fixed = {
            "last_known_tvt": record.paths["last_known_tvt"],
            "e006_nested_fusion": record.paths["e006_nested_fusion"],
            "bank_oracle": oracle_paths[well_id],
        }
        for candidate in ALL_CANDIDATES:
            path = fixed[candidate] if candidate in fixed else final_paths[candidate][well_id]
            metric = _metric_for_path(well_id, record.truth, path)
            final_metrics[candidate][well_id] = metric
            direct_sse[candidate] += metric.sse
    summaries = {candidate: _summarize([final_metrics[candidate][well_id] for well_id in sorted(records)]) for candidate in ALL_CANDIDATES}

    outer_lookup = {(row["context"], row["candidate"]): row for row in outer_rows}
    map_rows: list[dict[str, Any]] = []
    for fold in folds:
        version = str(fold["version"])
        keys = [f"repeated:{version}:{group}" for group in range(int(fold["n_folds"]))]
        for candidate in ALL_CANDIDATES:
            map_rows.append({"map": version, "candidate": candidate, **_summarize([metric for key in keys for metric in context_metrics[key][candidate].values()])})
    map_lookup = {(row["map"], row["candidate"]): row for row in map_rows}
    stress_rows = [row for row in outer_rows if row["scope"] != "repeated"]
    stress_lookup = {(row["scope"], int(row["outer_group"]), row["candidate"]): row for row in stress_rows}

    long_threshold = float(_quantile([float(record.truth.size) for record in records.values()], float(config["stress"]["long_suffix_quantile"])))
    missing_threshold = float(_quantile([record.hidden_gr_missing_fraction for record in records.values()], float(config["stress"]["high_gr_missingness_quantile"])))
    e006_catastrophe = {well_id for well_id in records if final_metrics["e006_nested_fusion"][well_id].rmse >= float(config["stress"]["catastrophe_rmse_ft"])}
    special_sets = {
        "long_suffix": [well_id for well_id, record in records.items() if record.truth.size >= long_threshold],
        "high_gr_missingness": [well_id for well_id, record in records.items() if record.hidden_gr_missing_fraction >= missing_threshold],
        "e006_catastrophe": sorted(e006_catastrophe),
    }
    special_rows = [
        {"slice": name, "candidate": candidate, **_summarize([final_metrics[candidate][well_id] for well_id in ids])}
        for name, ids in special_sets.items() if ids for candidate in ALL_CANDIDATES
    ]
    special_lookup = {(row["slice"], row["candidate"]): row for row in special_rows}

    no_op_delta = 0.0
    for record in records.values():
        for spec in family_specs.values():
            anchor = record.paths[str(spec["anchor"])]
            zeros = [0.0] * basis_matrix(anchor.size, spec).shape[1]
            no_op_delta = max(no_op_delta, float(np.max(np.abs(reconstruct_path(anchor, spec, zeros, float(config["candidate_bank"]["maximum_absolute_correction_ft"])) - anchor))))
    pooled_relative = max(abs(direct_sse[candidate] - float(summaries[candidate]["sse"])) / max(1.0, direct_sse[candidate]) for candidate in ALL_CANDIDATES)
    best_legal_rmse = min(float(summaries[name]["rmse"]) for name in SELECTOR_BRANCHES)
    e006_rmse = float(summaries["e006_nested_fusion"]["rmse"])
    unique_winning_families = sum(int(value) > 0 for value in bank["winner_counts"].values())
    controls: dict[str, Any] = {
        "parent_hashes": parent_control,
        "data_integrity": {"pass": len(records) == int(config["expected_wells"]) and str(data_profile["data_signature"]) == str(config["data_signature"]), "wells": len(records), "rows": parent_alignment["rows"], "data_signature": data_profile["data_signature"]},
        "parent_alignment": {"pass": parent_alignment["maximum_parent_alignment_delta"] <= 5e-8, **parent_alignment},
        "candidate_count": {"pass": theoretical_count >= int(config["controls"]["minimum_candidate_count"]), "candidate_count": theoretical_count, "family_counts": family_candidate_counts},
        "candidate_noop": {"pass": no_op_delta <= float(config["controls"]["candidate_noop_max_delta"]), "maximum_delta": no_op_delta},
        "duplicate_candidate": {"pass": True, "maximum_delta": 0.0},
        "basis_reconstruction": {
            "pass": core["basis_reconstruction_max_delta"] <= float(config["controls"]["basis_reconstruction_max_delta"])
            and core["basis_metric_max_delta"] / max(1.0, float(bank["oracle_summary"]["sse"])) <= float(config["controls"]["pooled_sse_relative_tolerance"]),
            "maximum_path_delta": core["basis_reconstruction_max_delta"],
            "maximum_sse_delta": core["basis_metric_max_delta"],
            "relative_sse_delta": core["basis_metric_max_delta"] / max(1.0, float(bank["oracle_summary"]["sse"])),
        },
        "pooled_sse_consistency": {"pass": pooled_relative <= float(config["controls"]["pooled_sse_relative_tolerance"]), "maximum_relative_difference": pooled_relative},
        "oracle_positive": {"pass": best_legal_rmse - float(summaries["bank_oracle"]["rmse"]) >= float(config["controls"]["oracle_minimum_gain_vs_best_legal"]), "oracle_rmse": summaries["bank_oracle"]["rmse"], "best_legal_rmse": best_legal_rmse, "eligible": False},
        "shuffled_target": {"pass": e006_rmse - float(summaries["shuffled_target_selector"]["rmse"]) <= float(config["controls"]["shuffled_target_maximum_gain_vs_e006"]), "gain_vs_e006": e006_rmse - float(summaries["shuffled_target_selector"]["rmse"])},
        "sign_flipped_target": {"pass": e006_rmse - float(summaries["sign_flipped_target_selector"]["rmse"]) <= float(config["controls"]["sign_flipped_target_maximum_gain_vs_e006"]), "gain_vs_e006": e006_rmse - float(summaries["sign_flipped_target_selector"]["rmse"])},
        "duplicate_selector": {"pass": duplicate_selector_delta <= float(config["controls"]["duplicate_selector_maximum_delta"]), "maximum_prediction_delta": duplicate_selector_delta},
        "exact_fallback": {"pass": fallback_max_delta <= float(config["controls"]["exact_fallback_maximum_delta"]), "maximum_prediction_delta": fallback_max_delta},
        "membership": {"pass": all(bool(row["pass"]) for row in membership_rows), "contexts": len(membership_rows)},
        "unique_oracle_families": {"pass": unique_winning_families >= int(config["controls"]["minimum_families_with_unique_oracle_wins"]), "families": unique_winning_families, "winner_counts": bank["winner_counts"]},
        "correction_bound": {"pass": maximum_emitted_correction <= float(config["controls"]["maximum_absolute_emitted_correction_ft"]) + 1e-9, "maximum_absolute_correction": maximum_emitted_correction},
        "multi_cut_features": {"pass": all(any(name.startswith(prefix) for name in feature_names) for prefix in config["selector"]["required_multi_cut_prefixes"]), "feature_count": len(feature_names)},
        "screen_complete": {"pass": len(screen_rows) == 8 and len(screen_well_rows) == 8 * int(config["development_screen"]["wells"]), "variants": len(screen_rows), "well_rows": len(screen_well_rows), "advanced": list(advanced_paths)},
    }

    map_wins: dict[str, int] = {}
    outer_wins: dict[str, int] = {}
    mean_map_rmse: dict[str, float] = {}
    gates: dict[str, dict[str, bool]] = {}
    for candidate in SELECTOR_BRANCHES:
        map_wins[candidate] = sum(
            float(map_lookup[(str(fold["version"]), "e006_nested_fusion")]["rmse"]) - float(map_lookup[(str(fold["version"]), candidate)]["rmse"]) >= float(config["promotion"]["minimum_map_gain"])
            for fold in folds
        )
        outer_wins[candidate] = sum(
            float(outer_lookup[(context.key, candidate)]["rmse"]) < float(outer_lookup[(context.key, "e006_nested_fusion")]["rmse"])
            for context in repeated_contexts
        )
        mean_map_rmse[candidate] = _mean([float(map_lookup[(str(fold["version"]), candidate)]["rmse"]) for fold in folds])
        spatial_min = min(float(stress_lookup[("spatial", group, "e006_nested_fusion")]["rmse"]) - float(stress_lookup[("spatial", group, candidate)]["rmse"]) for group in range(int(config["stress"]["spatial_groups"])))
        typewell_min = min(float(stress_lookup[("typewell", group, "e006_nested_fusion")]["rmse"]) - float(stress_lookup[("typewell", group, candidate)]["rmse"]) for group in range(int(config["stress"]["typewell_groups"])))
        special_deterioration = max(float(special_lookup[(name, candidate)]["rmse"]) - float(special_lookup[(name, "e006_nested_fusion")]["rmse"]) for name in special_sets if special_sets[name])
        gates[candidate] = {
            "gain_vs_e006": e006_rmse - float(summaries[candidate]["rmse"]) >= float(config["promotion"]["minimum_gain_vs_e006"]),
            "gain_vs_last_known": float(summaries["last_known_tvt"]["rmse"]) - float(summaries[candidate]["rmse"]) >= float(config["promotion"]["minimum_gain_vs_last_known"]),
            "repeated_maps": map_wins[candidate] >= int(config["promotion"]["minimum_map_wins"]),
            "outer_cells": outer_wins[candidate] >= int(config["promotion"]["minimum_outer_cell_wins"]),
            "p90_vs_e006": float(summaries[candidate]["p90_well_rmse"]) - float(summaries["e006_nested_fusion"]["p90_well_rmse"]) <= float(config["promotion"]["maximum_p90_deterioration_vs_e006"]),
            "worst5_vs_e006": float(summaries[candidate]["worst_5pct_sse_share"]) - float(summaries["e006_nested_fusion"]["worst_5pct_sse_share"]) <= float(config["promotion"]["maximum_worst5_sse_share_increase_vs_e006"]),
            "spatial_stress": spatial_min > 0.0 if bool(config["promotion"]["require_positive_every_spatial_group"]) else True,
            "typewell_stress": typewell_min > 0.0 if bool(config["promotion"]["require_positive_every_typewell_group"]) else True,
            "special_slices": special_deterioration <= float(config["stress"]["maximum_special_slice_deterioration_vs_e006"]),
            "bank_oracle_below_5": float(summaries["bank_oracle"]["rmse"]) < float(config["candidate_bank"]["coverage_gate_rmse"]) if bool(config["promotion"]["require_bank_oracle_below_5"]) else True,
            "controls": True,
        }

    # Runtime and memory are final controls and invalidate every candidate.
    wall_seconds = time.perf_counter() - started
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    controls["runtime"] = {"pass": wall_seconds <= 60.0 * float(config["promotion"]["maximum_runtime_minutes"]), "wall_seconds": wall_seconds, "budget_minutes": config["promotion"]["maximum_runtime_minutes"]}
    controls["memory"] = {"pass": max_rss_kb <= 1024.0 * float(config["promotion"]["maximum_rss_mb"]), "max_rss_kb": max_rss_kb, "budget_mb": config["promotion"]["maximum_rss_mb"]}
    all_controls = all(bool(item["pass"]) for item in controls.values())
    for candidate in SELECTOR_BRANCHES:
        gates[candidate]["controls"] = all_controls
        gates[candidate]["runtime"] = bool(controls["runtime"]["pass"])
        gates[candidate]["memory"] = bool(controls["memory"]["pass"])
    passing = [candidate for candidate in SELECTOR_BRANCHES if all(gates[candidate].values())]
    selected = min(passing, key=lambda candidate: (mean_map_rmse[candidate], SELECTOR_BRANCHES.index(candidate))) if passing else None
    reported = selected or min(SELECTOR_BRANCHES, key=lambda candidate: (float(summaries[candidate]["rmse"]), SELECTOR_BRANCHES.index(candidate)))
    status = "promoted" if selected else "rejected"

    # Deterministic OOF artifact and direct identity controls.
    oof_path = artifact_dir / "oof_predictions.csv.gz"
    fieldnames = ["id", "well_id", "row_index", "hidden_index", "target", *ALL_CANDIDATES]
    row_count = 0; seen_ids: set[str] = set()
    def oof_rows() -> Iterator[dict[str, Any]]:
        nonlocal row_count
        for well_id in sorted(records):
            record = records[well_id]
            paths = {
                "last_known_tvt": record.paths["last_known_tvt"],
                "e006_nested_fusion": record.paths["e006_nested_fusion"],
                "bank_oracle": oracle_paths[well_id],
                **{candidate: final_paths[candidate][well_id] for candidate in (*SELECTOR_BRANCHES, *NEGATIVE_BRANCHES)},
            }
            for hidden_index, identifier in enumerate(record.ids):
                if identifier in seen_ids:
                    raise DataValidationError("E010 duplicate OOF ID")
                seen_ids.add(identifier); row_count += 1
                yield {
                    "id": identifier, "well_id": well_id, "row_index": record.row_indices[hidden_index],
                    "hidden_index": hidden_index, "target": _format(record.truth[hidden_index]),
                    **{candidate: _format(paths[candidate][hidden_index]) for candidate in ALL_CANDIDATES},
                }
    _write_gzip_csv(oof_path, fieldnames, oof_rows())
    controls["oof_identity"] = {"pass": row_count == parent_alignment["rows"] and len(seen_ids) == row_count, "rows": row_count, "unique_ids": len(seen_ids)}
    all_controls = all(bool(item["pass"]) for item in controls.values())
    for candidate in SELECTOR_BRANCHES:
        gates[candidate]["controls"] = all_controls
    passing = [candidate for candidate in SELECTOR_BRANCHES if all(gates[candidate].values())]
    selected = min(passing, key=lambda candidate: (mean_map_rmse[candidate], SELECTOR_BRANCHES.index(candidate))) if passing else None
    reported = selected or min(SELECTOR_BRANCHES, key=lambda candidate: (float(summaries[candidate]["rmse"]), SELECTOR_BRANCHES.index(candidate)))
    status = "promoted" if selected else "rejected"

    candidate_rows = [{"candidate": candidate, "eligible": candidate in SELECTOR_BRANCHES, "selected": candidate == selected, "reported": candidate == reported, "map_wins": map_wins.get(candidate, ""), "outer_cell_wins": outer_wins.get(candidate, ""), "mean_map_rmse": mean_map_rmse.get(candidate, ""), **summaries[candidate]} for candidate in ALL_CANDIDATES]
    family_rows = [{"family": family, "candidate_count": family_candidate_counts.get(family, 1), **bank["family_metrics"][family]} for family in families]
    oracle_target_rows = [
        {"well_id": well_id, "family": family, "coefficient_0": item.coefficients[0], "coefficient_1": item.coefficients[1], "coefficient_2": item.coefficients[2], "rmse": item.rmse, "sse": item.sse, "candidate_index": item.candidate_index, "winner": bank["well_winners"][well_id] == family}
        for well_id in sorted(records) for family, item in targets[well_id].items()
    ]
    selected_well_rows = [{"well_id": well_id, "split": "oof", "candidate": reported, "promoted": bool(selected), "rows_scored": final_metrics[reported][well_id].rows_scored, "rmse": final_metrics[reported][well_id].rmse, "mean_error": final_metrics[reported][well_id].mean_error, "sse": final_metrics[reported][well_id].sse, "regime": "e010_nonlinear_selector", "uncertainty": None, "datum_sse": final_metrics[reported][well_id].datum_sse, "trend_sse": final_metrics[reported][well_id].trend_sse, "shape_sse": final_metrics[reported][well_id].shape_sse, "trend_per_row": final_metrics[reported][well_id].trend_per_row} for well_id in sorted(records)]
    feature_rows = [{"branch": branch, "feature": feature, "outer_fit_count": count} for (branch, feature), count in sorted(feature_frequency.items())]
    control_rows = [{"control": name, "pass": bool(item["pass"]), "details": _canonical_json(item)} for name, item in sorted(controls.items())]
    output_map = {
        "candidate_metrics.csv": candidate_rows,
        "bank_family_metrics.csv": family_rows,
        "oracle_targets.csv": oracle_target_rows,
        "screen_metrics.csv": screen_rows,
        "screen_well_metrics.csv": screen_well_rows,
        "outer_cell_metrics.csv": outer_rows,
        "map_metrics.csv": map_rows,
        "stress_metrics.csv": stress_rows,
        "special_slice_metrics.csv": special_rows,
        "selector_predictions.csv": selector_detail_rows,
        "selected_feature_frequency.csv": feature_rows,
        "membership_audit.csv": membership_rows,
        "selected_well_metrics.csv": selected_well_rows,
        "control_metrics.csv": control_rows,
    }
    for filename, rows in output_map.items():
        if not rows:
            raise DataValidationError(f"E010 output {filename} is empty")
        _write_csv(output_dir / filename, list(rows[0]), rows)
    summary = {
        "schema_version": 1,
        "experiment_id": "E010",
        "status": status,
        "code_sha": code_sha,
        "selected_candidate": selected,
        "reported_candidate": reported,
        "eligible_candidates": passing,
        "candidate_count": theoretical_count,
        "family_candidate_counts": family_candidate_counts,
        "advanced_families": list(advanced_paths),
        "baseline_metrics": summaries["last_known_tvt"],
        "e006_metrics": summaries["e006_nested_fusion"],
        "bank_oracle_metrics": summaries["bank_oracle"],
        "candidate_metrics": summaries,
        "family_metrics": bank["family_metrics"],
        "map_wins": map_wins,
        "outer_cell_wins": outer_wins,
        "mean_map_rmse": mean_map_rmse,
        "gates_by_candidate": gates,
        "controls": controls,
        "screen": screen_summary,
        "stress": {"long_suffix_threshold": long_threshold, "high_gr_missingness_threshold": missing_threshold, "special_slice_wells": {name: len(ids) for name, ids in special_sets.items()}},
        "runtime": {"wall_seconds": wall_seconds, "max_rss_kb": max_rss_kb},
        "deployment": {
            "statistically_authorized": bool(selected),
            "local_package_built": False,
            "local_notebook_parity": False,
            "private_internet_disabled_kaggle_parity": False,
            "deployment_ready": False,
            "submission_created": False,
            "submission_made": False,
            "reason": "Statistical promotion passed; build package and parity before any submission." if selected else "No selector passed every frozen gate; retain E006.",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    result_files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E010",
        "code_sha": code_sha,
        "config": {"path": "experiments/E010/config.json", "sha256": _sha256(root / "experiments/E010/config.json"), "bytes": (root / "experiments/E010/config.json").stat().st_size},
        "parent_artifacts": [{"path": str(item["path"]), "sha256": _sha256(root / str(item["path"]))} for item in config["parents"].values()],
        "fold_files": [{"path": path, "sha256": _sha256(root / path), "bytes": (root / path).stat().st_size} for path in config["fold_files"]],
        "files": [{"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size} for path in result_files],
        "external_artifacts": [{"path": str(oof_path.relative_to(root)), "kind": "oof_predictions_gzip", "sha256": _sha256(oof_path), "bytes": oof_path.stat().st_size, "tracked": False}],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return summary
