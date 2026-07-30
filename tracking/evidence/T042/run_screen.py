#!/usr/bin/env python3
"""T042 posterior-state public-core candidate-loss screen.

The `features` phase is target-free and consumes only T041 score-blind caches,
E011 predictions, and test-available well/typewell columns. The `finalize` phase
opens hidden targets and evaluates the preregistered cross-fitted model bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "tracking/evidence/T042/config.json"
DEFAULT_FEATURES = ROOT / "tracking/evidence/T042/features.csv"
DEFAULT_OUTPUT = ROOT / "tracking/evidence/T042/results"
RESULT_PATH = ROOT / "tracking/evidence/T042/RESULT.md"
RAW_TRACKERS = ("pf", "hmm_stable", "hmm_edge")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def robust_scale(values: np.ndarray) -> float:
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    median = float(np.median(array))
    return float(1.4826 * np.median(np.abs(array - median)))


def safe_stat(values: np.ndarray, operation: str) -> float:
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    if operation == "mean":
        return float(np.mean(array))
    if operation == "std":
        return float(np.std(array))
    if operation == "median":
        return float(np.median(array))
    if operation == "p90":
        return float(np.quantile(array, 0.90))
    if operation == "max":
        return float(np.max(array))
    if operation == "min":
        return float(np.min(array))
    raise ValueError(operation)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_selected() -> list[str]:
    summary = json.loads((ROOT / "tracking/evidence/T041/results/summary.json").read_text(encoding="utf-8"))
    selected = [str(value) for value in summary["sample_wells"]]
    if len(selected) != 100 or len(set(selected)) != 100:
        raise AssertionError("T041 sample identity mismatch")
    return selected


def load_e011(selected: set[str], include_target: bool) -> pd.DataFrame:
    config = load_config()
    columns = ["well_id", "row_index", "spline4_ridge_equal_s075"]
    if include_target:
        columns.append("target")
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(ROOT / config["baseline_oof"], usecols=columns, chunksize=250_000):
        keep = chunk["well_id"].isin(selected)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    if not parts:
        raise RuntimeError("E011 selected rows not found")
    return pd.concat(parts, ignore_index=True).sort_values(["well_id", "row_index"]).reset_index(drop=True)


def load_frames(well_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    data_dir = ROOT / config["data_dir"]
    horizontal = pd.read_csv(
        data_dir / f"{well_id}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    typewell = pd.read_csv(data_dir / f"{well_id}__typewell.csv", usecols=["TVT", "GR"])
    return horizontal, typewell


def load_cache(well_id: str) -> dict[str, Any]:
    config = load_config()
    path = ROOT / config["t041_cache_dir"] / "wells" / f"{well_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        output: dict[str, Any] = {"row_index": data["row_index"].astype(np.int64)}
        for tracker in RAW_TRACKERS:
            prediction = data[f"pred_{tracker}"].astype(float)
            if not np.isfinite(prediction).all():
                raise AssertionError(f"nonfinite {tracker} cache {well_id}")
            output[tracker] = prediction
        for tracker in ("hmm_stable", "hmm_edge"):
            output[f"{tracker}_mean_std"] = float(data[f"{tracker}_mean_std"].item())
            output[f"{tracker}_p90_std"] = float(data[f"{tracker}_p90_std"].item())
            output[f"{tracker}_loglik"] = float(data[f"{tracker}_loglik"].item())
        output["input_immutable"] = bool(data["input_immutable"].item())
    return output


def path_features(prefix: str, path: np.ndarray, reference: np.ndarray, md_hidden: np.ndarray) -> dict[str, float]:
    delta = np.asarray(path, float) - np.asarray(reference, float)
    absolute = np.abs(delta)
    md_span = max(float(md_hidden[-1] - md_hidden[0]), 1.0) if md_hidden.size > 1 else 1.0
    path_slope = float((path[-1] - path[0]) / md_span) if path.size > 1 else 0.0
    reference_slope = float((reference[-1] - reference[0]) / md_span) if reference.size > 1 else 0.0
    differences = np.diff(path)
    return {
        f"{prefix}_mean_abs_vs_e011": safe_stat(absolute, "mean"),
        f"{prefix}_median_abs_vs_e011": safe_stat(absolute, "median"),
        f"{prefix}_p90_abs_vs_e011": safe_stat(absolute, "p90"),
        f"{prefix}_max_abs_vs_e011": safe_stat(absolute, "max"),
        f"{prefix}_terminal_abs_vs_e011": float(absolute[-1]) if absolute.size else 0.0,
        f"{prefix}_signed_mean_vs_e011": safe_stat(delta, "mean"),
        f"{prefix}_signed_terminal_vs_e011": float(delta[-1]) if delta.size else 0.0,
        f"{prefix}_path_slope": path_slope,
        f"{prefix}_slope_vs_e011": path_slope - reference_slope,
        f"{prefix}_roughness": safe_stat(differences, "std"),
        f"{prefix}_range": float(np.max(path) - np.min(path)) if path.size else 0.0,
    }


def pair_features(name: str, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    difference = np.asarray(first, float) - np.asarray(second, float)
    absolute = np.abs(difference)
    return {
        f"{name}_mean_abs": safe_stat(absolute, "mean"),
        f"{name}_p90_abs": safe_stat(absolute, "p90"),
        f"{name}_max_abs": safe_stat(absolute, "max"),
        f"{name}_terminal_abs": float(absolute[-1]) if absolute.size else 0.0,
        f"{name}_signed_mean": safe_stat(difference, "mean"),
        f"{name}_signed_terminal": float(difference[-1]) if difference.size else 0.0,
    }


def build_feature_row(well_id: str, base_rows: pd.DataFrame) -> dict[str, Any]:
    horizontal, typewell = load_frames(well_id)
    cache = load_cache(well_id)
    hidden = horizontal["TVT_input"].isna().to_numpy()
    known = ~hidden
    row_index = np.flatnonzero(hidden).astype(np.int64)
    if not np.array_equal(cache["row_index"], row_index):
        raise AssertionError(f"cache row mismatch {well_id}")
    group = base_rows[base_rows["well_id"] == well_id].sort_values("row_index")
    if not np.array_equal(group["row_index"].to_numpy(np.int64), row_index):
        raise AssertionError(f"E011 row mismatch {well_id}")
    e011 = group["spline4_ridge_equal_s075"].to_numpy(float)
    md = horizontal["MD"].to_numpy(float)
    z = horizontal["Z"].to_numpy(float)
    gr = horizontal["GR"].to_numpy(float)
    md_hidden = md[hidden]
    gr_hidden = gr[hidden]
    gr_known = gr[known]
    tvt_known = horizontal.loc[known, "TVT_input"].to_numpy(float)
    z_known = z[known]
    md_known = md[known]
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = typewell["GR"].ffill().bfill().to_numpy(float)
    tw_at_known = np.interp(tvt_known, tw_tvt, tw_gr) if tvt_known.size else np.asarray([], float)
    residual = gr_known - tw_at_known if gr_known.size else np.asarray([], float)
    tail_count = min(30, tvt_known.size)
    terminal_rate = 0.0
    if tail_count >= 4:
        tail_tvt = tvt_known[-tail_count:]
        tail_z = z_known[-tail_count:]
        tail_md = md_known[-tail_count:]
        delta_md = np.diff(tail_md)
        valid = delta_md > 0
        if int(valid.sum()) >= 3:
            terminal_rate = float(np.median((np.diff(tail_tvt)[valid] + np.diff(tail_z)[valid]) / delta_md[valid]))

    row: dict[str, Any] = {
        "well_id": well_id,
        "known_rows": int(known.sum()),
        "hidden_rows": int(hidden.sum()),
        "hidden_fraction": float(hidden.mean()),
        "md_span": float(np.max(md) - np.min(md)),
        "known_md_span": float(np.max(md_known) - np.min(md_known)) if md_known.size else 0.0,
        "hidden_md_span": float(np.max(md_hidden) - np.min(md_hidden)) if md_hidden.size else 0.0,
        "median_md_step": safe_stat(np.diff(md), "median"),
        "max_md_step": safe_stat(np.diff(md), "max"),
        "z_span": float(np.max(z) - np.min(z)),
        "hidden_z_span": float(np.max(z[hidden]) - np.min(z[hidden])) if hidden.any() else 0.0,
        "known_gr_missing_fraction": float(np.mean(~np.isfinite(gr_known))) if gr_known.size else 1.0,
        "hidden_gr_missing_fraction": float(np.mean(~np.isfinite(gr_hidden))) if gr_hidden.size else 1.0,
        "known_gr_mean": safe_stat(gr_known, "mean"),
        "known_gr_std": safe_stat(gr_known, "std"),
        "known_gr_robust_scale": robust_scale(gr_known),
        "hidden_gr_mean": safe_stat(gr_hidden, "mean"),
        "hidden_gr_std": safe_stat(gr_hidden, "std"),
        "hidden_gr_robust_scale": robust_scale(gr_hidden),
        "typewell_rows": int(len(typewell)),
        "typewell_tvt_span": float(np.max(tw_tvt) - np.min(tw_tvt)),
        "typewell_gr_mean": safe_stat(tw_gr, "mean"),
        "typewell_gr_std": safe_stat(tw_gr, "std"),
        "typewell_gr_robust_scale": robust_scale(tw_gr),
        "prefix_gr_residual_std": safe_stat(residual, "std"),
        "prefix_gr_residual_robust_scale": robust_scale(residual),
        "terminal_u_rate": terminal_rate,
        "hmm_stable_posterior_mean_std": float(cache["hmm_stable_mean_std"]),
        "hmm_stable_posterior_p90_std": float(cache["hmm_stable_p90_std"]),
        "hmm_stable_loglik_per_hidden_row": float(cache["hmm_stable_loglik"]) / max(1, int(hidden.sum())),
        "hmm_edge_posterior_mean_std": float(cache["hmm_edge_mean_std"]),
        "hmm_edge_posterior_p90_std": float(cache["hmm_edge_p90_std"]),
        "hmm_edge_loglik_per_hidden_row": float(cache["hmm_edge_loglik"]) / max(1, int(hidden.sum())),
        "hmm_loglik_difference_per_hidden_row": (float(cache["hmm_stable_loglik"]) - float(cache["hmm_edge_loglik"])) / max(1, int(hidden.sum())),
    }
    for tracker in RAW_TRACKERS:
        row.update(path_features(tracker, cache[tracker], e011, md_hidden))
    for first, second in combinations(RAW_TRACKERS, 2):
        row.update(pair_features(f"{first}_vs_{second}", cache[first], cache[second]))
    stack = np.vstack([cache[tracker] for tracker in RAW_TRACKERS])
    spread = np.std(stack, axis=0)
    row.update({
        "tracker_spread_mean": safe_stat(spread, "mean"),
        "tracker_spread_p90": safe_stat(spread, "p90"),
        "tracker_spread_max": safe_stat(spread, "max"),
        "tracker_spread_terminal": float(spread[-1]) if spread.size else 0.0,
        "cache_input_immutable": float(cache["input_immutable"]),
    })
    for key, value in row.items():
        if key != "well_id" and not np.isfinite(float(value)):
            raise AssertionError(f"nonfinite feature {key} for {well_id}")
    return row


def feature_phase(output_path: Path) -> int:
    started = time.time()
    config = load_config()
    selected = load_selected()
    base = load_e011(set(selected), include_target=False)
    assignments = json.loads((ROOT / config["fold_map"]).read_text(encoding="utf-8"))["assignments"]
    rows: list[dict[str, Any]] = []
    for index, well_id in enumerate(selected, 1):
        row = build_feature_row(well_id, base)
        row["fold"] = int(assignments[well_id])
        rows.append(row)
        if index % 10 == 0 or index == len(selected):
            print(f"features {index}/{len(selected)} {well_id}", flush=True)
    table = pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)
    forbidden = tuple(str(value) for value in config["forbidden_feature_prefixes"])
    bad = [column for column in table.columns if any(column.startswith(prefix) for prefix in forbidden)]
    if bad:
        raise AssertionError(f"forbidden feature columns present: {bad}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(output_path)
    metadata = {
        "schema_version": 1,
        "task_id": "T042",
        "rows": int(len(table)),
        "columns": int(len(table.columns)),
        "feature_columns": [column for column in table.columns if column not in {"well_id", "fold"}],
        "forbidden_columns": bad,
        "features_sha256": sha256_file(output_path),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runtime_seconds": time.time() - started,
        "hidden_target_read": False,
        "pseudo_cut_feature_used": False,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(output_path.with_name("FEATURES_FREEZE.json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


def build_model(name: str, spec: dict[str, Any]) -> Pipeline:
    family = str(spec["family"])
    if family == "ridge":
        estimator: Any = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(spec["alpha"]))),
        ])
    elif family == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            random_state=stable_seed(f"T042_{name}"),
            n_jobs=1,
        )
    elif family == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            random_state=stable_seed(f"T042_{name}"),
            n_jobs=1,
        )
    else:
        raise ValueError(family)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


def rank_spearman(first: np.ndarray, second: np.ndarray) -> float:
    left = pd.Series(np.asarray(first, float)).rank(method="average").to_numpy(float)
    right = pd.Series(np.asarray(second, float)).rank(method="average").to_numpy(float)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def crossfit_predictions(
    features: pd.DataFrame,
    target_log_mse: np.ndarray,
    model_name: str,
    model_spec: dict[str, Any],
    corruption: str | None = None,
) -> np.ndarray:
    x_columns = [column for column in features.columns if column not in {"well_id", "fold"}]
    x = features[x_columns].to_numpy(float)
    folds = features["fold"].to_numpy(int)
    output = np.zeros_like(target_log_mse, dtype=float)
    for fold in range(5):
        train = folds != fold
        valid = folds == fold
        y_train = target_log_mse[train].copy()
        if corruption == "shuffled_training_targets":
            rng = np.random.default_rng(stable_seed(f"T042_shuffle_fold_{fold}"))
            y_train = y_train[rng.permutation(y_train.shape[0])]
        elif corruption == "rotated_candidate_targets":
            y_train = np.roll(y_train, 1, axis=1)
        elif corruption is not None:
            raise ValueError(corruption)
        model = build_model(model_name, model_spec)
        model.fit(x[train], y_train)
        output[valid] = np.asarray(model.predict(x[valid]), float)
    if output.shape != target_log_mse.shape or not np.isfinite(output).all():
        raise AssertionError(f"invalid crossfit predictions {model_name} {corruption}")
    return output


def candidate_paths(
    selected: list[str],
    e011: pd.DataFrame,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, int]]:
    config = load_config()
    order = [str(value) for value in config["candidate_order"]]
    paths: dict[str, dict[str, np.ndarray]] = {name: {} for name in order}
    targets: dict[str, np.ndarray] = {}
    rows: dict[str, int] = {}
    groups = {str(well_id): group.sort_values("row_index") for well_id, group in e011.groupby("well_id", sort=True)}
    for well_id in selected:
        group = groups[well_id]
        base = group["spline4_ridge_equal_s075"].to_numpy(float)
        target = group["target"].to_numpy(float)
        cache = load_cache(well_id)
        if not np.array_equal(group["row_index"].to_numpy(np.int64), cache["row_index"]):
            raise AssertionError(f"row mismatch {well_id}")
        raw = {tracker: np.asarray(cache[tracker], float) for tracker in RAW_TRACKERS}
        paths["e011"][well_id] = base
        for tracker in RAW_TRACKERS:
            paths[tracker][well_id] = raw[tracker]
        paths["blend_e011_pf"][well_id] = 0.5 * base + 0.5 * raw["pf"]
        paths["blend_e011_hmm_stable"][well_id] = 0.5 * base + 0.5 * raw["hmm_stable"]
        paths["blend_e011_hmm_edge"][well_id] = 0.5 * base + 0.5 * raw["hmm_edge"]
        targets[well_id] = target
        rows[well_id] = int(target.size)
    return paths, targets, rows


def apply_placement(
    predicted_log_mse: np.ndarray,
    placement: str,
    config: dict[str, Any],
    order: list[str],
    paths: dict[str, dict[str, np.ndarray]],
    selected: list[str],
) -> dict[str, np.ndarray]:
    reference_index = order.index(str(config["reference_candidate"]))
    output: dict[str, np.ndarray] = {}
    for index, well_id in enumerate(selected):
        values = predicted_log_mse[index]
        best_index = int(np.argmin(values))
        if placement == "hard":
            output[well_id] = paths[order[best_index]][well_id]
        elif placement == "guarded":
            margin = float(config["placements"]["guarded"]["minimum_predicted_log_mse_gain"])
            chosen = best_index if values[reference_index] - values[best_index] >= margin else reference_index
            output[well_id] = paths[order[chosen]][well_id]
        elif placement == "soft":
            temperature = float(config["placements"]["soft"]["temperature"])
            weights = np.exp(-(values - values.min()) / temperature)
            weights /= weights.sum()
            stack = np.vstack([paths[name][well_id] for name in order])
            output[well_id] = np.sum(weights[:, None] * stack, axis=0)
        else:
            raise ValueError(placement)
    return output


def summarize_predictions(
    predictions: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
    assignments: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for candidate, candidate_paths_map in predictions.items():
        for well_id, prediction in candidate_paths_map.items():
            error = np.asarray(prediction, float) - targets[well_id]
            sse = float(np.square(error).sum())
            rows.append({
                "candidate": candidate,
                "well_id": well_id,
                "fold": int(assignments[well_id]),
                "rows": int(error.size),
                "sse": sse,
                "rmse": math.sqrt(sse / error.size),
            })
    table = pd.DataFrame(rows)
    summaries: dict[str, dict[str, Any]] = {}
    for candidate, group in table.groupby("candidate", sort=True):
        total_sse = float(group["sse"].sum())
        total_rows = int(group["rows"].sum())
        worst_count = max(1, int(math.ceil(0.20 * len(group))))
        summaries[str(candidate)] = {
            "rmse": math.sqrt(total_sse / total_rows),
            "sse": total_sse,
            "rows": total_rows,
            "wells": int(len(group)),
            "median_well_rmse": float(group["rmse"].median()),
            "p90_well_rmse": float(group["rmse"].quantile(0.90)),
            "max_well_rmse": float(group["rmse"].max()),
            "worst20_sse_share": float(group.nlargest(worst_count, "sse")["sse"].sum() / total_sse),
        }
    return table, summaries


def subset_rmse(table: pd.DataFrame, candidate: str, ids: set[str]) -> float:
    part = table[(table["candidate"] == candidate) & table["well_id"].isin(ids)]
    return math.sqrt(float(part["sse"].sum()) / int(part["rows"].sum()))


def finalize(features_path: Path, output_dir: Path) -> int:
    started = time.time()
    config = load_config()
    features = pd.read_csv(features_path).sort_values("well_id").reset_index(drop=True)
    selected = features["well_id"].astype(str).tolist()
    if selected != sorted(load_selected()):
        raise AssertionError("feature sample identity mismatch")
    forbidden = tuple(str(value) for value in config["forbidden_feature_prefixes"])
    bad = [column for column in features.columns if any(column.startswith(prefix) for prefix in forbidden)]
    if bad:
        raise AssertionError(f"forbidden feature columns: {bad}")
    e011 = load_e011(set(selected), include_target=True)
    paths, targets, row_counts = candidate_paths(selected, e011)
    order = [str(value) for value in config["candidate_order"]]
    assignments = {str(key): int(value) for key, value in json.loads((ROOT / config["fold_map"]).read_text())["assignments"].items()}

    actual_mse = np.zeros((len(selected), len(order)), dtype=float)
    for well_index, well_id in enumerate(selected):
        for candidate_index, candidate in enumerate(order):
            actual_mse[well_index, candidate_index] = float(np.mean(np.square(paths[candidate][well_id] - targets[well_id])))
    target_log_mse = np.log1p(actual_mse)

    predicted_by_model: dict[str, np.ndarray] = {}
    all_predictions: dict[str, dict[str, np.ndarray]] = {
        "reference": {well_id: paths[str(config["reference_candidate"])][well_id] for well_id in selected}
    }
    model_quality_rows: list[dict[str, Any]] = []
    for model_name, model_spec in config["models"].items():
        predicted = crossfit_predictions(features, target_log_mse, model_name, model_spec)
        predicted_by_model[model_name] = predicted
        spearman_values = [rank_spearman(predicted[index], target_log_mse[index]) for index in range(len(selected))]
        hit_rate = float(np.mean(np.argmin(predicted, axis=1) == np.argmin(target_log_mse, axis=1)))
        model_quality_rows.append({
            "model": model_name,
            "mean_candidate_rank_spearman": float(np.mean(spearman_values)),
            "candidate_hit_rate": hit_rate,
        })
        for placement in config["placements"]:
            candidate_name = f"{model_name}_{placement}"
            all_predictions[candidate_name] = apply_placement(predicted, placement, config, order, paths, selected)

    legal_table, legal_summaries = summarize_predictions(all_predictions, targets, assignments)
    registered = [name for name in legal_summaries if name != "reference"]
    selected_candidate = min(registered, key=lambda name: (legal_summaries[name]["rmse"], name))
    selected_model, selected_placement = selected_candidate.rsplit("_", 1)
    selected_quality = next(row for row in model_quality_rows if row["model"] == selected_model)

    control_predictions: dict[str, dict[str, np.ndarray]] = {}
    for corruption in config["controls"]:
        corrupted = crossfit_predictions(
            features,
            target_log_mse,
            selected_model,
            config["models"][selected_model],
            corruption=corruption,
        )
        control_predictions[f"control_{corruption}"] = apply_placement(
            corrupted, selected_placement, config, order, paths, selected
        )
    control_table, control_summaries = summarize_predictions(control_predictions, targets, assignments)
    combined_table = pd.concat([legal_table, control_table], ignore_index=True)
    summaries = {**legal_summaries, **control_summaries}

    reference_name = "reference"
    fold_rows: list[dict[str, Any]] = []
    fold_wins = 0
    cell_rows: list[dict[str, Any]] = []
    cell_wins = 0
    for fold in range(5):
        fold_ids = sorted(well_id for well_id in selected if assignments[well_id] == fold)
        ids = set(fold_ids)
        selected_rmse = subset_rmse(combined_table, selected_candidate, ids)
        reference_rmse = subset_rmse(combined_table, reference_name, ids)
        win = selected_rmse < reference_rmse
        fold_wins += int(win)
        fold_rows.append({"fold": fold, "wells": len(ids), "reference_rmse": reference_rmse, "selected_rmse": selected_rmse, "gain": reference_rmse - selected_rmse, "win": win})
        for cell, values in enumerate(np.array_split(np.asarray(fold_ids, dtype=object), 5)):
            cell_ids = set(str(value) for value in values.tolist())
            candidate_rmse = subset_rmse(combined_table, selected_candidate, cell_ids)
            base_rmse = subset_rmse(combined_table, reference_name, cell_ids)
            cell_win = candidate_rmse < base_rmse
            cell_wins += int(cell_win)
            cell_rows.append({"fold": fold, "cell": cell, "wells": len(cell_ids), "reference_rmse": base_rmse, "selected_rmse": candidate_rmse, "gain": base_rmse - candidate_rmse, "win": cell_win})

    oracle_sse = 0.0
    oracle_rows = 0
    oracle_counts = {candidate: 0 for candidate in order}
    for index, well_id in enumerate(selected):
        best = int(np.argmin(actual_mse[index]))
        candidate = order[best]
        oracle_counts[candidate] += 1
        oracle_sse += actual_mse[index, best] * row_counts[well_id]
        oracle_rows += row_counts[well_id]
    component_oracle_rmse = math.sqrt(oracle_sse / oracle_rows)

    control_rows: list[dict[str, Any]] = []
    minimum_control_loss = float("inf")
    for control in control_predictions:
        loss = summaries[control]["rmse"] - summaries[selected_candidate]["rmse"]
        minimum_control_loss = min(minimum_control_loss, loss)
        control_rows.append({"control": control, "rmse": summaries[control]["rmse"], "selected_rmse": summaries[selected_candidate]["rmse"], "loss_vs_selected": loss})

    gates = config["gates"]
    scientific_gates = {
        "gain_vs_reference": summaries[reference_name]["rmse"] - summaries[selected_candidate]["rmse"] >= float(gates["minimum_gain_vs_reference_rmse"]),
        "fold_wins": fold_wins >= int(gates["minimum_fold_wins"]),
        "cell_wins": cell_wins >= int(gates["minimum_cell_wins"]),
        "p90": summaries[selected_candidate]["p90_well_rmse"] <= summaries[reference_name]["p90_well_rmse"] + float(gates["maximum_p90_deterioration"]),
        "worst20_share": summaries[selected_candidate]["worst20_sse_share"] <= summaries[reference_name]["worst20_sse_share"] + float(gates["maximum_worst20_share_increase"]),
        "candidate_rank_spearman": float(selected_quality["mean_candidate_rank_spearman"]) >= float(gates["minimum_mean_candidate_rank_spearman"]),
        "candidate_hit_rate": float(selected_quality["candidate_hit_rate"]) >= float(gates["minimum_candidate_hit_rate"]),
        "component_oracle": component_oracle_rmse <= float(gates["maximum_component_oracle_rmse"]),
        "controls": minimum_control_loss >= float(gates["minimum_control_loss_rmse"]),
    }
    structural_gates = {
        "feature_hash": sha256_file(features_path) == json.loads(features_path.with_name("FEATURES_FREEZE.json").read_text())["features_sha256"],
        "forbidden_features_absent": not bad,
        "complete_crossfit_predictions": all(prediction.shape == target_log_mse.shape and np.isfinite(prediction).all() for prediction in predicted_by_model.values()),
        "complete_rows": summaries[selected_candidate]["rows"] == summaries[reference_name]["rows"],
        "finite_paths": all(np.isfinite(values).all() for candidates in all_predictions.values() for values in candidates.values()),
        "rmse_sse_identity": abs(summaries[selected_candidate]["rmse"] ** 2 * summaries[selected_candidate]["rows"] - summaries[selected_candidate]["sse"]) <= max(1e-6, 1e-10 * summaries[selected_candidate]["sse"]),
    }
    passed = all(bool(value) for value in scientific_gates.values()) and all(bool(value) for value in structural_gates.values())
    decision = "PASS_FREEZE_FOR_673_WELL_HOLDOUT" if passed else "REJECT_H034_EXACT_DIAGNOSTIC_SELECTOR"

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = [
        {
            "candidate": name,
            **summary,
            "gain_vs_reference": summaries[reference_name]["rmse"] - summary["rmse"],
            "selected": name == selected_candidate,
        }
        for name, summary in summaries.items()
    ]
    pd.DataFrame(candidate_rows).sort_values(["rmse", "candidate"]).to_csv(output_dir / "candidate_metrics.csv", index=False)
    combined_table.to_csv(output_dir / "per_well_metrics.csv", index=False)
    pd.DataFrame(model_quality_rows).to_csv(output_dir / "model_quality.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(output_dir / "cell_metrics.csv", index=False)
    pd.DataFrame(control_rows).to_csv(output_dir / "control_metrics.csv", index=False)
    pd.DataFrame({"well_id": selected, **{f"actual_log_mse_{order[index]}": target_log_mse[:, index] for index in range(len(order))}, **{f"predicted_log_mse_{order[index]}": predicted_by_model[selected_model][:, index] for index in range(len(order))}}).to_csv(output_dir / "selected_model_predictions.csv", index=False)

    summary = {
        "schema_version": 1,
        "task_id": "T042",
        "hypothesis_id": "H034",
        "decision": decision,
        "passed": passed,
        "selected_candidate": selected_candidate,
        "selected_model": selected_model,
        "selected_placement": selected_placement,
        "reference": summaries[reference_name],
        "selected": summaries[selected_candidate],
        "gain_vs_reference": summaries[reference_name]["rmse"] - summaries[selected_candidate]["rmse"],
        "candidate_metrics": summaries,
        "model_quality": model_quality_rows,
        "selected_mean_candidate_rank_spearman": float(selected_quality["mean_candidate_rank_spearman"]),
        "selected_candidate_hit_rate": float(selected_quality["candidate_hit_rate"]),
        "fold_wins": fold_wins,
        "cell_wins": cell_wins,
        "component_oracle_rmse": component_oracle_rmse,
        "component_oracle_counts": oracle_counts,
        "minimum_control_loss_rmse": minimum_control_loss,
        "scientific_gates": {key: bool(value) for key, value in scientific_gates.items()},
        "structural_gates": {key: bool(value) for key, value in structural_gates.items()},
        "feature_columns": [column for column in features.columns if column not in {"well_id", "fold"}],
        "features_sha256": sha256_file(features_path),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "runtime_seconds": time.time() - started,
        "observed_max_rss_gb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 / 1024.0,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kaggle_run": False,
        "submission_made": False,
    }
    atomic_json(output_dir / "summary.json", summary)
    lines = [
        "# T042 result — posterior-state public-core candidate selector",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Selected: `{selected_candidate}`",
        f"- Reference RMSE: `{summaries[reference_name]['rmse']:.12f}`",
        f"- Selected RMSE: `{summaries[selected_candidate]['rmse']:.12f}`",
        f"- Gain: `{summaries[reference_name]['rmse'] - summaries[selected_candidate]['rmse']:.12f}`",
        f"- Fold wins: `{fold_wins}/5`; cell wins: `{cell_wins}/25`",
        f"- Rank Spearman: `{float(selected_quality['mean_candidate_rank_spearman']):.6f}`",
        f"- Candidate hit rate: `{float(selected_quality['candidate_hit_rate']):.6f}`",
        f"- Component oracle RMSE: `{component_oracle_rmse:.12f}`",
        f"- Minimum control loss: `{minimum_control_loss:.12f}`",
        "",
        "Scientific gates:",
        *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in scientific_gates.items()],
        "",
        "Structural gates:",
        *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in structural_gates.items()],
        "",
        "No Kaggle run or submission was made.",
    ]
    atomic_text(RESULT_PATH, "\n".join(lines) + "\n")
    manifest: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(output_dir / "artifact_manifest.json", {"schema_version": 1, "files": manifest})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("features", "finalize"))
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.phase == "features":
        return feature_phase(args.features_path.resolve())
    return finalize(args.features_path.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
