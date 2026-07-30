#!/usr/bin/env python3
"""T041 public-frontier visible-prefix arbitration worth audit.

Prediction mode is score-blind: it reads only test-available columns and writes
per-well tracker and pseudo-cut caches. Finalization is the only phase that opens
E011 OOF targets and computes hidden-suffix metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "tracking/evidence/T041/config.json"
DEFAULT_CACHE = ROOT / "scratch/agents/public-frontier-uplift-20260730/t041_cache"
DEFAULT_OUTPUT = ROOT / "tracking/evidence/T041/results"
RESULT_PATH = ROOT / "tracking/evidence/T041/RESULT.md"
TRACKERS = ("pf", "hmm_stable", "hmm_edge")
PROFILES = ("conservative", "balanced")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(frame.columns).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(np.uint64).tobytes())
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def split_ids(ids: list[str], chunk_index: int, chunks: int) -> list[str]:
    if chunks <= 0 or chunk_index < 0 or chunk_index >= chunks:
        raise ValueError("invalid chunk specification")
    return [str(x) for x in np.array_split(np.asarray(ids, dtype=object), chunks)[chunk_index].tolist()]


def load_public_functions(source: Path) -> tuple[dict[str, Any], str]:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if len(cells) <= 10:
        raise ValueError("unexpected public notebook structure")
    namespace: dict[str, Any] = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "njit": njit,
        "prange": prange,
    }
    for index in (4, 8, 10):
        code = "".join(cells[index].get("source", []))
        code = code.replace("cache=True", "cache=False")
        exec(compile(code, f"{source.name}:cell{index}", "exec"), namespace)
    missing = [name for name in ("run_hmm2", "lik_pf") if name not in namespace]
    if missing:
        raise RuntimeError(f"public source functions missing: {missing}")
    return namespace, sha256_file(source)


def common() -> tuple[dict[str, Any], Path, str, str, dict[str, int], list[str]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = ROOT / config["public_source"]
    config_sha = sha256_file(CONFIG_PATH)
    source_sha = sha256_file(source)
    fold = json.loads((ROOT / config["sample"]["fold_map"]).read_text(encoding="utf-8"))
    assignments = {str(key): int(value) for key, value in fold["assignments"].items()}
    selected: list[str] = []
    per_fold = int(config["sample"]["wells_per_fold"])
    for fold_id in range(5):
        ids = sorted(key for key, value in assignments.items() if value == fold_id)
        indices = np.linspace(0, len(ids) - 1, per_fold, dtype=int)
        selected.extend(ids[int(index)] for index in indices)
    if len(selected) != int(config["sample"]["expected_wells"]) or len(set(selected)) != len(selected):
        raise AssertionError("sample identity mismatch")
    return config, source, config_sha, source_sha, assignments, selected


def load_frames(data_dir: Path, well_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(
        data_dir / f"{well_id}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    typewell = pd.read_csv(
        data_dir / f"{well_id}__typewell.csv",
        usecols=["TVT", "GR"],
    )
    return horizontal, typewell


def run_tracker(
    namespace: dict[str, Any],
    tracker: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    if tracker == "pf":
        params = config["trackers"][tracker]
        result = namespace["lik_pf"](
            horizontal.copy(deep=True),
            typewell.copy(deep=True),
            n_particles=int(params["n_particles"]),
            n_seeds=int(params["n_seeds"]),
            scale=float(params["scale"]),
            init_spr=float(params["init_spread"]),
        )
        diagnostics = {"mean_std": math.nan, "p90_std": math.nan, "loglik": math.nan}
    else:
        params = dict(config["trackers"][tracker])
        result = namespace["run_hmm2"](
            horizontal.copy(deep=True),
            typewell.copy(deep=True),
            **params,
        )
        std = np.asarray(result.get("std_eval", []), dtype=float)
        diagnostics = {
            "mean_std": float(np.nanmean(std)) if std.size else math.nan,
            "p90_std": float(np.nanquantile(std, 0.90)) if std.size else math.nan,
            "loglik": float(result.get("loglik", math.nan)),
        }
    prediction = np.asarray(result["pred"], dtype=float)
    if prediction.shape != (len(horizontal),) or not np.isfinite(prediction).all():
        raise AssertionError(f"invalid {tracker} prediction")
    visible = horizontal["TVT_input"].notna().to_numpy()
    if not np.array_equal(prediction[visible], horizontal.loc[visible, "TVT_input"].to_numpy(float)):
        raise AssertionError(f"{tracker} changed visible prefix")
    return prediction, diagnostics


def pseudo_profile(
    namespace: dict[str, Any],
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
    fractions: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible_indices = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())
    if visible_indices.size < 6:
        raise AssertionError("insufficient visible prefix")
    visible_end = int(visible_indices[-1]) + 1
    visible_frame = horizontal.iloc[:visible_end].copy(deep=True)
    original_tvt = visible_frame["TVT_input"].to_numpy(float)
    mse_sums = np.zeros(len(TRACKERS), dtype=float)
    valid_cuts = np.zeros(len(TRACKERS), dtype=np.int64)
    pseudo_rows = np.zeros(len(TRACKERS), dtype=np.int64)
    minimum_rows = int(config["minimum_pseudo_rows"])

    for fraction in fractions:
        cut_count = int(math.floor(float(fraction) * visible_indices.size))
        cut_count = max(3, min(cut_count, visible_indices.size - 1))
        held_indices = visible_indices[cut_count:]
        if held_indices.size < minimum_rows:
            continue
        cut_start = int(held_indices[0])
        pseudo = visible_frame.copy(deep=True)
        pseudo.loc[cut_start:, "TVT_input"] = np.nan
        for tracker_index, tracker in enumerate(TRACKERS):
            prediction, _ = run_tracker(namespace, tracker, pseudo, typewell, config)
            errors = prediction[held_indices] - original_tvt[held_indices]
            if not np.isfinite(errors).all():
                raise AssertionError(f"non-finite pseudo errors for {tracker}")
            mse_sums[tracker_index] += float(np.mean(np.square(errors)))
            valid_cuts[tracker_index] += 1
            pseudo_rows[tracker_index] += int(held_indices.size)

    if np.any(valid_cuts <= 0):
        raise AssertionError("profile has no valid pseudo-cut for at least one tracker")
    return mse_sums / valid_cuts, valid_cuts, pseudo_rows


def read_cache(
    path: Path,
    *,
    row_index: np.ndarray,
    config_sha: str,
    source_sha: str,
    horizontal_sha: str,
    typewell_sha: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        identities = {
            "config_sha256": config_sha,
            "source_sha256": source_sha,
            "horizontal_sha256": horizontal_sha,
            "typewell_sha256": typewell_sha,
        }
        for key, expected in identities.items():
            if str(data[key].item()) != expected:
                raise AssertionError(f"stale cache identity {key}: {path}")
        cached_index = data["row_index"].astype(np.int64)
        if not np.array_equal(cached_index, row_index):
            raise AssertionError(f"row identity mismatch: {path}")
        output: dict[str, Any] = {
            "runtime_seconds": float(data["runtime_seconds"].item()),
            "rss_kb": int(data["rss_kb"].item()),
            "input_immutable": bool(data["input_immutable"].item()),
        }
        for tracker in TRACKERS:
            values = data[f"pred_{tracker}"].astype(float)
            if values.shape != row_index.shape or not np.isfinite(values).all():
                raise AssertionError(f"invalid cached {tracker}: {path}")
            output[f"pred_{tracker}"] = values
        for profile in PROFILES:
            mse = data[f"mse_{profile}"].astype(float)
            cuts = data[f"cuts_{profile}"].astype(np.int64)
            rows = data[f"rows_{profile}"].astype(np.int64)
            if mse.shape != (len(TRACKERS),) or not np.isfinite(mse).all() or np.any(cuts <= 0):
                raise AssertionError(f"invalid cached pseudo profile {profile}: {path}")
            output[f"mse_{profile}"] = mse
            output[f"cuts_{profile}"] = cuts
            output[f"rows_{profile}"] = rows
        return output


def predict_chunk(cache_dir: Path, chunk_index: int, chunks: int) -> int:
    config, source, config_sha, source_sha, _, selected = common()
    namespace, loaded_source_sha = load_public_functions(source)
    if loaded_source_sha != source_sha:
        raise AssertionError("source changed during load")
    data_dir = ROOT / config["data_dir"]
    ids = split_ids(selected, chunk_index, chunks)
    processed = 0
    skipped = 0

    for position, well_id in enumerate(ids, 1):
        horizontal, typewell = load_frames(data_dir, well_id)
        horizontal_sha = frame_digest(horizontal)
        typewell_sha = frame_digest(typewell)
        hidden = horizontal["TVT_input"].isna().to_numpy()
        row_index = np.flatnonzero(hidden).astype(np.int64)
        path = cache_dir / "wells" / f"{well_id}.npz"
        if path.exists():
            read_cache(
                path,
                row_index=row_index,
                config_sha=config_sha,
                source_sha=source_sha,
                horizontal_sha=horizontal_sha,
                typewell_sha=typewell_sha,
            )
            skipped += 1
        else:
            started = time.time()
            arrays: dict[str, Any] = {}
            diagnostics: dict[str, float] = {}
            for tracker in TRACKERS:
                full, diag = run_tracker(namespace, tracker, horizontal, typewell, config)
                arrays[f"pred_{tracker}"] = full[hidden]
                for key, value in diag.items():
                    diagnostics[f"{tracker}_{key}"] = value
            for profile in PROFILES:
                mse, cuts, rows = pseudo_profile(
                    namespace,
                    horizontal,
                    typewell,
                    config,
                    [float(value) for value in config["cut_profiles"][profile]],
                )
                arrays[f"mse_{profile}"] = mse
                arrays[f"cuts_{profile}"] = cuts
                arrays[f"rows_{profile}"] = rows
            immutable = horizontal_sha == frame_digest(horizontal) and typewell_sha == frame_digest(typewell)
            if not immutable:
                raise AssertionError(f"input mutation for {well_id}")
            atomic_npz(
                path,
                row_index=row_index,
                **arrays,
                **{key: np.asarray(value) for key, value in diagnostics.items()},
                config_sha256=np.asarray(config_sha),
                source_sha256=np.asarray(source_sha),
                horizontal_sha256=np.asarray(horizontal_sha),
                typewell_sha256=np.asarray(typewell_sha),
                runtime_seconds=np.asarray(float(time.time() - started)),
                rss_kb=np.asarray(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
                input_immutable=np.asarray(True),
            )
            processed += 1
        if position % 2 == 0 or position == len(ids):
            print(f"predict chunk={chunk_index}/{chunks} {position}/{len(ids)} {well_id}", flush=True)

    receipt = {
        "phase": "predict",
        "chunk_index": chunk_index,
        "chunks": chunks,
        "selected_wells": len(ids),
        "processed": processed,
        "skipped": skipped,
        "config_sha256": config_sha,
        "source_sha256": source_sha,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(cache_dir / "receipts" / f"predict_{chunk_index:03d}_of_{chunks:03d}.json", receipt)
    return 0


def pooled_summary(per_well: pd.DataFrame, candidate: str) -> dict[str, Any]:
    table = per_well[per_well["candidate"] == candidate].copy()
    total_sse = float(table["sse"].sum())
    total_rows = int(table["rows"].sum())
    worst_count = max(1, int(math.ceil(0.20 * len(table))))
    return {
        "rmse": math.sqrt(total_sse / total_rows),
        "sse": total_sse,
        "rows": total_rows,
        "wells": int(len(table)),
        "median_well_rmse": float(table["rmse"].median()),
        "p90_well_rmse": float(table["rmse"].quantile(0.90)),
        "max_well_rmse": float(table["rmse"].max()),
        "worst20_sse_share": float(table.nlargest(worst_count, "sse")["sse"].sum() / total_sse),
    }


def subset_rmse(per_well: pd.DataFrame, candidate: str, ids: set[str]) -> float:
    table = per_well[(per_well["candidate"] == candidate) & (per_well["well_id"].isin(ids))]
    return math.sqrt(float(table["sse"].sum()) / int(table["rows"].sum()))


def inverse_mse_weights(mse: np.ndarray, floor: float) -> np.ndarray:
    weights = 1.0 / (np.asarray(mse, float) + float(floor))
    return weights / weights.sum()


def soft_weights(mse: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(mse, float)
    centered = values - values.min()
    weights = np.exp(-centered / float(temperature))
    return weights / weights.sum()


def rank_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(np.asarray(left, float)).rank(method="average").to_numpy(float)
    right_rank = pd.Series(np.asarray(right, float)).rank(method="average").to_numpy(float)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def load_e011(path: Path, selected: set[str]) -> pd.DataFrame:
    columns = ["well_id", "row_index", "target", "spline4_ridge_equal_s075"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        keep = chunk["well_id"].isin(selected)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    if not parts:
        raise RuntimeError("selected E011 rows not found")
    return pd.concat(parts, ignore_index=True).sort_values(["well_id", "row_index"]).reset_index(drop=True)


def finalize(cache_dir: Path, output_dir: Path) -> int:
    started = time.time()
    config, source, config_sha, source_sha, assignments, selected_list = common()
    selected = set(selected_list)
    cache_paths = sorted((cache_dir / "wells").glob("*.npz"))
    if len(cache_paths) != len(selected_list):
        raise AssertionError(f"expected {len(selected_list)} caches, found {len(cache_paths)}")

    data_dir = ROOT / config["data_dir"]
    e011 = load_e011(ROOT / config["baseline_oof"], selected)
    e011_groups = {str(well_id): group.sort_values("row_index") for well_id, group in e011.groupby("well_id", sort=True)}
    candidate_predictions: dict[str, dict[str, np.ndarray]] = {
        "e011": {},
        "pf": {},
        "hmm_stable": {},
        "hmm_edge": {},
        "fixed_pf_blend": {},
        "inverse_mse_conservative": {},
        "hard_conservative": {},
        "inverse_mse_balanced": {},
        "soft_t4_conservative": {},
        "soft_t8_conservative": {},
        "uniform_tracker": {},
        "control_shuffled_well_pseudo": {},
        "control_rotated_candidate_labels": {},
    }
    targets: dict[str, np.ndarray] = {}
    pseudo_by_well: dict[str, dict[str, np.ndarray]] = {}
    actual_tracker_rmse: dict[str, np.ndarray] = {}
    weight_rows: list[dict[str, Any]] = []
    total_runtime = 0.0
    maximum_rss_kb = 0
    input_immutable = True

    cache_data: dict[str, dict[str, Any]] = {}
    for well_id in selected_list:
        horizontal, typewell = load_frames(data_dir, well_id)
        hidden = horizontal["TVT_input"].isna().to_numpy()
        row_index = np.flatnonzero(hidden).astype(np.int64)
        cache = read_cache(
            cache_dir / "wells" / f"{well_id}.npz",
            row_index=row_index,
            config_sha=config_sha,
            source_sha=source_sha,
            horizontal_sha=frame_digest(horizontal),
            typewell_sha=frame_digest(typewell),
        )
        cache_data[well_id] = cache
        total_runtime += float(cache["runtime_seconds"])
        maximum_rss_kb = max(maximum_rss_kb, int(cache["rss_kb"]))
        input_immutable &= bool(cache["input_immutable"])

    rng = np.random.default_rng(stable_seed("T041_shuffled_well_pseudo"))
    permutation = rng.permutation(len(selected_list))
    if np.array_equal(permutation, np.arange(len(selected_list))):
        permutation = np.roll(permutation, 1)
    shuffled_source = {well_id: selected_list[int(permutation[index])] for index, well_id in enumerate(selected_list)}

    blend_weight = float(config["e011_weight"])
    floor = float(config["inverse_mse_floor"])
    for well_id in selected_list:
        group = e011_groups[well_id]
        cache = cache_data[well_id]
        y = group["target"].to_numpy(float)
        base = group["spline4_ridge_equal_s075"].to_numpy(float)
        trackers = np.vstack([cache[f"pred_{tracker}"] for tracker in TRACKERS])
        if trackers.shape[1] != y.size or not np.isfinite(trackers).all():
            raise AssertionError(f"prediction identity mismatch {well_id}")
        targets[well_id] = y
        candidate_predictions["e011"][well_id] = base
        for index, tracker in enumerate(TRACKERS):
            candidate_predictions[tracker][well_id] = trackers[index]
        candidate_predictions["fixed_pf_blend"][well_id] = blend_weight * base + (1.0 - blend_weight) * trackers[0]

        mse_conservative = np.asarray(cache["mse_conservative"], float)
        mse_balanced = np.asarray(cache["mse_balanced"], float)
        pseudo_by_well[well_id] = {"conservative": mse_conservative, "balanced": mse_balanced}
        weights_primary = inverse_mse_weights(mse_conservative, floor)
        weights_hard = np.zeros(len(TRACKERS), dtype=float)
        weights_hard[int(np.argmin(mse_conservative))] = 1.0
        weights_balanced = inverse_mse_weights(mse_balanced, floor)
        weights_t4 = soft_weights(mse_conservative, float(config["soft_temperatures"][0]))
        weights_t8 = soft_weights(mse_conservative, float(config["soft_temperatures"][1]))
        weights_uniform = np.full(len(TRACKERS), 1.0 / len(TRACKERS))
        control_shuffled = inverse_mse_weights(
            np.asarray(cache_data[shuffled_source[well_id]]["mse_conservative"], float), floor
        )
        control_rotated = inverse_mse_weights(np.roll(mse_conservative, 1), floor)
        schemes = {
            "inverse_mse_conservative": weights_primary,
            "hard_conservative": weights_hard,
            "inverse_mse_balanced": weights_balanced,
            "soft_t4_conservative": weights_t4,
            "soft_t8_conservative": weights_t8,
            "uniform_tracker": weights_uniform,
            "control_shuffled_well_pseudo": control_shuffled,
            "control_rotated_candidate_labels": control_rotated,
        }
        for name, weights in schemes.items():
            tracker_path = np.sum(weights[:, None] * trackers, axis=0)
            candidate_predictions[name][well_id] = blend_weight * base + (1.0 - blend_weight) * tracker_path
        actual = np.asarray([math.sqrt(float(np.mean(np.square(trackers[index] - y)))) for index in range(len(TRACKERS))])
        actual_tracker_rmse[well_id] = actual
        weight_rows.append({
            "well_id": well_id,
            "fold": assignments[well_id],
            **{f"pseudo_mse_{TRACKERS[index]}": float(mse_conservative[index]) for index in range(len(TRACKERS))},
            **{f"actual_rmse_{TRACKERS[index]}": float(actual[index]) for index in range(len(TRACKERS))},
            **{f"weight_{TRACKERS[index]}": float(weights_primary[index]) for index in range(len(TRACKERS))},
            "pseudo_best": TRACKERS[int(np.argmin(mse_conservative))],
            "actual_best": TRACKERS[int(np.argmin(actual))],
            "rank_spearman": rank_spearman(mse_conservative, actual),
        })

    per_well_rows: list[dict[str, Any]] = []
    for candidate, predictions in candidate_predictions.items():
        for well_id in selected_list:
            error = predictions[well_id] - targets[well_id]
            sse = float(np.square(error).sum())
            per_well_rows.append({
                "candidate": candidate,
                "well_id": well_id,
                "fold": assignments[well_id],
                "rows": int(error.size),
                "sse": sse,
                "rmse": math.sqrt(sse / error.size),
            })
    per_well = pd.DataFrame(per_well_rows)
    summaries = {candidate: pooled_summary(per_well, candidate) for candidate in candidate_predictions}
    primary_name = str(config["primary_candidate"])
    reference_name = "fixed_pf_blend"

    fold_rows: list[dict[str, Any]] = []
    fold_wins = 0
    cell_rows: list[dict[str, Any]] = []
    cell_wins = 0
    for fold_id in range(5):
        fold_ids = sorted(well_id for well_id in selected_list if assignments[well_id] == fold_id)
        ids = set(fold_ids)
        primary_rmse = subset_rmse(per_well, primary_name, ids)
        reference_rmse = subset_rmse(per_well, reference_name, ids)
        win = primary_rmse < reference_rmse
        fold_wins += int(win)
        fold_rows.append({
            "fold": fold_id,
            "wells": len(ids),
            "reference_rmse": reference_rmse,
            "primary_rmse": primary_rmse,
            "gain": reference_rmse - primary_rmse,
            "win": win,
        })
        for cell_id, cell_values in enumerate(np.array_split(np.asarray(fold_ids, dtype=object), 5)):
            cell_ids = set(str(value) for value in cell_values.tolist())
            candidate_rmse = subset_rmse(per_well, primary_name, cell_ids)
            base_rmse = subset_rmse(per_well, reference_name, cell_ids)
            cell_win = candidate_rmse < base_rmse
            cell_wins += int(cell_win)
            cell_rows.append({
                "fold": fold_id,
                "cell": cell_id,
                "wells": len(cell_ids),
                "reference_rmse": base_rmse,
                "primary_rmse": candidate_rmse,
                "gain": base_rmse - candidate_rmse,
                "win": cell_win,
            })

    weights_table = pd.DataFrame(weight_rows)
    mean_spearman = float(weights_table["rank_spearman"].mean())
    tracker_hit_rate = float(np.mean(weights_table["pseudo_best"] == weights_table["actual_best"]))

    oracle_candidates: dict[str, dict[str, np.ndarray]] = {
        "e011": candidate_predictions["e011"],
        "pf": candidate_predictions["pf"],
        "hmm_stable": candidate_predictions["hmm_stable"],
        "hmm_edge": candidate_predictions["hmm_edge"],
        "blend_e011_pf": candidate_predictions["fixed_pf_blend"],
        "blend_e011_hmm_stable": {
            well_id: 0.5 * candidate_predictions["e011"][well_id] + 0.5 * candidate_predictions["hmm_stable"][well_id]
            for well_id in selected_list
        },
        "blend_e011_hmm_edge": {
            well_id: 0.5 * candidate_predictions["e011"][well_id] + 0.5 * candidate_predictions["hmm_edge"][well_id]
            for well_id in selected_list
        },
    }
    oracle_rows: list[dict[str, Any]] = []
    oracle_sse = 0.0
    oracle_count = 0
    for well_id in selected_list:
        candidate_sse = {
            name: float(np.square(predictions[well_id] - targets[well_id]).sum())
            for name, predictions in oracle_candidates.items()
        }
        selected_candidate = min(candidate_sse, key=candidate_sse.get)
        oracle_sse += candidate_sse[selected_candidate]
        oracle_count += len(targets[well_id])
        oracle_rows.append({
            "well_id": well_id,
            "selected_candidate": selected_candidate,
            "selected_sse": candidate_sse[selected_candidate],
            "rows": len(targets[well_id]),
            **{f"sse_{name}": value for name, value in candidate_sse.items()},
        })
    component_oracle_rmse = math.sqrt(oracle_sse / oracle_count)

    control_rows = []
    minimum_control_loss = float("inf")
    for control in ("control_shuffled_well_pseudo", "control_rotated_candidate_labels"):
        loss = summaries[control]["rmse"] - summaries[primary_name]["rmse"]
        minimum_control_loss = min(minimum_control_loss, loss)
        control_rows.append({
            "control": control,
            "rmse": summaries[control]["rmse"],
            "primary_rmse": summaries[primary_name]["rmse"],
            "loss_vs_primary": loss,
            "passed": loss >= float(config["gates"]["minimum_control_loss_rmse"]),
        })

    projected_runtime = total_runtime / len(selected_list) * 773.0
    finalization_runtime = time.time() - started
    observed_rss_kb = max(maximum_rss_kb, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    observed_rss_gb = observed_rss_kb / 1024.0 / 1024.0
    gates = config["gates"]
    scientific_gates = {
        "gain_vs_e011": summaries["e011"]["rmse"] - summaries[primary_name]["rmse"] >= float(gates["minimum_gain_vs_e011_rmse"]),
        "gain_vs_fixed_pf_blend": summaries[reference_name]["rmse"] - summaries[primary_name]["rmse"] >= float(gates["minimum_gain_vs_fixed_pf_blend_rmse"]),
        "fold_wins": fold_wins >= int(gates["minimum_fold_wins_vs_fixed_pf_blend"]),
        "cell_wins": cell_wins >= int(gates["minimum_cell_wins_vs_fixed_pf_blend"]),
        "p90": summaries[primary_name]["p90_well_rmse"] <= summaries[reference_name]["p90_well_rmse"] + float(gates["maximum_p90_deterioration_vs_fixed_pf_blend"]),
        "worst20_share": summaries[primary_name]["worst20_sse_share"] <= summaries[reference_name]["worst20_sse_share"] + float(gates["maximum_worst20_share_increase_vs_fixed_pf_blend"]),
        "rank_spearman": mean_spearman >= float(gates["minimum_mean_rank_spearman"]),
        "tracker_hit_rate": tracker_hit_rate >= float(gates["minimum_tracker_hit_rate"]),
        "component_oracle": component_oracle_rmse <= float(gates["maximum_component_oracle_rmse"]),
        "controls": minimum_control_loss >= float(gates["minimum_control_loss_rmse"]),
        "runtime": projected_runtime <= float(gates["maximum_projected_full_runtime_hours"]) * 3600.0,
        "memory": observed_rss_gb <= float(gates["maximum_rss_gb"]),
    }
    structural_gates = {
        "all_caches_complete": len(cache_paths) == len(selected_list),
        "source_identity": sha256_file(source) == source_sha,
        "input_immutability": input_immutable,
        "finite_predictions": all(np.isfinite(values).all() for predictions in candidate_predictions.values() for values in predictions.values()),
        "complete_rows": int(summaries[primary_name]["rows"]) == int(e011.shape[0]),
        "rmse_sse_identity": abs(summaries[primary_name]["rmse"] ** 2 * summaries[primary_name]["rows"] - summaries[primary_name]["sse"]) <= max(1e-6, 1e-10 * summaries[primary_name]["sse"]),
        "valid_pseudo_profiles": all(np.all(cache_data[well_id][f"cuts_{profile}"] > 0) for well_id in selected_list for profile in PROFILES),
    }
    passed = all(bool(value) for value in scientific_gates.values()) and all(bool(value) for value in structural_gates.values())
    decision = "PASS_AUTHORIZE_CLEAN_ROOM_E014" if passed else "REJECT_H033_EXACT_ARBITRATION"

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = [
        {
            "candidate": candidate,
            **summary,
            "gain_vs_e011": summaries["e011"]["rmse"] - summary["rmse"],
            "gain_vs_fixed_pf_blend": summaries[reference_name]["rmse"] - summary["rmse"],
            "primary": candidate == primary_name,
        }
        for candidate, summary in summaries.items()
    ]
    pd.DataFrame(candidate_rows).sort_values(["rmse", "candidate"]).to_csv(output_dir / "candidate_metrics.csv", index=False)
    per_well.to_csv(output_dir / "per_well_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(output_dir / "cell_metrics.csv", index=False)
    weights_table.to_csv(output_dir / "arbitration_metrics.csv", index=False)
    pd.DataFrame(control_rows).to_csv(output_dir / "control_metrics.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(output_dir / "component_oracle.csv", index=False)

    summary = {
        "schema_version": 1,
        "task_id": "T041",
        "hypothesis_id": "H033",
        "decision": decision,
        "passed": passed,
        "primary_candidate": primary_name,
        "reference_candidate": reference_name,
        "baseline": summaries["e011"],
        "reference": summaries[reference_name],
        "primary": summaries[primary_name],
        "gain_vs_e011": summaries["e011"]["rmse"] - summaries[primary_name]["rmse"],
        "gain_vs_fixed_pf_blend": summaries[reference_name]["rmse"] - summaries[primary_name]["rmse"],
        "candidate_metrics": summaries,
        "fold_wins": fold_wins,
        "cell_wins": cell_wins,
        "mean_rank_spearman": mean_spearman,
        "tracker_hit_rate": tracker_hit_rate,
        "component_oracle_rmse": component_oracle_rmse,
        "minimum_control_loss_rmse": minimum_control_loss,
        "prediction_runtime_seconds": total_runtime,
        "projected_773_runtime_seconds": projected_runtime,
        "projected_773_runtime_hours": projected_runtime / 3600.0,
        "finalization_runtime_seconds": finalization_runtime,
        "observed_max_rss_kb": observed_rss_kb,
        "observed_max_rss_gb": observed_rss_gb,
        "scientific_gates": {key: bool(value) for key, value in scientific_gates.items()},
        "structural_gates": {key: bool(value) for key, value in structural_gates.items()},
        "sample_wells": selected_list,
        "config_sha256": config_sha,
        "source_sha256": source_sha,
        "runner_sha256": sha256_file(Path(__file__)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kaggle_run": False,
        "submission_made": False,
    }
    atomic_json(output_dir / "summary.json", summary)

    result_lines = [
        "# T041 result — public-frontier visible-prefix arbitration worth audit",
        "",
        f"Decision: **{decision}**",
        "",
        f"- E011 RMSE: `{summaries['e011']['rmse']:.12f}`",
        f"- Fixed 50/50 E011–PF RMSE: `{summaries[reference_name]['rmse']:.12f}`",
        f"- Primary arbitration RMSE: `{summaries[primary_name]['rmse']:.12f}`",
        f"- Gain versus E011: `{summaries['e011']['rmse'] - summaries[primary_name]['rmse']:.12f}`",
        f"- Gain versus fixed PF blend: `{summaries[reference_name]['rmse'] - summaries[primary_name]['rmse']:.12f}`",
        f"- Fold wins: `{fold_wins}/5`; cell wins: `{cell_wins}/25`",
        f"- Mean rank Spearman: `{mean_spearman:.6f}`; tracker hit rate: `{tracker_hit_rate:.6f}`",
        f"- Component oracle RMSE: `{component_oracle_rmse:.12f}`",
        f"- Minimum control loss: `{minimum_control_loss:.12f}`",
        f"- Projected 773-well runtime: `{projected_runtime / 3600.0:.3f}` hours",
        f"- Observed max RSS: `{observed_rss_gb:.3f}` GB",
        "",
        "Scientific gates:",
        *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in scientific_gates.items()],
        "",
        "Structural gates:",
        *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in structural_gates.items()],
        "",
        "This research source audit cannot promote public code or predictions. No Kaggle run or submission was made.",
    ]
    atomic_text(RESULT_PATH, "\n".join(result_lines) + "\n")

    manifest: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(output_dir / "artifact_manifest.json", {"schema_version": 1, "files": manifest})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "finalize"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "predict":
        return predict_chunk(args.cache_dir.resolve(), args.chunk_index, args.chunks)
    return finalize(args.cache_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
