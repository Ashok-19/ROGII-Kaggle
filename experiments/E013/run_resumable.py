#!/usr/bin/env python3
"""Resumable operational wrapper for the frozen E013 scientific computation.

Prediction phases never read hidden targets or compute scores. Finalization runs
only after every registered full-well and control prediction cache is complete.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/E013"))
from rogii_validation.likelihood_pf import LikelihoodPFConfig, predict_likelihood_pf
import run_experiment as core

CONFIG_PATH = ROOT / "experiments/E013/config.json"
MODULE_PATH = ROOT / "src/rogii_validation/likelihood_pf.py"
DEFAULT_CACHE = ROOT / "scratch/agents/public-frontier-sprint-20260728/e013_cache"
DEFAULT_OUTPUT = ROOT / "experiments/E013/results"
RESULT_MD = ROOT / "experiments/E013/RESULT.md"
CONTROL_MODES = (
    "no_emission",
    "reversed_typewell_gr",
    "circular_shift_hidden_gr",
    "permuted_hidden_gr",
    "duplicate_determinism",
    "reversed_seed_order",
    "all_missing_hidden_gr",
    "short_visible_gr_support",
)


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


def cache_metadata(cfg_sha: str, module_sha: str, hw_sha: str, tw_sha: str, runtime: float) -> dict[str, Any]:
    return {
        "config_sha256": np.asarray(cfg_sha),
        "module_sha256": np.asarray(module_sha),
        "horizontal_digest": np.asarray(hw_sha),
        "typewell_digest": np.asarray(tw_sha),
        "runtime_seconds": np.asarray(float(runtime)),
        "rss_kb": np.asarray(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
        "input_immutable": np.asarray(True),
    }


def read_cache(
    path: Path,
    *,
    row_index: np.ndarray,
    cfg_sha: str,
    module_sha: str,
    hw_sha: str,
    tw_sha: str,
) -> tuple[np.ndarray, float, int, bool]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["config_sha256"].item()) != cfg_sha or str(data["module_sha256"].item()) != module_sha:
            raise AssertionError(f"stale cache implementation identity: {path}")
        if str(data["horizontal_digest"].item()) != hw_sha or str(data["typewell_digest"].item()) != tw_sha:
            raise AssertionError(f"stale cache input identity: {path}")
        cached_index = data["row_index"].astype(np.int64)
        prediction = data["prediction"].astype(float)
        if not np.array_equal(cached_index, row_index):
            raise AssertionError(f"cache row identity mismatch: {path}")
        if prediction.shape != row_index.shape or not np.isfinite(prediction).all():
            raise AssertionError(f"invalid cached prediction: {path}")
        return (
            prediction,
            float(data["runtime_seconds"].item()),
            int(data["rss_kb"].item()),
            bool(data["input_immutable"].item()),
        )


def common() -> tuple[dict[str, Any], LikelihoodPFConfig, Path, str, str, list[dict[str, Any]], list[str], list[str]]:
    cfg = json.loads(CONFIG_PATH.read_text())
    pf_cfg = LikelihoodPFConfig.from_mapping(cfg["particle_filter"])
    data_dir = ROOT / cfg["data_dir"]
    cfg_sha = core.sha256_file(CONFIG_PATH)
    module_sha = core.sha256_file(MODULE_PATH)
    fold_maps = [json.loads((ROOT / path).read_text()) for path in cfg["fold_files"]]
    all_ids = sorted(fold_maps[0]["assignments"])
    if len(all_ids) != int(cfg["expected_wells"]):
        raise AssertionError("fold-map well identity mismatch")
    control_ids = core.choose_panel(
        {k: int(v) for k, v in fold_maps[0]["assignments"].items()},
        int(cfg["control_panel"]["wells_per_fold"]),
    )
    return cfg, pf_cfg, data_dir, cfg_sha, module_sha, fold_maps, all_ids, control_ids


def compute_full(cache_dir: Path, chunk_index: int, chunks: int) -> int:
    cfg, pf_cfg, data_dir, cfg_sha, module_sha, _, all_ids, _ = common()
    selected = split_ids(all_ids, chunk_index, chunks)
    processed = 0
    skipped = 0
    for i, well_id in enumerate(selected, 1):
        path = cache_dir / "full" / f"{well_id}.npz"
        hw = pd.read_csv(data_dir / f"{well_id}__horizontal_well.csv")
        tw = pd.read_csv(data_dir / f"{well_id}__typewell.csv")
        hw_sha = core.digest_frame(hw)
        tw_sha = core.digest_frame(tw)
        hidden = hw["TVT_input"].isna().to_numpy()
        row_index = np.flatnonzero(hidden).astype(np.int64)
        if path.exists():
            read_cache(path, row_index=row_index, cfg_sha=cfg_sha, module_sha=module_sha, hw_sha=hw_sha, tw_sha=tw_sha)
            skipped += 1
        else:
            started = time.time()
            prediction = predict_likelihood_pf(hw, tw, pf_cfg)[hidden]
            immutable = hw_sha == core.digest_frame(hw) and tw_sha == core.digest_frame(tw)
            if not immutable or prediction.shape != row_index.shape or not np.isfinite(prediction).all():
                raise AssertionError(f"invalid full prediction for {well_id}")
            atomic_npz(
                path,
                row_index=row_index,
                prediction=prediction,
                **cache_metadata(cfg_sha, module_sha, hw_sha, tw_sha, time.time() - started),
            )
            processed += 1
        if i % 20 == 0 or i == len(selected):
            print(f"full chunk={chunk_index}/{chunks} {i}/{len(selected)} {well_id}", flush=True)
    receipt = {
        "phase": "full",
        "chunk_index": chunk_index,
        "chunks": chunks,
        "selected_wells": len(selected),
        "processed": processed,
        "skipped": skipped,
        "config_sha256": cfg_sha,
        "module_sha256": module_sha,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    core.atomic_json(cache_dir / "receipts" / f"full_{chunk_index:03d}_of_{chunks:03d}.json", receipt)
    return 0


def compute_control(cache_dir: Path, mode: str, chunk_index: int, chunks: int) -> int:
    if mode not in CONTROL_MODES:
        raise ValueError(mode)
    cfg, pf_cfg, data_dir, cfg_sha, module_sha, _, _, control_ids = common()
    eligible = control_ids[:10] if mode in {"all_missing_hidden_gr", "short_visible_gr_support"} else control_ids
    selected = split_ids(eligible, chunk_index, chunks)
    processed = 0
    skipped = 0
    reverse_ids = tuple(reversed(pf_cfg.seed_ids))
    for i, well_id in enumerate(selected, 1):
        path = cache_dir / "control" / mode / f"{well_id}.npz"
        hw = pd.read_csv(data_dir / f"{well_id}__horizontal_well.csv")
        tw = pd.read_csv(data_dir / f"{well_id}__typewell.csv")
        hw_sha = core.digest_frame(hw)
        tw_sha = core.digest_frame(tw)
        hidden = hw["TVT_input"].isna().to_numpy()
        row_index = np.flatnonzero(hidden).astype(np.int64)
        if path.exists():
            read_cache(path, row_index=row_index, cfg_sha=cfg_sha, module_sha=module_sha, hw_sha=hw_sha, tw_sha=tw_sha)
            skipped += 1
        else:
            started = time.time()
            if mode == "no_emission":
                prediction = predict_likelihood_pf(hw, tw, pf_cfg, use_emission=False)[hidden]
            elif mode == "duplicate_determinism":
                prediction = predict_likelihood_pf(hw, tw, pf_cfg)[hidden]
            elif mode == "reversed_seed_order":
                prediction = predict_likelihood_pf(hw, tw, pf_cfg, seed_ids=reverse_ids)[hidden]
            else:
                modified_h, modified_t = core.modify_frames(hw, tw, mode, well_id)
                prediction = predict_likelihood_pf(modified_h, modified_t, pf_cfg)[hidden]
            immutable = hw_sha == core.digest_frame(hw) and tw_sha == core.digest_frame(tw)
            if not immutable or prediction.shape != row_index.shape or not np.isfinite(prediction).all():
                raise AssertionError(f"invalid {mode} prediction for {well_id}")
            atomic_npz(
                path,
                row_index=row_index,
                prediction=prediction,
                **cache_metadata(cfg_sha, module_sha, hw_sha, tw_sha, time.time() - started),
            )
            processed += 1
        if i % 20 == 0 or i == len(selected):
            print(f"control={mode} chunk={chunk_index}/{chunks} {i}/{len(selected)} {well_id}", flush=True)
    receipt = {
        "phase": "control",
        "mode": mode,
        "chunk_index": chunk_index,
        "chunks": chunks,
        "selected_wells": len(selected),
        "processed": processed,
        "skipped": skipped,
        "config_sha256": cfg_sha,
        "module_sha256": module_sha,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    core.atomic_json(cache_dir / "receipts" / f"control_{mode}_{chunk_index:03d}_of_{chunks:03d}.json", receipt)
    return 0


def finalize(cache_dir: Path, out: Path) -> int:
    finalize_started = time.time()
    cfg, pf_cfg, data_dir, cfg_sha, module_sha, fold_maps, all_ids, control_ids = common()
    out.mkdir(parents=True, exist_ok=True)
    e011 = core.load_e011(ROOT / cfg["baseline_oof"])
    e011_groups = {wid: g.copy() for wid, g in e011.groupby("well_id", sort=True)}
    if len(e011_groups) != int(cfg["expected_wells"]) or len(e011) != int(cfg["expected_hidden_rows"]):
        raise AssertionError("E011 identity mismatch")

    membership = pd.read_csv(ROOT / cfg["group_membership_source"])
    group_cols = ["legacy_spatial", "legacy_typewell", "legal_covariate_kmeans", "spatial_2d_kmeans", "horizon_quintile"]
    if len(membership) != int(cfg["expected_wells"]) or set(membership["well_id"]) != set(e011_groups):
        raise AssertionError("group-membership identity mismatch")
    membership = membership[["well_id", *group_cols]].drop_duplicates("well_id")
    membership_map = membership.set_index("well_id").to_dict("index")

    candidates = ["e011", "pf", "blend_w0.50", "blend_w0.75"]
    rows: list[dict[str, Any]] = []
    observables: list[dict[str, Any]] = []
    full_predictions: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    baselines: dict[str, np.ndarray] = {}
    row_indices: dict[str, np.ndarray] = {}
    total_compute_runtime = 0.0
    max_cached_rss_kb = 0
    input_immutable = True
    exact_e011_identity = True

    oof_path = out / "oof_predictions.csv.gz"
    oof_tmp = out / "oof_predictions.csv.gz.tmp"
    with oof_tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(["id", "well_id", "row_index", "target", "e011", "pf", "blend_w0.50", "blend_w0.75"])
                for i, well_id in enumerate(all_ids, 1):
                    cache_path = cache_dir / "full" / f"{well_id}.npz"
                    if not cache_path.exists():
                        raise AssertionError(f"missing full cache {well_id}")
                    hw = pd.read_csv(data_dir / f"{well_id}__horizontal_well.csv")
                    tw = pd.read_csv(data_dir / f"{well_id}__typewell.csv")
                    hw_sha = core.digest_frame(hw)
                    tw_sha = core.digest_frame(tw)
                    hidden = hw["TVT_input"].isna().to_numpy()
                    idx = np.flatnonzero(hidden).astype(np.int64)
                    pf, runtime, rss_kb, immutable = read_cache(
                        cache_path,
                        row_index=idx,
                        cfg_sha=cfg_sha,
                        module_sha=module_sha,
                        hw_sha=hw_sha,
                        tw_sha=tw_sha,
                    )
                    total_compute_runtime += runtime
                    max_cached_rss_kb = max(max_cached_rss_kb, rss_kb)
                    input_immutable &= immutable
                    g = e011_groups[well_id].sort_values("row_index")
                    if not np.array_equal(g["row_index"].to_numpy(np.int64), idx):
                        raise AssertionError(f"E011 row identity mismatch {well_id}")
                    y = g["target"].to_numpy(float)
                    base = g["spline4_ridge_equal_s075"].to_numpy(float)
                    pred = {
                        "e011": base.copy(),
                        "pf": pf,
                        "blend_w0.50": 0.50 * base + 0.50 * pf,
                        "blend_w0.75": 0.25 * base + 0.75 * pf,
                    }
                    exact_e011_identity &= np.array_equal(pred["e011"], base)
                    if any(v.shape != y.shape or not np.isfinite(v).all() for v in pred.values()):
                        raise AssertionError(f"invalid final prediction {well_id}")
                    for name in candidates:
                        err = pred[name] - y
                        sse = float(np.square(err).sum())
                        rows.append({"well_id": well_id, "candidate": name, "rows": len(y), "sse": sse, "rmse": math.sqrt(sse / len(y)), **membership_map[well_id]})
                    for row_index, target, b, p, c50, c75 in zip(idx, y, base, pf, pred["blend_w0.50"], pred["blend_w0.75"]):
                        writer.writerow([f"{well_id}_{row_index}", well_id, int(row_index), f"{target:.10f}", f"{b:.10f}", f"{p:.10f}", f"{c50:.10f}", f"{c75:.10f}"])
                    hidden_gr = hw.loc[hidden, "GR"].to_numpy(float)
                    observables.append({"well_id": well_id, "hidden_rows": int(hidden.sum()), "hidden_gr_missing_fraction": float(np.mean(~np.isfinite(hidden_gr)))})
                    full_predictions[well_id] = pf
                    targets[well_id] = y
                    baselines[well_id] = base
                    row_indices[well_id] = idx
                    if i % 100 == 0 or i == len(all_ids):
                        print(f"finalize full {i}/{len(all_ids)} {well_id}", flush=True)
    oof_tmp.replace(oof_path)

    per_well = pd.DataFrame(rows)
    selected = "blend_w0.75"
    summary_by_candidate = {name: core.summarize(per_well, name) for name in candidates}
    base_summary = summary_by_candidate["e011"]
    selected_summary = summary_by_candidate[selected]

    map_rows: list[dict[str, Any]] = []
    map_wins = 0
    cell_wins = 0
    for map_index, fmap in enumerate(fold_maps, 1):
        assignments = {k: int(v) for k, v in fmap["assignments"].items()}
        wins_this_map = 0
        for fold in range(int(fmap["n_folds"])):
            ids = {k for k, v in assignments.items() if v == fold}
            b_rmse, _, n = core.subset_metric(per_well, "e011", ids)
            c_rmse, _, _ = core.subset_metric(per_well, selected, ids)
            win = c_rmse < b_rmse
            wins_this_map += int(win)
            cell_wins += int(win)
            map_rows.append({"map": map_index, "fold": fold, "rows": n, "baseline_rmse": b_rmse, "candidate_rmse": c_rmse, "gain": b_rmse - c_rmse, "win": win})
        map_wins += int(wins_this_map >= 3)

    group_rows: list[dict[str, Any]] = []
    every_legacy_positive = True
    for column in group_cols:
        for group in sorted(membership[column].unique()):
            ids = set(membership.loc[membership[column] == group, "well_id"])
            b, _, n = core.subset_metric(per_well, "e011", ids)
            c, _, _ = core.subset_metric(per_well, selected, ids)
            gain = b - c
            group_rows.append({"group_family": column, "group": int(group), "wells": len(ids), "rows": n, "baseline_rmse": b, "candidate_rmse": c, "gain": gain})
            if column in {"legacy_spatial", "legacy_typewell"}:
                every_legacy_positive &= gain > 0

    obs = pd.DataFrame(observables)
    baseline_well = per_well[per_well["candidate"] == "e011"][["well_id", "rmse"]].rename(columns={"rmse": "baseline_well_rmse"})
    obs = obs.merge(baseline_well, on="well_id", validate="one_to_one")
    suffix_threshold = float(obs["hidden_rows"].quantile(0.80))
    missing_threshold = float(obs["hidden_gr_missing_fraction"].quantile(0.80))
    slices = {
        "long_suffix": set(obs.loc[obs["hidden_rows"] >= suffix_threshold, "well_id"]),
        "high_hidden_gr_missingness": set(obs.loc[obs["hidden_gr_missing_fraction"] >= missing_threshold, "well_id"]),
        "e011_catastrophe_ge20": set(obs.loc[obs["baseline_well_rmse"] >= 20.0, "well_id"]),
    }
    slice_rows: list[dict[str, Any]] = []
    for name, ids in slices.items():
        b, _, n = core.subset_metric(per_well, "e011", ids)
        c, _, _ = core.subset_metric(per_well, selected, ids)
        slice_rows.append({"slice": name, "wells": len(ids), "rows": n, "baseline_rmse": b, "candidate_rmse": c, "gain": b - c})

    def fixed_blend(base: np.ndarray, pf: np.ndarray) -> np.ndarray:
        return 0.25 * base + 0.75 * pf

    def pooled_control(predictions: dict[str, np.ndarray]) -> float:
        sse = sum(float(np.square(predictions[wid] - targets[wid]).sum()) for wid in control_ids)
        rows_count = sum(len(targets[wid]) for wid in control_ids)
        return math.sqrt(sse / rows_count)

    legal_predictions = {wid: fixed_blend(baselines[wid], full_predictions[wid]) for wid in control_ids}
    legal_control_rmse = pooled_control(legal_predictions)
    control_rows: list[dict[str, Any]] = []
    destructive_pass = True
    minimum_advantage = float("inf")
    all_control_caches_complete = True
    for mode in cfg["destructive_controls"]:
        predictions: dict[str, np.ndarray] = {}
        for wid in control_ids:
            hw = pd.read_csv(data_dir / f"{wid}__horizontal_well.csv")
            tw = pd.read_csv(data_dir / f"{wid}__typewell.csv")
            path = cache_dir / "control" / mode / f"{wid}.npz"
            if not path.exists():
                raise AssertionError(f"missing {mode} cache {wid}")
            control_pf, runtime, rss_kb, immutable = read_cache(
                path,
                row_index=row_indices[wid],
                cfg_sha=cfg_sha,
                module_sha=module_sha,
                hw_sha=core.digest_frame(hw),
                tw_sha=core.digest_frame(tw),
            )
            total_compute_runtime += runtime
            max_cached_rss_kb = max(max_cached_rss_kb, rss_kb)
            input_immutable &= immutable
            predictions[wid] = fixed_blend(baselines[wid], control_pf)
        rmse = pooled_control(predictions)
        advantage = rmse - legal_control_rmse
        passed = advantage >= float(cfg["gates"]["minimum_control_advantage_rmse"])
        destructive_pass &= passed
        minimum_advantage = min(minimum_advantage, advantage)
        control_rows.append({"control": mode, "rmse": rmse, "legal_rmse": legal_control_rmse, "legal_advantage": advantage, "passed": passed})

    duplicate_max_delta = 0.0
    reverse_seed_max_delta = 0.0
    for mode in ("duplicate_determinism", "reversed_seed_order"):
        predictions: dict[str, np.ndarray] = {}
        for wid in control_ids:
            hw = pd.read_csv(data_dir / f"{wid}__horizontal_well.csv")
            tw = pd.read_csv(data_dir / f"{wid}__typewell.csv")
            path = cache_dir / "control" / mode / f"{wid}.npz"
            if not path.exists():
                raise AssertionError(f"missing {mode} cache {wid}")
            control_pf, runtime, rss_kb, immutable = read_cache(
                path,
                row_index=row_indices[wid],
                cfg_sha=cfg_sha,
                module_sha=module_sha,
                hw_sha=core.digest_frame(hw),
                tw_sha=core.digest_frame(tw),
            )
            total_compute_runtime += runtime
            max_cached_rss_kb = max(max_cached_rss_kb, rss_kb)
            input_immutable &= immutable
            predictions[wid] = fixed_blend(baselines[wid], control_pf)
            delta = float(np.max(np.abs(predictions[wid] - legal_predictions[wid])))
            if mode == "duplicate_determinism":
                duplicate_max_delta = max(duplicate_max_delta, delta)
            else:
                reverse_seed_max_delta = max(reverse_seed_max_delta, delta)
        control_rows.append({
            "control": mode,
            "rmse": pooled_control(predictions),
            "legal_rmse": legal_control_rmse,
            "legal_advantage": 0.0,
            "passed": duplicate_max_delta <= 1e-10 if mode == "duplicate_determinism" else reverse_seed_max_delta <= 1e-9,
            "max_abs_delta": duplicate_max_delta if mode == "duplicate_determinism" else reverse_seed_max_delta,
        })

    missing_pass = True
    for mode in ("all_missing_hidden_gr", "short_visible_gr_support"):
        finite = True
        for wid in control_ids[:10]:
            hw = pd.read_csv(data_dir / f"{wid}__horizontal_well.csv")
            tw = pd.read_csv(data_dir / f"{wid}__typewell.csv")
            path = cache_dir / "control" / mode / f"{wid}.npz"
            if not path.exists():
                raise AssertionError(f"missing {mode} cache {wid}")
            prediction, runtime, rss_kb, immutable = read_cache(
                path,
                row_index=row_indices[wid],
                cfg_sha=cfg_sha,
                module_sha=module_sha,
                hw_sha=core.digest_frame(hw),
                tw_sha=core.digest_frame(tw),
            )
            total_compute_runtime += runtime
            max_cached_rss_kb = max(max_cached_rss_kb, rss_kb)
            input_immutable &= immutable
            finite &= prediction.shape == targets[wid].shape and np.isfinite(prediction).all()
        missing_pass &= finite
        control_rows.append({"control": mode, "rmse": np.nan, "legal_rmse": legal_control_rmse, "legal_advantage": np.nan, "passed": finite})

    finalization_runtime = time.time() - finalize_started
    total_runtime = total_compute_runtime + finalization_runtime
    current_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_kb = max(current_rss_kb, max_cached_rss_kb)
    rss_gb = rss_kb / 1024.0 / 1024.0
    gates = cfg["gates"]
    scientific_gates = {
        "go_rmse": selected_summary["rmse"] <= float(gates["go_rmse"]),
        "map_wins": map_wins >= int(gates["minimum_map_wins"]),
        "cell_wins": cell_wins >= int(gates["minimum_cell_wins"]),
        "every_legacy_group_positive": every_legacy_positive,
        "p90": selected_summary["p90_well_rmse"] <= base_summary["p90_well_rmse"] + float(gates["maximum_p90_deterioration"]),
        "worst5_share": selected_summary["worst5_sse_share"] <= base_summary["worst5_sse_share"] + float(gates["maximum_worst5_sse_share_increase"]),
        "destructive_controls": destructive_pass,
        "runtime": total_runtime <= float(gates["maximum_runtime_hours"]) * 3600.0,
        "memory": rss_gb <= float(gates["maximum_rss_gb"]),
    }
    freeze = json.loads((ROOT / "experiments/E013/IMPLEMENTATION_FREEZE.json").read_text())
    structural_gates = {
        "source_parity_smoke": bool(freeze.get("source_parity", {}).get("all_pass", False)),
        "all_full_caches_complete": len(list((cache_dir / "full").glob("*.npz"))) == int(cfg["expected_wells"]),
        "all_control_caches_complete": all_control_caches_complete,
        "duplicate_determinism": duplicate_max_delta <= 1e-10,
        "reversed_seed_order": reverse_seed_max_delta <= 1e-9,
        "missing_data": missing_pass,
        "input_immutability": input_immutable,
        "exact_e011_fallback": exact_e011_identity,
        "finite_complete_rows": int(selected_summary["rows"]) == int(cfg["expected_hidden_rows"]),
        "rmse_sse_identity": abs(selected_summary["rmse"] ** 2 * selected_summary["rows"] - selected_summary["sse"]) <= max(1e-6, 1e-10 * selected_summary["sse"]),
        "group_membership_identity": len(membership) == int(cfg["expected_wells"]),
    }
    scientific_gates = {key: bool(value) for key, value in scientific_gates.items()}
    structural_gates = {key: bool(value) for key, value in structural_gates.items()}
    go = bool(all(scientific_gates.values()) and all(structural_gates.values()))
    breakthrough = bool(go and selected_summary["rmse"] <= float(gates["breakthrough_rmse"]))
    decision = "BREAKTHROUGH_SUB6" if breakthrough else ("GO_CLEAN_ROOM_PF" if go else "REJECT_E013")

    candidate_rows = [{"candidate": name, **summary_by_candidate[name], "gain_vs_e011": base_summary["rmse"] - summary_by_candidate[name]["rmse"], "primary": name == selected} for name in candidates]
    pd.DataFrame(candidate_rows).to_csv(out / "candidate_metrics.csv", index=False)
    pd.DataFrame(map_rows).to_csv(out / "map_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "group_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(out / "slice_metrics.csv", index=False)
    pd.DataFrame(control_rows).to_csv(out / "control_metrics.csv", index=False)
    per_well.to_csv(out / "selected_well_metrics.csv", index=False)

    summary = {
        "schema_version": 1,
        "experiment_id": "E013",
        "execution_mode": "resumable_score_blind_prediction_caches_then_single_finalization",
        "decision": decision,
        "go": go,
        "breakthrough": breakthrough,
        "primary_candidate": selected,
        "baseline": base_summary,
        "primary": selected_summary,
        "gain_vs_e011": base_summary["rmse"] - selected_summary["rmse"],
        "candidate_metrics": summary_by_candidate,
        "map_wins": map_wins,
        "cell_wins": cell_wins,
        "every_legacy_group_positive": every_legacy_positive,
        "control_panel_wells": control_ids,
        "legal_control_rmse": legal_control_rmse,
        "minimum_control_advantage": minimum_advantage,
        "scientific_gates": scientific_gates,
        "structural_gates": structural_gates,
        "compute_runtime_seconds": total_compute_runtime,
        "finalization_runtime_seconds": finalization_runtime,
        "runtime_seconds": total_runtime,
        "runtime_hours": total_runtime / 3600.0,
        "observed_max_rss_kb": rss_kb,
        "observed_max_rss_gb": rss_gb,
        "config_sha256": cfg_sha,
        "module_sha256": module_sha,
        "resumable_runner_sha256": core.sha256_file(Path(__file__)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kaggle_run": False,
        "submission_made": False,
    }
    core.atomic_json(out / "summary.json", summary)
    lines = [
        "# E013 result — clean-room sequential likelihood-PF",
        "",
        f"Decision: **{decision}**",
        "",
        f"- E011 RMSE: `{base_summary['rmse']:.12f}`",
        f"- Fixed 0.75 PF blend RMSE: `{selected_summary['rmse']:.12f}`",
        f"- Gain: `{base_summary['rmse'] - selected_summary['rmse']:.12f}`",
        f"- Map wins: `{map_wins}/5`",
        f"- Cell wins: `{cell_wins}/25`",
        f"- p90: `{selected_summary['p90_well_rmse']:.12f}` versus `{base_summary['p90_well_rmse']:.12f}`",
        f"- worst-5% SSE share: `{selected_summary['worst5_sse_share']:.12f}` versus `{base_summary['worst5_sse_share']:.12f}`",
        f"- minimum destructive-control advantage: `{minimum_advantage:.12f}`",
        f"- accumulated compute plus finalization runtime: `{total_runtime / 3600.0:.3f}` hours",
        f"- max RSS: `{rss_gb:.3f}` GB",
        "",
        "Scientific gates:",
        *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in scientific_gates.items()],
        "",
        "Structural gates:",
        *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in structural_gates.items()],
        "",
        "No Kaggle run or submission was made.",
    ]
    core.atomic_text(RESULT_MD, "\n".join(lines) + "\n")
    manifest = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": core.sha256_file(path)}
    core.atomic_json(out / "artifact_manifest.json", {"schema_version": 1, "files": manifest})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("full", "control", "finalize"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--mode", choices=CONTROL_MODES)
    args = parser.parse_args()
    cache_dir = args.cache_dir.resolve()
    if args.phase == "full":
        return compute_full(cache_dir, args.chunk_index, args.chunks)
    if args.phase == "control":
        if args.mode is None:
            parser.error("--mode is required for control phase")
        return compute_control(cache_dir, args.mode, args.chunk_index, args.chunks)
    return finalize(cache_dir, args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
