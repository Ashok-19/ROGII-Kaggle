#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
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

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rogii_validation.likelihood_pf import LikelihoodPFConfig, predict_likelihood_pf

CONFIG_PATH = ROOT / "experiments/E013/config.json"
RESULT_DIR = ROOT / "experiments/E013/results"
RESULT_MD = ROOT / "experiments/E013/RESULT.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def digest_frame(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("|".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy(np.uint64).tobytes())
    return h.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def choose_panel(assignments: dict[str, int], per_fold: int) -> list[str]:
    out: list[str] = []
    for fold in range(5):
        ids = sorted(k for k, v in assignments.items() if int(v) == fold)
        idx = np.linspace(0, len(ids) - 1, per_fold, dtype=int)
        out.extend(ids[int(i)] for i in idx)
    if len(out) != 5 * per_fold or len(set(out)) != len(out):
        raise AssertionError("control-panel identity failure")
    return out


def load_e011(path: Path) -> pd.DataFrame:
    cols = ["well_id", "row_index", "target", "spline4_ridge_equal_s075"]
    df = pd.read_csv(path, usecols=cols)
    if df.duplicated(["well_id", "row_index"]).any():
        raise AssertionError("duplicate E011 rows")
    return df.sort_values(["well_id", "row_index"]).reset_index(drop=True)


def modify_frames(hw: pd.DataFrame, tw: pd.DataFrame, mode: str, well_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = hw.copy(deep=True)
    t = tw.copy(deep=True)
    hidden = h["TVT_input"].isna().to_numpy()
    idx = np.flatnonzero(hidden)
    if mode == "reversed_typewell_gr":
        t.loc[:, "GR"] = t["GR"].to_numpy()[::-1]
    elif mode == "circular_shift_hidden_gr":
        vals = h.loc[idx, "GR"].to_numpy(copy=True)
        h.loc[idx, "GR"] = np.roll(vals, max(1, len(vals) // 3))
    elif mode == "permuted_hidden_gr":
        vals = h.loc[idx, "GR"].to_numpy(copy=True)
        rng = np.random.default_rng(stable_seed(well_id + mode))
        h.loc[idx, "GR"] = vals[rng.permutation(len(vals))]
    elif mode == "all_missing_hidden_gr":
        h.loc[idx, "GR"] = np.nan
    elif mode == "short_visible_gr_support":
        known = np.flatnonzero(~hidden)
        finite = known[np.isfinite(h.loc[known, "GR"].to_numpy(float))]
        keep = set(finite[-5:].tolist())
        h.loc[[i for i in known if i not in keep], "GR"] = np.nan
    else:
        raise ValueError(mode)
    return h, t


def summarize(per_well: pd.DataFrame, candidate: str) -> dict[str, float | int]:
    g = per_well[per_well["candidate"] == candidate]
    sse = float(g["sse"].sum())
    rows = int(g["rows"].sum())
    ordered = g.sort_values("sse", ascending=False)
    n5 = max(1, math.ceil(0.05 * len(g)))
    n10 = max(1, math.ceil(0.10 * len(g)))
    return {
        "rmse": math.sqrt(sse / rows),
        "sse": sse,
        "rows": rows,
        "wells": int(len(g)),
        "median_well_rmse": float(g["rmse"].median()),
        "p90_well_rmse": float(g["rmse"].quantile(0.90)),
        "p95_well_rmse": float(g["rmse"].quantile(0.95)),
        "max_well_rmse": float(g["rmse"].max()),
        "worst5_sse_share": float(ordered.head(n5)["sse"].sum() / sse),
        "worst10_sse_share": float(ordered.head(n10)["sse"].sum() / sse),
    }


def subset_metric(per_well: pd.DataFrame, candidate: str, ids: set[str]) -> tuple[float, float, int]:
    g = per_well[(per_well["candidate"] == candidate) & (per_well["well_id"].isin(ids))]
    sse = float(g["sse"].sum())
    rows = int(g["rows"].sum())
    return math.sqrt(sse / rows), sse, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text())
    pf_cfg = LikelihoodPFConfig.from_mapping(cfg["particle_filter"])
    data_dir = ROOT / cfg["data_dir"]
    e011 = load_e011(ROOT / cfg["baseline_oof"])
    e011_groups = {wid: g.copy() for wid, g in e011.groupby("well_id", sort=True)}
    if len(e011_groups) != int(cfg["expected_wells"]) or len(e011) != int(cfg["expected_hidden_rows"]):
        raise AssertionError("E011 identity mismatch")

    membership = pd.read_csv(ROOT / cfg["group_membership_source"])
    group_cols = ["legacy_spatial", "legacy_typewell", "legal_covariate_kmeans", "spatial_2d_kmeans", "horizon_quintile"]
    if len(membership) != int(cfg["expected_wells"]) or set(membership["well_id"]) != set(e011_groups):
        raise AssertionError("group-membership identity mismatch")
    membership = membership[["well_id", *group_cols]].drop_duplicates("well_id")
    membership_map = membership.set_index("well_id").to_dict("index")

    fold_maps: list[dict[str, Any]] = []
    for path in cfg["fold_files"]:
        raw = json.loads((ROOT / path).read_text())
        fold_maps.append(raw)
    control_ids = choose_panel({k: int(v) for k, v in fold_maps[0]["assignments"].items()}, int(cfg["control_panel"]["wells_per_fold"]))
    if len(control_ids) != int(cfg["control_panel"]["expected_wells"]):
        raise AssertionError("control-panel size mismatch")

    candidates = ["e011", "pf", "blend_w0.50", "blend_w0.75"]
    weights = {"blend_w0.50": 0.50, "blend_w0.75": 0.75}
    rows: list[dict[str, Any]] = []
    observables: list[dict[str, Any]] = []
    control_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]] = {}
    input_immutable = True
    oof_path = out / "oof_predictions.csv.gz"
    oof_tmp = out / "oof_predictions.csv.gz.tmp"
    started = time.time()
    exact_e011_identity = True

    with oof_tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["id", "well_id", "row_index", "target", "e011", "pf", "blend_w0.50", "blend_w0.75"])
                for i, well_id in enumerate(sorted(e011_groups), 1):
                    hw = pd.read_csv(data_dir / f"{well_id}__horizontal_well.csv")
                    tw = pd.read_csv(data_dir / f"{well_id}__typewell.csv")
                    h0, t0 = digest_frame(hw), digest_frame(tw)
                    hidden = hw["TVT_input"].isna().to_numpy()
                    idx = np.flatnonzero(hidden)
                    g = e011_groups[well_id].sort_values("row_index")
                    if not np.array_equal(g["row_index"].to_numpy(int), idx):
                        raise AssertionError(f"row identity mismatch {well_id}")
                    y = g["target"].to_numpy(float)
                    base = g["spline4_ridge_equal_s075"].to_numpy(float)
                    full_pf = predict_likelihood_pf(hw, tw, pf_cfg)
                    pf = full_pf[hidden]
                    pred = {
                        "e011": base.copy(),
                        "pf": pf,
                        "blend_w0.50": 0.50 * base + 0.50 * pf,
                        "blend_w0.75": 0.25 * base + 0.75 * pf,
                    }
                    exact_e011_identity &= np.array_equal(pred["e011"], base)
                    if any(v.shape != y.shape or not np.isfinite(v).all() for v in pred.values()):
                        raise AssertionError(f"invalid output {well_id}")
                    for name in candidates:
                        err = pred[name] - y
                        sse = float(np.square(err).sum())
                        rows.append({"well_id": well_id, "candidate": name, "rows": len(y), "sse": sse, "rmse": math.sqrt(sse / len(y)), **membership_map[well_id]})
                    for r, target, b, p, c50, c75 in zip(idx, y, base, pf, pred["blend_w0.50"], pred["blend_w0.75"]):
                        writer.writerow([f"{well_id}_{r}", well_id, int(r), f"{target:.10f}", f"{b:.10f}", f"{p:.10f}", f"{c50:.10f}", f"{c75:.10f}"])
                    hidden_gr = hw.loc[hidden, "GR"].to_numpy(float)
                    observables.append({"well_id": well_id, "hidden_rows": int(hidden.sum()), "hidden_gr_missing_fraction": float(np.mean(~np.isfinite(hidden_gr)))})
                    input_immutable &= h0 == digest_frame(hw) and t0 == digest_frame(tw)
                    if well_id in set(control_ids):
                        control_cache[well_id] = (hw, tw, hidden, y, base)
                    if i % 25 == 0 or i == len(e011_groups):
                        print(f"full {i}/{len(e011_groups)} {well_id}", flush=True)
    oof_tmp.replace(oof_path)

    per_well = pd.DataFrame(rows)
    selected = "blend_w0.75"
    summary_by_candidate = {name: summarize(per_well, name) for name in candidates}

    map_rows: list[dict[str, Any]] = []
    map_wins = 0
    cell_wins = 0
    for map_index, fmap in enumerate(fold_maps, 1):
        assignments = {k: int(v) for k, v in fmap["assignments"].items()}
        wins_this_map = 0
        for fold in range(int(fmap["n_folds"])):
            ids = {k for k, v in assignments.items() if v == fold}
            b_rmse, _, n = subset_metric(per_well, "e011", ids)
            c_rmse, _, _ = subset_metric(per_well, selected, ids)
            win = c_rmse < b_rmse
            wins_this_map += int(win)
            cell_wins += int(win)
            map_rows.append({"map": map_index, "fold": fold, "rows": n, "baseline_rmse": b_rmse, "candidate_rmse": c_rmse, "gain": b_rmse - c_rmse, "win": win})
        map_wins += int(wins_this_map >= 3)

    group_rows: list[dict[str, Any]] = []
    every_legacy_positive = True
    for col in group_cols:
        for group in sorted(membership[col].unique()):
            ids = set(membership.loc[membership[col] == group, "well_id"])
            b, _, n = subset_metric(per_well, "e011", ids)
            c, _, _ = subset_metric(per_well, selected, ids)
            gain = b - c
            group_rows.append({"group_family": col, "group": int(group), "wells": len(ids), "rows": n, "baseline_rmse": b, "candidate_rmse": c, "gain": gain})
            if col in {"legacy_spatial", "legacy_typewell"}:
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
        b, _, n = subset_metric(per_well, "e011", ids)
        c, _, _ = subset_metric(per_well, selected, ids)
        slice_rows.append({"slice": name, "wells": len(ids), "rows": n, "baseline_rmse": b, "candidate_rmse": c, "gain": b - c})

    legal_control_predictions: dict[str, np.ndarray] = {}
    legal_y: dict[str, np.ndarray] = {}
    legal_base: dict[str, np.ndarray] = {}
    for well_id in control_ids:
        hw, tw, hidden, y, base = control_cache[well_id]
        legal_pf = predict_likelihood_pf(hw, tw, pf_cfg)[hidden]
        legal_control_predictions[well_id] = 0.25 * base + 0.75 * legal_pf
        legal_y[well_id] = y
        legal_base[well_id] = base

    def control_rmse(preds: dict[str, np.ndarray]) -> float:
        sse = sum(float(np.square(preds[w] - legal_y[w]).sum()) for w in control_ids)
        n = sum(len(legal_y[w]) for w in control_ids)
        return math.sqrt(sse / n)

    legal_control_rmse = control_rmse(legal_control_predictions)
    control_rows: list[dict[str, Any]] = []
    destructive_pass = True
    min_advantage = float("inf")
    for mode in cfg["destructive_controls"]:
        preds: dict[str, np.ndarray] = {}
        for j, well_id in enumerate(control_ids, 1):
            hw, tw, hidden, y, base = control_cache[well_id]
            if mode == "no_emission":
                pf = predict_likelihood_pf(hw, tw, pf_cfg, use_emission=False)[hidden]
            else:
                mh, mt = modify_frames(hw, tw, mode, well_id)
                pf = predict_likelihood_pf(mh, mt, pf_cfg)[hidden]
            preds[well_id] = 0.25 * base + 0.75 * pf
            if j % 25 == 0:
                print(f"control {mode} {j}/{len(control_ids)}", flush=True)
        rmse = control_rmse(preds)
        advantage = rmse - legal_control_rmse
        passed = advantage >= float(cfg["gates"]["minimum_control_advantage_rmse"])
        destructive_pass &= passed
        min_advantage = min(min_advantage, advantage)
        control_rows.append({"control": mode, "rmse": rmse, "legal_rmse": legal_control_rmse, "legal_advantage": advantage, "passed": passed})

    duplicate_preds: dict[str, np.ndarray] = {}
    reverse_seed_preds: dict[str, np.ndarray] = {}
    max_duplicate_delta = 0.0
    max_seed_order_delta = 0.0
    reverse_ids = tuple(reversed(pf_cfg.seed_ids))
    for j, well_id in enumerate(control_ids, 1):
        hw, tw, hidden, y, base = control_cache[well_id]
        duplicate_pf = predict_likelihood_pf(hw, tw, pf_cfg)[hidden]
        reversed_pf = predict_likelihood_pf(hw, tw, pf_cfg, seed_ids=reverse_ids)[hidden]
        duplicate_preds[well_id] = 0.25 * base + 0.75 * duplicate_pf
        reverse_seed_preds[well_id] = 0.25 * base + 0.75 * reversed_pf
        max_duplicate_delta = max(max_duplicate_delta, float(np.max(np.abs(duplicate_preds[well_id] - legal_control_predictions[well_id]))))
        max_seed_order_delta = max(max_seed_order_delta, float(np.max(np.abs(reverse_seed_preds[well_id] - legal_control_predictions[well_id]))))
        if j % 25 == 0:
            print(f"structural reruns {j}/{len(control_ids)}", flush=True)
    duplicate_pass = max_duplicate_delta <= 1e-10
    seed_order_pass = max_seed_order_delta <= 1e-9
    control_rows.append({"control": "duplicate_determinism", "rmse": control_rmse(duplicate_preds), "legal_rmse": legal_control_rmse, "legal_advantage": 0.0, "passed": duplicate_pass, "max_abs_delta": max_duplicate_delta})
    control_rows.append({"control": "reversed_seed_order", "rmse": control_rmse(reverse_seed_preds), "legal_rmse": legal_control_rmse, "legal_advantage": 0.0, "passed": seed_order_pass, "max_abs_delta": max_seed_order_delta})

    missing_pass = True
    for mode in ("all_missing_hidden_gr", "short_visible_gr_support"):
        finite = True
        for well_id in control_ids[:10]:
            hw, tw, hidden, y, base = control_cache[well_id]
            mh, mt = modify_frames(hw, tw, mode, well_id)
            pred = predict_likelihood_pf(mh, mt, pf_cfg)[hidden]
            finite &= pred.shape == y.shape and np.isfinite(pred).all()
        missing_pass &= finite
        control_rows.append({"control": mode, "rmse": np.nan, "legal_rmse": legal_control_rmse, "legal_advantage": np.nan, "passed": finite})

    base_summary = summary_by_candidate["e011"]
    selected_summary = summary_by_candidate[selected]
    elapsed = time.time() - started
    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
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
        "runtime": elapsed <= float(gates["maximum_runtime_hours"]) * 3600,
        "memory": rss_gb <= float(gates["maximum_rss_gb"]),
    }
    freeze_path = ROOT / "experiments/E013/IMPLEMENTATION_FREEZE.json"
    freeze = json.loads(freeze_path.read_text()) if freeze_path.exists() else {}
    source_parity_pass = bool(freeze.get("source_parity", {}).get("all_pass", False))
    structural_gates = {
        "source_parity_smoke": source_parity_pass,
        "duplicate_determinism": duplicate_pass,
        "reversed_seed_order": seed_order_pass,
        "missing_data": missing_pass,
        "input_immutability": input_immutable,
        "exact_e011_fallback": exact_e011_identity,
        "finite_complete_rows": int(selected_summary["rows"]) == int(cfg["expected_hidden_rows"]),
        "rmse_sse_identity": abs(selected_summary["rmse"] ** 2 * selected_summary["rows"] - selected_summary["sse"]) <= max(1e-6, 1e-10 * selected_summary["sse"]),
        "group_membership_identity": len(membership) == int(cfg["expected_wells"]),
    }
    go = all(scientific_gates.values()) and all(structural_gates.values())
    breakthrough = go and selected_summary["rmse"] <= float(gates["breakthrough_rmse"])
    decision = "BREAKTHROUGH_SUB6" if breakthrough else ("GO_CLEAN_ROOM_PF" if go else "REJECT_E013")

    candidate_rows = []
    for name in candidates:
        s = summary_by_candidate[name]
        candidate_rows.append({"candidate": name, **s, "gain_vs_e011": base_summary["rmse"] - s["rmse"], "primary": name == selected})
    pd.DataFrame(candidate_rows).to_csv(out / "candidate_metrics.csv", index=False)
    pd.DataFrame(map_rows).to_csv(out / "map_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "group_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(out / "slice_metrics.csv", index=False)
    pd.DataFrame(control_rows).to_csv(out / "control_metrics.csv", index=False)
    per_well.to_csv(out / "selected_well_metrics.csv", index=False)

    summary = {
        "schema_version": 1,
        "experiment_id": "E013",
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
        "minimum_control_advantage": min_advantage,
        "scientific_gates": scientific_gates,
        "structural_gates": structural_gates,
        "runtime_seconds": elapsed,
        "runtime_hours": elapsed / 3600.0,
        "observed_max_rss_kb": rss_kb,
        "observed_max_rss_gb": rss_gb,
        "config_sha256": sha256_file(CONFIG_PATH),
        "module_sha256": sha256_file(ROOT / "src/rogii_validation/likelihood_pf.py"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kaggle_run": False,
        "submission_made": False,
    }
    atomic_json(out / "summary.json", summary)
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
        f"- minimum destructive-control advantage: `{min_advantage:.12f}`",
        f"- runtime: `{elapsed / 3600.0:.3f}` hours",
        f"- max RSS: `{rss_gb:.3f}` GB",
        "",
        "Scientific gates:",
        *[f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in scientific_gates.items()],
        "",
        "Structural gates:",
        *[f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in structural_gates.items()],
        "",
        "No Kaggle run or submission was made.",
    ]
    atomic_text(RESULT_MD, "\n".join(lines) + "\n")
    manifest = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(out / "artifact_manifest.json", {"schema_version": 1, "files": manifest})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
