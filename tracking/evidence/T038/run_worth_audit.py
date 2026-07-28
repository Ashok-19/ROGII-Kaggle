#!/usr/bin/env python3
"""T038 bounded source audit for public HMM/PF trajectory mechanisms.

This script intentionally treats the downloaded public notebook as a quarantined
research source. Its predictions can only decide whether a separate clean-room
experiment is worth implementing; they cannot be promoted or deployed.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "tracking/evidence/T038/config.json"
DEFAULT_OUTPUT = ROOT / "scratch/agents/public-frontier-sprint-20260728/t038_worth"


@dataclass
class WellData:
    well_id: str
    fold: int
    is_primary: bool
    is_edge: bool
    hw: pd.DataFrame
    tw: pd.DataFrame
    hidden_mask: np.ndarray
    row_index: np.ndarray
    target: np.ndarray
    e011: np.ndarray


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def pooled_rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(p, float) - np.asarray(y, float)))))


def frame_digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("|".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy(np.uint64).tobytes())
    return h.hexdigest()


def load_public_functions(source: Path) -> tuple[dict[str, Any], str]:
    if not source.exists():
        raise FileNotFoundError(f"Public source notebook not found: {source}")
    nb = json.loads(source.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    if len(cells) <= 10:
        raise ValueError("Unexpected public notebook structure")
    ns: dict[str, Any] = {"np": np, "pd": pd, "Path": Path, "njit": njit, "prange": prange}
    for idx in (4, 8, 10):
        code = "".join(cells[idx].get("source", []))
        code = code.replace("cache=True", "cache=False")
        exec(compile(code, f"{source.name}:cell{idx}", "exec"), ns)
    required = ["load_well", "run_hmm2", "lik_pf"]
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError(f"Missing public functions: {missing}")
    return ns, sha256_file(source)


def choose_primary(assignments: dict[str, int], per_fold: int) -> list[str]:
    out: list[str] = []
    for fold in range(5):
        wells = sorted(w for w, f in assignments.items() if int(f) == fold)
        if len(wells) < per_fold:
            raise ValueError(f"Fold {fold} has fewer than {per_fold} wells")
        idx = np.linspace(0, len(wells) - 1, per_fold, dtype=int)
        out.extend(wells[int(i)] for i in idx)
    if len(out) != 5 * per_fold or len(set(out)) != len(out):
        raise AssertionError("Primary sample is not the expected unique fold-stratified set")
    return out


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - med)))


def build_edge_table(data_dir: Path, well_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, well_id in enumerate(well_ids, 1):
        path = data_dir / f"{well_id}__horizontal_well.csv"
        df = pd.read_csv(path, usecols=["MD", "GR", "TVT_input"])
        hidden = df["TVT_input"].isna().to_numpy()
        known = ~hidden
        hidden_gr = df.loc[hidden, "GR"].to_numpy(float)
        md = df["MD"].to_numpy(float)
        rows.append(
            {
                "well_id": well_id,
                "hidden_row_count": int(hidden.sum()),
                "known_row_count": int(known.sum()),
                "hidden_gr_missing_fraction": float(np.mean(~np.isfinite(hidden_gr))) if hidden_gr.size else 1.0,
                "hidden_gr_robust_scale": robust_scale(hidden_gr),
                "maximum_md_step": float(np.nanmax(np.diff(md))) if md.size > 1 else 0.0,
            }
        )
        if i % 100 == 0:
            print(f"edge scan {i}/{len(well_ids)}", flush=True)
    return pd.DataFrame(rows)


def choose_edges(edge_table: pd.DataFrame, primary: set[str]) -> list[str]:
    selectors = [
        ("hidden_row_count", False),
        ("known_row_count", True),
        ("hidden_gr_missing_fraction", False),
        ("hidden_gr_robust_scale", True),
        ("maximum_md_step", False),
    ]
    out: list[str] = []
    for column, ascending in selectors:
        ranked = edge_table.sort_values([column, "well_id"], ascending=[ascending, True])
        added = 0
        for well_id in ranked["well_id"]:
            if well_id not in primary and well_id not in out:
                out.append(str(well_id))
                added += 1
                if added == 2:
                    break
        if added != 2:
            raise AssertionError(f"Could not select two unique edge wells for {column}")
    if len(out) != 10:
        raise AssertionError(f"Expected 10 unique edge wells, got {len(out)}")
    return out


def load_e011_rows(path: Path, wanted: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    cols = ["well_id", "row_index", "target", "spline4_ridge_equal_s075"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=250_000):
        keep = chunk["well_id"].isin(wanted)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    if not parts:
        raise RuntimeError("No E011 rows found for selected wells")
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["well_id", "row_index"]).reset_index(drop=True)


def load_well_data(
    ns: dict[str, Any],
    data_dir: Path,
    assignments: dict[str, int],
    primary: set[str],
    edges: set[str],
    e011: pd.DataFrame,
) -> dict[str, WellData]:
    out: dict[str, WellData] = {}
    for well_id in sorted(primary | edges):
        hw, tw = ns["load_well"](well_id, data_dir)
        hidden = hw["TVT_input"].isna().to_numpy()
        idx = np.flatnonzero(hidden)
        g = e011[e011["well_id"] == well_id].sort_values("row_index")
        if not np.array_equal(g["row_index"].to_numpy(int), idx):
            raise AssertionError(f"E011 row identity mismatch for {well_id}")
        target = g["target"].to_numpy(float)
        base = g["spline4_ridge_equal_s075"].to_numpy(float)
        if target.size != hidden.sum() or not np.isfinite(target).all() or not np.isfinite(base).all():
            raise AssertionError(f"Invalid scored arrays for {well_id}")
        out[well_id] = WellData(
            well_id=well_id,
            fold=int(assignments[well_id]),
            is_primary=well_id in primary,
            is_edge=well_id in edges,
            hw=hw,
            tw=tw,
            hidden_mask=hidden,
            row_index=idx,
            target=target,
            e011=base,
        )
    return out


def run_hmm(ns: dict[str, Any], wd: WellData, params: dict[str, Any]) -> np.ndarray:
    kwargs = {k: v for k, v in params.items() if k != "name"}
    res = ns["run_hmm2"](wd.hw.copy(deep=True), wd.tw.copy(deep=True), **kwargs)
    pred = np.asarray(res["pred"], float)[wd.hidden_mask]
    if pred.shape != wd.target.shape or not np.isfinite(pred).all():
        raise AssertionError(f"Invalid HMM output for {wd.well_id}")
    return pred


def run_pf(ns: dict[str, Any], wd: WellData, params: dict[str, Any]) -> np.ndarray:
    res = ns["lik_pf"](
        wd.hw.copy(deep=True),
        wd.tw.copy(deep=True),
        n_particles=int(params["n_particles"]),
        n_seeds=int(params["n_seeds"]),
        scale=float(params["scale"]),
        init_spr=float(params["init_spread"]),
    )
    pred = np.asarray(res["pred"], float)[wd.hidden_mask]
    if pred.shape != wd.target.shape or not np.isfinite(pred).all():
        raise AssertionError(f"Invalid PF output for {wd.well_id}")
    return pred


def aggregate_metrics(wells: dict[str, WellData], predictions: dict[str, np.ndarray], selected: set[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for well_id in sorted(selected):
        wd = wells[well_id]
        p = predictions[well_id]
        err = p - wd.target
        sse = float(np.square(err).sum())
        rows.append({"well_id": well_id, "fold": wd.fold, "n": int(err.size), "sse": sse, "rmse": math.sqrt(sse / err.size)})
        ys.append(wd.target)
        ps.append(p)
    table = pd.DataFrame(rows)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    total_sse = float(np.square(p - y).sum())
    pooled = math.sqrt(total_sse / y.size)
    n_worst = max(1, int(math.ceil(0.20 * len(table))))
    worst_share = float(table.nlargest(n_worst, "sse")["sse"].sum() / total_sse) if total_sse > 0 else 0.0
    return {
        "rmse": float(pooled),
        "sse": total_sse,
        "rows": int(y.size),
        "wells": int(len(table)),
        "median_well_rmse": float(table["rmse"].median()),
        "p90_well_rmse": float(table["rmse"].quantile(0.90)),
        "worst20_sse_share": worst_share,
        "per_well": table,
    }


def make_modified_well(wd: WellData, mode: str) -> WellData:
    hw = wd.hw.copy(deep=True)
    tw = wd.tw.copy(deep=True)
    hidden_idx = np.flatnonzero(wd.hidden_mask)
    if mode == "reversed_typewell_gr":
        tw.loc[:, "GR"] = tw["GR"].to_numpy()[::-1]
    elif mode == "circular_shift_hidden_gr":
        vals = hw.loc[hidden_idx, "GR"].to_numpy(copy=True)
        shift = max(1, len(vals) // 3)
        hw.loc[hidden_idx, "GR"] = np.roll(vals, shift)
    elif mode == "permuted_hidden_gr":
        vals = hw.loc[hidden_idx, "GR"].to_numpy(copy=True)
        rng = np.random.default_rng(stable_seed(wd.well_id + mode))
        hw.loc[hidden_idx, "GR"] = vals[rng.permutation(len(vals))]
    elif mode == "all_missing_hidden_gr":
        hw.loc[hidden_idx, "GR"] = np.nan
    elif mode == "short_gr_support":
        known_idx = np.flatnonzero(~wd.hidden_mask)
        finite_known = known_idx[np.isfinite(hw.loc[known_idx, "GR"].to_numpy(float))]
        keep = set(finite_known[-5:].tolist())
        drop = [i for i in known_idx if i not in keep]
        hw.loc[drop, "GR"] = np.nan
    else:
        raise ValueError(mode)
    return WellData(
        well_id=wd.well_id,
        fold=wd.fold,
        is_primary=wd.is_primary,
        is_edge=wd.is_edge,
        hw=hw,
        tw=tw,
        hidden_mask=wd.hidden_mask.copy(),
        row_index=wd.row_index.copy(),
        target=wd.target.copy(),
        e011=wd.e011.copy(),
    )


def candidate_components(name: str, best_hmm: str, pf_name: str) -> set[str]:
    components: set[str] = set()
    if best_hmm in name or name.startswith("hmm_"):
        components.add(best_hmm if not name.startswith("hmm_") else name)
    if pf_name in name or "pf" in name:
        components.add(pf_name)
    return components


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = ROOT / config["public_source"]
    data_dir = ROOT / "data/train"
    fold_path = ROOT / "folds/v1.json"
    e011_path = ROOT / "artifacts/E011/oof_predictions.csv.gz"
    ns, source_sha = load_public_functions(source)
    assignments = {k: int(v) for k, v in json.loads(fold_path.read_text())["assignments"].items()}
    all_well_ids = sorted(assignments)

    primary_list = choose_primary(assignments, int(config["primary_sample"]["wells_per_fold"]))
    primary = set(primary_list)
    edge_table = build_edge_table(data_dir, all_well_ids)
    edge_list = choose_edges(edge_table, primary)
    edges = set(edge_list)
    wanted = primary | edges
    e011 = load_e011_rows(e011_path, wanted)
    wells = load_well_data(ns, data_dir, assignments, primary, edges, e011)

    baseline = {wid: wd.e011.copy() for wid, wd in wells.items()}
    predictions: dict[str, dict[str, np.ndarray]] = {"e011": baseline}
    runtimes: dict[str, float] = {"e011": 0.0}
    input_hashes = {wid: (frame_digest(wd.hw), frame_digest(wd.tw)) for wid, wd in wells.items()}

    for branch in config["hmm_branches"]:
        name = branch["name"]
        print(f"START {name}", flush=True)
        t0 = time.time()
        branch_pred: dict[str, np.ndarray] = {}
        for i, (wid, wd) in enumerate(sorted(wells.items()), 1):
            branch_pred[wid] = run_hmm(ns, wd, branch)
            print(f"{name} {i}/{len(wells)} {wid}", flush=True)
        predictions[name] = branch_pred
        runtimes[name] = time.time() - t0
        print(f"DONE {name} {runtimes[name]:.3f}s", flush=True)

    pf_cfg = config["pf_branch"]
    pf_name = pf_cfg["name"]
    print(f"START {pf_name}", flush=True)
    t0 = time.time()
    predictions[pf_name] = {}
    for i, (wid, wd) in enumerate(sorted(wells.items()), 1):
        predictions[pf_name][wid] = run_pf(ns, wd, pf_cfg)
        print(f"{pf_name} {i}/{len(wells)} {wid}", flush=True)
    runtimes[pf_name] = time.time() - t0
    print(f"DONE {pf_name} {runtimes[pf_name]:.3f}s", flush=True)

    primary_metrics_pre: dict[str, dict[str, Any]] = {}
    for name, pred in predictions.items():
        primary_metrics_pre[name] = aggregate_metrics(wells, pred, primary)
    best_hmm = min((b["name"] for b in config["hmm_branches"]), key=lambda n: primary_metrics_pre[n]["rmse"])

    for tracker in [b["name"] for b in config["hmm_branches"]] + [pf_name]:
        for w in config["fixed_blend_weights"]:
            name = f"blend_e011_{tracker}_w{float(w):.2f}"
            predictions[name] = {wid: (1.0 - float(w)) * baseline[wid] + float(w) * predictions[tracker][wid] for wid in wells}
            runtimes[name] = runtimes[tracker]

    name = f"blend_{best_hmm}_{pf_name}_w050"
    predictions[name] = {wid: 0.5 * predictions[best_hmm][wid] + 0.5 * predictions[pf_name][wid] for wid in wells}
    runtimes[name] = runtimes[best_hmm] + runtimes[pf_name]

    for weights in config["three_way_weights"]:
        wb, wh, wp = map(float, weights)
        name = f"blend3_e011_{best_hmm}_{pf_name}_{wb:.2f}_{wh:.2f}_{wp:.2f}"
        predictions[name] = {
            wid: wb * baseline[wid] + wh * predictions[best_hmm][wid] + wp * predictions[pf_name][wid]
            for wid in wells
        }
        runtimes[name] = runtimes[best_hmm] + runtimes[pf_name]

    branch_rows: list[dict[str, Any]] = []
    per_well_rows: list[dict[str, Any]] = []
    all_metrics: dict[str, dict[str, Any]] = {}
    baseline_primary = aggregate_metrics(wells, baseline, primary)
    baseline_edge = aggregate_metrics(wells, baseline, edges) if edges else None

    for name, pred in predictions.items():
        pm = aggregate_metrics(wells, pred, primary)
        em = aggregate_metrics(wells, pred, edges) if edges else None
        all_metrics[name] = {"primary": pm, "edge": em}
        fold_wins = 0
        fold_gains: dict[str, float] = {}
        for fold in range(5):
            fold_wells = {wid for wid in primary if wells[wid].fold == fold}
            bm = aggregate_metrics(wells, baseline, fold_wells)
            cm = aggregate_metrics(wells, pred, fold_wells)
            gain = bm["rmse"] - cm["rmse"]
            fold_gains[str(fold)] = float(gain)
            fold_wins += int(gain > 0)
        row = {
            "candidate": name,
            "primary_rmse": pm["rmse"],
            "primary_gain_vs_e011": baseline_primary["rmse"] - pm["rmse"],
            "primary_p90": pm["p90_well_rmse"],
            "primary_worst20_sse_share": pm["worst20_sse_share"],
            "edge_rmse": em["rmse"] if em else np.nan,
            "edge_gain_vs_e011": (baseline_edge["rmse"] - em["rmse"]) if em and baseline_edge else np.nan,
            "fold_wins": fold_wins,
            "fold_gains_json": json.dumps(fold_gains, sort_keys=True),
            "runtime_seconds": runtimes[name],
        }
        branch_rows.append(row)
        for panel, metrics in (("primary", pm), ("edge", em)):
            if metrics is None:
                continue
            table = metrics["per_well"].copy()
            table.insert(0, "candidate", name)
            table.insert(1, "panel", panel)
            per_well_rows.extend(table.to_dict("records"))

    branch_df = pd.DataFrame(branch_rows).sort_values(["primary_rmse", "candidate"]).reset_index(drop=True)
    selected = str(branch_df.iloc[0]["candidate"])
    selected_metrics = all_metrics[selected]["primary"]

    best_hmm_params = next(dict(x) for x in config["hmm_branches"] if x["name"] == best_hmm)
    control_wells = primary_list[: int(config["control_wells"])]
    control_set = set(control_wells)
    legal_control_pred = {wid: predictions[best_hmm][wid] for wid in control_wells}
    legal_control_metrics = aggregate_metrics(wells, legal_control_pred, control_set)
    control_rows: list[dict[str, Any]] = []

    def run_control(name: str, mode: str | None = None, lam: float | None = None) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for wid in control_wells:
            wd = wells[wid]
            c_wd = make_modified_well(wd, mode) if mode else wd
            params = dict(best_hmm_params)
            if lam is not None:
                params["lam"] = lam
            out[wid] = run_hmm(ns, c_wd, params)
        return out

    destructive_specs = [
        ("transition_only_lam0", None, 0.0),
        ("reversed_typewell_gr", "reversed_typewell_gr", None),
        ("circular_shift_hidden_gr", "circular_shift_hidden_gr", None),
        ("permuted_hidden_gr", "permuted_hidden_gr", None),
    ]
    destructive_pass = True
    min_control_advantage = float("inf")
    for name, mode, lam in destructive_specs:
        t0 = time.time()
        cp = run_control(name, mode=mode, lam=lam)
        cm = aggregate_metrics(wells, cp, control_set)
        advantage = cm["rmse"] - legal_control_metrics["rmse"]
        min_control_advantage = min(min_control_advantage, advantage)
        passed = advantage >= float(config["worth_gates"]["minimum_advantage_over_each_destructive_control"])
        destructive_pass &= passed
        control_rows.append({"control": name, "rmse": cm["rmse"], "legal_rmse": legal_control_metrics["rmse"], "legal_advantage": advantage, "passed": passed, "runtime_seconds": time.time() - t0})

    rerun_pred = run_control("duplicate_and_deterministic_rerun")
    max_rerun_delta = max(float(np.max(np.abs(rerun_pred[wid] - legal_control_pred[wid]))) for wid in control_wells)
    rerun_pass = max_rerun_delta <= 1e-10
    control_rows.append({"control": "duplicate_and_deterministic_rerun", "rmse": aggregate_metrics(wells, rerun_pred, control_set)["rmse"], "legal_rmse": legal_control_metrics["rmse"], "legal_advantage": 0.0, "passed": rerun_pass, "max_abs_delta": max_rerun_delta})

    missing_controls_pass = True
    for mode in ("all_missing_hidden_gr", "short_gr_support"):
        cp = run_control(mode, mode=mode)
        finite = all(np.isfinite(v).all() and v.shape == wells[k].target.shape for k, v in cp.items())
        missing_controls_pass &= finite
        control_rows.append({"control": mode, "rmse": aggregate_metrics(wells, cp, control_set)["rmse"], "legal_rmse": legal_control_metrics["rmse"], "legal_advantage": np.nan, "passed": finite})

    mutation_pass = all(input_hashes[wid] == (frame_digest(wd.hw), frame_digest(wd.tw)) for wid, wd in wells.items())
    fallback_pass = all(np.array_equal(predictions["e011"][wid], wells[wid].e011) for wid in wells)
    rmse_identity_pass = abs(selected_metrics["rmse"] ** 2 * selected_metrics["rows"] - selected_metrics["sse"]) <= max(1e-6, 1e-10 * selected_metrics["sse"])
    finite_pass = all(np.isfinite(v).all() for pred in predictions.values() for v in pred.values())

    selected_row = branch_df[branch_df["candidate"] == selected].iloc[0]
    edge_deterioration = -float(selected_row["edge_gain_vs_e011"]) if edges else 0.0
    projected_seconds = float(selected_row["runtime_seconds"]) / max(1, len(wells)) * 773.0
    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_gb = rss_kb / (1024.0 * 1024.0)
    gates = config["worth_gates"]
    scientific_gates = {
        "minimum_primary_gain": float(selected_row["primary_gain_vs_e011"]) >= float(gates["minimum_primary_gain_rmse"]),
        "fold_stratum_wins": int(selected_row["fold_wins"]) >= int(gates["minimum_fold_stratum_wins"]),
        "p90_not_worse": float(selected_row["primary_p90"]) <= baseline_primary["p90_well_rmse"] + float(gates["maximum_p90_deterioration"]),
        "worst20_share": float(selected_row["primary_worst20_sse_share"]) <= baseline_primary["worst20_sse_share"] + float(gates["maximum_worst20_sse_share_increase"]),
        "edge_panel": edge_deterioration <= float(gates["maximum_edge_panel_rmse_deterioration"]),
        "destructive_controls": destructive_pass,
        "runtime": projected_seconds <= float(gates["maximum_projected_full_runtime_hours"]) * 3600.0,
        "memory": rss_gb <= float(gates["maximum_rss_gb"]),
    }
    structural_gates = {
        "deterministic_rerun": rerun_pass,
        "missing_data_controls": missing_controls_pass,
        "input_immutability": mutation_pass,
        "exact_fallback": fallback_pass,
        "rmse_sse_identity": rmse_identity_pass,
        "finite_complete_outputs": finite_pass,
        "all_registered_branches_completed": len(predictions) == 1 + len(config["hmm_branches"]) + 1 + (len(config["hmm_branches"]) + 1) * len(config["fixed_blend_weights"]) + 1 + len(config["three_way_weights"]),
    }
    passed = all(scientific_gates.values()) and all(structural_gates.values())
    decision = "PASS_AUTHORIZE_CLEAN_ROOM_E013" if passed else "REJECT_H030_EXACT_FAMILY"

    edge_table = edge_table.copy()
    edge_table["is_primary"] = edge_table["well_id"].isin(primary)
    edge_table["is_edge_selected"] = edge_table["well_id"].isin(edges)
    branch_df.to_csv(output_dir / "branch_metrics.csv", index=False)
    pd.DataFrame(per_well_rows).to_csv(output_dir / "per_well_metrics.csv", index=False)
    pd.DataFrame(control_rows).to_csv(output_dir / "control_metrics.csv", index=False)
    edge_table.to_csv(output_dir / "edge_selection.csv", index=False)

    summary = {
        "schema_version": 1,
        "task_id": "T038",
        "decision": decision,
        "passed": passed,
        "selected_candidate": selected,
        "best_hmm_branch": best_hmm,
        "primary_wells": primary_list,
        "edge_wells": edge_list,
        "baseline_primary": {k: v for k, v in baseline_primary.items() if k != "per_well"},
        "baseline_edge": {k: v for k, v in baseline_edge.items() if k != "per_well"} if baseline_edge else None,
        "selected_primary": {k: v for k, v in selected_metrics.items() if k != "per_well"},
        "selected_edge": {k: v for k, v in all_metrics[selected]["edge"].items() if k != "per_well"} if edges else None,
        "selected_gain_vs_e011": float(selected_row["primary_gain_vs_e011"]),
        "selected_fold_wins": int(selected_row["fold_wins"]),
        "selected_edge_gain_vs_e011": float(selected_row["edge_gain_vs_e011"]),
        "legal_control_rmse": legal_control_metrics["rmse"],
        "minimum_control_advantage": min_control_advantage,
        "projected_full_runtime_seconds": projected_seconds,
        "projected_full_runtime_hours": projected_seconds / 3600.0,
        "observed_max_rss_kb": rss_kb,
        "observed_max_rss_gb": rss_gb,
        "scientific_gates": scientific_gates,
        "structural_gates": structural_gates,
        "source_notebook_sha256": source_sha,
        "config_sha256": sha256_file(CONFIG_PATH),
        "branch_count": int(len(predictions)),
        "control_count": int(len(control_rows)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(output_dir / "summary.json", summary)

    result_lines = [
        "# T038 result — public HMM/PF mechanism worth audit",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Selected fixed candidate: `{selected}`",
        f"- Primary E011 RMSE: `{baseline_primary['rmse']:.12f}`",
        f"- Selected primary RMSE: `{selected_metrics['rmse']:.12f}`",
        f"- Gain versus E011: `{float(selected_row['primary_gain_vs_e011']):.12f}`",
        f"- Fold-stratum wins: `{int(selected_row['fold_wins'])}/5`",
        f"- Edge gain versus E011: `{float(selected_row['edge_gain_vs_e011']):.12f}`",
        f"- Primary p90: `{selected_metrics['p90_well_rmse']:.12f}` versus `{baseline_primary['p90_well_rmse']:.12f}`",
        f"- Worst-20% SSE share: `{selected_metrics['worst20_sse_share']:.12f}` versus `{baseline_primary['worst20_sse_share']:.12f}`",
        f"- Minimum advantage over destructive controls: `{min_control_advantage:.12f}`",
        f"- Projected 773-well runtime: `{projected_seconds / 3600.0:.3f}` hours",
        f"- Observed max RSS: `{rss_gb:.3f}` GB",
        "",
        "Scientific gates:",
    ]
    result_lines.extend(f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in scientific_gates.items())
    result_lines.append("")
    result_lines.append("Structural gates:")
    result_lines.extend(f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in structural_gates.items())
    result_lines.extend(
        [
            "",
            "This source-audit task cannot promote public code or predictions. A pass only authorizes a separate clean-room E013 implementation and full outer-isolated validation.",
        ]
    )
    atomic_write_text(output_dir / "RESULT.md", "\n".join(result_lines) + "\n")

    manifest = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_write_json(output_dir / "artifact_manifest.json", {"schema_version": 1, "files": manifest})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
