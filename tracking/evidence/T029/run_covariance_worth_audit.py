#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
T025 = ROOT / "tracking/evidence/T025"
T027 = ROOT / "tracking/evidence/T027"
T016 = ROOT / "tracking/evidence/T016"
OUT = ROOT / "scratch/agents/t029-independent-mechanism-20260724/covariance_worth_audit.json"

E011_RMSE = 12.550756295689673
T025_RMSE = 12.42492760849511
MIN_GAIN_VS_T025 = 0.03
MIN_GAIN_VS_E011 = 0.15
MAX_ABS_DIRECTION_CORRELATION = 0.95
GRID_STEP = 0.025
LEG_NAMES = ("t025_pseudo_coeff", "t027_original_coeff", "t016_nested_datum")


def load_module() -> Any:
    path = T027 / "run_screen.py"
    spec = importlib.util.spec_from_file_location("t027_screen_for_t029", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load T027 module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_coefficients(path: Path) -> tuple[dict[str, np.ndarray], float, str]:
    result: dict[str, np.ndarray] = {}
    weights: set[float] = set()
    candidates: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"well_id", "weight", "candidate", *(f"predicted_coefficient_{i}" for i in range(4))}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{path}: coefficient schema differs")
        for row in reader:
            well_id = str(row["well_id"])
            if well_id in result:
                raise RuntimeError(f"{path}: duplicate well {well_id}")
            result[well_id] = np.asarray([float(row[f"predicted_coefficient_{i}"]) for i in range(4)], dtype=np.float64)
            weights.add(float(row["weight"])); candidates.add(str(row["candidate"]))
    if len(weights) != 1 or len(candidates) != 1:
        raise RuntimeError(f"{path}: selected candidate identity differs")
    return result, next(iter(weights)), next(iter(candidates))


def read_t016(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    correction: dict[str, float] = {}; reported_rmse: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"well_id", "nested_real_correction", "nested_real_rmse"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("T016 per-well schema differs")
        for row in reader:
            well_id = str(row["well_id"])
            if well_id in correction:
                raise RuntimeError(f"duplicate T016 well {well_id}")
            correction[well_id] = float(row["nested_real_correction"])
            reported_rmse[well_id] = float(row["nested_real_rmse"])
    return correction, reported_rmse


def direction_terms(record: Any, c: np.ndarray) -> tuple[float, float, float]:
    linear = float(record.e_dot_d) + float(np.dot(record.basis_dot_e, c))
    quadratic = float(record.d_sse) + 2.0 * float(np.dot(record.basis_dot_d, c)) + float(c @ record.basis_cross @ c)
    total = float(record.d_sum) + float(np.dot(record.basis_sum, c))
    return linear, max(0.0, quadratic), total


def exact_box_optimum(g: np.ndarray, h: np.ndarray, active_legs: Sequence[int]) -> np.ndarray:
    active = tuple(sorted(int(value) for value in active_legs))
    best_w: np.ndarray | None = None; best_obj = math.inf
    for states in itertools.product((-1, 0, 1), repeat=len(active)):
        w = np.zeros(3, dtype=np.float64)
        free: list[int] = []; fixed: list[int] = []
        for position, leg in enumerate(active):
            state = states[position]
            if state == -1: free.append(leg)
            else: w[leg] = float(state); fixed.append(leg)
        if free:
            ff = np.ix_(free, free)
            rhs = -g[free]
            if fixed:
                rhs = rhs - h[np.ix_(free, fixed)] @ w[fixed]
            try:
                solved = np.linalg.solve(h[ff], rhs)
            except np.linalg.LinAlgError:
                solved = np.linalg.lstsq(h[ff], rhs, rcond=None)[0]
            if np.any(solved < -1e-10) or np.any(solved > 1.0 + 1e-10):
                continue
            w[free] = np.clip(solved, 0.0, 1.0)
        obj = float(2.0 * np.dot(g, w) + w @ h @ w)
        if obj < best_obj - 1e-9 or (abs(obj-best_obj) <= 1e-9 and (best_w is None or tuple(w) < tuple(best_w))):
            best_obj = obj; best_w = w.copy()
    if best_w is None:
        raise RuntimeError("box QP produced no feasible point")
    return best_w


def rmse_from_terms(base_sse: float, rows: int, g: np.ndarray, h: np.ndarray, w: np.ndarray) -> float:
    sse = base_sse + 2.0 * float(np.dot(g, w)) + float(w @ h @ w)
    if sse < -1e-6 * max(1.0, base_sse):
        raise RuntimeError("materially negative SSE")
    return math.sqrt(max(0.0, sse) / rows)


def aggregate(indices: np.ndarray, base: np.ndarray, rows: np.ndarray, g: np.ndarray, h: np.ndarray) -> tuple[float, int, np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    return float(base[idx].sum()), int(rows[idx].sum()), g[idx].sum(axis=0), h[idx].sum(axis=0)


def main() -> None:
    module = load_module()
    config = json.loads((T027 / "config.json").read_text(encoding="utf-8"))
    records = module.load_records(ROOT / config["data_dir"], int(config["expected_wells"]), float(config["target"]["coefficient_absolute_bound_ft"]))
    coverage = module.attach_e011_sufficient(records, ROOT / config["e011_oof_path"], int(config["expected_hidden_rows"]))
    _features, _names, spatial, typewell = module.load_compact(ROOT / config["compact_path"], records, int(config["feature_contract"]["expected_features"]))

    t025, t025_weight, t025_candidate = read_coefficients(T025 / "selected_well_metrics.csv")
    t027, t027_weight, t027_candidate = read_coefficients(T027 / "selected_well_metrics.csv")
    t016, t016_reported = read_t016(T016 / "per_well_final.csv")
    well_ids = [record.well_id for record in records]
    expected = set(well_ids)
    if set(t025) != expected or set(t027) != expected or set(t016) != expected:
        raise RuntimeError("leg coverage differs")
    if abs(t025_weight - 0.5) > 1e-12 or abs(t027_weight - 0.5) > 1e-12:
        raise RuntimeError("coefficient leg selected weight differs")

    n = len(records)
    base = np.asarray([float(record.e011_sse) for record in records], dtype=np.float64)
    rows = np.asarray([int(record.hidden_rows) for record in records], dtype=np.int64)
    g = np.zeros((n, 3), dtype=np.float64)
    h = np.zeros((n, 3, 3), dtype=np.float64)
    direction_sum = np.zeros((n, 3), dtype=np.float64)
    max_t016_rmse_delta = 0.0
    max_t025_sse_delta = 0.0
    max_t027_sse_delta = 0.0

    for i, record in enumerate(records):
        ca = t025[record.well_id]; cb = t027[record.well_id]; datum = float(t016[record.well_id])
        ga, haa, suma = direction_terms(record, ca)
        gb, hbb, sumb = direction_terms(record, cb)
        gc = datum * float(record.e011_sum)
        hcc = datum * datum * record.hidden_rows
        hab = float(record.d_sse) + float(np.dot(record.basis_dot_d, ca + cb)) + float(ca @ record.basis_cross @ cb)
        hac = datum * suma; hbc = datum * sumb
        g[i] = (ga, gb, gc)
        h[i] = ((haa, hab, hac), (hab, hbb, hbc), (hac, hbc, hcc))
        direction_sum[i] = (suma, sumb, datum * record.hidden_rows)
        if np.linalg.eigvalsh(h[i]).min() < -1e-5:
            raise RuntimeError(f"{record.well_id}: direction Gram matrix is not PSD")
        t025_sse = base[i] + 2*0.5*ga + 0.25*haa
        t027_sse = base[i] + 2*0.5*gb + 0.25*hbb
        max_t025_sse_delta = max(max_t025_sse_delta, abs(t025_sse - record.metric(ca, 0.5)["sse"]))
        max_t027_sse_delta = max(max_t027_sse_delta, abs(t027_sse - record.metric(cb, 0.5)["sse"]))
        t016_sse = base[i] + 2*gc + hcc
        t016_rmse = math.sqrt(max(0.0, t016_sse) / record.hidden_rows)
        max_t016_rmse_delta = max(max_t016_rmse_delta, abs(t016_rmse - t016_reported[record.well_id]))

    all_idx = np.arange(n, dtype=np.int64)
    base_sse, total_rows, G, H = aggregate(all_idx, base, rows, g, h)
    if abs(math.sqrt(base_sse/total_rows)-E011_RMSE) > 1e-9:
        raise RuntimeError("E011 base differs")

    # Row-centered correction-direction correlation.
    sums = direction_sum.sum(axis=0)
    centered = H - np.outer(sums, sums) / total_rows
    scales = np.sqrt(np.maximum(np.diag(centered), 0.0))
    corr = np.divide(centered, np.outer(scales, scales), out=np.eye(3), where=np.outer(scales, scales) > 1e-12)

    candidates: list[dict[str, Any]] = []
    for active in ((0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)):
        w = exact_box_optimum(G, H, active)
        candidates.append({
            "name": "+".join(LEG_NAMES[i] for i in active),
            "active_legs": list(active),
            "weights": w.tolist(),
            "rmse": rmse_from_terms(base_sse, total_rows, G, H, w),
        })

    domain_terms: list[dict[str, Any]] = []
    for scope, labels in (("spatial", spatial), ("typewell", typewell)):
        for group in range(5):
            idx = np.flatnonzero(labels == group)
            b, r, gg, hh = aggregate(idx, base, rows, g, h)
            domain_terms.append({"scope": scope, "group": group, "base_sse": b, "rows": r, "g": gg, "h": hh, "base_rmse": math.sqrt(b/r)})

    values = np.arange(0.0, 1.0 + GRID_STEP/2.0, GRID_STEP)
    best_pooled: tuple[float, tuple[float,float,float], list[float]] | None = None
    best_minimax: tuple[float, float, tuple[float,float,float], list[float]] | None = None
    best_all_domain: tuple[float, tuple[float,float,float], list[float]] | None = None
    grid_points = 0
    for a in values:
        for b in values:
            max_c = 1.0 - a - b
            if max_c < -1e-12: continue
            for c in values[values <= max_c + 1e-12]:
                w = np.asarray((a,b,c), dtype=np.float64); grid_points += 1
                pooled = rmse_from_terms(base_sse,total_rows,G,H,w)
                gains = []
                for domain in domain_terms:
                    candidate_rmse = rmse_from_terms(float(domain["base_sse"]),int(domain["rows"]),domain["g"],domain["h"],w)
                    gains.append(float(domain["base_rmse"])-candidate_rmse)
                min_gain = min(gains)
                key = tuple(float(x) for x in w)
                if best_pooled is None or pooled < best_pooled[0]-1e-12 or (abs(pooled-best_pooled[0])<=1e-12 and key<best_pooled[1]):
                    best_pooled=(pooled,key,gains)
                minimax_key=(min_gain,-pooled)
                if best_minimax is None or minimax_key>(best_minimax[0],best_minimax[1]) or (minimax_key==(best_minimax[0],best_minimax[1]) and key<best_minimax[2]):
                    best_minimax=(min_gain,-pooled,key,gains)
                if min_gain>=-1e-12 and (best_all_domain is None or pooled<best_all_domain[0]-1e-12 or (abs(pooled-best_all_domain[0])<=1e-12 and key<best_all_domain[1])):
                    best_all_domain=(pooled,key,gains)

    if best_pooled is None or best_minimax is None:
        raise RuntimeError("simplex grid is empty")

    def grid_record(name: str, item: Any) -> dict[str, Any] | None:
        if item is None: return None
        if name=="minimax":
            min_gain, neg_rmse, weights_key, gains=item; rmse=-neg_rmse
        else:
            rmse, weights_key, gains=item; min_gain=min(gains)
        return {
            "weights": list(weights_key), "rmse": rmse,
            "gain_vs_e011": E011_RMSE-rmse, "gain_vs_t025": T025_RMSE-rmse,
            "minimum_domain_gain": min_gain,
            "domain_gains": [{"scope": domain_terms[i]["scope"],"group":domain_terms[i]["group"],"gain":gains[i]} for i in range(10)],
        }

    pooled_grid = grid_record("pooled",best_pooled)
    minimax_grid = grid_record("minimax",best_minimax)
    all_domain_grid = grid_record("all_domain",best_all_domain)
    offdiag = [abs(float(corr[i,j])) for i in range(3) for j in range(i+1,3)]
    gate_results = {
        "all_domain_candidate_exists": all_domain_grid is not None,
        "gain_vs_t025": all_domain_grid is not None and float(all_domain_grid["gain_vs_t025"]) >= MIN_GAIN_VS_T025,
        "gain_vs_e011": all_domain_grid is not None and float(all_domain_grid["gain_vs_e011"]) >= MIN_GAIN_VS_E011,
        "every_domain_positive": all_domain_grid is not None and float(all_domain_grid["minimum_domain_gain"]) >= -1e-12,
        "independent_direction_pair": min(offdiag) <= MAX_ABS_DIRECTION_CORRELATION,
    }
    result = {
        "schema_version":1,"task_id":"T029","source_commit":"edc45e1eb7467f0ec4dd68b83db619f7a5d03602","diagnostic_only":True,
        "frozen_contract":{"grid_step":GRID_STEP,"simplex":"weights nonnegative and sum <= 1","minimum_gain_vs_t025":MIN_GAIN_VS_T025,"minimum_gain_vs_e011":MIN_GAIN_VS_E011,"maximum_abs_direction_correlation":MAX_ABS_DIRECTION_CORRELATION},
        "coverage":coverage,
        "leg_identity":{"t025":{"candidate":t025_candidate,"selected_weight":t025_weight},"t027":{"candidate":t027_candidate,"selected_weight":t027_weight},"t016":{"candidate":"nested_real_correction","selected_weight":1.0}},
        "reproduction_controls":{"maximum_t025_sse_delta":max_t025_sse_delta,"maximum_t027_sse_delta":max_t027_sse_delta,"maximum_t016_rmse_delta":max_t016_rmse_delta},
        "direction_correlation":{"names":list(LEG_NAMES),"matrix":corr.tolist(),"minimum_absolute_offdiagonal":min(offdiag)},
        "exact_box_candidates":candidates,
        "simplex_grid_points":grid_points,
        "pooled_best_simplex":pooled_grid,
        "minimax_simplex":minimax_grid,
        "best_all_domain_simplex":all_domain_grid,
        "gate_results":gate_results,
        "worth_screen_authorized":bool(all(gate_results.values())),
    }
    result["decision"]="preregister_complementary_action_ensemble" if result["worth_screen_authorized"] else "reject_saved_action_ensemble"
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"decision":result["decision"],"direction_correlation":result["direction_correlation"],"best_all_domain_simplex":all_domain_grid,"pooled_best_simplex":pooled_grid,"minimax_simplex":minimax_grid,"gate_results":gate_results},indent=2,sort_keys=True))


if __name__=="__main__": main()
