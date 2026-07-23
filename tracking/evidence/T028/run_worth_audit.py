#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
T027 = ROOT / "tracking/evidence/T027"
MODULE_PATH = T027 / "run_screen.py"
SELECTED_PATH = T027 / "selected_well_metrics.csv"
OUT = ROOT / "scratch/agents/t028-domain-robust-original-20260723/worth_audit.json"

E011_RMSE = 12.550756295689673
CURRENT_WEIGHT = 0.50
MIN_ADDITIONAL_HEADROOM = 0.05
MAX_SIMPLE_PLACEMENT_GAIN = 0.02
MIN_DOMAIN_ORACLE_HEADROOM = 0.05
MIN_MATERIAL_DOMAIN_WEIGHT_GAPS = 3
DOMAIN_WEIGHT_GAP = 0.10


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("t027_screen", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load T027 screen module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pooled_rmse(sse: float, rows: int) -> float:
    if rows <= 0 or sse < 0 or not math.isfinite(sse):
        raise RuntimeError("invalid pooled metric")
    return math.sqrt(sse / rows)


def direction_terms(record: Any, coefficients: np.ndarray) -> tuple[float, float]:
    c = np.asarray(coefficients, dtype=np.float64)
    if c.shape != (4,) or not np.all(np.isfinite(c)):
        raise RuntimeError(f"{record.well_id}: invalid coefficient vector")
    linear = float(record.e_dot_d) + float(np.dot(record.basis_dot_e, c))
    quadratic = (
        float(record.d_sse)
        + 2.0 * float(np.dot(record.basis_dot_d, c))
        + float(c @ record.basis_cross @ c)
    )
    if not math.isfinite(linear) or not math.isfinite(quadratic) or quadratic < -1e-8:
        raise RuntimeError(f"{record.well_id}: invalid quadratic direction")
    return linear, max(0.0, quadratic)


def optimum_weight(linear: float, quadratic: float) -> float:
    if quadratic <= 1e-15:
        return 0.0 if linear >= 0 else 1.0
    return float(np.clip(-linear / quadratic, 0.0, 1.0))


def group_summary(indices: Sequence[int], records: Sequence[Any], linear: np.ndarray, quadratic: np.ndarray, weight: float) -> dict[str, float | int]:
    idx = np.asarray(indices, dtype=np.int64)
    rows = int(sum(records[int(i)].hidden_rows for i in idx))
    base_sse = float(sum(float(records[int(i)].e011_sse) for i in idx))
    l = float(linear[idx].sum())
    q = float(quadratic[idx].sum())
    sse = base_sse + 2.0 * weight * l + weight * weight * q
    return {
        "wells": int(len(idx)),
        "rows": rows,
        "base_rmse": pooled_rmse(base_sse, rows),
        "candidate_rmse": pooled_rmse(max(0.0, sse), rows),
        "gain_vs_e011": pooled_rmse(base_sse, rows) - pooled_rmse(max(0.0, sse), rows),
        "linear": l,
        "quadratic": q,
    }


def main() -> None:
    module = load_module()
    config = json.loads((T027 / "config.json").read_text(encoding="utf-8"))
    records = module.load_records(ROOT / config["data_dir"], int(config["expected_wells"]), float(config["target"]["coefficient_absolute_bound_ft"]))
    coverage = module.attach_e011_sufficient(records, ROOT / config["e011_oof_path"], int(config["expected_hidden_rows"]))
    features, feature_names, spatial, typewell = module.load_compact(ROOT / config["compact_path"], records, int(config["feature_contract"]["expected_features"]))
    if features.shape != (773, 142) or len(feature_names) != 142:
        raise RuntimeError("compact feature contract differs")

    selected: dict[str, np.ndarray] = {}
    targets_from_csv: dict[str, np.ndarray] = {}
    with SELECTED_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"well_id", *(f"predicted_coefficient_{i}" for i in range(4)), *(f"target_coefficient_{i}" for i in range(4))}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("selected-well schema differs")
        for row in reader:
            well_id = str(row["well_id"])
            if well_id in selected:
                raise RuntimeError(f"duplicate selected well {well_id}")
            selected[well_id] = np.asarray([float(row[f"predicted_coefficient_{i}"]) for i in range(4)], dtype=np.float64)
            targets_from_csv[well_id] = np.asarray([float(row[f"target_coefficient_{i}"]) for i in range(4)], dtype=np.float64)
    if set(selected) != {record.well_id for record in records}:
        raise RuntimeError("selected-well coverage differs")

    coefficient_matrix = np.vstack([selected[record.well_id] for record in records])
    target_matrix = np.vstack([record.target_coefficients for record in records])
    csv_target_matrix = np.vstack([targets_from_csv[record.well_id] for record in records])
    target_delta = float(np.max(np.abs(target_matrix - csv_target_matrix)))
    if target_delta > 1e-10:
        raise RuntimeError(f"selected targets differ from raw targets: {target_delta}")

    terms = [direction_terms(record, coefficient_matrix[index]) for index, record in enumerate(records)]
    linear = np.asarray([item[0] for item in terms], dtype=np.float64)
    quadratic = np.asarray([item[1] for item in terms], dtype=np.float64)
    base_sse = np.asarray([float(record.e011_sse) for record in records], dtype=np.float64)
    rows = np.asarray([int(record.hidden_rows) for record in records], dtype=np.int64)
    total_rows = int(rows.sum())

    global_linear = float(linear.sum())
    global_quadratic = float(quadratic.sum())
    global_optimum = optimum_weight(global_linear, global_quadratic)
    current_sse = float(np.sum(base_sse + 2.0 * CURRENT_WEIGHT * linear + CURRENT_WEIGHT**2 * quadratic))
    global_optimum_sse = float(np.sum(base_sse + 2.0 * global_optimum * linear + global_optimum**2 * quadratic))
    current_rmse = pooled_rmse(current_sse, total_rows)
    global_optimum_rmse = pooled_rmse(global_optimum_sse, total_rows)

    positive_limits: list[float] = []
    domain_rows: list[dict[str, Any]] = []
    domain_oracle_sse = {"spatial": 0.0, "typewell": 0.0}
    material_gaps = 0
    all_domains_negative_derivative = True
    for scope, labels in (("spatial", spatial), ("typewell", typewell)):
        for group in range(5):
            idx = np.flatnonzero(labels == group)
            l = float(linear[idx].sum())
            q = float(quadratic[idx].sum())
            opt = optimum_weight(l, q)
            if abs(opt - global_optimum) >= DOMAIN_WEIGHT_GAP:
                material_gaps += 1
            if l >= 0.0:
                all_domains_negative_derivative = False
                positive_limit = 0.0
            elif q <= 1e-15:
                positive_limit = 1.0
            else:
                positive_limit = float(np.clip(-2.0 * l / q, 0.0, 1.0))
            positive_limits.append(positive_limit)
            base_group = group_summary(idx, records, linear, quadratic, 0.0)
            current_group = group_summary(idx, records, linear, quadratic, CURRENT_WEIGHT)
            optimal_group = group_summary(idx, records, linear, quadratic, opt)
            domain_oracle_sse[scope] += float(optimal_group["candidate_rmse"]) ** 2 * int(optimal_group["rows"])
            residual = target_matrix[idx] - coefficient_matrix[idx]
            domain_rows.append({
                "scope": scope,
                "group": group,
                "wells": int(len(idx)),
                "rows": int(base_group["rows"]),
                "first_derivative_at_zero": 2.0 * l,
                "optimal_weight": opt,
                "positive_weight_upper_bound": positive_limit,
                "current_gain_vs_e011": float(current_group["gain_vs_e011"]),
                "oracle_weight_gain_vs_e011": float(optimal_group["gain_vs_e011"]),
                "residual_mean": residual.mean(axis=0).tolist(),
                "residual_mean_l2": float(np.linalg.norm(residual.mean(axis=0))),
                "residual_rmse": float(np.sqrt(np.mean(residual * residual))),
            })

    common_positive_upper = float(min(positive_limits))
    domain_oracle_rmse = {scope: pooled_rmse(value, total_rows) for scope, value in domain_oracle_sse.items()}
    domain_oracle_gain = {scope: current_rmse - value for scope, value in domain_oracle_rmse.items()}
    weakest_domain_oracle_gain = min(domain_oracle_gain.values())

    well_weights = np.asarray([optimum_weight(linear[i], quadratic[i]) for i in range(len(records))], dtype=np.float64)
    per_well_oracle_sse = float(np.sum(base_sse + 2.0 * well_weights * linear + well_weights * well_weights * quadratic))
    per_well_oracle_rmse = pooled_rmse(per_well_oracle_sse, total_rows)

    residuals = target_matrix - coefficient_matrix
    result = {
        "schema_version": 1,
        "task_id": "T028",
        "hypothesis_id": "H021",
        "source_commit": "d36e52d5b66571655f8033ab4345c5a873ea4b64",
        "diagnostic_only": True,
        "legal_warning": "Per-domain and per-well oracle weights use hidden labels and are forbidden at inference; they measure opportunity only.",
        "frozen_worth_gates": {
            "minimum_additional_headroom_rmse": MIN_ADDITIONAL_HEADROOM,
            "maximum_simple_global_placement_gain_rmse": MAX_SIMPLE_PLACEMENT_GAIN,
            "minimum_domain_oracle_headroom_rmse": MIN_DOMAIN_ORACLE_HEADROOM,
            "minimum_material_domain_weight_gaps": MIN_MATERIAL_DOMAIN_WEIGHT_GAPS,
            "material_domain_weight_gap": DOMAIN_WEIGHT_GAP,
        },
        "coverage": coverage,
        "feature_contract": {"wells": int(features.shape[0]), "features": int(features.shape[1]), "target_csv_max_delta": target_delta},
        "baseline": {"e011_rmse": E011_RMSE, "current_original_only_rmse": current_rmse, "current_weight": CURRENT_WEIGHT},
        "global_direction": {
            "linear": global_linear,
            "quadratic": global_quadratic,
            "optimal_weight": global_optimum,
            "optimal_rmse": global_optimum_rmse,
            "gain_from_reoptimizing_weight": current_rmse - global_optimum_rmse,
            "gain_vs_e011_at_optimal_weight": E011_RMSE - global_optimum_rmse,
            "all_domains_negative_first_derivative": all_domains_negative_derivative,
            "common_positive_weight_interval": [0.0, common_positive_upper],
            "common_positive_interval_nonempty": common_positive_upper > 0.0,
        },
        "oracle_headroom": {
            "domain_specific_weight_rmse": domain_oracle_rmse,
            "domain_specific_weight_gain_vs_current": domain_oracle_gain,
            "weakest_domain_specific_gain_vs_current": weakest_domain_oracle_gain,
            "per_well_weight_rmse": per_well_oracle_rmse,
            "per_well_weight_gain_vs_current": current_rmse - per_well_oracle_rmse,
            "per_well_weight_quantiles": {str(q): float(np.quantile(well_weights, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        },
        "coefficient_residual": {
            "pooled_rmse": float(np.sqrt(np.mean(residuals * residuals))),
            "mean": residuals.mean(axis=0).tolist(),
            "mean_l2": float(np.linalg.norm(residuals.mean(axis=0))),
            "maximum_domain_residual_mean_l2": max(float(row["residual_mean_l2"]) for row in domain_rows),
            "minimum_domain_residual_mean_l2": min(float(row["residual_mean_l2"]) for row in domain_rows),
        },
        "domain_diagnostics": domain_rows,
        "material_domain_weight_gaps": material_gaps,
    }
    gate_results = {
        "additional_per_well_headroom": result["oracle_headroom"]["per_well_weight_gain_vs_current"] >= MIN_ADDITIONAL_HEADROOM,
        "simple_global_placement_not_sufficient": result["global_direction"]["gain_from_reoptimizing_weight"] <= MAX_SIMPLE_PLACEMENT_GAIN,
        "domain_conflict_headroom": result["oracle_headroom"]["weakest_domain_specific_gain_vs_current"] >= MIN_DOMAIN_ORACLE_HEADROOM,
        "material_domain_weight_heterogeneity": material_gaps >= MIN_MATERIAL_DOMAIN_WEIGHT_GAPS,
    }
    result["gate_results"] = gate_results
    result["worth_screen_authorized"] = bool(all(gate_results.values()))
    result["decision"] = "preregister_t028" if result["worth_screen_authorized"] else "close_h021_without_implementation"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "current_rmse": current_rmse,
        "global_optimum_weight": global_optimum,
        "global_optimum_rmse": global_optimum_rmse,
        "common_positive_interval": result["global_direction"]["common_positive_weight_interval"],
        "domain_oracle_rmse": domain_oracle_rmse,
        "domain_oracle_gain_vs_current": domain_oracle_gain,
        "per_well_oracle_rmse": per_well_oracle_rmse,
        "material_domain_weight_gaps": material_gaps,
        "gate_results": gate_results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
