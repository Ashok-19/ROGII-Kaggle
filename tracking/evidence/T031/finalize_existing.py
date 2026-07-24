#!/usr/bin/env python3
"""Finalize an already-computed T031 run after a writer-only JSON defect.

This script never trains a model and never changes predictions, metrics, branch
selection, gates, or the scientific summary. It independently validates the
existing outputs and writes only closure artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "scratch/agents/t031-state-gate-20260724/main"
CONFIG_PATH = Path(__file__).with_name("config.json")
RUN_SCRIPT = Path(__file__).with_name("run_screen.py")

SCIENTIFIC_FILES = [
    "implementation_receipt.json",
    "tensor_manifest.json",
    "summary.json",
    "branch_metrics.csv",
    "context_metrics.csv",
    "group_metrics.csv",
    "selected_hyperparameters.csv",
    "control_metrics.csv",
    "predictions.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return {
        "pearson": float(pearsonr(y, pred).statistic),
        "spearman": float(spearmanr(y, pred).statistic),
        "mae_ft": float(np.mean(np.abs(y - pred))),
        "rmse_ft": float(np.sqrt(np.mean((y - pred) ** 2))),
        "count": int(len(y)),
    }


def close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    return bool(abs(float(a) - float(b)) <= tol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stderr-log",
        type=Path,
        default=ROOT / "scratch/agents/t031-state-gate-20260724/run2/stderr.log",
    )
    args = parser.parse_args()
    out = args.output_dir
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    before = {
        name: {"bytes": (out / name).stat().st_size, "sha256": sha256_file(out / name)}
        for name in SCIENTIFIC_FILES
    }

    branch = pd.read_csv(out / "branch_metrics.csv")
    context = pd.read_csv(out / "context_metrics.csv")
    groups = pd.read_csv(out / "group_metrics.csv")
    hyper = pd.read_csv(out / "selected_hyperparameters.csv")
    controls = pd.read_csv(out / "control_metrics.csv")
    pred = pd.read_csv(out / "predictions.csv")

    metadata_columns = {
        "well_id", "target", "legacy_spatial", "legacy_typewell",
        "legal_covariate_kmeans", "spatial_2d_kmeans",
    }
    expected_prediction_columns = {
        "mean_target", "summary_ridge", "raw_binned_ridge", "state_tcn",
        "state_tcn_no_typewell", "state_tcn_no_hidden_gr", "coordinate_only_control",
        "selected_branch_prediction", "permuted_target_prediction",
    }
    prediction_columns = set(pred.columns) - metadata_columns
    y = pred["target"].to_numpy(float)
    tensor_cache = np.load(out / "tensors_v1.npz", allow_pickle=False)
    cached_target = tensor_cache["target"].astype(float)

    recomputed = {
        "selected_metrics": metrics(y, pred["selected_branch_prediction"].to_numpy(float)),
        "summary_ridge_metrics": metrics(y, pred["summary_ridge"].to_numpy(float)),
        "coordinate_control_metrics": metrics(y, pred["coordinate_only_control"].to_numpy(float)),
        "permuted_target_metrics": metrics(y, pred["permuted_target_prediction"].to_numpy(float)),
    }
    metric_match = {
        key: all(close(actual[field], summary[key][field]) for field in actual)
        for key, actual in recomputed.items()
    }

    selected_name = str(summary["selected_branch"])
    checks: dict[str, bool] = {
        "all_scientific_files_exist": all((out / name).is_file() for name in SCIENTIFIC_FILES),
        "wells_exactly_773": len(pred) == 773,
        "well_ids_unique": pred["well_id"].nunique() == 773,
        "target_finite": bool(np.isfinite(y).all()),
        "target_hash_matches": sha256_array(cached_target.astype("<f8")) == config["target"]["sha256_float64_well_order"],
        "csv_target_matches_cache": bool(np.allclose(y, cached_target, rtol=0.0, atol=5e-13)),
        "prediction_columns_exact": prediction_columns == expected_prediction_columns,
        "all_predictions_finite": bool(np.isfinite(pred[sorted(prediction_columns)].to_numpy(float)).all()),
        "selected_prediction_identity": bool(np.array_equal(
            pred["selected_branch_prediction"].to_numpy(float),
            pred[selected_name].to_numpy(float),
        )),
        "headline_metrics_exact": all(metric_match.values()),
        "branch_rows_42": len(branch) == 42,
        "context_rows_150": len(context) == 150,
        "group_rows_140": len(groups) == 140,
        "hyperparameter_rows_150": len(hyper) == 150,
        "control_rows_6": len(controls) == 6,
        "context_count_25": context["context"].nunique() == 25,
        "six_modeled_context_branches": context["branch"].nunique() == 6,
        "seven_reported_branches": branch["branch"].nunique() == 7 and groups["branch"].nunique() == 7,
        "four_group_systems": groups["group_system"].nunique() == 4,
        "five_groups_each": bool((groups.groupby(["branch", "group_system"])["group"].nunique() == 5).all()),
        "all_table_metrics_finite": bool(np.isfinite(
            branch[["pearson", "spearman", "mae_ft", "rmse_ft"]].to_numpy(float)
        ).all()) and bool(np.isfinite(
            context[["pearson", "spearman", "mae_ft", "rmse_ft"]].to_numpy(float)
        ).all()) and bool(np.isfinite(
            groups[["pearson", "spearman", "mae_ft", "rmse_ft"]].to_numpy(float)
        ).all()),
        "permutation_control_below_cap": abs(float(summary["permuted_target_metrics"]["pearson"])) <= float(config["go_gate"]["maximum_permuted_abs_pearson"]),
        "hard_stop_matches_threshold": bool(
            float(summary["selected_metrics"]["pearson"]) < float(config["stop_gate"]["maximum_pearson_for_hard_stop"])
            or float(summary["selected_metrics"]["mae_ft"]) > float(config["stop_gate"]["minimum_mae_ft_for_hard_stop"])
        ),
        "decision_is_hard_stop": summary["decision"] == "HARD_STOP_CLOSE_COMPACT_RAW_SEQUENCE_STATE_RECOVERY" and bool(summary["hard_stop"]),
    }

    edge_cases = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "row_counts": {
            "branch_metrics": int(len(branch)),
            "context_metrics": int(len(context)),
            "group_metrics": int(len(groups)),
            "selected_hyperparameters": int(len(hyper)),
            "control_metrics": int(len(controls)),
            "predictions": int(len(pred)),
        },
        "metric_recomputation": recomputed,
        "metric_match": metric_match,
    }
    dump_json(out / "edge_cases.json", edge_cases)

    lines = [
        "# T031 Result — Raw-sequence latent formation-displacement identifiability gate",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        f"Selected legal raw-sequence branch: `{selected_name}`",
        f"- Pearson: {summary['selected_metrics']['pearson']:.6f}",
        f"- Spearman: {summary['selected_metrics']['spearman']:.6f}",
        f"- MAE: {summary['selected_metrics']['mae_ft']:.6f} ft",
        f"- RMSE: {summary['selected_metrics']['rmse_ft']:.6f} ft",
        f"- Legacy group Pearson floor: {summary['legacy_group_pearson_floor']:.6f}",
        f"- New target-free group Pearson floor: {summary['new_group_pearson_floor']:.6f}",
        "",
        "## Comparators",
        "",
        f"- Summary ridge Pearson/MAE: {summary['summary_ridge_metrics']['pearson']:.6f} / {summary['summary_ridge_metrics']['mae_ft']:.6f} ft",
        f"- Coordinate-only Pearson/MAE: {summary['coordinate_control_metrics']['pearson']:.6f} / {summary['coordinate_control_metrics']['mae_ft']:.6f} ft",
        f"- Permuted-target Pearson: {summary['permuted_target_metrics']['pearson']:.6f}",
        "",
        "## Gate outcomes",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in summary["gates"].items()
    )
    lines.extend([
        "",
        "The best raw sequence branch is materially worse than the frozen 142-summary ridge and falls below the preregistered hard-stop Pearson threshold. Compact raw-sequence latent-state recovery is closed; no T033 joint TVT model is authorized.",
        "",
        "This task is an auxiliary-state gate only. It does not emit a competition prediction.",
    ])
    (out / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {
        name: {"bytes": (out / name).stat().st_size, "sha256": sha256_file(out / name)}
        for name in SCIENTIFIC_FILES
    }
    correction = {
        "schema_version": 1,
        "type": "score_independent_writer_correction",
        "scientific_outputs_unchanged": before == after,
        "scientific_files_before": before,
        "scientific_files_after": after,
        "failure": "NumPy boolean in edge_cases could not be serialized by the original json_dump implementation after summary and all scientific CSVs were written.",
        "correction": "Convert NumPy scalars during JSON serialization and finalize existing outputs without model fitting or changes to stored metrics.",
        "stderr_log_sha256": sha256_file(args.stderr_log) if args.stderr_log.exists() else None,
        "run_script_sha256_after_writer_fix": sha256_file(RUN_SCRIPT),
        "finalizer_sha256": sha256_file(Path(__file__)),
        "edge_case_status": edge_cases["status"],
    }
    dump_json(out / "WRITER_CORRECTION.json", correction)

    manifest_names = SCIENTIFIC_FILES + [
        "edge_cases.json", "RESULT.md", "WRITER_CORRECTION.json",
    ]
    manifest = {
        "schema_version": 1,
        "files": [
            {"name": name, "bytes": (out / name).stat().st_size, "sha256": sha256_file(out / name)}
            for name in manifest_names
        ],
    }
    dump_json(out / "artifact_manifest.json", manifest)

    result = {
        "status": "PASS" if edge_cases["status"] == "PASS" and correction["scientific_outputs_unchanged"] else "FAIL",
        "decision": summary["decision"],
        "selected_branch": selected_name,
        "selected_metrics": summary["selected_metrics"],
        "scientific_outputs_unchanged": correction["scientific_outputs_unchanged"],
        "edge_case_status": edge_cases["status"],
        "manifest_files": len(manifest_names),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
