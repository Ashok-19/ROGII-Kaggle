#!/usr/bin/env python3
"""Build the E011 model, sealed private input dataset, and canonical notebook.

This tool never executes the notebook and never creates a Kaggle competition
submission.  It performs deterministic model fitting, committed-source
packaging, hash verification, notebook authoring, and static code-cell checks.
The user alone imports, attaches, and runs the notebook on Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "2"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.e011_inference import (  # noqa: E402
    load_e011_model,
    predict_spline_coefficients,
    profile_e011_test_input,
    validate_e011_model,
)

DEFAULT_DATASET_REF = "ashok205/rogii-e011-deployment-inputs"
DEFAULT_DATASET_VERSION = 1
BUNDLE_FILENAME = "rogii-e011-deployment-inputs-v1.zip.bin"
INPUT_RECEIPT_FILENAME = "e011-input-receipt-v1.json"
NOTEBOOK_PATH = Path("notebooks/training_and_submission/e011_spline4_deployment_kaggle.ipynb")
CONTRACT_PATH = Path("experiments/E011/deployment/kaggle_contract.json")
RESULT_ARCHIVE_FILENAME = "rogii-e011-deployment-results-v1.zip"
OUTPUT_MANIFEST_FILENAME = "e011-output-manifest.json"
RUN_RECEIPT_FILENAME = "e011-run-receipt.json"
WELL_DIAGNOSTICS_FILENAME = "e011-well-predictions.json"
SUBMISSION_FILENAME = "submission.csv"
THREAD_LIMIT = 2
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
TRAIN_WELLS = 773
TRAIN_SIGNATURE = "6ebe65b403f80fefd97dcd7bbfce7314252e779a1837c97364cf55761590fe77"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))


def safe_member_name(name: str) -> str:
    raw = str(name).replace("\\", "/")
    raw_parts = raw.split("/")
    normalized = PurePosixPath(raw)
    if (
        normalized.is_absolute()
        or not raw
        or any(part in {"", ".", ".."} for part in raw_parts)
        or not normalized.parts
        or any(part in {"", ".", ".."} for part in normalized.parts)
    ):
        raise ValueError(f"unsafe archive member: {name!r}")
    return normalized.as_posix()


def zip_info(name: str, *, compressed: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(safe_member_name(name), date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def git_bytes(root: Path, commit: str, path: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{path} is not available at source commit {commit}") from exc


def current_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def fit_model(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / "experiments/E011/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verification_path = root / "experiments/E011/verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (
        verification.get("decision") != "promote_statistically"
        or verification.get("metrics", {}).get("selected_candidate") != "spline4_ridge_equal_s075"
    ):
        raise ValueError("E011 verification does not authorize spline4_ridge_equal_s075")
    compact_spec = config["compact_input"]
    compact_path = root / compact_spec["path"]
    if compact_path.stat().st_size != int(compact_spec["bytes"]) or sha256_file(compact_path) != compact_spec["sha256"]:
        raise ValueError("E011 compact input identity mismatch")
    legal_spec = config["parent_artifacts"]["e008_legal_features"]
    legal_path = root / legal_spec["path"]
    if legal_path.stat().st_size != int(legal_spec["bytes"]) or sha256_file(legal_path) != legal_spec["sha256"]:
        raise ValueError("E008 legal feature identity mismatch")
    e008_config_path = root / "experiments/E008/config.json"
    e008_config = json.loads(e008_config_path.read_text(encoding="utf-8"))
    e006_model_path = root / e008_config["parents"]["e006_model"]
    if sha256_file(e006_model_path) != e008_config["parents"]["e006_model_sha256"]:
        raise ValueError("E006 deployment model identity mismatch")
    e006_model = json.loads(e006_model_path.read_text(encoding="utf-8"))
    e007_config_path = root / "experiments/E007/config.json"
    e007_config = json.loads(e007_config_path.read_text(encoding="utf-8"))

    arrays = np.load(compact_path, allow_pickle=False)
    feature_names = [str(value) for value in arrays["feature_names"].tolist()]
    well_ids = [str(value) for value in arrays["well_ids"].tolist()]
    features = np.asarray(arrays["features"], dtype=np.float64)
    targets = np.asarray(arrays["spline4__target_coefficients"], dtype=np.float64)
    if feature_names != sorted(feature_names) or len(feature_names) != 142 or len(set(feature_names)) != 142:
        raise ValueError("compact feature order differs from the frozen 142-feature contract")
    if features.shape != (773, 142) or targets.shape != (773, 4) or len(well_ids) != 773:
        raise ValueError("compact E011 model arrays have unexpected dimensions")
    if not np.all(np.isfinite(targets)):
        raise ValueError("compact spline4 coefficient targets are non-finite")

    medians = np.zeros(features.shape[1], dtype=np.float64)
    imputed = features.copy()
    nonfinite_cells = int(np.size(features) - np.count_nonzero(np.isfinite(features)))
    for column in range(features.shape[1]):
        finite = features[np.isfinite(features[:, column]), column]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
        imputed[:, column] = np.where(np.isfinite(features[:, column]), features[:, column], medians[column])
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    standardized = (imputed - means) / scales
    target_means = targets.mean(axis=0)
    target_scales = targets.std(axis=0)
    target_scales[~np.isfinite(target_scales) | (target_scales < 1e-9)] = 1.0
    scaled_targets = (targets - target_means) / target_scales

    x_mean = standardized.mean(axis=0)
    y_mean = scaled_targets.mean(axis=0)
    centered_x = standardized - x_mean
    centered_y = scaled_targets - y_mean
    gram = centered_x.T @ centered_x
    right = centered_x.T @ centered_y
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    alpha = 1.0
    ridge = eigenvectors @ ((eigenvectors.T @ right) / (eigenvalues[:, None] + alpha))
    scaled_intercept = y_mean - x_mean @ ridge
    raw_predictions = (standardized @ ridge + scaled_intercept) * target_scales + target_means
    if not np.all(np.isfinite(raw_predictions)):
        raise ValueError("full E011 fit emitted non-finite coefficients")

    model = {
        "schema_version": 1,
        "experiment_id": "E011",
        "model_name": "spline4_ridge_equal_s075",
        "representation": "spline4",
        "statistical_source_commit": verification["source_commit"],
        "training_wells": 773,
        "training_feature_cells": int(features.size),
        "training_nonfinite_feature_cells": nonfinite_cells,
        "selected_features": feature_names,
        "feature_medians": medians.tolist(),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "target_means": target_means.tolist(),
        "target_scales": target_scales.tolist(),
        "ridge_alpha": alpha,
        "ridge_coefficients": ridge.tolist(),
        "ridge_scaled_intercept": scaled_intercept.tolist(),
        "sample_weight": "equal_well",
        "shrinkage": 0.75,
        "coefficient_absolute_bound_ft": 80.0,
        "emitted_correction_absolute_bound_ft": 120.0,
        "spline_knots": [0.25, 0.5, 0.75, 1.0],
        "e006_model": e006_model,
        "e007_config": {"self_correlation": e007_config["self_correlation"]},
        "e008_config": {"feature_contract": e008_config["feature_contract"]},
        "surfaces_required": False,
        "absolute_spatial_coordinates_used": False,
        "direct_typewell_features_used": False,
        "hidden_labels_used_at_inference": False,
        "external_artifacts_required": False,
        "internet_required": False,
        "submission_authorized": False,
        "source_identities": {
            "compact_stats_sha256": sha256_file(compact_path),
            "legal_features_sha256": sha256_file(legal_path),
            "e006_model_sha256": sha256_file(e006_model_path),
            "e007_config_sha256": sha256_file(e007_config_path),
            "e008_config_sha256": sha256_file(e008_config_path),
            "e011_config_sha256": sha256_file(config_path),
            "e011_verification_sha256": sha256_file(verification_path),
            "e011_inference_sha256": sha256_file(root / "src/rogii_validation/e011_inference.py"),
        },
    }
    validate_e011_model(model)
    runtime_predictions = np.vstack(
        [
            predict_spline_coefficients(
                model,
                {
                    name: (
                        None
                        if not math.isfinite(float(features[row, column]))
                        else float(features[row, column])
                    )
                    for column, name in enumerate(feature_names)
                },
            )
            for row in range(features.shape[0])
        ]
    )
    expected = np.clip(0.75 * raw_predictions, -80.0, 80.0)
    runtime_delta = float(np.max(np.abs(runtime_predictions - expected)))
    if runtime_delta > 1e-10:
        raise ValueError(f"runtime/full-fit coefficient parity failed: {runtime_delta}")
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "candidate": model["model_name"],
        "wells": 773,
        "features": 142,
        "targets": 4,
        "alpha": alpha,
        "shrinkage": 0.75,
        "coefficient_bound_ft": 80.0,
        "nonfinite_feature_cells_imputed": nonfinite_cells,
        "feature_order_sha256": sha256_bytes("\n".join(feature_names).encode("utf-8")),
        "well_order_sha256": sha256_bytes("\n".join(well_ids).encode("utf-8")),
        "ridge_matrix_shape": list(ridge.shape),
        "runtime_coefficient_max_abs_delta": runtime_delta,
        "raw_prediction_minimum": float(np.min(raw_predictions)),
        "raw_prediction_maximum": float(np.max(raw_predictions)),
        "bounded_prediction_minimum": float(np.min(expected)),
        "bounded_prediction_maximum": float(np.max(expected)),
        "source_identities": model["source_identities"],
    }
    return model, receipt


def prepare_model(root: Path) -> dict[str, Any]:
    deployment_dir = root / "experiments/E011/deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    model, receipt = fit_model(root)
    model_path = deployment_dir / "model.json"
    write_json(model_path, model)
    receipt["model_sha256"] = sha256_file(model_path)
    write_json(deployment_dir / "model_fit_receipt.json", receipt)
    loaded = load_e011_model(model_path)
    validate_e011_model(loaded)
    return {
        "model": {
            "path": str(model_path.relative_to(root)),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },
        "fit_receipt": {
            "path": "experiments/E011/deployment/model_fit_receipt.json",
            "bytes": (deployment_dir / "model_fit_receipt.json").stat().st_size,
            "sha256": sha256_file(deployment_dir / "model_fit_receipt.json"),
        },
    }


def bundle_paths(root: Path) -> tuple[str, ...]:
    package = sorted(path.relative_to(root).as_posix() for path in (root / "src/rogii_validation").glob("*.py"))
    fixed = [
        "tools/run_e011_inference.py",
        "experiments/E011/config.json",
        "experiments/E011/manifest.json",
        "experiments/E011/verification.json",
        "experiments/E011/deployment/model.json",
        "experiments/E011/deployment/model_fit_receipt.json",
        "experiments/E011/deployment/feature_parity.json",
        "experiments/E011/deployment/edge_case_receipt.json",
        "experiments/E011/deployment/pseudo_hidden_benchmark.json",
        "experiments/E011/deployment/validation_receipt.json",
    ]
    return tuple(dict.fromkeys([*package, *fixed]))


def committed_entries(root: Path, source_commit: str, paths: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    entries: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for relative in paths:
        safe_member_name(relative)
        payload = git_bytes(root, source_commit, relative)
        payloads[relative] = payload
        entries.append({"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)})
    return entries, payloads


def write_bundle(path: Path, manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        archive.writestr(zip_info("bundle_manifest.json"), canonical_json(manifest))
        for entry in manifest["files"]:
            relative = safe_member_name(entry["path"])
            archive.writestr(zip_info(relative), payloads[relative])
    temporary.replace(path)


def notebook_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    constants_source = "CONTRACT = " + json.dumps(contract, indent=2, sort_keys=True) + "\n"
    preflight = r'''import datetime as _dt
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import traceback
import zipfile
from pathlib import Path, PurePosixPath

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = str(CONTRACT["THREAD_LIMIT"])
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

WORKING = Path("/kaggle/working")
RUNTIME_ROOT = WORKING / "rogii-e011-runtime"
PREFLIGHT_PATH = WORKING / "e011-preflight.json"
RUN_RECEIPT_PATH = WORKING / CONTRACT["RUN_RECEIPT_FILENAME"]
SUBMISSION_PATH = WORKING / CONTRACT["SUBMISSION_FILENAME"]
WELL_DIAGNOSTICS_PATH = WORKING / CONTRACT["WELL_DIAGNOSTICS_FILENAME"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe(name: str) -> str:
    raw = str(name).replace("\\", "/")
    raw_parts = raw.split("/")
    value = PurePosixPath(raw)
    if value.is_absolute() or not raw or any(part in {"", ".", ".."} for part in raw_parts) or not value.parts:
        raise RuntimeError(f"unsafe archive member {name!r}")
    return value.as_posix()


def _write_failure(stage: str, exc: BaseException) -> None:
    failure = {
        "schema_version": 1,
        "experiment_id": "E011",
        "status": "FAILED",
        "stage": stage,
        "source_commit": CONTRACT["SOURCE_COMMIT"],
        "failed_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "submission_made": False,
    }
    RUN_RECEIPT_PATH.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")

try:
    bundle_matches = sorted(Path("/kaggle/input").rglob(CONTRACT["BUNDLE_FILENAME"]))
    receipt_matches = sorted(Path("/kaggle/input").rglob(CONTRACT["INPUT_RECEIPT_FILENAME"]))
    if len(bundle_matches) != 1 or len(receipt_matches) != 1:
        raise RuntimeError({
            "bundle_matches": [str(path) for path in bundle_matches],
            "receipt_matches": [str(path) for path in receipt_matches],
        })
    BUNDLE_PATH = bundle_matches[0]
    INPUT_RECEIPT_PATH = receipt_matches[0]
    if BUNDLE_PATH.stat().st_size != int(CONTRACT["BUNDLE_BYTES"]) or _sha256(BUNDLE_PATH) != CONTRACT["BUNDLE_SHA256"]:
        raise RuntimeError("sealed E011 bundle identity mismatch")
    if _sha256(INPUT_RECEIPT_PATH) != CONTRACT["INPUT_RECEIPT_SHA256"]:
        raise RuntimeError("E011 input receipt identity mismatch")
    input_receipt = json.loads(INPUT_RECEIPT_PATH.read_text(encoding="utf-8"))
    if (
        input_receipt["dataset"]["ref"] != CONTRACT["DATASET_REF"]
        or int(input_receipt["dataset"]["version"]) != int(CONTRACT["DATASET_VERSION"])
    ):
        raise RuntimeError("attached E011 dataset ref/version differs from notebook contract")

    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate E011 archive members")
        for name in names:
            _safe(name)
        archive.extractall(RUNTIME_ROOT)
    manifest = json.loads((RUNTIME_ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest["source_commit"] != CONTRACT["SOURCE_COMMIT"] or manifest["model_sha256"] != CONTRACT["MODEL_SHA256"]:
        raise RuntimeError("E011 bundle source/model contract mismatch")
    for entry in manifest["files"]:
        path = RUNTIME_ROOT / _safe(entry["path"])
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"E011 extracted member mismatch: {entry['path']}")

    for module_name in tuple(sys.modules):
        if module_name == "rogii_validation" or module_name.startswith("rogii_validation."):
            del sys.modules[module_name]
    importlib.invalidate_caches()
    sys.path.insert(0, str(RUNTIME_ROOT / "src"))
    from rogii_validation.harness import scan_profiles
    from rogii_validation.e011_inference import load_e011_model, profile_e011_test_input, write_e011_submission

    preferred_roots = (
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    )
    COMPETITION_ROOT = next(
        (
            root
            for root in preferred_roots
            if (root / "train").is_dir() and (root / "test").is_dir() and (root / "sample_submission.csv").is_file()
        ),
        None,
    )
    if COMPETITION_ROOT is None:
        fallback_roots = sorted(
            sample.parent
            for sample in Path("/kaggle/input").rglob("sample_submission.csv")
            if (sample.parent / "train").is_dir() and (sample.parent / "test").is_dir()
        )
        if len(fallback_roots) != 1:
            raise RuntimeError({"competition_roots_found": [str(root) for root in fallback_roots]})
        COMPETITION_ROOT = fallback_roots[0]
    TRAIN_DIR = COMPETITION_ROOT / "train"
    TEST_DIR = COMPETITION_ROOT / "test"
    SAMPLE_SUBMISSION = COMPETITION_ROOT / "sample_submission.csv"
    _, TRAIN_PROFILE = scan_profiles(TRAIN_DIR)
    if int(TRAIN_PROFILE["well_count"]) != int(CONTRACT["TRAIN_WELLS"]) or TRAIN_PROFILE["data_signature"] != CONTRACT["TRAIN_SIGNATURE"]:
        raise RuntimeError({
            "competition_root": str(COMPETITION_ROOT),
            "expected_train_wells": CONTRACT["TRAIN_WELLS"],
            "expected_train_signature": CONTRACT["TRAIN_SIGNATURE"],
            "actual_train_profile": TRAIN_PROFILE,
        })
    TEST_PROFILE = profile_e011_test_input(TEST_DIR, SAMPLE_SUBMISSION)
    if int(TEST_PROFILE["wells"]) == int(CONTRACT["VISIBLE_FIXTURE"]["wells"]):
        fixture = CONTRACT["VISIBLE_FIXTURE"]
        if (
            TEST_PROFILE["input_files_signature"] != fixture["input_files_signature"]
            or TEST_PROFILE["sample_id_order_sha256"] != fixture["sample_id_order_sha256"]
            or int(TEST_PROFILE["sample_rows"]) != int(fixture["sample_rows"])
        ):
            raise RuntimeError("visible E011 competition fixture identity mismatch")
        TEST_MODE = "visible_fixture"
    else:
        TEST_MODE = "private_rerun"
    MODEL_PATH = RUNTIME_ROOT / "experiments/E011/deployment/model.json"
    if _sha256(MODEL_PATH) != CONTRACT["MODEL_SHA256"]:
        raise RuntimeError("E011 deployment model hash mismatch")
    MODEL = load_e011_model(MODEL_PATH)

    PREFLIGHT = {
        "schema_version": 1,
        "experiment_id": "E011",
        "status": "PASS",
        "checked_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_commit": CONTRACT["SOURCE_COMMIT"],
        "model_sha256": CONTRACT["MODEL_SHA256"],
        "bundle": {"path": str(BUNDLE_PATH), "bytes": BUNDLE_PATH.stat().st_size, "sha256": _sha256(BUNDLE_PATH)},
        "input_receipt": {"path": str(INPUT_RECEIPT_PATH), "sha256": _sha256(INPUT_RECEIPT_PATH)},
        "dataset": {"ref": CONTRACT["DATASET_REF"], "version": CONTRACT["DATASET_VERSION"]},
        "competition_root": str(COMPETITION_ROOT),
        "train_profile": TRAIN_PROFILE,
        "test_mode": TEST_MODE,
        "test_profile": TEST_PROFILE,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "thread_limit": CONTRACT["THREAD_LIMIT"],
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                )
            },
            "internet_expected": False,
            "accelerator_expected": "CPU",
        },
        "submission_authorized": False,
    }
    PREFLIGHT_PATH.write_text(json.dumps(PREFLIGHT, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(PREFLIGHT, indent=2, sort_keys=True))
except Exception as exc:
    _write_failure("preflight", exc)
    raise
'''
    execution = r'''import datetime as _dt
import json
import resource
import traceback

import numpy as np
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

STARTED_AT = _dt.datetime.now(_dt.timezone.utc)
try:
    with threadpool_limits(limits=CONTRACT["THREAD_LIMIT"]):
        THREAD_POOLS_BEFORE = threadpool_info()
        if any(int(item.get("num_threads", 0) or 0) > int(CONTRACT["THREAD_LIMIT"]) for item in THREAD_POOLS_BEFORE):
            raise RuntimeError({"thread_pool_limit_failed": THREAD_POOLS_BEFORE})
        RESULT = write_e011_submission(MODEL, TEST_DIR, SAMPLE_SUBMISSION, SUBMISSION_PATH)
        WELL_DIAGNOSTICS_PATH.write_text(
            json.dumps(RESULT["well_predictions"], indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        THREAD_POOLS_AFTER = threadpool_info()
        if any(int(item.get("num_threads", 0) or 0) > int(CONTRACT["THREAD_LIMIT"]) for item in THREAD_POOLS_AFTER):
            raise RuntimeError({"thread_pool_limit_failed_after": THREAD_POOLS_AFTER})
    ENDED_AT = _dt.datetime.now(_dt.timezone.utc)
    print(json.dumps({
        "status": "PASS",
        "test_mode": TEST_MODE,
        "wells": RESULT["wells"],
        "rows": RESULT["rows"],
        "submission_sha256": RESULT["sha256"],
        "wall_seconds": (ENDED_AT - STARTED_AT).total_seconds(),
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }, indent=2, sort_keys=True))
except Exception as exc:
    _write_failure("inference", exc)
    raise
'''
    packaging = r'''import hashlib
import json
import resource
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def _output_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _output_safe(name: str) -> str:
    raw = str(name).replace("\\", "/")
    parts = raw.split("/")
    value = PurePosixPath(raw)
    if value.is_absolute() or not raw or any(part in {"", ".", ".."} for part in parts) or not value.parts:
        raise RuntimeError(f"unsafe output member {name!r}")
    return value.as_posix()

try:
    output_sources = [SUBMISSION_PATH, WELL_DIAGNOSTICS_PATH, PREFLIGHT_PATH]
    if not all(path.is_file() for path in output_sources):
        raise RuntimeError("E011 expected output file is missing before packaging")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E011",
        "source_commit": CONTRACT["SOURCE_COMMIT"],
        "model_sha256": CONTRACT["MODEL_SHA256"],
        "input_bundle_sha256": CONTRACT["BUNDLE_SHA256"],
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _output_sha256(path)}
            for path in output_sources
        ],
    }
    MANIFEST_PATH = WORKING / CONTRACT["OUTPUT_MANIFEST_FILENAME"]
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ARCHIVE_PATH = WORKING / CONTRACT["RESULT_ARCHIVE_FILENAME"]
    temporary = ARCHIVE_PATH.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [*output_sources, MANIFEST_PATH]:
            info = zipfile.ZipInfo(_output_safe(path.name), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    temporary.replace(ARCHIVE_PATH)
    RECEIPT = {
        "schema_version": 1,
        "experiment_id": "E011",
        "status": "COMPLETE",
        "source_commit": CONTRACT["SOURCE_COMMIT"],
        "model_sha256": CONTRACT["MODEL_SHA256"],
        "input_bundle": {"sha256": CONTRACT["BUNDLE_SHA256"], "bytes": CONTRACT["BUNDLE_BYTES"]},
        "dataset": {"ref": CONTRACT["DATASET_REF"], "version": CONTRACT["DATASET_VERSION"]},
        "competition_root": str(COMPETITION_ROOT),
        "train_profile": TRAIN_PROFILE,
        "test_mode": TEST_MODE,
        "test_profile": TEST_PROFILE,
        "started_at_utc": STARTED_AT.isoformat(),
        "ended_at_utc": ENDED_AT.isoformat(),
        "wall_seconds": (ENDED_AT - STARTED_AT).total_seconds(),
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "thread_limit": CONTRACT["THREAD_LIMIT"],
        "thread_pools_before": THREAD_POOLS_BEFORE,
        "thread_pools_after": THREAD_POOLS_AFTER,
        "internet_expected": False,
        "accelerator_expected": "CPU",
        "prediction": {key: value for key, value in RESULT.items() if key != "well_predictions"},
        "output_manifest": {"filename": MANIFEST_PATH.name, "bytes": MANIFEST_PATH.stat().st_size, "sha256": _output_sha256(MANIFEST_PATH)},
        "result_archive": {"filename": ARCHIVE_PATH.name, "bytes": ARCHIVE_PATH.stat().st_size, "sha256": _output_sha256(ARCHIVE_PATH)},
        "submission_file_created": True,
        "kaggle_submission_made": False,
    }
    RUN_RECEIPT_PATH.write_text(json.dumps(RECEIPT, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(RECEIPT, indent=2, sort_keys=True))
except Exception as exc:
    _write_failure("packaging", exc)
    raise
'''
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "e011-intro",
                "metadata": {},
                "source": [
                    "# E011 spline4 deployment parity\n",
                    "Private, internet-disabled, CPU-only deployment notebook. Attach the exact private E011 input dataset version and the ROGII competition data manually. The user is the only authorized notebook runner. This notebook never submits to the competition.\n",
                ],
            },
            {"cell_type": "code", "id": "e011-contract", "execution_count": None, "metadata": {}, "outputs": [], "source": constants_source.splitlines(keepends=True)},
            {"cell_type": "code", "id": "e011-preflight", "execution_count": None, "metadata": {}, "outputs": [], "source": preflight.splitlines(keepends=True)},
            {"cell_type": "code", "id": "e011-inference", "execution_count": None, "metadata": {}, "outputs": [], "source": execution.splitlines(keepends=True)},
            {"cell_type": "code", "id": "e011-package", "execution_count": None, "metadata": {}, "outputs": [], "source": packaging.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "rogii": {
                "experiment_id": "E011",
                "source_commit": contract["SOURCE_COMMIT"],
                "model_sha256": contract["MODEL_SHA256"],
                "bundle_sha256": contract["BUNDLE_SHA256"],
                "dataset_ref": contract["DATASET_REF"],
                "dataset_version": contract["DATASET_VERSION"],
                "internet_required": False,
                "accelerator": "CPU",
                "maximum_threads": THREAD_LIMIT,
                "user_execution_required": True,
                "assistant_execution_authorized": False,
                "submission_authorized": False,
                "expected_outputs": [
                    RUN_RECEIPT_FILENAME,
                    OUTPUT_MANIFEST_FILENAME,
                    RESULT_ARCHIVE_FILENAME,
                    SUBMISSION_FILENAME,
                    WELL_DIAGNOSTICS_FILENAME,
                ],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def static_validate_notebook(notebook: Mapping[str, Any]) -> dict[str, Any]:
    if int(notebook.get("nbformat", 0)) != 4:
        raise ValueError("E011 notebook must use nbformat 4")
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(code_cells) != 4:
        raise ValueError("E011 notebook must contain exactly four code cells")
    for cell in code_cells:
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        compile(text, f"notebook:{cell.get('id', 'unknown')}", "exec")
    combined = "\n".join("".join(cell["source"]) for cell in code_cells)
    forbidden = ["kaggle_create_code_competition_submission", "competitions submit", "kaggle competitions submit"]
    present = [token for token in forbidden if token in combined]
    if present:
        raise ValueError(f"E011 notebook contains submission operation tokens: {present}")
    return {"status": "PASS", "code_cells": len(code_cells), "submission_operation_tokens": []}


def regenerate_artifact_manifest(root: Path, notebook_path: Path | None = None) -> dict[str, Any]:
    deployment_dir = root / "experiments/E011/deployment"
    files = [path for path in deployment_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files)
        ],
    }
    if notebook_path is not None and notebook_path.is_file():
        payload["notebook"] = {
            "path": str(notebook_path.relative_to(root)),
            "bytes": notebook_path.stat().st_size,
            "sha256": sha256_file(notebook_path),
        }
    write_json(deployment_dir / "artifact_manifest.json", payload)
    return payload


def build_package(
    *,
    root: Path,
    source_commit: str,
    staging_dir: Path,
    notebook_path: Path,
    dataset_ref: str,
    dataset_version: int,
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    paths = bundle_paths(root)
    entries, payloads = committed_entries(root, source_commit, paths)
    model_entry = next(entry for entry in entries if entry["path"] == "experiments/E011/deployment/model.json")
    model_payload = json.loads(payloads[model_entry["path"]].decode("utf-8"))
    validate_e011_model(model_payload)
    inference_sha = next(entry["sha256"] for entry in entries if entry["path"] == "src/rogii_validation/e011_inference.py")
    if model_payload["source_identities"]["e011_inference_sha256"] != inference_sha:
        raise ValueError("committed E011 inference source differs from the fitted model receipt")
    bundle_manifest = {
        "schema_version": 1,
        "experiment_id": "E011",
        "role": "spline4_deployment_runtime",
        "source_commit": source_commit,
        "model_sha256": model_entry["sha256"],
        "train_wells": TRAIN_WELLS,
        "train_signature": TRAIN_SIGNATURE,
        "maximum_threads": THREAD_LIMIT,
        "internet_required": False,
        "external_artifacts_required": False,
        "submission_authorized": False,
        "files": entries,
    }
    staging_dir.mkdir(parents=True, exist_ok=True)
    allowed_names = {BUNDLE_FILENAME, INPUT_RECEIPT_FILENAME, "dataset-metadata.json"}
    for path in staging_dir.iterdir():
        if path.name not in allowed_names:
            raise ValueError(f"unexpected file in E011 staging directory: {path}")
    bundle_path = staging_dir / BUNDLE_FILENAME
    write_bundle(bundle_path, bundle_manifest, payloads)
    bundle_sha = sha256_file(bundle_path)
    visible_profile = profile_e011_test_input(root / "data/test", root / "data/sample_submission.csv")
    input_receipt = {
        "schema_version": 1,
        "experiment_id": "E011",
        "role": "spline4_deployment_inputs",
        "dataset": {"ref": dataset_ref, "version": dataset_version, "private": True},
        "source_commit": source_commit,
        "model_sha256": model_entry["sha256"],
        "bundle": {"filename": BUNDLE_FILENAME, "bytes": bundle_path.stat().st_size, "sha256": bundle_sha},
        "train_wells": TRAIN_WELLS,
        "train_signature": TRAIN_SIGNATURE,
        "visible_fixture": {
            key: visible_profile[key]
            for key in ("wells", "rows", "known_rows", "hidden_rows", "sample_rows", "well_ids_sha256", "sample_id_order_sha256", "input_files_signature")
        },
        "maximum_threads": THREAD_LIMIT,
        "competition_data_included": False,
        "competition_data_attachment_required": True,
        "user_execution_required": True,
        "assistant_execution_authorized": False,
        "submission_authorized": False,
    }
    receipt_path = staging_dir / INPUT_RECEIPT_FILENAME
    write_json(receipt_path, input_receipt)
    metadata = {
        "title": "ROGII E011 Deployment Inputs",
        "id": dataset_ref,
        "licenses": [{"name": "other"}],
    }
    write_json(staging_dir / "dataset-metadata.json", metadata)
    contract = {
        "SOURCE_COMMIT": source_commit,
        "MODEL_SHA256": model_entry["sha256"],
        "BUNDLE_FILENAME": BUNDLE_FILENAME,
        "BUNDLE_SHA256": bundle_sha,
        "BUNDLE_BYTES": bundle_path.stat().st_size,
        "INPUT_RECEIPT_FILENAME": INPUT_RECEIPT_FILENAME,
        "INPUT_RECEIPT_SHA256": sha256_file(receipt_path),
        "DATASET_REF": dataset_ref,
        "DATASET_VERSION": dataset_version,
        "TRAIN_WELLS": TRAIN_WELLS,
        "TRAIN_SIGNATURE": TRAIN_SIGNATURE,
        "VISIBLE_FIXTURE": input_receipt["visible_fixture"],
        "THREAD_LIMIT": THREAD_LIMIT,
        "RESULT_ARCHIVE_FILENAME": RESULT_ARCHIVE_FILENAME,
        "OUTPUT_MANIFEST_FILENAME": OUTPUT_MANIFEST_FILENAME,
        "RUN_RECEIPT_FILENAME": RUN_RECEIPT_FILENAME,
        "WELL_DIAGNOSTICS_FILENAME": WELL_DIAGNOSTICS_FILENAME,
        "SUBMISSION_FILENAME": SUBMISSION_FILENAME,
    }
    notebook = notebook_payload(contract)
    static_receipt = static_validate_notebook(notebook)
    write_json(notebook_path, notebook)
    durable_contract = {
        "schema_version": 1,
        "status": "prepared_for_user_run",
        "source_commit": source_commit,
        "model": {"path": model_entry["path"], "bytes": model_entry["bytes"], "sha256": model_entry["sha256"]},
        "dataset": {
            "ref": dataset_ref,
            "version": dataset_version,
            "private": True,
            "bundle_filename": BUNDLE_FILENAME,
            "bundle_bytes": bundle_path.stat().st_size,
            "bundle_sha256": bundle_sha,
            "input_receipt_filename": INPUT_RECEIPT_FILENAME,
            "input_receipt_sha256": sha256_file(receipt_path),
        },
        "notebook": {
            "path": str(notebook_path.relative_to(root)),
            "bytes": notebook_path.stat().st_size,
            "sha256": sha256_file(notebook_path),
            "static_validation": static_receipt,
        },
        "required_manual_attachments": [
            f"private dataset {dataset_ref} version {dataset_version}",
            "ROGII competition data",
        ],
        "runtime": {"private": True, "internet": False, "accelerator": "CPU", "maximum_threads": THREAD_LIMIT},
        "expected_outputs": [
            RUN_RECEIPT_FILENAME,
            OUTPUT_MANIFEST_FILENAME,
            RESULT_ARCHIVE_FILENAME,
            SUBMISSION_FILENAME,
            WELL_DIAGNOSTICS_FILENAME,
        ],
        "user_execution_required": True,
        "assistant_execution_authorized": False,
        "notebook_executed": False,
        "kaggle_submission_authorized": False,
        "kaggle_submission_made": False,
    }
    write_json(root / CONTRACT_PATH, durable_contract)
    regenerate_artifact_manifest(root, notebook_path)
    return {
        "status": "PREPARED_NOT_EXECUTED",
        "source_commit": source_commit,
        "dataset_ref": dataset_ref,
        "dataset_version": dataset_version,
        "staging_dir": str(staging_dir),
        "bundle": durable_contract["dataset"],
        "notebook": durable_contract["notebook"],
        "required_manual_attachments": durable_contract["required_manual_attachments"],
        "expected_outputs": durable_contract["expected_outputs"],
        "user_execution_required": True,
        "assistant_execution_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=ROOT)
    command.add_argument("--prepare-model-only", action="store_true")
    command.add_argument("--source-commit", default="")
    command.add_argument("--staging-dir", type=Path, default=Path("scratch/agents/t020-e011-kaggle/input-v1"))
    command.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    command.add_argument("--dataset-ref", default=DEFAULT_DATASET_REF)
    command.add_argument("--dataset-version", type=int, default=DEFAULT_DATASET_VERSION)
    return command


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    root = arguments.root.resolve()
    if arguments.prepare_model_only:
        result = prepare_model(root)
        regenerate_artifact_manifest(root)
        print(json.dumps({"status": "MODEL_PREPARED", **result}, indent=2, sort_keys=True))
        return 0
    source_commit = arguments.source_commit.strip()
    if not source_commit:
        raise SystemExit("--source-commit is required unless --prepare-model-only is used")
    result = build_package(
        root=root,
        source_commit=source_commit,
        staging_dir=resolve(root, arguments.staging_dir),
        notebook_path=resolve(root, arguments.notebook),
        dataset_ref=str(arguments.dataset_ref),
        dataset_version=int(arguments.dataset_version),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
