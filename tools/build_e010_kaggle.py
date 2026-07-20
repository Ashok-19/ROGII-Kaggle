#!/usr/bin/env python3
"""Build the sealed E010 private-Kaggle input bundle and canonical notebook.

The heavy E010 validation itself is not executed locally. This builder performs
only deterministic packaging, hash verification, and notebook authoring. The
notebook imports the exact bundled ``src`` package, caps native CPU pools at two
threads, locates the official competition data by frozen data signature, runs
fail closed, and emits a result ZIP plus machine-readable receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_COMMIT = "49ca3995269638026561c5593ad2628f8f08a4e9"
DEFAULT_DATASET_REF = "ashok205/rogii-e010-selector-inputs"
DEFAULT_DATASET_VERSION = 2
BUNDLE_FILENAME = "rogii-e010-inputs-v2.zip.bin"
INPUT_RECEIPT_FILENAME = "e010-input-receipt-v2.json"
NOTEBOOK_PATH = Path("notebooks/training_and_submission/e010_candidate_selector_kaggle.ipynb")
RESULT_ARCHIVE_FILENAME = "rogii-e010-results-v2.zip"
OUTPUT_MANIFEST_FILENAME = "e010-output-manifest.json"
RUN_RECEIPT_FILENAME = "e010-run-receipt.json"
THREAD_LIMIT = 2
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


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
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


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


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(safe_member_name(name), date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def git_bytes(root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)


def source_paths(root: Path, config: Mapping[str, Any]) -> tuple[str, ...]:
    package = sorted(path.relative_to(root).as_posix() for path in (root / "src/rogii_validation").glob("*.py"))
    fixed = [
        "tools/run_e010.py",
        "experiments/E010/config.json",
        "experiments/E010/manifest.json",
        "experiments/E010/PRE_REGISTRATION.md",
        *[str(path) for path in config["fold_files"]],
    ]
    parents = [str(item["path"]) for item in config["parents"].values()]
    return tuple(dict.fromkeys([*package, *fixed, *parents]))


def verify_inputs(
    root: Path,
    config: Mapping[str, Any],
    commit: str,
    paths: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    parent_hashes = {str(item["path"]): str(item["sha256"]) for item in config["parents"].values()}
    tracked_prefixes = ("src/", "tools/", "experiments/E010/", "folds/")
    entries: list[dict[str, Any]] = []
    committed_payloads: dict[str, bytes] = {}
    for relative in paths:
        safe_member_name(relative)
        path = root / relative
        if relative.startswith(tracked_prefixes):
            payload = git_bytes(root, commit, relative)
            committed_payloads[relative] = payload
            actual_sha = sha256_bytes(payload)
            actual_bytes = len(payload)
        else:
            if not path.is_file():
                raise FileNotFoundError(relative)
            actual_sha = sha256_file(path)
            actual_bytes = path.stat().st_size
        if relative in parent_hashes and actual_sha != parent_hashes[relative]:
            raise ValueError(f"parent hash mismatch for {relative}: {actual_sha} != {parent_hashes[relative]}")
        entries.append({"path": relative, "bytes": actual_bytes, "sha256": actual_sha})
    return entries, committed_payloads


def write_deterministic_bundle(
    root: Path,
    output_path: Path,
    manifest: Mapping[str, Any],
    committed_payloads: Mapping[str, bytes] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        archive.writestr(zip_info("bundle_manifest.json"), canonical_json(manifest))
        for entry in manifest["files"]:
            relative = safe_member_name(str(entry["path"]))
            payload = (committed_payloads or {}).get(relative)
            source_context = io.BytesIO(payload) if payload is not None else (root / relative).open("rb")
            with source_context as source, archive.open(zip_info(relative), "w", force_zip64=True) as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    temporary.replace(output_path)


def notebook_payload(*, source_commit: str, config_sha: str, bundle_sha: str, bundle_bytes: int,
                     receipt_sha: str, dataset_ref: str, dataset_version: int,
                     data_signature: str, expected_wells: int) -> dict[str, Any]:
    constants = {
        "SOURCE_COMMIT": source_commit,
        "CONFIG_SHA256": config_sha,
        "BUNDLE_FILENAME": BUNDLE_FILENAME,
        "BUNDLE_SHA256": bundle_sha,
        "BUNDLE_BYTES": bundle_bytes,
        "INPUT_RECEIPT_FILENAME": INPUT_RECEIPT_FILENAME,
        "INPUT_RECEIPT_SHA256": receipt_sha,
        "DATASET_REF": dataset_ref,
        "DATASET_VERSION": dataset_version,
        "DATA_SIGNATURE": data_signature,
        "EXPECTED_WELLS": expected_wells,
        "THREAD_LIMIT": THREAD_LIMIT,
        "RESULT_ARCHIVE_FILENAME": RESULT_ARCHIVE_FILENAME,
        "OUTPUT_MANIFEST_FILENAME": OUTPUT_MANIFEST_FILENAME,
        "RUN_RECEIPT_FILENAME": RUN_RECEIPT_FILENAME,
    }
    constants_source = "CONTRACT = " + json.dumps(constants, indent=2, sort_keys=True) + "\n"
    preflight = r'''import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = str(CONTRACT["THREAD_LIMIT"])
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

WORKING = Path("/kaggle/working")
RUNTIME_ROOT = WORKING / "rogii-e010-runtime"
RESULT_DIR = WORKING / "e010-results"
ARTIFACT_DIR = WORKING / "e010-artifacts"
PREFLIGHT_PATH = WORKING / "e010-preflight.json"


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

bundle_matches = sorted(Path("/kaggle/input").rglob(CONTRACT["BUNDLE_FILENAME"]))
receipt_matches = sorted(Path("/kaggle/input").rglob(CONTRACT["INPUT_RECEIPT_FILENAME"]))
if len(bundle_matches) != 1 or len(receipt_matches) != 1:
    raise RuntimeError({"bundle_matches": [str(p) for p in bundle_matches], "receipt_matches": [str(p) for p in receipt_matches]})
bundle_path = bundle_matches[0]
input_receipt_path = receipt_matches[0]
if bundle_path.stat().st_size != CONTRACT["BUNDLE_BYTES"] or _sha256(bundle_path) != CONTRACT["BUNDLE_SHA256"]:
    raise RuntimeError("sealed E010 bundle identity mismatch")
if _sha256(input_receipt_path) != CONTRACT["INPUT_RECEIPT_SHA256"]:
    raise RuntimeError("E010 input receipt identity mismatch")
input_receipt = json.loads(input_receipt_path.read_text(encoding="utf-8"))
if input_receipt["dataset"]["ref"] != CONTRACT["DATASET_REF"] or int(input_receipt["dataset"]["version"]) != int(CONTRACT["DATASET_VERSION"]):
    raise RuntimeError("attached E010 dataset ref/version differs from notebook contract")

if RUNTIME_ROOT.exists():
    shutil.rmtree(RUNTIME_ROOT)
RUNTIME_ROOT.mkdir(parents=True)
with zipfile.ZipFile(bundle_path) as archive:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate E010 archive members")
    for name in names:
        _safe(name)
    archive.extractall(RUNTIME_ROOT)
manifest = json.loads((RUNTIME_ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
if manifest["source_commit"] != CONTRACT["SOURCE_COMMIT"] or manifest["config_sha256"] != CONTRACT["CONFIG_SHA256"]:
    raise RuntimeError("E010 bundle source/config contract mismatch")
for entry in manifest["files"]:
    path = RUNTIME_ROOT / _safe(entry["path"])
    if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"E010 extracted member mismatch: {entry['path']}")

sys.path.insert(0, str(RUNTIME_ROOT / "src"))
from rogii_validation.harness import scan_profiles

preferred_roots = (
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
)
COMPETITION_ROOT = next(
    (
        root
        for root in preferred_roots
        if (root / "train").is_dir() and (root / "sample_submission.csv").is_file()
    ),
    None,
)
if COMPETITION_ROOT is None:
    fallback_roots = sorted(
        sample.parent
        for sample in Path("/kaggle/input").rglob("sample_submission.csv")
        if (sample.parent / "train").is_dir()
    )
    if len(fallback_roots) != 1:
        raise RuntimeError({"competition_roots_found": [str(root) for root in fallback_roots]})
    COMPETITION_ROOT = fallback_roots[0]
TRAIN_DIR = COMPETITION_ROOT / "train"
_, DATA_PROFILE = scan_profiles(TRAIN_DIR)
if int(DATA_PROFILE["well_count"]) != int(CONTRACT["EXPECTED_WELLS"]) or DATA_PROFILE["data_signature"] != CONTRACT["DATA_SIGNATURE"]:
    raise RuntimeError({
        "competition_root": str(COMPETITION_ROOT),
        "expected_wells": CONTRACT["EXPECTED_WELLS"],
        "expected_signature": CONTRACT["DATA_SIGNATURE"],
        "actual_profile": DATA_PROFILE,
    })

preflight = {
    "schema_version": 1,
    "experiment_id": "E010",
    "status": "PASS",
    "checked_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "source_commit": CONTRACT["SOURCE_COMMIT"],
    "config_sha256": CONTRACT["CONFIG_SHA256"],
    "bundle": {"path": str(bundle_path), "bytes": bundle_path.stat().st_size, "sha256": _sha256(bundle_path)},
    "input_receipt": {"path": str(input_receipt_path), "sha256": _sha256(input_receipt_path)},
    "dataset": {"ref": CONTRACT["DATASET_REF"], "version": CONTRACT["DATASET_VERSION"]},
    "competition_root": str(COMPETITION_ROOT),
    "data_profile": DATA_PROFILE,
    "runtime": {
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "thread_limit": CONTRACT["THREAD_LIMIT"],
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        )},
        "internet_expected": False,
        "accelerator_expected": "CPU",
    },
}
PREFLIGHT_PATH.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(preflight, indent=2, sort_keys=True))
'''
    execution = r'''import datetime as _dt
import json
import traceback
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits
from rogii_validation.nonlinear_selector import run_e010

started_at = _dt.datetime.now(_dt.timezone.utc)
config_path = RUNTIME_ROOT / "experiments/E010/config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
try:
    with threadpool_limits(limits=CONTRACT["THREAD_LIMIT"]):
        active_pools_before = threadpool_info()
        if any(int(item.get("num_threads", 0) or 0) > int(CONTRACT["THREAD_LIMIT"]) for item in active_pools_before):
            raise RuntimeError({"thread_pool_limit_failed": active_pools_before})
        summary = run_e010(
            root=RUNTIME_ROOT,
            train_dir=TRAIN_DIR,
            output_dir=RESULT_DIR,
            artifact_dir=ARTIFACT_DIR,
            config=config,
            code_sha=CONTRACT["SOURCE_COMMIT"],
        )
        active_pools_after = threadpool_info()
        if any(int(item.get("num_threads", 0) or 0) > int(CONTRACT["THREAD_LIMIT"]) for item in active_pools_after):
            raise RuntimeError({"thread_pool_limit_failed_after": active_pools_after})
except Exception as exc:
    failure = {
        "schema_version": 1,
        "experiment_id": "E010",
        "status": "FAILED",
        "source_commit": CONTRACT["SOURCE_COMMIT"],
        "started_at_utc": started_at.isoformat(),
        "failed_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    (Path("/kaggle/working") / CONTRACT["RUN_RECEIPT_FILENAME"]).write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise
ended_at = _dt.datetime.now(_dt.timezone.utc)
print(json.dumps({
    "status": summary["status"],
    "reported_candidate": summary["reported_candidate"],
    "selected_candidate": summary["selected_candidate"],
    "bank_oracle_rmse": summary["bank_oracle_metrics"]["rmse"],
    "reported_rmse": summary["candidate_metrics"][summary["reported_candidate"]]["rmse"],
    "wall_seconds": (ended_at - started_at).total_seconds(),
}, indent=2, sort_keys=True))
'''
    packaging = r'''import hashlib
import json
import platform
import shutil
import zipfile
from pathlib import Path, PurePosixPath

WORKING = Path("/kaggle/working")


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
        raise RuntimeError(f"unsafe output member {name!r}")
    return value.as_posix()

output_files = []
for base, prefix in ((RESULT_DIR, "results"), (ARTIFACT_DIR, "artifacts")):
    for path in sorted(base.rglob("*")):
        if path.is_file():
            output_files.append({
                "path": f"{prefix}/{path.relative_to(base).as_posix()}",
                "source": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
if not output_files or not any(item["path"] == "results/summary.json" for item in output_files):
    raise RuntimeError("E010 result set is incomplete")

manifest = {
    "schema_version": 1,
    "experiment_id": "E010",
    "source_commit": CONTRACT["SOURCE_COMMIT"],
    "config_sha256": CONTRACT["CONFIG_SHA256"],
    "input_bundle_sha256": CONTRACT["BUNDLE_SHA256"],
    "files": [{key: item[key] for key in ("path", "bytes", "sha256")} for item in output_files],
}
manifest_path = WORKING / CONTRACT["OUTPUT_MANIFEST_FILENAME"]
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

archive_path = WORKING / CONTRACT["RESULT_ARCHIVE_FILENAME"]
temporary = archive_path.with_suffix(".zip.tmp")
temporary.unlink(missing_ok=True)
with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
    for item in output_files:
        info = zipfile.ZipInfo(_safe(item["path"]), date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        with archive.open(info, "w", force_zip64=True) as destination, Path(item["source"]).open("rb") as source:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    info = zipfile.ZipInfo(CONTRACT["OUTPUT_MANIFEST_FILENAME"], date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, manifest_path.read_bytes())
temporary.replace(archive_path)

receipt = {
    "schema_version": 1,
    "experiment_id": "E010",
    "status": "COMPLETE",
    "source_commit": CONTRACT["SOURCE_COMMIT"],
    "config_sha256": CONTRACT["CONFIG_SHA256"],
    "input_bundle": {"sha256": CONTRACT["BUNDLE_SHA256"], "bytes": CONTRACT["BUNDLE_BYTES"]},
    "dataset": {"ref": CONTRACT["DATASET_REF"], "version": CONTRACT["DATASET_VERSION"]},
    "started_at_utc": started_at.isoformat(),
    "ended_at_utc": ended_at.isoformat(),
    "wall_seconds": (ended_at - started_at).total_seconds(),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__,
    "thread_limit": CONTRACT["THREAD_LIMIT"],
    "thread_pools_before": active_pools_before,
    "thread_pools_after": active_pools_after,
    "internet_expected": False,
    "accelerator_expected": "CPU",
    "summary": {
        "status": summary["status"],
        "selected_candidate": summary["selected_candidate"],
        "reported_candidate": summary["reported_candidate"],
        "eligible_candidates": summary["eligible_candidates"],
        "candidate_count": summary["candidate_count"],
        "advanced_families": summary["advanced_families"],
        "last_known_rmse": summary["baseline_metrics"]["rmse"],
        "e006_rmse": summary["e006_metrics"]["rmse"],
        "bank_oracle_rmse": summary["bank_oracle_metrics"]["rmse"],
        "reported_rmse": summary["candidate_metrics"][summary["reported_candidate"]]["rmse"],
        "all_controls_pass": all(bool(item["pass"]) for item in summary["controls"].values()),
        "runtime": summary["runtime"],
    },
    "output_manifest": {"filename": manifest_path.name, "bytes": manifest_path.stat().st_size, "sha256": _sha256(manifest_path)},
    "result_archive": {"filename": archive_path.name, "bytes": archive_path.stat().st_size, "sha256": _sha256(archive_path)},
    "submission_created": False,
    "submission_made": False,
}
receipt_path = WORKING / CONTRACT["RUN_RECEIPT_FILENAME"]
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2, sort_keys=True))
'''
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "e010-intro",
                "metadata": {},
                "source": [
                    "# E010 nonlinear candidate coverage and selector regret\n",
                    "Private, internet-disabled, CPU-only full validation. Attach the exact sealed E010 input dataset version and the ROGII competition data. The notebook caps all native CPU pools at two threads, imports the exact bundled `src/` package, and fails closed on any identity or control mismatch.\n",
                ],
            },
            {"cell_type": "code", "id": "e010-contract", "execution_count": None, "metadata": {}, "outputs": [], "source": constants_source.splitlines(keepends=True)},
            {"cell_type": "code", "id": "e010-preflight", "execution_count": None, "metadata": {}, "outputs": [], "source": preflight.splitlines(keepends=True)},
            {"cell_type": "code", "id": "e010-run", "execution_count": None, "metadata": {}, "outputs": [], "source": execution.splitlines(keepends=True)},
            {"cell_type": "code", "id": "e010-package", "execution_count": None, "metadata": {}, "outputs": [], "source": packaging.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "rogii": {
                "experiment_id": "E010",
                "source_commit": source_commit,
                "config_sha256": config_sha,
                "bundle_sha256": bundle_sha,
                "dataset_ref": dataset_ref,
                "dataset_version": dataset_version,
                "internet_required": False,
                "accelerator": "CPU",
                "maximum_threads": THREAD_LIMIT,
                "expected_outputs": [RUN_RECEIPT_FILENAME, OUTPUT_MANIFEST_FILENAME, RESULT_ARCHIVE_FILENAME],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))


def build(*, root: Path, staging_dir: Path, notebook_path: Path, source_commit: str,
          dataset_ref: str, dataset_version: int) -> dict[str, Any]:
    config_path = root / "experiments/E010/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = source_paths(root, config)
    entries, committed_payloads = verify_inputs(root, config, source_commit, paths)
    config_sha = sha256_file(config_path)
    bundle_manifest = {
        "schema_version": 1,
        "experiment_id": "E010",
        "source_commit": source_commit,
        "config_sha256": config_sha,
        "data_signature": config["data_signature"],
        "expected_wells": config["expected_wells"],
        "maximum_threads": THREAD_LIMIT,
        "files": entries,
    }
    staging_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = staging_dir / BUNDLE_FILENAME
    write_deterministic_bundle(root, bundle_path, bundle_manifest, committed_payloads)
    bundle_sha = sha256_file(bundle_path)
    input_receipt = {
        "schema_version": 1,
        "experiment_id": "E010",
        "role": "candidate_selector_full_validation_inputs",
        "dataset": {"ref": dataset_ref, "version": dataset_version, "private": True},
        "source_commit": source_commit,
        "config_sha256": config_sha,
        "bundle": {"filename": BUNDLE_FILENAME, "bytes": bundle_path.stat().st_size, "sha256": bundle_sha},
        "data_signature": config["data_signature"],
        "expected_wells": config["expected_wells"],
        "maximum_threads": THREAD_LIMIT,
        "competition_data_included": False,
        "competition_data_attachment_required": True,
        "submission_authorized": False,
    }
    receipt_path = staging_dir / INPUT_RECEIPT_FILENAME
    receipt_path.write_bytes(canonical_json(input_receipt))
    metadata = {
        "title": "ROGII E010 Candidate Selector Inputs",
        "id": dataset_ref,
        "licenses": [{"name": "other"}],
    }
    (staging_dir / "dataset-metadata.json").write_bytes(canonical_json(metadata))
    notebook = notebook_payload(
        source_commit=source_commit,
        config_sha=config_sha,
        bundle_sha=bundle_sha,
        bundle_bytes=bundle_path.stat().st_size,
        receipt_sha=sha256_file(receipt_path),
        dataset_ref=dataset_ref,
        dataset_version=dataset_version,
        data_signature=str(config["data_signature"]),
        expected_wells=int(config["expected_wells"]),
    )
    write_notebook(notebook_path, notebook)
    return {
        "source_commit": source_commit,
        "config_sha256": config_sha,
        "dataset_ref": dataset_ref,
        "dataset_version": dataset_version,
        "staging_dir": str(staging_dir),
        "bundle": {"path": str(bundle_path), "bytes": bundle_path.stat().st_size, "sha256": bundle_sha, "files": len(entries)},
        "input_receipt": {"path": str(receipt_path), "bytes": receipt_path.stat().st_size, "sha256": sha256_file(receipt_path)},
        "notebook": {"path": str(notebook_path), "bytes": notebook_path.stat().st_size, "sha256": sha256_file(notebook_path)},
        "required_attachments": [f"private dataset {dataset_ref} version {dataset_version}", "ROGII competition data"],
        "runtime": {"private": True, "internet": False, "accelerator": "CPU", "maximum_threads": THREAD_LIMIT},
        "expected_outputs": [RUN_RECEIPT_FILENAME, OUTPUT_MANIFEST_FILENAME, RESULT_ARCHIVE_FILENAME],
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=ROOT)
    command.add_argument("--staging-dir", type=Path, default=Path("scratch/agents/e010-kaggle-fix/T017/input-v2"))
    command.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    command.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    command.add_argument("--dataset-ref", default=DEFAULT_DATASET_REF)
    command.add_argument("--dataset-version", type=int, default=DEFAULT_DATASET_VERSION)
    return command


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    root = arguments.root.resolve()
    result = build(
        root=root,
        staging_dir=resolve(root, arguments.staging_dir),
        notebook_path=resolve(root, arguments.notebook),
        source_commit=str(arguments.source_commit),
        dataset_ref=str(arguments.dataset_ref),
        dataset_version=int(arguments.dataset_version),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
