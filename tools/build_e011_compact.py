#!/usr/bin/env python3
"""Build deterministic compact E011 sufficient statistics from verified E010 evidence."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.harness import _canonical_json, _sha256  # noqa: E402

DEFAULT_OUTPUT = Path("artifacts/E011/compact_stats_v1.npz")
DEFAULT_RECEIPT = Path("experiments/E011/COMPACT_INPUT_V1.json")
REPRESENTATIONS = ("shape3", "spline3", "spline4", "spline5", "spline7")
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if not arrays or any(not name or "/" in name or "\\" in name for name in arrays):
        raise ValueError("E011 compact array names are invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    members = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            payload = _npy_bytes(np.asarray(arrays[name]))
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
            members.append({
                "name": name,
                "member": info.filename,
                "dtype": str(np.asarray(arrays[name]).dtype),
                "shape": list(np.asarray(arrays[name]).shape),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    temporary.replace(path)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path), "members": members}


def _basis_columns(position: np.ndarray, name: str, config: Mapping[str, Any]) -> np.ndarray:
    s = np.asarray(position, dtype=np.float64)
    if s.ndim != 1 or not np.all(np.isfinite(s)) or np.any(s < -1e-12) or np.any(s > 1.0 + 1e-12):
        raise ValueError("E011 compact normalized positions are invalid")
    s = np.clip(s, 0.0, 1.0)
    spec = config["representations"][name]
    dimensions = int(spec["dimensions"])
    if name == "shape3":
        output = np.column_stack([s, 4.0 * s * (1.0 - s), 16.0 * s * (1.0 - s) * (s - 0.5)])
    elif name.startswith("spline"):
        knots = np.asarray(spec["knots"], dtype=np.float64)
        if knots.shape != (dimensions,) or not np.all(np.diff(knots) > 0.0) or abs(float(knots[-1]) - 1.0) > 1e-12:
            raise ValueError(f"E011 compact knot contract differs for {name}")
        output = np.zeros((s.size, dimensions), dtype=np.float64)
        for index, current in enumerate(knots):
            previous = 0.0 if index == 0 else float(knots[index - 1])
            rising = np.clip((s - previous) / max(float(current) - previous, 1e-15), 0.0, 1.0)
            if index + 1 < dimensions:
                following = float(knots[index + 1])
                falling = np.clip((following - s) / max(following - float(current), 1e-15), 0.0, 1.0)
                output[:, index] = np.minimum(rising, falling)
            else:
                output[:, index] = rising
    else:
        raise ValueError(f"E011 compact unsupported representation {name}")
    if output.shape != (s.size, dimensions) or not np.all(np.isfinite(output)):
        raise ValueError(f"E011 compact basis shape differs for {name}")
    return output


def _bincount(codes: np.ndarray, weights: np.ndarray, wells: int) -> np.ndarray:
    return np.bincount(codes, weights=np.asarray(weights, dtype=np.float64), minlength=wells).astype(np.float64, copy=False)


def _assignment_arrays(selector_path: Path, well_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(selector_path, usecols=["context", "scope", "branch", "well_id"], dtype="string")
    frame = frame[frame["scope"].isin(["spatial", "typewell"])].copy()
    if frame.empty:
        raise ValueError("E011 compact selector assignment rows are missing")
    frame["outer_group"] = frame["context"].str.rsplit(":", n=1).str[-1].astype(np.int16)
    consistency = frame.groupby(["scope", "well_id"], sort=True)["outer_group"].nunique()
    if int(consistency.max()) != 1:
        raise ValueError("E011 compact evaluator assignments differ across E010 branches")
    deduplicated = frame.drop_duplicates(["scope", "well_id", "outer_group"])
    mappings: dict[str, dict[str, int]] = {}
    for scope in ("spatial", "typewell"):
        part = deduplicated[deduplicated["scope"] == scope]
        mapping = {str(row.well_id): int(row.outer_group) for row in part.itertuples(index=False)}
        if set(mapping) != set(well_ids) or set(mapping.values()) != set(range(5)):
            raise ValueError(f"E011 compact {scope} assignment coverage differs")
        mappings[scope] = mapping
    spatial = np.asarray([mappings["spatial"][well_id] for well_id in well_ids], dtype=np.int8)
    typewell = np.asarray([mappings["typewell"][well_id] for well_id in well_ids], dtype=np.int8)
    return spatial, typewell, {
        "source_rows": int(frame.shape[0]),
        "deduplicated_rows": int(deduplicated.shape[0]),
        "branches": sorted(str(value) for value in frame["branch"].dropna().unique()),
        "spatial_counts": {str(group): int(np.sum(spatial == group)) for group in range(5)},
        "typewell_counts": {str(group): int(np.sum(typewell == group)) for group in range(5)},
    }


def _source_identity(root: Path, config: Mapping[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    item = config["parent_artifacts"][key]
    path = root / str(item["path"])
    if not path.is_file() or path.stat().st_size != int(item["bytes"]) or _sha256(path) != str(item["sha256"]):
        raise ValueError(f"E011 compact source identity differs: {key}")
    return path, {"path": str(item["path"]), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def build_compact(*, root: Path, config_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    output_path = output_path if output_path.is_absolute() else root / output_path
    receipt_path = receipt_path if receipt_path.is_absolute() else root / receipt_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "E011":
        raise ValueError("E011 compact config identity differs")
    legal_path, legal_identity = _source_identity(root, config, "e008_legal_features")
    oof_path, oof_identity = _source_identity(root, config, "e010_oof")
    selector_path, selector_identity = _source_identity(root, config, "e010_selector_predictions")

    legal = pd.read_csv(legal_path, dtype={"well_id": "string"})
    if "well_id" not in legal or legal["well_id"].duplicated().any():
        raise ValueError("E011 compact legal feature well IDs differ")
    legal = legal.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    well_ids = [str(value) for value in legal["well_id"]]
    if len(well_ids) != int(config["expected_wells"]):
        raise ValueError("E011 compact legal feature well count differs")
    feature_names = [name for name in legal.columns if name != "well_id"]
    if len(feature_names) + 1 != int(config["legal_feature_contract"]["expected_columns"]):
        raise ValueError("E011 compact legal feature column count differs")
    features = legal[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    hidden_rows_feature = pd.to_numeric(legal["hidden_rows"], errors="raise").to_numpy(dtype=np.int32)

    category = pd.CategoricalDtype(categories=well_ids, ordered=True)
    frame = pd.read_csv(
        oof_path,
        usecols=["well_id", "hidden_index", "target", "last_known_tvt", "e006_nested_fusion"],
        dtype={"well_id": category, "hidden_index": "int32", "target": "float64", "last_known_tvt": "float64", "e006_nested_fusion": "float64"},
    )
    codes = frame["well_id"].cat.codes.to_numpy(dtype=np.int32, copy=False)
    if np.any(codes < 0):
        raise ValueError("E011 compact OOF contains unknown wells")
    hidden_index = frame["hidden_index"].to_numpy(dtype=np.int32, copy=False)
    target = frame["target"].to_numpy(dtype=np.float64, copy=False)
    e006 = frame["e006_nested_fusion"].to_numpy(dtype=np.float64, copy=False)
    last = frame["last_known_tvt"].to_numpy(dtype=np.float64, copy=False)
    if frame.shape[0] != int(config["expected_hidden_rows"]) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(e006)) or not np.all(np.isfinite(last)):
        raise ValueError("E011 compact OOF row/finite contract differs")
    wells = len(well_ids)
    rows = np.bincount(codes, minlength=wells).astype(np.int32)
    if not np.array_equal(rows, hidden_rows_feature):
        raise ValueError("E011 compact OOF row counts differ from legal features")
    maximum_index = np.full(wells, -1, dtype=np.int32)
    np.maximum.at(maximum_index, codes, hidden_index)
    if not np.array_equal(maximum_index, rows - 1) or int(hidden_index.min()) != 0:
        raise ValueError("E011 compact hidden indices are not contiguous")
    denominator = np.maximum(rows[codes] - 1, 1).astype(np.float64)
    position = hidden_index.astype(np.float64) / denominator
    x = hidden_index.astype(np.float64)
    base_error = e006 - target
    last_error = last - target

    arrays: dict[str, np.ndarray] = {
        "well_ids": np.asarray(well_ids, dtype="U8"),
        "feature_names": np.asarray(feature_names, dtype=f"U{max(len(name) for name in feature_names)}"),
        "features": features,
        "rows": rows,
        "sum_x": _bincount(codes, x, wells),
        "sum_x_sq": _bincount(codes, x * x, wells),
        "base_sum": _bincount(codes, base_error, wells),
        "base_sse": _bincount(codes, base_error * base_error, wells),
        "base_x_sum": _bincount(codes, x * base_error, wells),
        "last_sum": _bincount(codes, last_error, wells),
        "last_sse": _bincount(codes, last_error * last_error, wells),
        "last_x_sum": _bincount(codes, x * last_error, wells),
    }
    representation_metrics: dict[str, Any] = {}
    total_rows = int(rows.sum())
    for name in REPRESENTATIONS:
        basis = _basis_columns(position, name, config)
        dimensions = basis.shape[1]
        basis_sum = np.column_stack([_bincount(codes, basis[:, column], wells) for column in range(dimensions)])
        basis_x_sum = np.column_stack([_bincount(codes, x * basis[:, column], wells) for column in range(dimensions)])
        basis_base = np.column_stack([_bincount(codes, base_error * basis[:, column], wells) for column in range(dimensions)])
        cross = np.empty((wells, dimensions, dimensions), dtype=np.float64)
        for left in range(dimensions):
            for right in range(left, dimensions):
                values = _bincount(codes, basis[:, left] * basis[:, right], wells)
                cross[:, left, right] = values
                cross[:, right, left] = values
        target_coefficients = np.einsum("wij,wj->wi", np.linalg.pinv(cross, rcond=1e-12), -basis_base)
        if not np.all(np.isfinite(target_coefficients)):
            raise ValueError(f"E011 compact target coefficients are non-finite for {name}")
        oracle_sse = arrays["base_sse"] + 2.0 * np.einsum("wi,wi->w", basis_base, target_coefficients) + np.einsum("wi,wij,wj->w", target_coefficients, cross, target_coefficients)
        oracle_sse = np.maximum(oracle_sse, 0.0)
        oracle_rmse = math.sqrt(float(oracle_sse.sum()) / total_rows)
        expected = float(config["representations"][name]["oracle_rmse"])
        if abs(oracle_rmse - expected) > 1e-8:
            raise ValueError(f"E011 compact {name} oracle RMSE differs: {oracle_rmse} vs {expected}")
        arrays[f"{name}__basis_sum"] = basis_sum
        arrays[f"{name}__basis_x_sum"] = basis_x_sum
        arrays[f"{name}__basis_base"] = basis_base
        arrays[f"{name}__basis_cross"] = cross
        arrays[f"{name}__target_coefficients"] = target_coefficients
        representation_metrics[name] = {"dimensions": dimensions, "oracle_rmse": oracle_rmse, "oracle_sse": float(oracle_sse.sum())}

    spatial, typewell, assignment_audit = _assignment_arrays(selector_path, well_ids)
    arrays["spatial_assignment"] = spatial
    arrays["typewell_assignment"] = typewell

    control_indices = np.asarray([0, wells // 2, wells - 1], dtype=np.int32)
    control_mask = np.isin(codes, control_indices)
    arrays["control_well_indices"] = control_indices
    arrays["control_codes"] = codes[control_mask]
    arrays["control_hidden_index"] = hidden_index[control_mask]
    arrays["control_target"] = target[control_mask]
    arrays["control_e006"] = e006[control_mask]

    source_manifest = {
        "schema_version": 1,
        "experiment_id": "E011",
        "role": "compact_sufficient_statistics",
        "expected_wells": wells,
        "expected_hidden_rows": total_rows,
        "representations": representation_metrics,
        "sources": {"e008_legal_features": legal_identity, "e010_oof": oof_identity, "e010_selector_predictions": selector_identity},
        "assignment_audit": assignment_audit,
        "control_well_ids": [well_ids[index] for index in control_indices],
        "builder": {"path": "tools/build_e011_compact.py", "pandas": pd.__version__, "numpy": np.__version__, "maximum_threads": 2},
    }
    manifest_payload = (_canonical_json(source_manifest) + "\n").encode("utf-8")
    arrays["manifest_json"] = np.frombuffer(manifest_payload, dtype=np.uint8).copy()
    artifact = write_deterministic_npz(output_path, arrays)
    artifact["path"] = output_path.relative_to(root).as_posix()
    receipt = {
        "schema_version": 1,
        "experiment_id": "E011",
        "record_type": "compact_sufficient_statistics",
        "status": "ready",
        "built_at": "2026-07-20T13:00:00Z",
        "artifact": artifact,
        "source_manifest": source_manifest,
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256_at_build": _sha256(config_path),
        "submission_authorized": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=ROOT)
    command.add_argument("--config", type=Path, default=Path("experiments/E011/config.json"))
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "2"
    arguments = parser().parse_args(argv)
    result = build_compact(root=arguments.root, config_path=arguments.config, output_path=arguments.output, receipt_path=arguments.receipt)
    print(json.dumps({"status": result["status"], "artifact": result["artifact"], "control_well_ids": result["source_manifest"]["control_well_ids"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
