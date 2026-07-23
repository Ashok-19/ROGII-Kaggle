#!/usr/bin/env python3
"""T025: preregistered multi-cut spline coefficient-dynamics worth screen."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 4 else Path.cwd()
CUTS = (0.40, 0.55, 0.70, 0.85)
KNOTS = (0.25, 0.50, 0.75, 1.00)
OUTPUTS = 4
CONTROL_MODES = ("permuted_wells", "shuffled_targets", "reversed_cuts")


class DataValidationError(ValueError):
    """Raised when a scientific or data contract is violated."""


@dataclass(frozen=True)
class Context:
    key: str
    scope: str
    label: str
    outer_group: int
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class WellRecord:
    well_id: str
    known_rows: int
    hidden_rows: int
    last_tvt: float
    hidden_gr_missing_fraction: float
    pseudo_coefficients: np.ndarray
    pseudo_rmse: np.ndarray
    pseudo_segment_rows: np.ndarray
    pseudo_features: np.ndarray
    reversed_features: np.ndarray
    latest_coefficients: np.ndarray
    linear_extrapolation: np.ndarray
    target_coefficients: np.ndarray
    raw_hidden_sum: float
    raw_hidden_sum_sq: float
    # Filled from the exact E011 OOF stream.
    e011_sse: float | None = None
    e011_sum: float | None = None
    e_dot_d: float | None = None
    d_sse: float | None = None
    d_sum: float | None = None
    basis_dot_e: np.ndarray | None = None
    basis_dot_d: np.ndarray | None = None
    basis_sum: np.ndarray | None = None
    basis_cross: np.ndarray | None = None

    def metric(self, coefficients: Sequence[float], weight: float) -> dict[str, Any]:
        c = np.asarray(coefficients, dtype=np.float64)
        if c.shape != (OUTPUTS,) or not np.all(np.isfinite(c)):
            raise DataValidationError(f"{self.well_id}: invalid coefficient vector")
        w = float(weight)
        if not math.isfinite(w) or w < -1e-12 or w > 1.0 + 1e-12:
            raise DataValidationError(f"{self.well_id}: invalid placement weight")
        if any(value is None for value in (self.e011_sse, self.e011_sum, self.e_dot_d, self.d_sse, self.d_sum, self.basis_dot_e, self.basis_dot_d, self.basis_sum, self.basis_cross)):
            raise DataValidationError(f"{self.well_id}: OOF sufficient statistics are incomplete")
        e011_sse = float(self.e011_sse)
        linear = float(self.e_dot_d) + float(np.dot(self.basis_dot_e, c))
        quadratic = float(self.d_sse) + 2.0 * float(np.dot(self.basis_dot_d, c)) + float(c @ self.basis_cross @ c)
        sse = e011_sse + 2.0 * w * linear + w * w * quadratic
        tolerance = 1e-8 * max(1.0, e011_sse, abs(2.0 * w * linear), abs(w * w * quadratic))
        if sse < -tolerance:
            raise DataValidationError(f"{self.well_id}: materially negative candidate SSE")
        sse = max(0.0, sse)
        error_sum = float(self.e011_sum) + w * (float(self.d_sum) + float(np.dot(self.basis_sum, c)))
        return {
            "well_id": self.well_id,
            "rows_scored": self.hidden_rows,
            "sse": sse,
            "rmse": math.sqrt(sse / self.hidden_rows),
            "mean_error": error_sum / self.hidden_rows,
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise DataValidationError(f"refusing to write empty CSV {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise DataValidationError(f"{path.name}: inconsistent row schema")
            writer.writerow(row)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DataValidationError("cannot compute quantile of empty values")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(metrics)
    if not rows:
        raise DataValidationError("cannot summarize empty metrics")
    total_rows = sum(int(row["rows_scored"]) for row in rows)
    total_sse = sum(float(row["sse"]) for row in rows)
    if total_rows <= 0 or total_sse < 0.0:
        raise DataValidationError("invalid aggregate metric totals")
    rmses = [float(row["rmse"]) for row in rows]
    ordered = sorted(rows, key=lambda row: (-float(row["sse"]), str(row["well_id"])))
    worst5 = max(1, math.ceil(0.05 * len(rows)))
    worst10 = max(1, math.ceil(0.10 * len(rows)))
    return {
        "rows_scored": total_rows,
        "wells_scored": len(rows),
        "sse": total_sse,
        "rmse": math.sqrt(total_sse / total_rows),
        "median_well_rmse": _quantile(rmses, 0.5),
        "p90_well_rmse": _quantile(rmses, 0.9),
        "p95_well_rmse": _quantile(rmses, 0.95),
        "max_well_rmse": max(rmses),
        "worst_5pct_sse_share": sum(float(row["sse"]) for row in ordered[:worst5]) / total_sse if total_sse else 0.0,
        "worst_10pct_sse_share": sum(float(row["sse"]) for row in ordered[:worst10]) / total_sse if total_sse else 0.0,
    }


def spline_basis(rows: int) -> np.ndarray:
    count = int(rows)
    if count <= 0:
        raise DataValidationError("spline basis requires positive rows")
    positions = np.arange(count, dtype=np.float64) / max(1, count - 1)
    grid = np.asarray((0.0, *KNOTS), dtype=np.float64)
    matrix = np.empty((count, OUTPUTS), dtype=np.float64)
    for index in range(OUTPUTS):
        controls = np.zeros(OUTPUTS + 1, dtype=np.float64)
        controls[index + 1] = 1.0
        matrix[:, index] = np.interp(positions, grid, controls)
    if matrix.shape != (count, OUTPUTS) or not np.all(np.isfinite(matrix)):
        raise DataValidationError("invalid spline basis")
    return matrix


def _solve_coefficients(matrix: np.ndarray, delta: np.ndarray, bound: float) -> tuple[np.ndarray, float]:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(delta, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != OUTPUTS or y.shape != (x.shape[0],):
        raise DataValidationError("coefficient solve dimensions differ")
    if x.shape[0] < OUTPUTS or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise DataValidationError("coefficient solve has insufficient or non-finite data")
    if np.linalg.matrix_rank(x) < OUTPUTS:
        raise DataValidationError("coefficient solve is rank deficient")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    coefficients = np.clip(coefficients, -float(bound), float(bound))
    residual = x @ coefficients - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    if coefficients.shape != (OUTPUTS,) or not np.all(np.isfinite(coefficients)) or not math.isfinite(rmse):
        raise DataValidationError("coefficient solve emitted non-finite values")
    return coefficients, rmse


def fit_segment(truth: Sequence[float], baseline: float, minimum_rows: int, bound: float) -> tuple[np.ndarray, float]:
    values = np.asarray(truth, dtype=np.float64)
    if values.ndim != 1 or len(values) < int(minimum_rows):
        raise DataValidationError("pseudo segment has insufficient support")
    if not math.isfinite(float(baseline)) or not np.all(np.isfinite(values)):
        raise DataValidationError("pseudo segment contains non-finite values")
    matrix = spline_basis(len(values))
    return _solve_coefficients(matrix, values - float(baseline), bound)


def validate_prefix(values: Sequence[float]) -> int:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any() or finite.all():
        raise DataValidationError("TVT_input must contain a known prefix and hidden suffix")
    first_hidden = int(np.flatnonzero(~finite)[0])
    if first_hidden <= 0 or finite[first_hidden:].any() or not finite[:first_hidden].all():
        raise DataValidationError("TVT_input is not a contiguous known prefix")
    return first_hidden


def build_pseudo_features(coefficients: np.ndarray, rmses: np.ndarray, segment_rows: np.ndarray, known_rows: int, hidden_rows: int) -> np.ndarray:
    c = np.asarray(coefficients, dtype=np.float64)
    r = np.asarray(rmses, dtype=np.float64)
    seg = np.asarray(segment_rows, dtype=np.float64)
    if c.shape != (len(CUTS), OUTPUTS) or r.shape != (len(CUTS),) or seg.shape != (len(CUTS),):
        raise DataValidationError("pseudo feature dimensions differ")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(seg)):
        raise DataValidationError("pseudo features contain non-finite values")
    deltas = np.diff(c, axis=0).reshape(-1)
    trends: list[float] = []
    x = np.asarray(CUTS, dtype=np.float64)
    for output in range(OUTPUTS):
        slope, intercept = np.polyfit(x, c[:, output], 1)
        trends.extend((float(slope), float(intercept + slope)))
    metadata = np.asarray(
        [
            float(known_rows),
            float(hidden_rows),
            float(hidden_rows) / max(1.0, float(known_rows)),
            *(seg / max(1.0, float(known_rows))),
        ],
        dtype=np.float64,
    )
    result = np.concatenate((c.reshape(-1), r, deltas, np.asarray(trends), metadata))
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise DataValidationError("constructed pseudo feature vector is invalid")
    return result


def linear_extrapolation(coefficients: np.ndarray, bound: float) -> np.ndarray:
    c = np.asarray(coefficients, dtype=np.float64)
    if c.shape != (len(CUTS), OUTPUTS) or not np.all(np.isfinite(c)):
        raise DataValidationError("invalid pseudo coefficients for extrapolation")
    x = np.asarray(CUTS, dtype=np.float64)
    output = np.empty(OUTPUTS, dtype=np.float64)
    for index in range(OUTPUTS):
        slope, intercept = np.polyfit(x, c[:, index], 1)
        output[index] = intercept + slope
    return np.clip(output, -bound, bound)


def load_raw_records(data_dir: Path, config: Mapping[str, Any]) -> list[WellRecord]:
    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if len(paths) != int(config["expected_wells"]):
        raise DataValidationError(f"expected {config['expected_wells']} training wells, found {len(paths)}")
    records: list[WellRecord] = []
    seen: set[str] = set()
    minimum = int(config["minimum_pseudo_segment_rows"])
    bound = float(config["coefficient_absolute_bound_ft"])
    cuts = tuple(float(value) for value in config["pseudo_cut_fractions"])
    if cuts != CUTS:
        raise DataValidationError("implementation cut grid differs from preregistration")
    for path in paths:
        well_id = path.name.split("__", 1)[0]
        if well_id in seen:
            raise DataValidationError(f"duplicate training well {well_id}")
        seen.add(well_id)
        frame = pd.read_csv(path, usecols=lambda name: name in {"TVT", "TVT_input", "GR"})
        if set(frame.columns) != {"TVT", "TVT_input", "GR"}:
            raise DataValidationError(f"{well_id}: required raw columns differ")
        truth = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(dtype=np.float64)
        tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(dtype=np.float64)
        gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(truth)):
            raise DataValidationError(f"{well_id}: truth contains non-finite values")
        known_rows = validate_prefix(tvt_input)
        known_values = tvt_input[:known_rows]
        hidden = truth[known_rows:]
        if len(hidden) <= 0:
            raise DataValidationError(f"{well_id}: hidden suffix is empty")
        if not np.all(np.isfinite(known_values)) or np.max(np.abs(known_values - truth[:known_rows])) > 1e-8:
            raise DataValidationError(f"{well_id}: known TVT_input differs from training truth")
        pseudo_coefficients: list[np.ndarray] = []
        pseudo_rmse: list[float] = []
        pseudo_rows: list[int] = []
        for fraction in cuts:
            cut_index = int(math.floor(fraction * known_rows))
            if cut_index <= 0 or cut_index >= known_rows:
                raise DataValidationError(f"{well_id}: pseudo cut is outside known prefix")
            segment = known_values[cut_index:known_rows]
            coefficients, rmse = fit_segment(segment, known_values[cut_index - 1], minimum, bound)
            pseudo_coefficients.append(coefficients)
            pseudo_rmse.append(rmse)
            pseudo_rows.append(len(segment))
        pseudo_array = np.vstack(pseudo_coefficients)
        rmse_array = np.asarray(pseudo_rmse, dtype=np.float64)
        segment_array = np.asarray(pseudo_rows, dtype=np.float64)
        final_coefficients, _final_rmse = fit_segment(hidden, known_values[-1], min(OUTPUTS, len(hidden)), bound)
        pseudo_features = build_pseudo_features(pseudo_array, rmse_array, segment_array, known_rows, len(hidden))
        reversed_features = build_pseudo_features(pseudo_array[::-1], rmse_array[::-1], segment_array[::-1], known_rows, len(hidden))
        hidden_gr = gr[known_rows:]
        missing = float(np.mean(~np.isfinite(hidden_gr))) if len(hidden_gr) else 1.0
        records.append(
            WellRecord(
                well_id=well_id,
                known_rows=known_rows,
                hidden_rows=len(hidden),
                last_tvt=float(known_values[-1]),
                hidden_gr_missing_fraction=missing,
                pseudo_coefficients=pseudo_array,
                pseudo_rmse=rmse_array,
                pseudo_segment_rows=segment_array,
                pseudo_features=pseudo_features,
                reversed_features=reversed_features,
                latest_coefficients=pseudo_array[-1].copy(),
                linear_extrapolation=linear_extrapolation(pseudo_array, bound),
                target_coefficients=final_coefficients,
                raw_hidden_sum=float(hidden.sum()),
                raw_hidden_sum_sq=float(np.dot(hidden, hidden)),
            )
        )
    return records


def attach_e011_sufficient(records: Sequence[WellRecord], oof_path: Path, expected_rows: int) -> dict[str, Any]:
    by_id = {record.well_id: record for record in records}
    if len(by_id) != len(records):
        raise DataValidationError("record well IDs are not unique")
    current = ""
    targets: list[float] = []
    e011_values: list[float] = []
    seen_ids: set[str] = set()
    seen_wells: set[str] = set()
    expected_hidden = 0
    total_rows = 0

    def finalize_group(well_id: str) -> None:
        if not well_id:
            return
        record = by_id.get(well_id)
        if record is None:
            raise DataValidationError(f"OOF contains unknown well {well_id}")
        y = np.asarray(targets, dtype=np.float64)
        pred = np.asarray(e011_values, dtype=np.float64)
        if len(y) != record.hidden_rows or pred.shape != y.shape:
            raise DataValidationError(f"{well_id}: OOF hidden length differs")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
            raise DataValidationError(f"{well_id}: OOF values are non-finite")
        sum_tolerance = 1e-6 * max(1.0, abs(record.raw_hidden_sum))
        square_tolerance = 1e-6 * max(1.0, abs(record.raw_hidden_sum_sq))
        if abs(float(y.sum()) - record.raw_hidden_sum) > sum_tolerance or abs(float(np.dot(y, y)) - record.raw_hidden_sum_sq) > square_tolerance:
            raise DataValidationError(f"{well_id}: OOF target differs from raw hidden truth")
        basis = spline_basis(record.hidden_rows)
        error = pred - y
        delta = record.last_tvt - pred
        record.e011_sse = float(np.dot(error, error))
        record.e011_sum = float(error.sum())
        record.e_dot_d = float(np.dot(error, delta))
        record.d_sse = float(np.dot(delta, delta))
        record.d_sum = float(delta.sum())
        record.basis_dot_e = basis.T @ error
        record.basis_dot_d = basis.T @ delta
        record.basis_sum = basis.sum(axis=0)
        record.basis_cross = basis.T @ basis
        seen_wells.add(well_id)

    with gzip.open(oof_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "well_id", "row_index", "hidden_index", "target", "spline4_ridge_equal_s075"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise DataValidationError("E011 OOF schema differs")
        for row in reader:
            well_id = str(row["well_id"])
            hidden_index = int(row["hidden_index"])
            row_index = int(row["row_index"])
            identifier = str(row["id"])
            if identifier in seen_ids:
                raise DataValidationError(f"duplicate OOF ID {identifier}")
            seen_ids.add(identifier)
            if well_id != current:
                finalize_group(current)
                if well_id in seen_wells:
                    raise DataValidationError(f"OOF well {well_id} is noncontiguous")
                current = well_id
                targets.clear()
                e011_values.clear()
                expected_hidden = 0
            if hidden_index != expected_hidden:
                raise DataValidationError(f"{well_id}: OOF hidden index is noncontiguous")
            record = by_id.get(well_id)
            if record is None or row_index != record.known_rows + hidden_index or identifier != f"{well_id}_{row_index}":
                raise DataValidationError(f"{well_id}: OOF ID/order differs from raw data")
            targets.append(float(row["target"]))
            e011_values.append(float(row["spline4_ridge_equal_s075"]))
            expected_hidden += 1
            total_rows += 1
    finalize_group(current)
    if total_rows != int(expected_rows) or len(seen_ids) != total_rows or len(seen_wells) != len(records):
        raise DataValidationError("OOF coverage is incomplete")
    return {"rows": total_rows, "wells": len(seen_wells), "unique_ids": len(seen_ids)}


def load_compact(path: Path, records: Sequence[WellRecord]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    required = {"well_ids", "feature_names", "features", "spatial_assignment", "typewell_assignment"}
    if not required.issubset(arrays.files):
        raise DataValidationError("E011 compact artifact schema differs")
    compact_ids = [str(value) for value in arrays["well_ids"]]
    record_ids = [record.well_id for record in records]
    if compact_ids != record_ids:
        raise DataValidationError("compact well order differs from raw well order")
    features = np.asarray(arrays["features"], dtype=np.float64)
    names = [str(value) for value in arrays["feature_names"]]
    spatial = np.asarray(arrays["spatial_assignment"], dtype=np.int64)
    typewell = np.asarray(arrays["typewell_assignment"], dtype=np.int64)
    if features.shape != (len(records), len(names)) or spatial.shape != (len(records),) or typewell.shape != (len(records),):
        raise DataValidationError("compact feature/group dimensions differ")
    if set(spatial.tolist()) != set(range(5)) or set(typewell.tolist()) != set(range(5)):
        raise DataValidationError("stress group labels differ from frozen five-group contract")
    return features, names, spatial, typewell


def load_contexts(root: Path, well_ids: Sequence[str], fold_files: Sequence[str], spatial: np.ndarray, typewell: np.ndarray) -> tuple[list[Context], list[Context]]:
    id_to_index = {well_id: index for index, well_id in enumerate(well_ids)}
    repeated: list[Context] = []
    for file_name in fold_files:
        payload = json.loads((root / file_name).read_text(encoding="utf-8"))
        version = str(payload["version"])
        assignments = payload["assignments"]
        if set(assignments) != set(well_ids) or int(payload["n_folds"]) != 5:
            raise DataValidationError(f"fold map {file_name} differs from well/fold contract")
        for fold in range(5):
            test = np.asarray([id_to_index[well_id] for well_id in well_ids if int(assignments[well_id]) == fold], dtype=np.int64)
            train = np.asarray([id_to_index[well_id] for well_id in well_ids if int(assignments[well_id]) != fold], dtype=np.int64)
            repeated.append(Context(f"repeated:{version}:{fold}", "repeated", version, fold, train, test))
    stress: list[Context] = []
    for scope, labels in (("spatial", spatial), ("typewell", typewell)):
        for group in range(5):
            test = np.flatnonzero(labels == group).astype(np.int64)
            train = np.flatnonzero(labels != group).astype(np.int64)
            stress.append(Context(f"{scope}:{group}", scope, scope, group, train, test))
    validate_contexts(repeated, stress, len(well_ids))
    return repeated, stress


def validate_contexts(repeated: Sequence[Context], stress: Sequence[Context], wells: int) -> None:
    if len(repeated) != 25 or len(stress) != 10:
        raise DataValidationError("context count differs")
    counts = np.zeros(wells, dtype=np.int64)
    for context in [*repeated, *stress]:
        train = set(context.train_indices.tolist())
        test = set(context.test_indices.tolist())
        if train & test or train | test != set(range(wells)) or not train or not test:
            raise DataValidationError(f"{context.key}: invalid membership")
        if context.scope == "repeated":
            counts[context.test_indices] += 1
    if not np.all(counts == 5):
        raise DataValidationError("repeated maps do not cover every well five times")


def _impute_scale(train: np.ndarray, test: np.ndarray, scale: bool) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(train, dtype=np.float64)
    x_test = np.asarray(test, dtype=np.float64)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1] or x_train.shape[0] <= 0:
        raise DataValidationError("feature matrix dimensions differ")
    finite = np.where(np.isfinite(x_train), x_train, np.nan)
    medians = np.nanmedian(finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    x_test = np.where(np.isfinite(x_test), x_test, medians)
    if scale:
        means = x_train.mean(axis=0)
        scales = x_train.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        x_train = (x_train - means) / scales
        x_test = (x_test - means) / scales
    if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
        raise DataValidationError("prepared features are non-finite")
    return x_train, x_test


def select_features(feature_set: str, pseudo: np.ndarray, e011: np.ndarray) -> np.ndarray:
    if feature_set == "pseudo":
        return pseudo
    if feature_set == "pseudo_e011":
        return np.hstack((pseudo, e011))
    if feature_set == "e011":
        return e011
    raise DataValidationError(f"unknown feature set {feature_set}")


def stable_seed(base: int, context_key: str, branch_name: str) -> int:
    digest = hashlib.sha256(f"{base}|{context_key}|{branch_name}".encode()).digest()
    return (int.from_bytes(digest[:4], "little") ^ int(base)) % (2**32 - 1)


def predict_branch(
    branch: Mapping[str, Any],
    context: Context,
    pseudo: np.ndarray,
    reversed_pseudo: np.ndarray,
    permuted_pseudo: np.ndarray,
    e011: np.ndarray,
    targets: np.ndarray,
    latest: np.ndarray,
    analytic_linear: np.ndarray,
    bound: float,
    control_mode: str | None,
    control_config: Mapping[str, Any],
) -> np.ndarray:
    name = str(branch["name"])
    family = str(branch["family"])
    if family == "analytic":
        if control_mode is not None:
            raise DataValidationError("analytic branches do not support learned controls")
        result = latest[context.test_indices] if name == "latest_cut_identity" else analytic_linear[context.test_indices]
        return np.clip(np.asarray(result, dtype=np.float64), -bound, bound)
    feature_set = str(branch["feature_set"])
    pseudo_source = pseudo
    if control_mode == "permuted_wells":
        pseudo_source = permuted_pseudo
    elif control_mode == "reversed_cuts":
        pseudo_source = reversed_pseudo
    x_all = select_features(feature_set, pseudo_source, e011)
    x_train_raw = x_all[context.train_indices]
    x_test_raw = x_all[context.test_indices]
    y_train = np.asarray(targets[context.train_indices], dtype=np.float64).copy()
    if control_mode == "shuffled_targets":
        rng = np.random.default_rng(stable_seed(int(control_config["target_shuffle_seed"]), context.key, name))
        y_train = y_train[rng.permutation(len(y_train))]
    if not np.all(np.isfinite(y_train)):
        raise DataValidationError(f"{name}: training targets are non-finite")
    if family == "ridge":
        x_train, x_test = _impute_scale(x_train_raw, x_test_raw, True)
        model = Ridge(alpha=float(branch["alpha"]), fit_intercept=True)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
    elif family == "extra_trees":
        x_train, x_test = _impute_scale(x_train_raw, x_test_raw, False)
        model = ExtraTreesRegressor(
            n_estimators=int(branch["n_estimators"]),
            min_samples_leaf=int(branch["min_samples_leaf"]),
            max_features=float(branch["max_features"]),
            random_state=int(branch["random_state"]),
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
    elif family == "hist_gradient":
        x_train, x_test = _impute_scale(x_train_raw, x_test_raw, True)
        columns: list[np.ndarray] = []
        for output in range(OUTPUTS):
            model = HistGradientBoostingRegressor(
                max_iter=int(branch["max_iter"]),
                max_leaf_nodes=int(branch["max_leaf_nodes"]),
                learning_rate=float(branch["learning_rate"]),
                l2_regularization=float(branch["l2_regularization"]),
                random_state=int(branch["random_state"]) + output,
            )
            model.fit(x_train, y_train[:, output])
            columns.append(model.predict(x_test))
        prediction = np.column_stack(columns)
    elif family == "knn":
        x_train, x_test = _impute_scale(x_train_raw, x_test_raw, True)
        neighbors = min(int(branch["n_neighbors"]), len(x_train))
        if neighbors <= 0:
            raise DataValidationError(f"{name}: no KNN training rows")
        model = KNeighborsRegressor(n_neighbors=neighbors, weights=str(branch["weights"]))
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
    else:
        raise DataValidationError(f"unknown branch family {family}")
    result = np.asarray(prediction, dtype=np.float64)
    if result.shape != (len(context.test_indices), OUTPUTS) or not np.all(np.isfinite(result)):
        raise DataValidationError(f"{name}: prediction dimensions or finiteness differ")
    return np.clip(result, -bound, bound)


def metrics_for_indices(records: Sequence[WellRecord], indices: Sequence[int], coefficients: np.ndarray, weight: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    array = np.asarray(coefficients, dtype=np.float64)
    if array.shape != (len(indices), OUTPUTS):
        raise DataValidationError("coefficient/test membership dimensions differ")
    rows = [records[int(index)].metric(array[position], weight) for position, index in enumerate(indices)]
    return summarize(rows), rows


def fixed_candidate_name(branch: str, weight: float) -> str:
    return f"{branch}__w{weight:.2f}"


def run_edge_tests() -> dict[str, Any]:
    passed: list[str] = []
    basis = spline_basis(128)
    assert basis.shape == (128, 4) and np.linalg.matrix_rank(basis) == 4
    passed.append("spline_basis_rank")

    try:
        fit_segment(np.arange(127.0), 0.0, 128, 80.0)
        raise AssertionError("127-row support did not fail")
    except DataValidationError:
        pass
    coefficients, rmse = fit_segment(np.arange(128.0), 0.0, 128, 80.0)
    assert coefficients.shape == (4,) and math.isfinite(rmse)
    passed.append("exact_support_boundary")

    for rows in (1, 2):
        try:
            fit_segment(np.arange(float(rows)), 0.0, 4, 80.0)
            raise AssertionError("one/two-row segment did not fail")
        except DataValidationError:
            pass
    passed.append("one_two_row_rejection")

    constant, constant_rmse = fit_segment(np.full(128, 7.0), 7.0, 128, 80.0)
    assert np.max(np.abs(constant)) < 1e-10 and constant_rmse < 1e-10
    passed.append("constant_tvt")

    try:
        fit_segment(np.r_[np.arange(127.0), np.nan], 0.0, 128, 80.0)
        raise AssertionError("non-finite segment did not fail")
    except DataValidationError:
        pass
    passed.append("nonfinite_tvt")

    try:
        _solve_coefficients(np.ones((128, 4)), np.zeros(128), 80.0)
        raise AssertionError("rank-deficient solve did not fail")
    except DataValidationError:
        pass
    passed.append("degenerate_spline_solve")

    assert validate_prefix([1.0, 2.0, np.nan, np.nan]) == 2
    for values in ([1.0, np.nan, 2.0, np.nan], [1.0, 2.0], [np.nan, np.nan]):
        try:
            validate_prefix(values)
            raise AssertionError("malformed prefix did not fail")
        except DataValidationError:
            pass
    passed.append("contiguous_prefix")

    x_train, x_test = _impute_scale(np.ones((3, 2)), np.asarray([[np.nan, 1.0]]), True)
    assert np.all(np.isfinite(x_train)) and np.all(np.isfinite(x_test))
    passed.append("zero_variance_features")

    tiny = Context("tiny", "repeated", "tiny", 0, np.asarray([0]), np.asarray([1]))
    branch = {"name": "ridge", "family": "ridge", "feature_set": "pseudo", "alpha": 1.0}
    pseudo = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    e011 = np.zeros((2, 1))
    targets = np.zeros((2, 4))
    latest = np.zeros((2, 4))
    pred = predict_branch(branch, tiny, pseudo, pseudo, pseudo, e011, targets, latest, latest, 80.0, None, {"target_shuffle_seed": 1})
    assert pred.shape == (1, 4)
    passed.append("tiny_ridge_partition")

    knn = {"name": "knn", "family": "knn", "feature_set": "pseudo", "n_neighbors": 25, "weights": "distance"}
    pred = predict_branch(knn, tiny, pseudo, pseudo, pseudo, e011, targets, latest, latest, 80.0, None, {"target_shuffle_seed": 1})
    assert pred.shape == (1, 4)
    passed.append("knn_neighbor_clipping")

    clipped, _ = _solve_coefficients(spline_basis(128), np.full(128, 1000.0), 5.0)
    assert np.max(np.abs(clipped)) <= 5.0
    passed.append("coefficient_clipping")

    record = WellRecord(
        "edgewell", 128, 128, 10.0, 0.0, np.zeros((4, 4)), np.zeros(4), np.full(4, 128),
        np.zeros(47), np.zeros(47), np.zeros(4), np.zeros(4), np.zeros(4), 0.0, 0.0,
        e011_sse=128.0, e011_sum=0.0, e_dot_d=0.0, d_sse=128.0, d_sum=0.0,
        basis_dot_e=np.zeros(4), basis_dot_d=np.zeros(4), basis_sum=spline_basis(128).sum(axis=0), basis_cross=spline_basis(128).T @ spline_basis(128),
    )
    fallback = record.metric(np.full(4, 80.0), 0.0)
    assert fallback["sse"] == 128.0
    passed.append("exact_zero_weight_fallback")

    names = ["a", "b", "c"]
    scores = [1.0, 1.0, 2.0]
    selected = min(range(len(names)), key=lambda index: (scores[index], index))
    assert names[selected] == "a"
    passed.append("deterministic_ties")

    repeated: list[Context] = []
    for version in range(5):
        for fold in range(5):
            test = np.flatnonzero(np.arange(10) % 5 == fold)
            train = np.flatnonzero(np.arange(10) % 5 != fold)
            repeated.append(Context(f"v{version}:{fold}", "repeated", f"v{version}", fold, train, test))
    stress = []
    for scope in ("spatial", "typewell"):
        for group in range(5):
            test = np.flatnonzero(np.arange(10) % 5 == group)
            train = np.flatnonzero(np.arange(10) % 5 != group)
            stress.append(Context(f"{scope}:{group}", scope, scope, group, train, test))
    validate_contexts(repeated, stress, 10)
    passed.append("fold_stress_membership")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.csv.gz"
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "well_id", "row_index", "hidden_index", "target", "spline4_ridge_equal_s075"])
            writer.writeheader()
            writer.writerow({"id": "edgewell_128", "well_id": "edgewell", "row_index": 128, "hidden_index": 0, "target": 1.0, "spline4_ridge_equal_s075": 1.0})
            writer.writerow({"id": "edgewell_128", "well_id": "edgewell", "row_index": 128, "hidden_index": 1, "target": 1.0, "spline4_ridge_equal_s075": 1.0})
        try:
            attach_e011_sufficient([record], path, 2)
            raise AssertionError("duplicate OOF ID did not fail")
        except DataValidationError:
            pass
    passed.append("duplicate_oof_ids")

    try:
        record.metric([0.0, 0.0, 0.0, np.nan], 1.0)
        raise AssertionError("non-finite coefficient did not fail")
    except DataValidationError:
        pass
    passed.append("nonfinite_predictions")

    assert len(passed) == 16
    return {"status": "PASS", "edge_groups": len(passed), "passed": passed}


def run_screen(root: Path, output_dir: Path, implementation_commit: str) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = root / "tracking/evidence/T025/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = load_raw_records(root / str(config["data_dir"]), config)
    coverage = attach_e011_sufficient(records, root / str(config["e011_oof"]), int(config["expected_hidden_rows"]))
    e011_features, e011_feature_names, spatial, typewell = load_compact(root / str(config["e011_compact"]), records)
    well_ids = [record.well_id for record in records]
    repeated, stress = load_contexts(root, well_ids, config["fold_files"], spatial, typewell)
    pseudo = np.vstack([record.pseudo_features for record in records])
    reversed_pseudo = np.vstack([record.reversed_features for record in records])
    targets = np.vstack([record.target_coefficients for record in records])
    latest = np.vstack([record.latest_coefficients for record in records])
    analytic_linear = np.vstack([record.linear_extrapolation for record in records])
    permutation_rng = np.random.default_rng(int(config["negative_controls"]["well_permutation_seed"]))
    permutation = permutation_rng.permutation(len(records))
    permuted_pseudo = pseudo[permutation]
    branches = list(config["branches"])
    branch_by_name = {str(branch["name"]): branch for branch in branches}
    if len(branch_by_name) != len(branches):
        raise DataValidationError("branch names are not unique")
    weights = [float(value) for value in config["placement_weights"]]
    if any(weight <= 0.0 or weight > 1.0 for weight in weights):
        raise DataValidationError("placement weights differ from positive fixed contract")
    bound = float(config["coefficient_absolute_bound_ft"])
    base_metrics = [record.metric(np.zeros(4), 0.0) for record in records]
    base_summary = summarize(base_metrics)
    oracle_metrics = [record.metric(record.target_coefficients, 1.0) for record in records]
    oracle_summary = summarize(oracle_metrics)

    real_context_rows: list[dict[str, Any]] = []
    map_coefficients = {
        label: {name: np.full((len(records), OUTPUTS), np.nan, dtype=np.float64) for name in branch_by_name}
        for label in sorted({context.label for context in repeated})
    }
    # Controls isolate the pseudo-trajectory mechanism. Pseudo+E011 branches are
    # excluded because intact E011 features could legitimately predict the target
    # even when the pseudo trajectory is corrupted.
    control_branches = [branch for branch in branches if str(branch["family"]) != "analytic" and str(branch["feature_set"]) == "pseudo"]
    control_sums = {
        (str(branch["name"]), mode): np.zeros((len(records), OUTPUTS), dtype=np.float64)
        for branch in control_branches for mode in CONTROL_MODES
    }
    control_counts = np.zeros(len(records), dtype=np.int64)
    base_context: dict[str, dict[str, Any]] = {}

    for context in repeated:
        base_context[context.key], _ = metrics_for_indices(records, context.test_indices, np.zeros((len(context.test_indices), 4)), 0.0)
        for branch in branches:
            name = str(branch["name"])
            prediction = predict_branch(branch, context, pseudo, reversed_pseudo, permuted_pseudo, e011_features, targets, latest, analytic_linear, bound, None, config["negative_controls"])
            map_coefficients[context.label][name][context.test_indices] = prediction
            for weight in weights:
                summary, _ = metrics_for_indices(records, context.test_indices, prediction, weight)
                real_context_rows.append({
                    "context": context.key,
                    "scope": context.scope,
                    "map": context.label,
                    "outer_group": context.outer_group,
                    "branch": name,
                    "weight": weight,
                    "candidate": fixed_candidate_name(name, weight),
                    **summary,
                })
        for branch in control_branches:
            name = str(branch["name"])
            for mode in CONTROL_MODES:
                prediction = predict_branch(branch, context, pseudo, reversed_pseudo, permuted_pseudo, e011_features, targets, latest, analytic_linear, bound, mode, config["negative_controls"])
                control_sums[(name, mode)][context.test_indices] += prediction
        control_counts[context.test_indices] += 1

    if not np.all(control_counts == 5):
        raise DataValidationError("control OOF coverage differs from five maps")
    for label, branch_map in map_coefficients.items():
        for name, values in branch_map.items():
            if not np.all(np.isfinite(values)):
                raise DataValidationError(f"{label}/{name}: map coefficient coverage is incomplete")

    map_rows: list[dict[str, Any]] = []
    final_coefficients: dict[str, np.ndarray] = {}
    for name in branch_by_name:
        stack = np.stack([map_coefficients[label][name] for label in sorted(map_coefficients)], axis=0)
        final_coefficients[name] = stack.mean(axis=0)
        for label in sorted(map_coefficients):
            for weight in weights:
                summary, _ = metrics_for_indices(records, np.arange(len(records)), map_coefficients[label][name], weight)
                map_rows.append({"map": label, "branch": name, "weight": weight, "candidate": fixed_candidate_name(name, weight), **summary})

    final_rows: list[dict[str, Any]] = []
    final_metrics_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for name, coefficients in final_coefficients.items():
        for weight in weights:
            candidate = fixed_candidate_name(name, weight)
            summary, well_metrics = metrics_for_indices(records, np.arange(len(records)), coefficients, weight)
            final_metrics_by_candidate[candidate] = well_metrics
            final_rows.append({
                "candidate": candidate,
                "branch": name,
                "weight": weight,
                "comparator": bool(branch_by_name[name].get("comparator", False)),
                "gain_vs_e011": float(base_summary["rmse"]) - float(summary["rmse"]),
                **summary,
            })

    context_lookup = {(row["context"], row["candidate"]): row for row in real_context_rows}
    map_lookup = {(row["map"], row["candidate"]): row for row in map_rows}
    base_map = {label: base_summary for label in map_coefficients}
    prelim: dict[str, dict[str, Any]] = {}
    for row in final_rows:
        candidate = str(row["candidate"])
        map_wins = sum(float(base_map[label]["rmse"]) - float(map_lookup[(label, candidate)]["rmse"]) > 0.0 for label in sorted(map_coefficients))
        cell_wins = sum(float(base_context[context.key]["rmse"]) - float(context_lookup[(context.key, candidate)]["rmse"]) > 0.0 for context in repeated)
        gates = {
            "gain": float(row["gain_vs_e011"]) >= float(config["preliminary"]["minimum_gain_vs_e011"]),
            "maps": map_wins >= int(config["preliminary"]["minimum_map_wins"]),
            "cells": cell_wins >= int(config["preliminary"]["minimum_outer_cell_wins"]),
            "p90": float(row["p90_well_rmse"]) - float(base_summary["p90_well_rmse"]) <= float(config["preliminary"]["maximum_p90_deterioration"]),
            "worst5": float(row["worst_5pct_sse_share"]) - float(base_summary["worst_5pct_sse_share"]) <= float(config["preliminary"]["maximum_worst5_share_increase"]),
        }
        row["map_wins"] = map_wins
        row["outer_cell_wins"] = cell_wins
        row["preliminary_pass"] = all(gates.values()) and not bool(row["comparator"])
        prelim[candidate] = gates
    preliminary_candidates = [str(row["candidate"]) for row in final_rows if bool(row["preliminary_pass"])]

    stress_rows: list[dict[str, Any]] = []
    stress_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    unique_prelim_branches = sorted({candidate.rsplit("__w", 1)[0] for candidate in preliminary_candidates})
    weights_by_branch: dict[str, list[float]] = {}
    for candidate in preliminary_candidates:
        branch_name, raw_weight = candidate.rsplit("__w", 1)
        weights_by_branch.setdefault(branch_name, []).append(float(raw_weight))
    for context in stress:
        base, _ = metrics_for_indices(records, context.test_indices, np.zeros((len(context.test_indices), 4)), 0.0)
        stress_rows.append({"context": context.key, "scope": context.scope, "outer_group": context.outer_group, "candidate": "e011", **base})
        stress_lookup[(context.key, "e011")] = base
        for name in unique_prelim_branches:
            branch = branch_by_name[name]
            prediction = predict_branch(branch, context, pseudo, reversed_pseudo, permuted_pseudo, e011_features, targets, latest, analytic_linear, bound, None, config["negative_controls"])
            for weight in sorted(set(weights_by_branch[name])):
                candidate = fixed_candidate_name(name, weight)
                summary, _ = metrics_for_indices(records, context.test_indices, prediction, weight)
                stress_rows.append({"context": context.key, "scope": context.scope, "outer_group": context.outer_group, "candidate": candidate, **summary})
                stress_lookup[(context.key, candidate)] = summary

    hidden_rows_values = np.asarray([record.hidden_rows for record in records], dtype=np.float64)
    missing_values = np.asarray([record.hidden_gr_missing_fraction for record in records], dtype=np.float64)
    long_threshold = float(np.quantile(hidden_rows_values, float(config["stress"]["long_suffix_quantile"])))
    missing_threshold = float(np.quantile(missing_values, float(config["stress"]["high_gr_missingness_quantile"])))
    base_well_rmse = np.asarray([metric["rmse"] for metric in base_metrics], dtype=np.float64)
    special_sets = {
        "long_suffix": np.flatnonzero(hidden_rows_values >= long_threshold),
        "high_gr_missingness": np.flatnonzero(missing_values >= missing_threshold),
        "e011_catastrophe": np.flatnonzero(base_well_rmse >= float(config["stress"]["e011_catastrophe_rmse"])),
    }
    special_rows: list[dict[str, Any]] = []
    special_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for slice_name, indices in special_sets.items():
        base = summarize([base_metrics[int(index)] for index in indices])
        special_rows.append({"slice": slice_name, "candidate": "e011", **base})
        special_lookup[(slice_name, "e011")] = base
        for candidate in preliminary_candidates:
            summary = summarize([final_metrics_by_candidate[candidate][int(index)] for index in indices])
            special_rows.append({"slice": slice_name, "candidate": candidate, **summary})
            special_lookup[(slice_name, candidate)] = summary

    control_rows: list[dict[str, Any]] = []
    maximum_control_gain = -math.inf
    for (name, mode), coefficient_sum in sorted(control_sums.items()):
        coefficients = coefficient_sum / control_counts[:, None]
        if not np.all(np.isfinite(coefficients)):
            raise DataValidationError(f"{name}/{mode}: control coefficient coverage differs")
        for weight in weights:
            summary, _ = metrics_for_indices(records, np.arange(len(records)), coefficients, weight)
            gain = float(base_summary["rmse"]) - float(summary["rmse"])
            maximum_control_gain = max(maximum_control_gain, gain)
            control_rows.append({
                "control": mode,
                "branch": name,
                "weight": weight,
                "candidate": f"{name}__{mode}__w{weight:.2f}",
                "gain_vs_e011": gain,
                **summary,
            })

    comparator_rows = [row for row in final_rows if bool(row["comparator"])]
    if not comparator_rows:
        raise DataValidationError("E011-only comparator rows are missing")
    best_comparator = min(comparator_rows, key=lambda row: (float(row["rmse"]), str(row["candidate"])))
    controls = {
        "source_inputs": {"pass": True, "config": str(config_path.relative_to(root))},
        "coverage": {"pass": coverage["rows"] == int(config["expected_hidden_rows"]) and coverage["wells"] == int(config["expected_wells"]), **coverage},
        "pseudo_support": {"pass": min(int(value) for record in records for value in record.pseudo_segment_rows) >= int(config["minimum_pseudo_segment_rows"]), "minimum_rows": min(int(value) for record in records for value in record.pseudo_segment_rows)},
        "finite_bounded": {"pass": all(np.all(np.isfinite(values)) and np.max(np.abs(values)) <= bound + 1e-9 for values in final_coefficients.values()), "bound": bound},
        "exact_fallback": {"pass": abs(float(base_summary["rmse"]) - 12.550756295689673) <= 1e-8, "rmse": base_summary["rmse"]},
        "negative_controls": {"pass": maximum_control_gain <= float(config["negative_controls"]["maximum_gain_vs_e011"]), "maximum_gain_vs_e011": maximum_control_gain},
        "context_completion": {"pass": len(real_context_rows) == 25 * len(branches) * len(weights), "rows": len(real_context_rows)},
        "edge_groups": run_edge_tests(),
    }
    all_controls = all(bool(item["pass"] if isinstance(item, Mapping) and "pass" in item else item.get("status") == "PASS") for item in controls.values())

    authorization_rows: list[dict[str, Any]] = []
    gates_by_candidate: dict[str, dict[str, bool]] = {}
    for row in final_rows:
        candidate = str(row["candidate"])
        if bool(row["comparator"]):
            continue
        spatial_gains: list[float] = []
        typewell_gains: list[float] = []
        special_gains: list[float] = []
        if candidate in preliminary_candidates:
            for context in stress:
                gain = float(stress_lookup[(context.key, "e011")]["rmse"]) - float(stress_lookup[(context.key, candidate)]["rmse"])
                (spatial_gains if context.scope == "spatial" else typewell_gains).append(gain)
            for slice_name in special_sets:
                special_gains.append(float(special_lookup[(slice_name, "e011")]["rmse"]) - float(special_lookup[(slice_name, candidate)]["rmse"]))
        incremental = float(best_comparator["rmse"]) - float(row["rmse"])
        gates = {
            "oracle": float(oracle_summary["rmse"]) <= float(config["authorization"]["maximum_oracle_rmse"]),
            "gain": float(row["gain_vs_e011"]) >= float(config["authorization"]["minimum_gain_vs_e011"]),
            "maps": int(row["map_wins"]) >= int(config["authorization"]["minimum_map_wins"]),
            "cells": int(row["outer_cell_wins"]) >= int(config["authorization"]["minimum_outer_cell_wins"]),
            "p90": float(row["p90_well_rmse"]) - float(base_summary["p90_well_rmse"]) <= float(config["authorization"]["maximum_p90_deterioration"]),
            "worst5": float(row["worst_5pct_sse_share"]) - float(base_summary["worst_5pct_sse_share"]) <= float(config["authorization"]["maximum_worst5_share_increase"]),
            "spatial": bool(spatial_gains) and min(spatial_gains) >= 0.0,
            "typewell": bool(typewell_gains) and min(typewell_gains) >= 0.0,
            "special_slices": bool(special_gains) and min(special_gains) >= 0.0,
            "incremental_pseudo": incremental >= float(config["authorization"]["minimum_incremental_gain_vs_best_e011_only"]),
            "controls": all_controls,
            "reproduction": False,
        }
        gates_by_candidate[candidate] = gates
        authorization_rows.append({
            "candidate": candidate,
            "incremental_gain_vs_best_e011_only": incremental,
            "minimum_spatial_gain": min(spatial_gains) if spatial_gains else "",
            "minimum_typewell_gain": min(typewell_gains) if typewell_gains else "",
            "minimum_special_slice_gain": min(special_gains) if special_gains else "",
            "preliminary_pass": candidate in preliminary_candidates,
            **{f"gate_{name}": value for name, value in gates.items()},
        })

    substantive_gates = {candidate: all(value for name, value in gates.items() if name != "reproduction") for candidate, gates in gates_by_candidate.items()}
    substantive_passers = [candidate for candidate, passed in substantive_gates.items() if passed]
    reported = min(final_rows, key=lambda row: (float(row["rmse"]), str(row["candidate"])))
    status = "awaiting_reproduction" if substantive_passers else "worth_screen_reject"
    decision = "await_independent_reproduction" if substantive_passers else "close_h018_without_formal_experiment"

    selected_well_rows: list[dict[str, Any]] = []
    reported_candidate = str(reported["candidate"])
    reported_branch = str(reported["branch"])
    reported_weight = float(reported["weight"])
    for index, record in enumerate(records):
        metric = final_metrics_by_candidate[reported_candidate][index]
        selected_well_rows.append({
            "well_id": record.well_id,
            "candidate": reported_candidate,
            "branch": reported_branch,
            "weight": reported_weight,
            "rows_scored": metric["rows_scored"],
            "sse": metric["sse"],
            "rmse": metric["rmse"],
            "mean_error": metric["mean_error"],
            "known_rows": record.known_rows,
            "hidden_rows": record.hidden_rows,
            "hidden_gr_missing_fraction": record.hidden_gr_missing_fraction,
            **{f"predicted_coefficient_{output}": final_coefficients[reported_branch][index, output] for output in range(OUTPUTS)},
            **{f"target_coefficient_{output}": record.target_coefficients[output] for output in range(OUTPUTS)},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidate_metrics.csv", final_rows)
    _write_csv(output_dir / "context_metrics.csv", real_context_rows)
    _write_csv(output_dir / "map_metrics.csv", map_rows)
    _write_csv(output_dir / "stress_metrics.csv", stress_rows if stress_rows else [{"context": "none", "scope": "none", "outer_group": -1, "candidate": "none", **base_summary}])
    _write_csv(output_dir / "special_slice_metrics.csv", special_rows if special_rows else [{"slice": "none", "candidate": "none", **base_summary}])
    _write_csv(output_dir / "negative_control_metrics.csv", control_rows)
    _write_csv(output_dir / "authorization_gates.csv", authorization_rows)
    _write_csv(output_dir / "selected_well_metrics.csv", selected_well_rows)
    edge = controls["edge_groups"]
    _write_json(output_dir / "edge_cases.json", edge)
    runtime = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "task_id": "T025",
        "hypothesis_id": "H018",
        "implementation_commit": implementation_commit,
        "status": status,
        "decision": decision,
        "formal_experiment_authorized": False,
        "reported_candidate": reported_candidate,
        "reported_metrics": reported,
        "base_summary": base_summary,
        "oracle_summary": oracle_summary,
        "best_e011_only_comparator": best_comparator,
        "preliminary_candidates": preliminary_candidates,
        "substantive_passers": substantive_passers,
        "gates_by_candidate": gates_by_candidate,
        "controls": controls,
        "maximum_negative_control_gain": maximum_control_gain,
        "thresholds": {"long_suffix": long_threshold, "high_gr_missingness": missing_threshold, "e011_catastrophe": config["stress"]["e011_catastrophe_rmse"]},
        "special_slice_wells": {name: len(indices) for name, indices in special_sets.items()},
        "feature_dimensions": {"pseudo": pseudo.shape[1], "e011": e011_features.shape[1], "pseudo_e011": pseudo.shape[1] + e011_features.shape[1]},
        "e011_feature_count": len(e011_feature_names),
        "runtime_seconds": runtime,
        "deployment": {"package_built": False, "kaggle_executed": False, "submission_created": False, "submission_made": False},
    }
    _write_json(output_dir / "summary.json", summary)
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "task_id": "T025",
        "implementation_commit": implementation_commit,
        "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--edge-only", action="store_true")
    args = parser.parse_args()
    if args.edge_only:
        print(json.dumps(run_edge_tests(), indent=2, sort_keys=True))
        return
    if not args.output_dir or not args.implementation_commit:
        parser.error("--output-dir and --implementation-commit are required unless --edge-only is used")
    root = ROOT
    summary = run_screen(root, root / args.output_dir, str(args.implementation_commit))
    print(json.dumps({
        "status": summary["status"],
        "decision": summary["decision"],
        "reported_candidate": summary["reported_candidate"],
        "reported_rmse": summary["reported_metrics"]["rmse"],
        "gain_vs_e011": summary["reported_metrics"]["gain_vs_e011"],
        "oracle_rmse": summary["oracle_summary"]["rmse"],
        "preliminary_candidates": len(summary["preliminary_candidates"]),
        "substantive_passers": len(summary["substantive_passers"]),
        "runtime_seconds": summary["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
