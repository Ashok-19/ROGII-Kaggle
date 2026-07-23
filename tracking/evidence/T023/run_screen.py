#!/usr/bin/env python3
"""T023: preregistered residual sequence-state worth screen around E011."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.harness import (  # noqa: E402
    DataValidationError,
    ErrorAccumulator,
    WellMetric,
    _canonical_json,
    _quantile,
    _summarize,
)

CONTROL_COUNTS = (4, 8, 16, 32, 64)
MAIN_COUNTS = (16, 32)
CHANNEL_NAMES = (
    "position",
    "md_normalized",
    "z_delta",
    "z_slope",
    "z_curvature",
    "gr_level",
    "gr_rolling_std",
    "gr_slope",
    "gr_missing",
    "e011_delta",
    "e011_slope",
    "e011_curvature",
    "e006_minus_e011",
    "disagreement_slope",
    "typewell_mismatch_e011",
    "typewell_mismatch_e006",
    "typewell_local_slope",
)
TYPEWELL_CHANNELS = (0, 5, 6, 7, 8, 14, 15, 16)
DISAGREEMENT_CHANNELS = (0, 9, 10, 11, 12, 13)
ACF_LAGS = (1, 4, 16, 64, 256)


@dataclass(frozen=True)
class Context:
    key: str
    scope: str
    label: str
    outer_group: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    inner_assignments: Mapping[str, int]


@dataclass
class SequenceStats:
    rows: int
    sum_x: float
    sum_x_sq: float
    base_sum: float
    base_sse: float
    base_x_sum: float
    basis_sum: np.ndarray
    basis_x_sum: np.ndarray
    basis_base: np.ndarray
    basis_cross: np.ndarray
    target_coefficients: np.ndarray

    def metric(self, well_id: str, coefficients: Sequence[float]) -> WellMetric:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != self.target_coefficients.shape or not np.all(np.isfinite(values)):
            raise DataValidationError(f"{well_id}: invalid T023 coefficient vector")
        sum_error = self.base_sum + float(self.basis_sum @ values)
        sum_error_sq = self.base_sse + 2.0 * float(self.basis_base @ values) + float(values @ self.basis_cross @ values)
        sum_x_error = self.base_x_sum + float(self.basis_x_sum @ values)
        tolerance = max(1e-9, abs(self.base_sse) * 1e-12)
        if sum_error_sq < 0.0 and abs(sum_error_sq) <= tolerance:
            sum_error_sq = 0.0
        if sum_error_sq < 0.0 or not math.isfinite(sum_error_sq):
            raise DataValidationError(f"{well_id}: invalid T023 sufficient-statistic SSE")
        return ErrorAccumulator(
            rows=self.rows,
            sum_error=sum_error,
            sum_error_sq=sum_error_sq,
            sum_x=self.sum_x,
            sum_x_sq=self.sum_x_sq,
            sum_x_error=sum_x_error,
        ).finalize(well_id)


@dataclass
class WellSequence:
    well_id: str
    rows: int
    profile_features: np.ndarray
    block_features: dict[int, np.ndarray]
    typewell_block_features: dict[int, np.ndarray]
    disagreement_block_features: dict[int, np.ndarray]
    statistics: dict[int, SequenceStats]
    base_metric: WellMetric
    hidden_gr_missing_fraction: float
    diagnostics: dict[str, float | None]
    spatial_group: int
    typewell_group: int


@dataclass(frozen=True)
class Selection:
    model_config: Mapping[str, Any]
    shrinkage: float
    cap: float
    inner_rmse: float
    effective_neighbors: int | None = None


def stable_order(ids: Sequence[str], salt: str) -> list[int]:
    return sorted(range(len(ids)), key=lambda index: (hashlib.sha256(f"{salt}|{ids[index]}".encode()).hexdigest(), ids[index]))


def shifted_permutation(ids: Sequence[str], salt: str) -> np.ndarray:
    order = stable_order(ids, salt)
    if len(order) > 1:
        order = order[1:] + order[:1]
    return np.asarray(order, dtype=int)


def finite_float(raw: Any, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"invalid {label}") from exc
    if not math.isfinite(value):
        raise DataValidationError(f"non-finite {label}")
    return value


def piecewise_basis(rows: int, controls: int) -> np.ndarray:
    n = int(rows)
    k = int(controls)
    if n <= 0 or k <= 0:
        raise DataValidationError("T023 basis requires positive rows and controls")
    position = np.linspace(0.0, 1.0, n, dtype=np.float64) if n > 1 else np.zeros(1, dtype=np.float64)
    scaled = position * k
    lower = np.floor(scaled + 1e-12).astype(int)
    lower = np.clip(lower, 0, k)
    upper = np.minimum(lower + 1, k)
    fraction = scaled - lower
    matrix = np.zeros((n, k), dtype=np.float64)
    row_index = np.arange(n)
    mask = lower >= 1
    matrix[row_index[mask], lower[mask] - 1] += 1.0 - fraction[mask]
    mask = (upper >= 1) & (upper != lower)
    matrix[row_index[mask], upper[mask] - 1] += fraction[mask]
    matrix[0, :] = 0.0
    if not np.all(np.isfinite(matrix)) or float(np.max(np.abs(matrix[0]))) != 0.0:
        raise DataValidationError("T023 basis boundary contract failed")
    return matrix


def metric_from_error(well_id: str, error: np.ndarray) -> WellMetric:
    values = np.asarray(error, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise DataValidationError(f"{well_id}: invalid error vector")
    x = np.arange(values.size, dtype=np.float64)
    return ErrorAccumulator(
        rows=int(values.size),
        sum_error=float(values.sum()),
        sum_error_sq=float(values @ values),
        sum_x=float(x.sum()),
        sum_x_sq=float(x @ x),
        sum_x_error=float(x @ values),
    ).finalize(well_id)


def build_stats(base_error: np.ndarray, controls: int) -> SequenceStats:
    error = np.asarray(base_error, dtype=np.float64)
    basis = piecewise_basis(len(error), controls)
    x = np.arange(len(error), dtype=np.float64)
    cross = basis.T @ basis
    base = basis.T @ error
    target = np.linalg.lstsq(cross, -base, rcond=None)[0]
    if target.shape != (controls,) or not np.all(np.isfinite(target)):
        raise DataValidationError("T023 oracle coefficients are invalid")
    return SequenceStats(
        rows=len(error),
        sum_x=float(x.sum()),
        sum_x_sq=float(x @ x),
        base_sum=float(error.sum()),
        base_sse=float(error @ error),
        base_x_sum=float(x @ error),
        basis_sum=basis.sum(axis=0),
        basis_x_sum=basis.T @ x,
        basis_base=base,
        basis_cross=cross,
        target_coefficients=target,
    )


def validate_md(md: np.ndarray) -> None:
    values = np.asarray(md, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise DataValidationError("invalid MD sequence")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise DataValidationError("MD must be strictly increasing for T023")


def fill_missing(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    if finite.all():
        return array.copy()
    index = np.arange(len(array), dtype=np.float64)
    return np.interp(index, index[finite], array[finite]).astype(np.float64)


def derivative(values: np.ndarray, md: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(md, dtype=np.float64)
    validate_md(x)
    if y.shape != x.shape or not np.all(np.isfinite(y)):
        raise DataValidationError("invalid derivative inputs")
    if len(y) == 1:
        return np.zeros(1, dtype=np.float64)
    return np.gradient(y, x, edge_order=1).astype(np.float64)


def rolling_std(values: np.ndarray, window: int = 31) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise DataValidationError("invalid rolling-standard-deviation input")
    return pd.Series(y).rolling(window=int(window), center=True, min_periods=1).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float64)


def read_typewell(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(dtype=np.float64)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(tvt) & np.isfinite(gr)
    if int(mask.sum()) < 2:
        raise DataValidationError(f"{path}: insufficient typewell TVT/GR")
    grouped = pd.DataFrame({"TVT": tvt[mask], "GR": gr[mask]}).groupby("TVT", sort=True, as_index=False)["GR"].mean()
    tvt = grouped["TVT"].to_numpy(dtype=np.float64)
    gr = grouped["GR"].to_numpy(dtype=np.float64)
    if len(tvt) < 2 or np.any(np.diff(tvt) <= 0.0) or not np.all(np.isfinite(gr)):
        raise DataValidationError(f"{path}: malformed typewell after duplicate aggregation")
    return tvt, gr


def typewell_gr(tvt_axis: np.ndarray, gr_axis: np.ndarray, predicted_tvt: np.ndarray) -> np.ndarray:
    values = np.asarray(predicted_tvt, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise DataValidationError("non-finite predicted TVT for typewell interpolation")
    return np.interp(values, tvt_axis, gr_axis, left=float(gr_axis[0]), right=float(gr_axis[-1]))


def linear_slope(y: np.ndarray, x: np.ndarray) -> float:
    values = np.asarray(y, dtype=np.float64)
    axis = np.asarray(x, dtype=np.float64)
    if len(values) < 2 or float(np.var(axis)) <= 1e-15:
        return 0.0
    return float(np.dot(axis - axis.mean(), values - values.mean()) / np.dot(axis - axis.mean(), axis - axis.mean()))


def profile_features(channels: np.ndarray) -> np.ndarray:
    matrix = np.asarray(channels, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(CHANNEL_NAMES) or not np.all(np.isfinite(matrix)):
        raise DataValidationError("invalid T023 channel matrix")
    position = np.linspace(0.0, 1.0, len(matrix), dtype=np.float64) if len(matrix) > 1 else np.zeros(1)
    output: list[float] = []
    phase = (np.arange(len(matrix), dtype=np.float64) + 0.5) / max(1, len(matrix))
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        mean = float(values.mean())
        std = float(values.std())
        output.extend([
            mean,
            std,
            float(values[0]),
            float(values[-1]),
            float(values[-1] - values[0]),
            linear_slope(values, position),
            float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0,
            float(np.quantile(values, 0.1)),
            float(np.quantile(values, 0.9)),
        ])
        normalized = (values - mean) / max(std, 1e-9)
        for frequency in range(1, 5):
            output.append(float(np.mean(normalized * np.cos(math.pi * frequency * phase))))
    result = np.asarray(output, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise DataValidationError("non-finite T023 profile features")
    return result


def summarize_channel(values: np.ndarray, position: np.ndarray) -> list[float]:
    y = np.asarray(values, dtype=np.float64)
    p = np.asarray(position, dtype=np.float64)
    return [
        float(y.mean()),
        float(y.std()),
        float(y[0]),
        float(y[-1]),
        float(y[-1] - y[0]),
        linear_slope(y, p),
    ]


def block_feature_matrix(channels: np.ndarray, controls: int, subset: Sequence[int] | None = None) -> np.ndarray:
    matrix = np.asarray(channels, dtype=np.float64)
    columns = tuple(range(matrix.shape[1])) if subset is None else tuple(int(index) for index in subset)
    if not columns or any(index < 0 or index >= matrix.shape[1] for index in columns):
        raise DataValidationError("invalid T023 block feature subset")
    n = len(matrix)
    edges = np.floor(np.linspace(0, n, controls + 1)).astype(int)
    edges[-1] = n
    global_position = np.linspace(0.0, 1.0, n, dtype=np.float64) if n > 1 else np.zeros(1)
    global_summary: list[float] = []
    for column in columns:
        values = matrix[:, column]
        global_summary.extend([float(values.mean()), float(values.std()), float(values[-1] - values[0]), linear_slope(values, global_position)])
    rows: list[list[float]] = []
    for block in range(controls):
        start = int(edges[block])
        stop = int(edges[block + 1])
        if stop <= start:
            start = min(start, n - 1)
            stop = start + 1
        local = matrix[start:stop]
        local_position = np.linspace(0.0, 1.0, len(local), dtype=np.float64) if len(local) > 1 else np.zeros(1)
        values: list[float] = [(block + 1) / controls, len(local) / max(1, n)]
        for column in columns:
            values.extend(summarize_channel(local[:, column], local_position))
        values.extend(global_summary)
        rows.append(values)
    result = np.asarray(rows, dtype=np.float64)
    if result.shape[0] != controls or not np.all(np.isfinite(result)):
        raise DataValidationError("invalid T023 block features")
    return result


def sequence_channels(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    e011: np.ndarray,
    e006: np.ndarray,
    boundary_md: float,
    boundary_z: float,
    boundary_tvt: float,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
) -> np.ndarray:
    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    gr = np.asarray(gr, dtype=np.float64)
    e011 = np.asarray(e011, dtype=np.float64)
    e006 = np.asarray(e006, dtype=np.float64)
    validate_md(np.concatenate(([float(boundary_md)], md)))
    if not (z.shape == gr.shape == e011.shape == e006.shape == md.shape):
        raise DataValidationError("T023 sequence lengths differ")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(e011)) or not np.all(np.isfinite(e006)):
        raise DataValidationError("T023 non-finite geometry or base prediction")
    missing = ~np.isfinite(gr)
    gr_filled = fill_missing(gr)
    full_md = np.concatenate(([float(boundary_md)], md))
    full_z = np.concatenate(([float(boundary_z)], z))
    full_e011 = np.concatenate(([float(boundary_tvt)], e011))
    full_e006 = np.concatenate(([float(boundary_tvt)], e006))
    z_slope = derivative(full_z, full_md)[1:]
    z_curvature = derivative(np.concatenate(([z_slope[0] if len(z_slope) else 0.0], z_slope)), full_md)[1:]
    gr_boundary = float(gr_filled[0]) if len(gr_filled) else 0.0
    gr_slope = derivative(np.concatenate(([gr_boundary], gr_filled)), full_md)[1:]
    e011_slope = derivative(full_e011, full_md)[1:]
    e011_curvature = derivative(np.concatenate(([e011_slope[0] if len(e011_slope) else 0.0], e011_slope)), full_md)[1:]
    disagreement = e006 - e011
    disagreement_slope = derivative(np.concatenate(([0.0], disagreement)), full_md)[1:]
    predicted_tw_e011 = typewell_gr(tw_tvt, tw_gr, e011)
    predicted_tw_e006 = typewell_gr(tw_tvt, tw_gr, e006)
    mismatch_e011 = np.where(missing, 0.0, gr_filled - predicted_tw_e011)
    mismatch_e006 = np.where(missing, 0.0, gr_filled - predicted_tw_e006)
    tw_slope = derivative(np.concatenate(([typewell_gr(tw_tvt, tw_gr, np.asarray([boundary_tvt]))[0]], predicted_tw_e011)), full_md)[1:]
    position = np.linspace(0.0, 1.0, len(md), dtype=np.float64) if len(md) > 1 else np.zeros(1)
    md_normalized = (md - float(boundary_md)) / max(float(md[-1] - boundary_md), 1e-9)
    result = np.column_stack((
        position,
        md_normalized,
        z - float(boundary_z),
        z_slope,
        z_curvature,
        gr_filled,
        rolling_std(gr_filled),
        gr_slope,
        missing.astype(np.float64),
        e011 - float(boundary_tvt),
        e011_slope,
        e011_curvature,
        disagreement,
        disagreement_slope,
        mismatch_e011,
        mismatch_e006,
        tw_slope,
    ))
    if result.shape != (len(md), len(CHANNEL_NAMES)) or not np.all(np.isfinite(result)):
        raise DataValidationError("T023 channel construction failed")
    return result


def acf(values: np.ndarray, lag: int) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= lag:
        return None
    left = array[:-lag]
    right = array[lag:]
    if float(left.std()) <= 1e-12 or float(right.std()) <= 1e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def residual_diagnostics(error: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(error, dtype=np.float64)
    centered = values - values.mean()
    spectrum = np.fft.rfft(centered)
    energy = np.abs(spectrum) ** 2
    low = float(energy[1 : min(len(energy), 17)].sum()) if len(energy) > 1 else 0.0
    total = float(energy[1:].sum()) if len(energy) > 1 else 0.0
    output: dict[str, float | None] = {f"acf_{lag}": acf(centered, lag) for lag in ACF_LAGS}
    output["low_frequency_energy_fraction"] = low / total if total > 0.0 else 0.0
    return output


def load_compact_groups(root: Path, e011_config: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(root / e011_config["compact_input"]["path"], allow_pickle=False) as payload:
        well_ids = tuple(str(value) for value in payload["well_ids"].tolist())
        rows = np.asarray(payload["rows"], dtype=np.int32)
        spatial = np.asarray(payload["spatial_assignment"], dtype=np.int8)
        typewell = np.asarray(payload["typewell_assignment"], dtype=np.int8)
        feature_names = tuple(str(value) for value in payload["feature_names"].tolist())
        features = np.asarray(payload["features"], dtype=np.float64)
    missing_index = feature_names.index("hidden_gr_missing_fraction")
    missing = features[:, missing_index]
    if len(well_ids) != 773 or int(rows.sum()) != 3783989:
        raise DataValidationError("T023 compact identity differs")
    return well_ids, rows, spatial, typewell, missing


def build_cache(root: Path, config: Mapping[str, Any], cache_path: Path) -> dict[str, WellSequence]:
    e011_config = json.loads((root / config["inputs"]["e011_config"]).read_text(encoding="utf-8"))
    well_ids, expected_rows, spatial, typewell, compact_missing = load_compact_groups(root, e011_config)
    index_by_well = {well_id: index for index, well_id in enumerate(well_ids)}
    records: dict[str, WellSequence] = {}
    required = {"id", "well_id", "row_index", "hidden_index", "target", "e006_nested_fusion", config["base_candidate"]}
    current: str | None = None
    block: dict[str, list[Any]] = defaultdict(list)
    seen: set[str] = set()
    total_rows = 0
    ordered_ids = hashlib.sha256()

    def finalize() -> None:
        nonlocal current, block
        if current is None:
            return
        well_id = current
        row_index = np.asarray(block["row_index"], dtype=int)
        hidden_index = np.asarray(block["hidden_index"], dtype=int)
        target = np.asarray(block["target"], dtype=np.float64)
        e006 = np.asarray(block["e006"], dtype=np.float64)
        e011 = np.asarray(block["e011"], dtype=np.float64)
        expected = int(expected_rows[index_by_well[well_id]])
        if len(target) != expected or not np.array_equal(hidden_index, np.arange(expected)):
            raise DataValidationError(f"{well_id}: T023 hidden row identity differs")
        horizontal_path = root / config["inputs"]["train_dir"] / f"{well_id}__horizontal_well.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=["MD", "Z", "GR", "TVT_input"])
        if row_index[0] <= 0 or not np.array_equal(row_index, np.arange(row_index[0], row_index[0] + expected)):
            raise DataValidationError(f"{well_id}: T023 parent row indices are not a contiguous suffix")
        if int(row_index[-1]) >= len(horizontal):
            raise DataValidationError(f"{well_id}: T023 row index exceeds horizontal data")
        boundary = horizontal.iloc[int(row_index[0]) - 1]
        boundary_tvt = finite_float(boundary["TVT_input"], "boundary TVT_input")
        hidden = horizontal.iloc[row_index]
        md = pd.to_numeric(hidden["MD"], errors="coerce").to_numpy(dtype=np.float64)
        z = pd.to_numeric(hidden["Z"], errors="coerce").to_numpy(dtype=np.float64)
        gr = pd.to_numeric(hidden["GR"], errors="coerce").to_numpy(dtype=np.float64)
        tw_tvt, tw_gr = read_typewell(root / config["inputs"]["train_dir"] / f"{well_id}__typewell.csv")
        channels = sequence_channels(
            md,
            z,
            gr,
            e011,
            e006,
            finite_float(boundary["MD"], "boundary MD"),
            finite_float(boundary["Z"], "boundary Z"),
            boundary_tvt,
            tw_tvt,
            tw_gr,
        )
        error = e011 - target
        statistics = {count: build_stats(error, count) for count in CONTROL_COUNTS}
        base_metric = metric_from_error(well_id, error)
        block_features = {count: block_feature_matrix(channels, count) for count in MAIN_COUNTS}
        typewell_features = {count: block_feature_matrix(channels, count, TYPEWELL_CHANNELS) for count in MAIN_COUNTS}
        disagreement_features = {count: block_feature_matrix(channels, count, DISAGREEMENT_CHANNELS) for count in MAIN_COUNTS}
        record = WellSequence(
            well_id=well_id,
            rows=expected,
            profile_features=profile_features(channels),
            block_features=block_features,
            typewell_block_features=typewell_features,
            disagreement_block_features=disagreement_features,
            statistics=statistics,
            base_metric=base_metric,
            hidden_gr_missing_fraction=float(compact_missing[index_by_well[well_id]]) if math.isfinite(float(compact_missing[index_by_well[well_id]])) else float(np.mean(~np.isfinite(gr))),
            diagnostics=residual_diagnostics(error),
            spatial_group=int(spatial[index_by_well[well_id]]),
            typewell_group=int(typewell[index_by_well[well_id]]),
        )
        records[well_id] = record
        current = None
        block = defaultdict(list)

    oof_path = root / config["inputs"]["e011_oof"]
    with gzip.open(oof_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise DataValidationError("T023 E011 OOF schema differs")
        for row in reader:
            well_id = str(row["well_id"])
            if current is None:
                if well_id in seen:
                    raise DataValidationError("T023 OOF well blocks repeat")
                current = well_id
                seen.add(well_id)
            elif well_id != current:
                finalize()
                if well_id in seen:
                    raise DataValidationError("T023 OOF well blocks repeat")
                current = well_id
                seen.add(well_id)
            block["row_index"].append(int(row["row_index"]))
            block["hidden_index"].append(int(row["hidden_index"]))
            block["target"].append(finite_float(row["target"], "target"))
            block["e006"].append(finite_float(row["e006_nested_fusion"], "E006"))
            block["e011"].append(finite_float(row[config["base_candidate"]], "E011"))
            ordered_ids.update(str(row["id"]).encode("utf-8"))
            ordered_ids.update(b"\n")
            total_rows += 1
    finalize()
    if tuple(sorted(records)) != well_ids or total_rows != int(config["expected_hidden_rows"]):
        raise DataValidationError("T023 well/row coverage differs")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_commit": config["source_commit"],
        "well_ids": well_ids,
        "rows": total_rows,
        "ordered_id_sha256": ordered_ids.hexdigest(),
        "records": records,
    }
    with cache_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=3, mtime=0) as handle:
            pickle.dump(payload, handle, protocol=5)
    return records


def load_or_build_cache(root: Path, config: Mapping[str, Any], cache_path: Path, rebuild: bool) -> tuple[dict[str, WellSequence], dict[str, Any]]:
    if rebuild or not cache_path.is_file():
        records = build_cache(root, config, cache_path)
    with gzip.open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema_version") != 1 or payload.get("source_commit") != config["source_commit"]:
        raise DataValidationError("T023 cache identity differs")
    records = payload["records"]
    if len(records) != int(config["expected_wells"]) or sum(record.rows for record in records.values()) != int(config["expected_hidden_rows"]):
        raise DataValidationError("T023 cache coverage differs")
    return records, {key: value for key, value in payload.items() if key != "records"}


def load_contexts(root: Path, config: Mapping[str, Any], records: Mapping[str, WellSequence]) -> list[Context]:
    ids = tuple(sorted(records))
    expected = set(ids)
    folds = []
    for relative in config["inputs"]["fold_files"]:
        fold = json.loads((root / relative).read_text(encoding="utf-8"))
        if set(fold.get("assignments", {})) != expected:
            raise DataValidationError(f"T023 fold membership differs: {relative}")
        folds.append(fold)
    contexts: list[Context] = []
    for fold in folds:
        assignments = {well_id: int(value) for well_id, value in fold["assignments"].items()}
        for outer in range(int(fold["n_folds"])):
            test = tuple(well_id for well_id in ids if assignments[well_id] == outer)
            train = tuple(well_id for well_id in ids if assignments[well_id] != outer)
            contexts.append(Context(f"repeated:{fold['version']}:{outer}", "repeated", str(fold["version"]), outer, train, test, assignments))
    inner = {well_id: int(folds[0]["assignments"][well_id]) for well_id in ids}
    for scope, attr in (("spatial", "spatial_group"), ("typewell", "typewell_group")):
        assignments = {well_id: int(getattr(record, attr)) for well_id, record in records.items()}
        if set(assignments.values()) != set(range(5)):
            raise DataValidationError(f"T023 {scope} evaluator groups differ")
        for outer in range(5):
            test = tuple(well_id for well_id in ids if assignments[well_id] == outer)
            train = tuple(well_id for well_id in ids if assignments[well_id] != outer)
            contexts.append(Context(f"{scope}:{outer}", scope, scope, outer, train, test, inner))
    for context in contexts:
        if not context.train_ids or not context.test_ids or set(context.train_ids) & set(context.test_ids):
            raise DataValidationError(f"{context.key}: invalid outer membership")
        coverage: set[str] = set()
        for group in sorted({int(context.inner_assignments[well_id]) for well_id in context.train_ids}):
            validation = {well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) == group}
            training = set(context.train_ids) - validation
            if not validation or not training or validation & set(context.test_ids):
                raise DataValidationError(f"{context.key}: invalid inner membership")
            coverage.update(validation)
        if coverage != set(context.train_ids):
            raise DataValidationError(f"{context.key}: incomplete inner coverage")
    return contexts


def target_matrix(records: Mapping[str, WellSequence], ids: Sequence[str], controls: int, mode: str, salt: str) -> np.ndarray:
    values = np.vstack([records[well_id].statistics[controls].target_coefficients for well_id in ids]).astype(np.float64)
    if mode == "shuffled_wells":
        values = values[shifted_permutation(ids, f"T023-target-shuffle|{salt}")]
    elif mode == "reversed_targets":
        values = values[:, ::-1]
    elif mode != "normal":
        raise DataValidationError(f"unknown T023 target mode {mode}")
    if not np.all(np.isfinite(values)):
        raise DataValidationError("non-finite T023 target matrix")
    return values


def feature_matrix(records: Mapping[str, WellSequence], ids: Sequence[str], branch: Mapping[str, Any], salt: str) -> tuple[np.ndarray, str]:
    family = str(branch["family"])
    controls = int(branch["controls"])
    if family in {"profile_ridge", "profile_knn", "negative_shuffled_profile"}:
        matrix = np.vstack([records[well_id].profile_features for well_id in ids])
        return matrix, "profile"
    if family in {"block_ridge", "block_hgb", "negative_reversed_profile"}:
        matrix = np.vstack([records[well_id].block_features[controls] for well_id in ids])
        return matrix, "block"
    if family in {"typewell_block_ridge", "negative_shuffled_gr"}:
        per_well = [records[well_id].typewell_block_features[controls] for well_id in ids]
        if family == "negative_shuffled_gr":
            permutation = shifted_permutation(ids, f"T023-feature-shuffle|{salt}")
            per_well = [per_well[index] for index in permutation]
        return np.vstack(per_well), "block"
    if family == "disagreement_block_ridge":
        return np.vstack([records[well_id].disagreement_block_features[controls] for well_id in ids]), "block"
    raise DataValidationError(f"unknown T023 family {family}")


def prepare_features(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train, dtype=np.float64)
    z = np.asarray(test, dtype=np.float64)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1]:
        raise DataValidationError("T023 feature shapes differ")
    medians = np.zeros(x.shape[1], dtype=np.float64)
    for column in range(x.shape[1]):
        finite = x[np.isfinite(x[:, column]), column]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
    x = np.where(np.isfinite(x), x, medians)
    z = np.where(np.isfinite(z), z, medians)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    x = (x - means) / scales
    z = (z - means) / scales
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)):
        raise DataValidationError("T023 prepared features are non-finite")
    return x, z


def fit_predict(
    records: Mapping[str, WellSequence],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    branch: Mapping[str, Any],
    model_config: Mapping[str, Any],
    salt: str,
) -> tuple[np.ndarray, int | None]:
    controls = int(branch["controls"])
    family = str(branch["family"])
    target_mode = "normal"
    if family == "negative_shuffled_profile":
        target_mode = "shuffled_wells"
    elif family == "negative_reversed_profile":
        target_mode = "reversed_targets"
    y_well = target_matrix(records, train_ids, controls, target_mode, salt)
    train_features, feature_kind = feature_matrix(records, train_ids, branch, salt + "|train")
    test_features, test_kind = feature_matrix(records, test_ids, branch, salt + "|test")
    if feature_kind != test_kind:
        raise DataValidationError("T023 train/test feature kinds differ")
    if feature_kind == "profile":
        x_train, x_test = prepare_features(train_features, test_features)
        if family in {"profile_ridge", "negative_shuffled_profile"}:
            model = Ridge(alpha=float(model_config["alpha"]), fit_intercept=True)
            model.fit(x_train, y_well)
            prediction = np.asarray(model.predict(x_test), dtype=np.float64)
            effective = None
        elif family == "profile_knn":
            requested = int(model_config["neighbors"])
            effective = min(requested, len(train_ids))
            model = KNeighborsRegressor(n_neighbors=effective, weights="distance", metric="euclidean")
            model.fit(x_train, y_well)
            prediction = np.asarray(model.predict(x_test), dtype=np.float64)
        else:
            raise DataValidationError("invalid T023 profile family")
    else:
        y = y_well.reshape(-1)
        x_train, x_test = prepare_features(train_features, test_features)
        if family in {"block_ridge", "typewell_block_ridge", "disagreement_block_ridge", "negative_reversed_profile", "negative_shuffled_gr"}:
            model = Ridge(alpha=float(model_config["alpha"]), fit_intercept=True)
            model.fit(x_train, y)
            flat = np.asarray(model.predict(x_test), dtype=np.float64)
            effective = None
        elif family == "block_hgb":
            model = HistGradientBoostingRegressor(
                max_depth=int(model_config["max_depth"]),
                max_iter=int(model_config["max_iter"]),
                learning_rate=float(model_config["learning_rate"]),
                min_samples_leaf=int(model_config["min_samples_leaf"]),
                l2_regularization=1.0,
                random_state=23023,
            )
            model.fit(x_train, y)
            flat = np.asarray(model.predict(x_test), dtype=np.float64)
            effective = None
        else:
            raise DataValidationError("invalid T023 block family")
        prediction = flat.reshape(len(test_ids), controls)
    if prediction.shape != (len(test_ids), controls) or not np.all(np.isfinite(prediction)):
        raise DataValidationError("T023 model emitted invalid coefficient profiles")
    return prediction, effective


def model_configs(branch: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    family = str(branch["family"])
    if "knn" in family:
        return [{"neighbors": int(value)} for value in config["model_grids"]["knn_neighbors"]]
    if family == "block_hgb":
        return [dict(item) for item in config["model_grids"]["hgb"]]
    return [{"alpha": float(value)} for value in config["model_grids"]["ridge_alphas"]]


def apply_bounds(values: np.ndarray, shrinkage: float, cap: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64) * float(shrinkage)
    array = np.clip(array, -float(cap), float(cap))
    if not np.all(np.isfinite(array)) or np.max(np.abs(array)) > float(cap) + 1e-12:
        raise DataValidationError("T023 coefficient bound failed")
    return array


def metrics_for_predictions(records: Mapping[str, WellSequence], ids: Sequence[str], controls: int, predictions: np.ndarray) -> list[WellMetric]:
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != (len(ids), controls):
        raise DataValidationError("T023 metric prediction shape differs")
    return [records[well_id].statistics[controls].metric(well_id, values[index]) for index, well_id in enumerate(ids)]


def select_and_fit_context(
    context: Context,
    records: Mapping[str, WellSequence],
    branch: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, Selection]:
    controls = int(branch["controls"])
    configs = model_configs(branch, config)
    predictions_by_config: list[dict[str, np.ndarray]] = [dict() for _ in configs]
    neighbors_by_config: list[list[int]] = [[] for _ in configs]
    groups = sorted({int(context.inner_assignments[well_id]) for well_id in context.train_ids})
    for group in groups:
        validation = tuple(well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) == group)
        training = tuple(well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) != group)
        for config_index, model_config in enumerate(configs):
            predicted, effective = fit_predict(records, training, validation, branch, model_config, f"{context.key}|inner:{group}|cfg:{config_index}")
            predictions_by_config[config_index].update({well_id: predicted[index] for index, well_id in enumerate(validation)})
            if effective is not None:
                neighbors_by_config[config_index].append(effective)
    if any(set(item) != set(context.train_ids) for item in predictions_by_config):
        raise DataValidationError(f"{context.key}: T023 inner OOF coverage failed")
    scored: list[tuple[float, int, int, int, float, float, Mapping[str, Any], int | None]] = []
    shrinkages = [float(value) for value in config["model_grids"]["shrinkages"]]
    caps = [float(value) for value in config["model_grids"]["correction_caps_ft"]]
    train_ids = tuple(context.train_ids)
    for config_index, by_well in enumerate(predictions_by_config):
        raw = np.vstack([by_well[well_id] for well_id in train_ids])
        for shrink_index, shrinkage in enumerate(shrinkages):
            for cap_index, cap in enumerate(caps):
                bounded = apply_bounds(raw, shrinkage, cap)
                summary = _summarize(metrics_for_predictions(records, train_ids, controls, bounded))
                effective = min(neighbors_by_config[config_index]) if neighbors_by_config[config_index] else None
                scored.append((float(summary["rmse"]), config_index, shrink_index, cap_index, shrinkage, cap, configs[config_index], effective))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    inner_rmse, config_index, _shrink_index, _cap_index, shrinkage, cap, selected_config, effective = scored[0]
    raw_outer, outer_effective = fit_predict(records, context.train_ids, context.test_ids, branch, selected_config, f"{context.key}|outer|cfg:{config_index}")
    if outer_effective is not None:
        effective = outer_effective
    outer = apply_bounds(raw_outer, shrinkage, cap)
    return outer, Selection(dict(selected_config), shrinkage, cap, inner_rmse, effective)


def aggregate_metrics(metrics: Iterable[WellMetric]) -> dict[str, Any]:
    return dict(_summarize(list(metrics)))


def diagnostic_summary(records: Mapping[str, WellSequence]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in [f"acf_{lag}" for lag in ACF_LAGS] + ["low_frequency_energy_fraction"]:
        values = [float(record.diagnostics[name]) for record in records.values() if record.diagnostics.get(name) is not None and math.isfinite(float(record.diagnostics[name]))]
        output[name] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "p10": float(np.quantile(values, 0.1)) if values else None,
            "p50": float(np.quantile(values, 0.5)) if values else None,
            "p90": float(np.quantile(values, 0.9)) if values else None,
        }
    base = aggregate_metrics(record.base_metric for record in records.values())
    total_sse = float(base["sse"])
    datum = sum(record.base_metric.datum_sse for record in records.values())
    trend = sum(record.base_metric.trend_sse for record in records.values())
    shape = sum(record.base_metric.shape_sse for record in records.values())
    output["residual_decomposition"] = {
        "datum_sse_share": datum / total_sse,
        "trend_sse_share": trend / total_sse,
        "shape_sse_share": shape / total_sse,
    }
    oracle = {}
    for controls in CONTROL_COUNTS:
        summary = aggregate_metrics(record.statistics[controls].metric(record.well_id, record.statistics[controls].target_coefficients) for record in records.values())
        oracle[str(controls)] = {**summary, "gain_vs_e011": float(base["rmse"]) - float(summary["rmse"])}
    output["oracle_profiles"] = oracle
    return output


def edge_tests(records: Mapping[str, WellSequence], config: Mapping[str, Any]) -> dict[str, Any]:
    passed: list[str] = []
    for rows in (1, 2):
        basis = piecewise_basis(rows, 16)
        assert basis.shape == (rows, 16) and np.max(np.abs(basis[0])) == 0.0
    passed.append("one_two_row_suffix")
    md = np.asarray([1.0, 2.0, 3.0])
    tw_tvt = np.asarray([0.0, 1.0])
    tw_gr = np.asarray([10.0, 20.0])
    channels = sequence_channels(md, np.asarray([0.0, 0.1, 0.2]), np.asarray([np.nan, np.nan, np.nan]), np.asarray([2.0, 2.1, 2.2]), np.asarray([2.0, 2.1, 2.2]), 0.0, -0.1, 1.9, tw_tvt, tw_gr)
    assert np.all(np.isfinite(channels)) and np.all(channels[:, 8] == 1.0)
    passed.append("all_gr_missing")
    channels = sequence_channels(md, np.asarray([0.0, 0.1, 0.2]), np.asarray([5.0, 5.0, 5.0]), np.asarray([2.0, 2.1, 2.2]), np.asarray([2.0, 2.1, 2.2]), 0.0, -0.1, 1.9, tw_tvt, tw_gr)
    assert np.all(np.isfinite(channels))
    passed.append("constant_gr")
    outside = typewell_gr(tw_tvt, tw_gr, np.asarray([-100.0, 100.0]))
    assert np.array_equal(outside, np.asarray([10.0, 20.0]))
    passed.append("typewell_outside_range")
    grouped = pd.DataFrame({"TVT": [0.0, 0.0, 1.0], "GR": [10.0, 14.0, 20.0]}).groupby("TVT", as_index=False)["GR"].mean()
    assert np.allclose(grouped["GR"].to_numpy(), [12.0, 20.0])
    passed.append("duplicate_typewell_tvt")
    for bad in (np.asarray([1.0, 1.0, 2.0]), np.asarray([2.0, 1.0])):
        try:
            validate_md(bad)
        except DataValidationError:
            pass
        else:
            raise AssertionError("invalid MD accepted")
    passed.append("invalid_md_rejected")
    try:
        sequence_channels(md, np.zeros(3), np.zeros(3), np.asarray([1.0, np.nan, 2.0]), np.ones(3), 0.0, 0.0, 1.0, tw_tvt, tw_gr)
    except DataValidationError:
        pass
    else:
        raise AssertionError("non-finite base prediction accepted")
    passed.append("nonfinite_base_rejected")
    x, z = prepare_features(np.ones((4, 3)), np.ones((2, 3)))
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(z))
    passed.append("zero_variance_features")
    model = Ridge(alpha=1.0).fit(np.arange(6, dtype=float)[:, None], np.ones((6, 3)))
    assert np.all(np.isfinite(model.predict(np.asarray([[2.0]]))))
    passed.append("constant_target_profile")
    model = KNeighborsRegressor(n_neighbors=1, weights="distance").fit(np.asarray([[0.0]]), np.asarray([[1.0, 2.0]]))
    assert np.allclose(model.predict(np.asarray([[0.0]])), [[1.0, 2.0]])
    passed.append("tiny_knn_partition")
    bounded = apply_bounds(np.asarray([[100.0, -100.0]]), 1.0, 20.0)
    assert np.max(np.abs(bounded)) == 20.0 and np.max(np.abs(piecewise_basis(5, 2)[0] @ bounded[0])) == 0.0
    passed.append("cap_zero_boundary")
    assert len(records) == int(config["expected_wells"]) and sum(record.rows for record in records.values()) == int(config["expected_hidden_rows"])
    passed.append("real_coverage")
    return {"status": "PASS", "edge_groups": len(passed), "passed": passed}


def run_screen(root: Path, config: Mapping[str, Any], output_dir: Path, cache_path: Path, rebuild_cache: bool) -> dict[str, Any]:
    started = time.perf_counter()
    records, cache_meta = load_or_build_cache(root, config, cache_path, rebuild_cache)
    contexts = load_contexts(root, config, records)
    repeated = [context for context in contexts if context.scope == "repeated"]
    stress = [context for context in contexts if context.scope != "repeated"]
    branches = [dict(item) for item in config["branches"]]
    names = [branch["name"] for branch in branches]
    if len(names) != len(set(names)):
        raise DataValidationError("T023 branch names are not unique")
    ids = tuple(sorted(records))
    index = {well_id: position for position, well_id in enumerate(ids)}
    repeated_sums = {name: np.zeros((len(ids), int(next(branch["controls"] for branch in branches if branch["name"] == name))), dtype=np.float64) for name in names}
    repeated_counts = {name: np.zeros(len(ids), dtype=np.int16) for name in names}
    context_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    context_metrics: dict[tuple[str, str], dict[str, WellMetric]] = {}

    for context in repeated:
        base_by_well = {well_id: records[well_id].base_metric for well_id in context.test_ids}
        context_metrics[(context.key, "e011")] = base_by_well
        context_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "candidate": "e011", **aggregate_metrics(base_by_well.values())})
        for branch in branches:
            prediction, selection = select_and_fit_context(context, records, branch, config)
            name = str(branch["name"])
            controls = int(branch["controls"])
            by_well = {well_id: records[well_id].statistics[controls].metric(well_id, prediction[position]) for position, well_id in enumerate(context.test_ids)}
            context_metrics[(context.key, name)] = by_well
            context_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "candidate": name, **aggregate_metrics(by_well.values())})
            selection_rows.append({"context": context.key, "scope": context.scope, "candidate": name, "model_config_json": _canonical_json(selection.model_config), "shrinkage": selection.shrinkage, "cap": selection.cap, "inner_rmse": selection.inner_rmse, "effective_neighbors": selection.effective_neighbors if selection.effective_neighbors is not None else ""})
            for position, well_id in enumerate(context.test_ids):
                repeated_sums[name][index[well_id]] += prediction[position]
                repeated_counts[name][index[well_id]] += 1

    final_predictions: dict[str, dict[str, np.ndarray]] = {}
    final_metrics: dict[str, dict[str, WellMetric]] = {"e011": {well_id: records[well_id].base_metric for well_id in ids}}
    for branch in branches:
        name = str(branch["name"])
        if np.any(repeated_counts[name] != 5):
            raise DataValidationError(f"T023 repeated coverage differs for {name}")
        values = repeated_sums[name] / repeated_counts[name][:, None]
        final_predictions[name] = {well_id: values[index[well_id]] for well_id in ids}
        controls = int(branch["controls"])
        final_metrics[name] = {well_id: records[well_id].statistics[controls].metric(well_id, final_predictions[name][well_id]) for well_id in ids}
    summaries = {name: aggregate_metrics(by_well.values()) for name, by_well in final_metrics.items()}
    base_rmse = float(summaries["e011"]["rmse"])

    row_lookup = {(row["context"], row["candidate"]): row for row in context_rows}
    map_rows: list[dict[str, Any]] = []
    map_wins: dict[str, int] = {}
    cell_wins: dict[str, int] = {}
    for label in sorted({context.label for context in repeated}):
        keys = [context.key for context in repeated if context.label == label]
        for candidate in ["e011", *names]:
            metrics = [metric for key in keys for metric in context_metrics[(key, candidate)].values()]
            map_rows.append({"map": label, "candidate": candidate, **aggregate_metrics(metrics)})
    map_lookup = {(row["map"], row["candidate"]): row for row in map_rows}
    for name in names:
        map_wins[name] = sum(float(map_lookup[(label, "e011")]["rmse"]) - float(map_lookup[(label, name)]["rmse"]) > 0.0 for label in sorted({context.label for context in repeated}))
        cell_wins[name] = sum(float(row_lookup[(context.key, "e011")]["rmse"]) - float(row_lookup[(context.key, name)]["rmse"]) > 0.0 for context in repeated)

    preliminary = {}
    preliminary_passers: list[str] = []
    branch_by_name = {str(branch["name"]): branch for branch in branches}
    for name in names:
        is_negative = bool(branch_by_name[name].get("negative_control", False))
        gates = {
            "not_negative_control": not is_negative,
            "gain": base_rmse - float(summaries[name]["rmse"]) >= float(config["preliminary_advancement"]["minimum_gain_vs_e011"]),
            "maps": map_wins[name] >= int(config["preliminary_advancement"]["minimum_map_wins"]),
            "cells": cell_wins[name] >= int(config["preliminary_advancement"]["minimum_outer_cell_wins"]),
            "p90": float(summaries[name]["p90_well_rmse"]) - float(summaries["e011"]["p90_well_rmse"]) <= float(config["preliminary_advancement"]["maximum_p90_deterioration"]),
            "worst5": float(summaries[name]["worst_5pct_sse_share"]) - float(summaries["e011"]["worst_5pct_sse_share"]) <= float(config["preliminary_advancement"]["maximum_worst5_sse_share_increase"]),
        }
        preliminary[name] = gates
        if all(gates.values()):
            preliminary_passers.append(name)

    stress_rows: list[dict[str, Any]] = []
    stress_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for context in stress:
        base = aggregate_metrics(records[well_id].base_metric for well_id in context.test_ids)
        stress_rows.append({"context": context.key, "scope": context.scope, "outer_group": context.outer_group, "candidate": "e011", **base})
        stress_lookup[(context.key, "e011")] = base
        for name in preliminary_passers:
            branch = branch_by_name[name]
            prediction, selection = select_and_fit_context(context, records, branch, config)
            controls = int(branch["controls"])
            metrics = metrics_for_predictions(records, context.test_ids, controls, prediction)
            summary = aggregate_metrics(metrics)
            stress_rows.append({"context": context.key, "scope": context.scope, "outer_group": context.outer_group, "candidate": name, **summary})
            stress_lookup[(context.key, name)] = summary
            selection_rows.append({"context": context.key, "scope": context.scope, "candidate": name, "model_config_json": _canonical_json(selection.model_config), "shrinkage": selection.shrinkage, "cap": selection.cap, "inner_rmse": selection.inner_rmse, "effective_neighbors": selection.effective_neighbors if selection.effective_neighbors is not None else ""})

    rows_values = np.asarray([records[well_id].rows for well_id in ids], dtype=float)
    missing_values = np.asarray([records[well_id].hidden_gr_missing_fraction for well_id in ids], dtype=float)
    special_sets = {
        "long_suffix": [well_id for well_id in ids if records[well_id].rows >= float(np.quantile(rows_values, 0.8))],
        "high_gr_missingness": [well_id for well_id in ids if records[well_id].hidden_gr_missing_fraction >= float(np.quantile(missing_values, 0.8))],
        "e011_catastrophe": [well_id for well_id in ids if records[well_id].base_metric.rmse >= 12.0],
    }
    special_rows: list[dict[str, Any]] = []
    special_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for slice_name, slice_ids in special_sets.items():
        for candidate in ["e011", *names]:
            summary = aggregate_metrics(final_metrics[candidate][well_id] for well_id in slice_ids)
            row = {"slice": slice_name, "candidate": candidate, **summary}
            special_rows.append(row)
            special_lookup[(slice_name, candidate)] = summary

    negative_gains = {name: base_rmse - float(summaries[name]["rmse"]) for name in names if bool(branch_by_name[name].get("negative_control", False))}
    authorization = {}
    authorized: list[str] = []
    for name in preliminary_passers:
        spatial_positive = all(float(stress_lookup[(f"spatial:{group}", "e011")]["rmse"]) - float(stress_lookup[(f"spatial:{group}", name)]["rmse"]) > 0.0 for group in range(5))
        typewell_positive = all(float(stress_lookup[(f"typewell:{group}", "e011")]["rmse"]) - float(stress_lookup[(f"typewell:{group}", name)]["rmse"]) > 0.0 for group in range(5))
        special_positive = all(float(special_lookup[(slice_name, "e011")]["rmse"]) - float(special_lookup[(slice_name, name)]["rmse"]) > 0.0 for slice_name in special_sets)
        gates = {
            "gain": base_rmse - float(summaries[name]["rmse"]) >= float(config["authorization_gate_for_new_experiment"]["minimum_gain_vs_e011"]),
            "maps": map_wins[name] >= int(config["authorization_gate_for_new_experiment"]["minimum_map_wins"]),
            "cells": cell_wins[name] >= int(config["authorization_gate_for_new_experiment"]["minimum_outer_cell_wins"]),
            "p90": float(summaries[name]["p90_well_rmse"]) - float(summaries["e011"]["p90_well_rmse"]) <= float(config["authorization_gate_for_new_experiment"]["maximum_p90_deterioration"]),
            "worst5": float(summaries[name]["worst_5pct_sse_share"]) - float(summaries["e011"]["worst_5pct_sse_share"]) <= float(config["authorization_gate_for_new_experiment"]["maximum_worst5_sse_share_increase"]),
            "spatial": spatial_positive,
            "typewell": typewell_positive,
            "special_slices": special_positive,
            "negative_controls": max(negative_gains.values(), default=-math.inf) <= float(config["authorization_gate_for_new_experiment"]["maximum_negative_control_gain"]),
        }
        authorization[name] = gates
        if all(gates.values()):
            authorized.append(name)

    diagnostics = diagnostic_summary(records)
    headroom_pass = float(diagnostics["oracle_profiles"]["16"]["gain_vs_e011"]) >= float(config["diagnostics"]["minimum_b16_oracle_gain"])
    b32_increment = float(diagnostics["oracle_profiles"]["16"]["rmse"]) - float(diagnostics["oracle_profiles"]["32"]["rmse"])
    controls = {
        "source_commit": {"pass": config["source_commit"] == "f2a295606132a8a2f067c064823c71907d33c0bb", "value": config["source_commit"]},
        "coverage": {"pass": len(records) == int(config["expected_wells"]) and sum(record.rows for record in records.values()) == int(config["expected_hidden_rows"]), "wells": len(records), "rows": sum(record.rows for record in records.values())},
        "context_completion": {"pass": len(context_rows) == len(repeated) * (1 + len(branches)), "rows": len(context_rows), "expected": len(repeated) * (1 + len(branches))},
        "finite_metrics": {"pass": all(math.isfinite(float(value)) for summary in summaries.values() for value in (summary["rmse"], summary["sse"]))},
        "bounds": {"pass": all(np.max(np.abs(values)) <= 40.0 + 1e-12 for by_well in final_predictions.values() for values in by_well.values())},
        "zero_fallback": {"pass": max(abs(records[well_id].statistics[16].metric(well_id, np.zeros(16)).sse - records[well_id].base_metric.sse) for well_id in ids) <= 1e-8},
        "negative_controls": {"pass": max(negative_gains.values(), default=-math.inf) <= float(config["authorization_gate_for_new_experiment"]["maximum_negative_control_gain"]), "gains": negative_gains},
        "oracle_headroom": {"pass": headroom_pass, "b16_gain": diagnostics["oracle_profiles"]["16"]["gain_vs_e011"], "b32_incremental_gain": b32_increment},
    }
    edges = edge_tests(records, config)
    controls["edge_groups"] = {"pass": edges["status"] == "PASS", "count": edges["edge_groups"]}
    status = "worth_screen_authorize_experiment" if authorized and all(item["pass"] for item in controls.values()) else "worth_screen_reject"
    result = {
        "schema_version": 1,
        "task_id": "T023",
        "hypothesis_id": "H017",
        "source_commit": config["source_commit"],
        "status": status,
        "decision": "authorize_new_sequence_experiment" if status == "worth_screen_authorize_experiment" else "close_h017_without_new_experiment",
        "runtime_seconds": time.perf_counter() - started,
        "cache": cache_meta,
        "diagnostics": diagnostics,
        "candidate_summaries": {name: {**summaries[name], "gain_vs_e011": base_rmse - float(summaries[name]["rmse"]), "map_wins": map_wins[name], "cell_wins": cell_wins[name]} for name in names},
        "preliminary_gates": preliminary,
        "preliminary_passers": preliminary_passers,
        "authorization_gates": authorization,
        "authorized_candidates": authorized,
        "negative_control_gains": negative_gains,
        "special_slice_wells": {name: len(values) for name, values in special_sets.items()},
        "controls": controls,
        "edge_cases": edges,
        "context_rows": context_rows,
        "map_rows": map_rows,
        "stress_rows": stress_rows,
        "special_slice_rows": special_rows,
        "selections": selection_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screen_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("tracking/evidence/T023/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/agents/t023-sequence-screen-20260723/main"))
    parser.add_argument("--cache", type=Path, default=Path("scratch/agents/t023-sequence-screen-20260723/sequence_cache.pkl.gz"))
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    cache_path = args.cache if args.cache.is_absolute() else root / args.cache
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_screen(root, config, output_dir, cache_path, bool(args.rebuild_cache))
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"],
        "runtime_seconds": result["runtime_seconds"],
        "b16_oracle_gain": result["diagnostics"]["oracle_profiles"]["16"]["gain_vs_e011"],
        "preliminary_passers": result["preliminary_passers"],
        "authorized_candidates": result["authorized_candidates"],
        "best_candidate": min(result["candidate_summaries"], key=lambda name: result["candidate_summaries"][name]["rmse"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
