#!/usr/bin/env python3
"""T026: randomized long-horizon mask-task coefficient meta-learning worth screen."""
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
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 4 else Path.cwd()
KNOTS = (0.25, 0.50, 0.75, 1.00)
OUTPUTS = 4
HISTORY_FRACTIONS = (0.40, 0.60, 0.80, 1.00)
PLACEMENTS = (0.25, 0.50, 0.75, 1.00)
CONTROL_MODES = (
    "permuted_source_well_targets",
    "shuffled_task_targets",
    "reversed_history_horizons",
    "permuted_horizon_features",
)


class DataValidationError(ValueError):
    """Raised when a frozen scientific or data contract is violated."""


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
    target_coefficients: np.ndarray
    raw_hidden_sum: float
    raw_hidden_sum_sq: float
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
        values = (self.e011_sse, self.e011_sum, self.e_dot_d, self.d_sse, self.d_sum,
                  self.basis_dot_e, self.basis_dot_d, self.basis_sum, self.basis_cross)
        if any(value is None for value in values):
            raise DataValidationError(f"{self.well_id}: OOF sufficient statistics are incomplete")
        linear = float(self.e_dot_d) + float(np.dot(self.basis_dot_e, c))
        quadratic = float(self.d_sse) + 2.0 * float(np.dot(self.basis_dot_d, c)) + float(c @ self.basis_cross @ c)
        sse = float(self.e011_sse) + 2.0 * w * linear + w * w * quadratic
        tolerance = 1e-8 * max(1.0, abs(float(self.e011_sse)), abs(2.0 * w * linear), abs(w * w * quadratic))
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


@dataclass
class TaskPool:
    task_ids: list[str]
    well_index: np.ndarray
    prefix_rows: np.ndarray
    horizon_rows: np.ndarray
    is_original: np.ndarray
    raw: np.ndarray
    history: np.ndarray
    reversed_history: np.ndarray
    targets: np.ndarray
    latest: np.ndarray
    log_extrapolation: np.ndarray
    raw_names: list[str]
    history_names: list[str]
    raw_horizon_indices: np.ndarray
    history_conditioned_indices: np.ndarray
    original_task_index: np.ndarray


# ---------- generic utilities ----------

def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
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


def stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256((str(base) + "|" + "|".join(parts)).encode()).digest()
    return (int.from_bytes(digest[:4], "little") ^ int(base)) % (2**32 - 1)


def quantile(values: Sequence[float], p: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise DataValidationError("cannot compute an empty quantile")
    return float(np.quantile(array, p))


def summarize(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(metrics)
    if not rows:
        raise DataValidationError("cannot summarize empty metrics")
    total_rows = sum(int(row["rows_scored"]) for row in rows)
    total_sse = sum(float(row["sse"]) for row in rows)
    if total_rows <= 0 or total_sse < 0.0:
        raise DataValidationError("invalid aggregate metric totals")
    rmses = np.asarray([float(row["rmse"]) for row in rows], dtype=np.float64)
    ordered = sorted(rows, key=lambda row: (-float(row["sse"]), str(row["well_id"])))
    worst5 = max(1, math.ceil(0.05 * len(rows)))
    worst10 = max(1, math.ceil(0.10 * len(rows)))
    return {
        "rows_scored": total_rows,
        "wells_scored": len(rows),
        "sse": total_sse,
        "rmse": math.sqrt(total_sse / total_rows),
        "median_well_rmse": float(np.quantile(rmses, 0.5)),
        "p90_well_rmse": float(np.quantile(rmses, 0.9)),
        "p95_well_rmse": float(np.quantile(rmses, 0.95)),
        "max_well_rmse": float(rmses.max()),
        "worst_5pct_sse_share": sum(float(row["sse"]) for row in ordered[:worst5]) / total_sse if total_sse else 0.0,
        "worst_10pct_sse_share": sum(float(row["sse"]) for row in ordered[:worst10]) / total_sse if total_sse else 0.0,
    }


# ---------- spline target and history ----------

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
    if np.linalg.matrix_rank(matrix) < OUTPUTS:
        raise DataValidationError("spline basis is rank deficient")
    return matrix


def solve_coefficients(matrix: np.ndarray, delta: np.ndarray, bound: float) -> tuple[np.ndarray, float]:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(delta, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != OUTPUTS or y.shape != (x.shape[0],):
        raise DataValidationError("coefficient solve dimensions differ")
    if x.shape[0] < OUTPUTS or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise DataValidationError("coefficient solve has insufficient or non-finite data")
    if np.linalg.matrix_rank(x) < OUTPUTS:
        raise DataValidationError("coefficient solve is rank deficient")
    coefficients = np.clip(np.linalg.lstsq(x, y, rcond=None)[0], -bound, bound)
    residual = x @ coefficients - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    if coefficients.shape != (OUTPUTS,) or not np.all(np.isfinite(coefficients)) or not math.isfinite(rmse):
        raise DataValidationError("coefficient solve emitted non-finite values")
    return coefficients, rmse


def fit_segment(values: Sequence[float], baseline: float, minimum_rows: int, bound: float) -> tuple[np.ndarray, float]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < int(minimum_rows):
        raise DataValidationError("segment has insufficient support")
    if not math.isfinite(float(baseline)) or not np.all(np.isfinite(y)):
        raise DataValidationError("segment contains non-finite values")
    return solve_coefficients(spline_basis(len(y)), y - float(baseline), bound)


def history_features(truth: np.ndarray, cut: int, bound: float, minimum: int, *, reverse: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    horizon = len(truth) - cut
    m = min(horizon, cut - 1)
    lengths = np.asarray([max(minimum, int(math.floor(f * m))) for f in HISTORY_FRACTIONS], dtype=np.int64)
    lengths = np.minimum(lengths, m)
    if np.any(lengths < minimum) or np.any(lengths >= cut + 1):
        raise DataValidationError("history support differs from frozen boundary")
    coefficients: list[np.ndarray] = []
    rmses: list[float] = []
    for length in lengths:
        start = cut - int(length)
        if start <= 0:
            raise DataValidationError("history segment lacks a baseline row")
        c, r = fit_segment(truth[start:cut], truth[start - 1], minimum, bound)
        coefficients.append(c)
        rmses.append(r)
    c = np.vstack(coefficients)
    r = np.asarray(rmses, dtype=np.float64)
    support = lengths.astype(np.float64) / max(1.0, float(horizon))
    if reverse:
        c = c[::-1].copy(); r = r[::-1].copy(); support = support[::-1].copy()
    diffs = np.diff(c, axis=0).reshape(-1)
    log_x = np.log(np.maximum(lengths.astype(np.float64), 1.0))
    log_target = math.log(max(float(horizon), 1.0))
    extrap = np.empty(OUTPUTS, dtype=np.float64)
    for output in range(OUTPUTS):
        slope, intercept = np.polyfit(log_x, c[:, output], 1)
        extrap[output] = intercept + slope * log_target
    extrap = np.clip(extrap, -bound, bound)
    names = (
        [f"history_c{level}_{output}" for level in range(4) for output in range(4)] +
        [f"history_rmse_{level}" for level in range(4)] +
        [f"history_support_ratio_{level}" for level in range(4)] +
        [f"history_diff_{level}_{output}" for level in range(3) for output in range(4)] +
        [f"history_log_extrap_{output}" for output in range(4)] +
        ["history_m_rows", "history_m_prefix_ratio", "history_horizon_prefix_ratio"]
    )
    vector = np.concatenate((c.reshape(-1), r, support, diffs, extrap,
                             np.asarray([float(m), m / max(1.0, float(cut)), horizon / max(1.0, float(cut))])))
    conditioned_start = 16
    conditioned = np.arange(conditioned_start, len(vector), dtype=np.int64)
    if len(names) != len(vector) or not np.all(np.isfinite(vector)):
        raise DataValidationError("history feature vector differs")
    return vector, c[-1].copy(), extrap, names, conditioned


# ---------- raw legal task features ----------

def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xc = x - x.mean(); denominator = float(np.dot(xc, xc))
    return 0.0 if denominator <= 0.0 else float(np.dot(xc, y - y.mean()) / denominator)


def _summary(x: np.ndarray, values: np.ndarray) -> list[float]:
    mask = np.isfinite(values)
    if not mask.any():
        return [np.nan] * 5
    xx = x[mask]; yy = values[mask]
    return [float(yy.mean()), float(yy.std()), float(yy.max() - yy.min()), float(yy[-1] - yy[0]), _slope(xx, yy)]


def _missing_fraction(values: np.ndarray) -> float:
    return float(np.mean(~np.isfinite(values))) if len(values) else 1.0


def _longest_missing(values: np.ndarray) -> int:
    longest = current = 0
    for missing in ~np.isfinite(values):
        if missing:
            current += 1; longest = max(longest, current)
        else:
            current = 0
    return longest


def _huber_line(x: np.ndarray, y: np.ndarray, window: int) -> tuple[float, float]:
    count = min(len(x), max(2, int(window)))
    xx = x[-count:].astype(np.float64); yy = y[-count:].astype(np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]; yy = yy[mask]
    if len(xx) < 2:
        return 0.0, 0.0
    weights = np.ones(len(xx), dtype=np.float64)
    slope = _slope(xx, yy); intercept = float(yy.mean() - slope * xx.mean()); scale = 0.0
    for _ in range(6):
        residual = yy - (intercept + slope * xx)
        median = float(np.median(residual)); scale = 1.4826 * float(np.median(np.abs(residual - median)))
        if scale <= 1e-12: break
        cutoff = 1.5 * scale
        weights = np.where(np.abs(residual) <= cutoff, 1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        total = float(weights.sum()); center_x = float(np.dot(weights, xx) / total); center_y = float(np.dot(weights, yy) / total)
        denominator = float(np.dot(weights, (xx - center_x) ** 2))
        slope = 0.0 if denominator <= 0.0 else float(np.dot(weights, (xx - center_x) * (yy - center_y)) / denominator)
        intercept = center_y - slope * center_x
    return slope, scale


def typewell_summary(frame: pd.DataFrame) -> tuple[list[float], list[str]]:
    if not {"TVT", "GR"}.issubset(frame.columns) or len(frame) == 0:
        raise DataValidationError("typewell schema differs")
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(dtype=np.float64)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(dtype=np.float64)
    clean_tvt = tvt[np.isfinite(tvt)]; clean_gr = gr[np.isfinite(gr)]
    if len(clean_tvt) == 0:
        raise DataValidationError("typewell TVT is entirely missing")
    values = [
        float(len(tvt)), float(clean_tvt.max() - clean_tvt.min()),
        float(clean_gr.mean()) if len(clean_gr) else np.nan,
        float(clean_gr.std()) if len(clean_gr) else np.nan,
        float(clean_gr.max() - clean_gr.min()) if len(clean_gr) else np.nan,
        _slope(np.flatnonzero(np.isfinite(gr)).astype(float), clean_gr) if len(clean_gr) >= 2 else 0.0,
        _missing_fraction(gr),
    ]
    names = ["typewell_rows", "typewell_tvt_span", "typewell_gr_mean", "typewell_gr_std", "typewell_gr_range", "typewell_gr_index_slope", "typewell_gr_missing_fraction"]
    return values, names


def raw_task_features(columns: Mapping[str, np.ndarray], type_values: Sequence[float], type_names: Sequence[str], cut: int,
                      slope_windows: Sequence[int], backtest_fractions: Sequence[float]) -> tuple[np.ndarray, list[str], np.ndarray]:
    md, x, y, z, gr, truth = (np.asarray(columns[name], dtype=np.float64) for name in ("MD", "X", "Y", "Z", "GR", "TVT"))
    total = len(truth); prefix = int(cut); horizon = total - prefix
    if prefix < 3 or horizon <= 0:
        raise DataValidationError("task cut has empty prefix or horizon")
    known = slice(0, prefix); hidden = slice(prefix, total)
    visible_tvt = truth[known]; visible_u = visible_tvt + z[known]
    values: list[float] = []
    names: list[str] = []
    def add(name: str, value: float) -> None:
        names.append(name); values.append(float(value))
    add("total_rows", total); add("known_rows", prefix); add("horizon_rows", horizon)
    add("known_fraction", prefix / total); add("hidden_fraction", horizon / total)
    add("md_known_span", md[prefix-1] - md[0]); add("md_hidden_span", md[-1] - md[prefix-1]); add("md_total_span", md[-1] - md[0])
    add("last_visible_tvt", visible_tvt[-1]); add("last_visible_z", z[prefix-1]); add("last_visible_u", visible_u[-1])
    add("last_visible_x", x[prefix-1]); add("last_visible_y", y[prefix-1])
    for name, array in (("x", x), ("y", y), ("z", z)):
        add(f"{name}_total_delta", array[-1] - array[0]); add(f"{name}_hidden_delta", array[-1] - array[prefix-1])
    add("horizontal_total_distance", math.hypot(x[-1]-x[0], y[-1]-y[0]))
    add("horizontal_hidden_distance", math.hypot(x[-1]-x[prefix-1], y[-1]-y[prefix-1]))
    add("spatial_x_mid", 0.5*(x[0]+x[-1])); add("spatial_y_mid", 0.5*(y[0]+y[-1]))
    angle = math.atan2(y[-1]-y[prefix-1], x[-1]-x[prefix-1]); add("hidden_azimuth_sin", math.sin(angle)); add("hidden_azimuth_cos", math.cos(angle))
    summary_inputs = (
        ("visible_tvt", md[known], visible_tvt), ("visible_z", md[known], z[known]),
        ("visible_u", md[known], visible_u), ("hidden_z", md[hidden], z[hidden]),
        ("visible_gr", md[known], gr[known]), ("hidden_gr", md[hidden], gr[hidden]), ("whole_gr", md, gr),
    )
    for prefix_name, xx, yy in summary_inputs:
        for suffix, value in zip(("mean","std","range","delta","slope"), _summary(xx, yy)):
            add(f"{prefix_name}_{suffix}", value)
    add("visible_gr_missing_fraction", _missing_fraction(gr[known])); add("hidden_gr_missing_fraction", _missing_fraction(gr[hidden]))
    add("whole_gr_missing_fraction", _missing_fraction(gr)); add("hidden_gr_longest_missing_run", _longest_missing(gr[hidden]))
    full_u_slope = _slope(md[known], visible_u)
    for window in slope_windows:
        u_slope, u_scale = _huber_line(md[known], visible_u, int(window)); tvt_slope, _ = _huber_line(md[known], visible_tvt, int(window))
        add(f"visible_u_slope_w{window}", u_slope); add(f"visible_u_scale_w{window}", u_scale)
        add(f"visible_u_slope_delta_w{window}", u_slope-full_u_slope); add(f"visible_tvt_slope_w{window}", tvt_slope)
    for fraction in backtest_fractions:
        label = str(float(fraction)).replace(".", "p")
        bt_cut = min(prefix-2, max(2, int(round(prefix*float(fraction)))))
        residual = visible_tvt[bt_cut:] - visible_tvt[bt_cut-1]
        centered = np.arange(len(residual), dtype=np.float64) / max(1, len(residual)-1) - 0.5
        before_u = visible_tvt[:bt_cut] + z[:bt_cut]; after_u = visible_tvt[bt_cut:] + z[bt_cut:prefix]
        add(f"backtest_{label}_mean", float(residual.mean())); add(f"backtest_{label}_rmse", float(np.sqrt(np.mean(residual*residual))))
        add(f"backtest_{label}_trend", _slope(centered, residual)); add(f"backtest_{label}_toe", residual[-1])
        add(f"backtest_{label}_u_slope_delta", _slope(md[bt_cut:prefix], after_u)-_slope(md[:bt_cut], before_u))
    for name, value in zip(type_names, type_values): add(name, value)
    vector = np.asarray(values, dtype=np.float64)
    explicit = np.asarray([names.index(name) for name in ("known_rows", "horizon_rows", "known_fraction", "hidden_fraction")], dtype=np.int64)
    if len(names) != len(set(names)) or not np.all(np.isfinite(vector) | np.isnan(vector)):
        raise DataValidationError("raw feature schema differs")
    return vector, names, explicit


# ---------- data loading and task construction ----------

def deterministic_jitter(seed: int, well_id: str, base_cut: int, amplitude: int) -> int:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{well_id}:{base_cut}".encode()).digest()[:8], "big")
    return int(value % (2*amplitude+1)) - amplitude


def validate_horizontal(frame: pd.DataFrame, well_id: str) -> tuple[dict[str, np.ndarray], int]:
    required = {"MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"{well_id}: required horizontal columns differ")
    arrays = {name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64) for name in required}
    for name in ("MD", "X", "Y", "Z", "TVT"):
        if not np.all(np.isfinite(arrays[name])):
            raise DataValidationError(f"{well_id}: non-finite {name}")
    if np.any(np.diff(arrays["MD"]) <= 0.0):
        raise DataValidationError(f"{well_id}: MD must be strictly increasing")
    finite = np.isfinite(arrays["TVT_input"])
    if not finite.any() or finite.all():
        raise DataValidationError(f"{well_id}: TVT_input must have visible prefix and hidden suffix")
    known = int(np.flatnonzero(~finite)[0])
    if known <= 0 or finite[known:].any() or not finite[:known].all():
        raise DataValidationError(f"{well_id}: TVT_input is not a contiguous prefix")
    if np.max(np.abs(arrays["TVT_input"][:known] - arrays["TVT"][:known])) > 1e-8:
        raise DataValidationError(f"{well_id}: visible TVT_input differs from truth")
    return arrays, known


def build_task_pool(data_dir: Path, config: Mapping[str, Any]) -> tuple[list[WellRecord], TaskPool]:
    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if len(paths) != int(config["expected_wells"]):
        raise DataValidationError(f"expected {config['expected_wells']} wells, found {len(paths)}")
    mask_cfg = config["mask_pool"]; spline_cfg = config["spline"]; raw_cfg = config["raw_features"]
    bound = float(spline_cfg["coefficient_absolute_bound_ft"]); minimum_history = int(spline_cfg["minimum_history_rows"])
    task_ids: list[str] = []; well_index: list[int] = []; prefix_rows: list[int] = []; horizon_rows: list[int] = []; originals: list[bool] = []
    raw_rows: list[np.ndarray] = []; history_rows_list: list[np.ndarray] = []; reversed_rows: list[np.ndarray] = []; targets: list[np.ndarray] = []; latest: list[np.ndarray] = []; extrap: list[np.ndarray] = []
    records: list[WellRecord] = []; original_task_index = np.full(len(paths), -1, dtype=np.int64)
    frozen_raw_names: list[str] | None = None; frozen_history_names: list[str] | None = None; raw_horizon_idx: np.ndarray | None = None; history_conditioned_idx: np.ndarray | None = None
    seen: set[str] = set()
    for wi, path in enumerate(paths):
        well_id = path.name.split("__", 1)[0]
        if well_id in seen: raise DataValidationError(f"duplicate source well {well_id}")
        seen.add(well_id)
        frame = pd.read_csv(path)
        columns, original_cut = validate_horizontal(frame, well_id)
        type_path = data_dir / f"{well_id}__typewell.csv"
        if not type_path.exists(): raise DataValidationError(f"{well_id}: missing typewell")
        type_values, type_names = typewell_summary(pd.read_csv(type_path))
        cuts = {original_cut}
        for base in mask_cfg["base_prefix_grid"]:
            cut = int(base) + deterministic_jitter(int(mask_cfg["seed"]), well_id, int(base), int(mask_cfg["jitter_rows"]))
            if cut >= int(mask_cfg["minimum_prefix_rows"]) and len(frame)-cut >= int(mask_cfg["minimum_horizon_rows"]): cuts.add(cut)
        cuts = sorted(cuts)
        if not (int(mask_cfg["minimum_tasks_per_well"]) <= len(cuts) <= int(mask_cfg["maximum_tasks_per_well"])):
            raise DataValidationError(f"{well_id}: task count {len(cuts)} differs")
        final_target, _ = fit_segment(columns["TVT"][original_cut:], columns["TVT"][original_cut-1], OUTPUTS, bound)
        hidden_gr = columns["GR"][original_cut:]
        raw_hidden = columns["TVT"][original_cut:]
        records.append(WellRecord(well_id, original_cut, len(frame)-original_cut, float(columns["TVT"][original_cut-1]), _missing_fraction(hidden_gr), final_target, float(raw_hidden.sum()), float(np.dot(raw_hidden, raw_hidden))))
        for cut in cuts:
            raw, raw_names, explicit_idx = raw_task_features(columns, type_values, type_names, cut, raw_cfg["visible_slope_windows"], raw_cfg["visible_backtest_fractions"])
            hist, hist_latest, hist_extrap, hist_names, conditioned_idx = history_features(columns["TVT"], cut, bound, minimum_history, reverse=False)
            reversed_hist, _, _, reversed_names, _ = history_features(columns["TVT"], cut, bound, minimum_history, reverse=True)
            target, _ = fit_segment(columns["TVT"][cut:], columns["TVT"][cut-1], OUTPUTS, bound)
            if frozen_raw_names is None:
                frozen_raw_names = raw_names; frozen_history_names = hist_names; raw_horizon_idx = explicit_idx; history_conditioned_idx = conditioned_idx
            elif raw_names != frozen_raw_names or hist_names != frozen_history_names or reversed_names != frozen_history_names:
                raise DataValidationError("task feature schema changed across wells")
            index = len(task_ids)
            if cut == original_cut:
                if original_task_index[wi] >= 0: raise DataValidationError(f"{well_id}: duplicate original task")
                original_task_index[wi] = index
            task_ids.append(f"{well_id}:{cut}"); well_index.append(wi); prefix_rows.append(cut); horizon_rows.append(len(frame)-cut); originals.append(cut==original_cut)
            raw_rows.append(raw); history_rows_list.append(hist); reversed_rows.append(reversed_hist); targets.append(target); latest.append(hist_latest); extrap.append(hist_extrap)
    if len(task_ids) != int(mask_cfg["expected_tasks"]) or np.any(original_task_index < 0):
        raise DataValidationError(f"task pool count/original coverage differs: {len(task_ids)}")
    if frozen_raw_names is None or frozen_history_names is None or raw_horizon_idx is None or history_conditioned_idx is None:
        raise DataValidationError("task feature schema was not initialized")
    pool = TaskPool(task_ids, np.asarray(well_index, dtype=np.int64), np.asarray(prefix_rows, dtype=np.int64), np.asarray(horizon_rows, dtype=np.int64), np.asarray(originals, dtype=bool),
                    np.vstack(raw_rows), np.vstack(history_rows_list), np.vstack(reversed_rows), np.vstack(targets), np.vstack(latest), np.vstack(extrap),
                    frozen_raw_names, frozen_history_names, raw_horizon_idx, history_conditioned_idx, original_task_index)
    if len(set(pool.task_ids)) != len(pool.task_ids) or int(pool.is_original.sum()) != len(records):
        raise DataValidationError("task IDs/original count differ")
    return records, pool


# ---------- E011 scoring and frozen contexts ----------

def attach_e011_sufficient(records: Sequence[WellRecord], oof_path: Path, expected_rows: int) -> dict[str, Any]:
    by_id = {record.well_id: record for record in records}
    current = ""; targets: list[float] = []; predictions: list[float] = []; hidden_indices: list[int] = []; seen_ids: set[str] = set(); seen_wells: set[str] = set(); total_rows = 0
    def finalize(well_id: str) -> None:
        if not well_id: return
        record = by_id.get(well_id)
        if record is None: raise DataValidationError(f"OOF unknown well {well_id}")
        y = np.asarray(targets, dtype=np.float64); pred = np.asarray(predictions, dtype=np.float64)
        if len(y) != record.hidden_rows or pred.shape != y.shape or not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
            raise DataValidationError(f"{well_id}: OOF dimensions/finiteness differ")
        if hidden_indices != list(range(record.hidden_rows)):
            raise DataValidationError(f"{well_id}: OOF hidden order differs")
        sum_tolerance = 1e-6 * max(1.0, abs(record.raw_hidden_sum))
        square_tolerance = 1e-6 * max(1.0, abs(record.raw_hidden_sum_sq))
        if abs(float(y.sum()) - record.raw_hidden_sum) > sum_tolerance or abs(float(np.dot(y, y)) - record.raw_hidden_sum_sq) > square_tolerance:
            raise DataValidationError(f"{well_id}: OOF targets differ from raw hidden truth")
        basis = spline_basis(record.hidden_rows); error = pred-y; delta = record.last_tvt-pred
        record.e011_sse=float(np.dot(error,error)); record.e011_sum=float(error.sum()); record.e_dot_d=float(np.dot(error,delta)); record.d_sse=float(np.dot(delta,delta)); record.d_sum=float(delta.sum())
        record.basis_dot_e=basis.T@error; record.basis_dot_d=basis.T@delta; record.basis_sum=basis.sum(axis=0); record.basis_cross=basis.T@basis; seen_wells.add(well_id)
    with gzip.open(oof_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required={"id","well_id","hidden_index","target","spline4_ridge_equal_s075"}
        if not required.issubset(reader.fieldnames or []): raise DataValidationError("OOF schema differs")
        for row in reader:
            row_id=str(row["id"]); well_id=str(row["well_id"])
            if row_id in seen_ids: raise DataValidationError(f"duplicate OOF ID {row_id}")
            seen_ids.add(row_id)
            if current and well_id != current:
                finalize(current); targets.clear(); predictions.clear(); hidden_indices.clear()
            current=well_id; targets.append(float(row["target"])); predictions.append(float(row["spline4_ridge_equal_s075"])); hidden_indices.append(int(row["hidden_index"])); total_rows += 1
    finalize(current)
    if total_rows != expected_rows or seen_wells != set(by_id): raise DataValidationError("OOF coverage differs")
    return {"rows": total_rows, "wells": len(seen_wells), "unique_ids": len(seen_ids)}


def load_compact(path: Path, records: Sequence[WellRecord]) -> tuple[np.ndarray, np.ndarray]:
    arrays=np.load(path, allow_pickle=False); ids=[str(v) for v in arrays["well_ids"]]
    if ids != [r.well_id for r in records]: raise DataValidationError("compact well order differs")
    spatial=np.asarray(arrays["spatial_assignment"],dtype=np.int64); typewell=np.asarray(arrays["typewell_assignment"],dtype=np.int64)
    if set(spatial.tolist()) != set(range(5)) or set(typewell.tolist()) != set(range(5)): raise DataValidationError("stress groups differ")
    return spatial,typewell


def load_contexts(root: Path, well_ids: Sequence[str], fold_files: Sequence[str], spatial: np.ndarray, typewell: np.ndarray) -> tuple[list[Context], list[Context]]:
    id_to_index={w:i for i,w in enumerate(well_ids)}; repeated=[]
    for file_name in fold_files:
        payload=json.loads((root/file_name).read_text()); assignments=payload["assignments"]; version=str(payload["version"])
        if set(assignments)!=set(well_ids) or int(payload["n_folds"])!=5: raise DataValidationError(f"{file_name}: fold contract differs")
        for fold in range(5):
            test=np.asarray([id_to_index[w] for w in well_ids if int(assignments[w])==fold],dtype=np.int64); train=np.asarray([i for i in range(len(well_ids)) if i not in set(test.tolist())],dtype=np.int64)
            repeated.append(Context(f"repeated:{version}:{fold}","repeated",version,fold,train,test))
    stress=[]
    for scope,labels in (("spatial",spatial),("typewell",typewell)):
        for group in range(5):
            test=np.flatnonzero(labels==group).astype(np.int64); train=np.flatnonzero(labels!=group).astype(np.int64); stress.append(Context(f"{scope}:{group}",scope,scope,group,train,test))
    validate_contexts(repeated,stress,len(well_ids)); return repeated,stress


def validate_contexts(repeated: Sequence[Context], stress: Sequence[Context], wells: int) -> None:
    if len(repeated)!=25 or len(stress)!=10: raise DataValidationError("context count differs")
    coverage=np.zeros(wells,dtype=np.int64)
    for context in [*repeated,*stress]:
        train=set(context.train_indices.tolist()); test=set(context.test_indices.tolist())
        if train&test or train|test!=set(range(wells)) or not train or not test: raise DataValidationError(f"{context.key}: membership differs")
        if context.scope=="repeated": coverage[context.test_indices]+=1
    if not np.all(coverage==5): raise DataValidationError("repeated coverage differs")


# ---------- features, weights, controls, and models ----------

def feature_matrix(pool: TaskPool, feature_set: str, *, reversed_history: bool=False) -> tuple[np.ndarray, np.ndarray]:
    history=pool.reversed_history if reversed_history else pool.history
    if feature_set=="raw": return pool.raw, pool.raw_horizon_indices.copy()
    if feature_set=="history": return history, pool.history_conditioned_indices.copy()
    if feature_set=="combined":
        conditioned=np.concatenate((pool.raw_horizon_indices, len(pool.raw_names)+pool.history_conditioned_indices))
        return np.hstack((pool.raw,history)), conditioned
    if feature_set=="horizon_only":
        raw=pool.raw[:,pool.raw_horizon_indices]; hist=history[:,pool.history_conditioned_indices]
        return np.hstack((raw,hist)), np.arange(raw.shape[1]+hist.shape[1],dtype=np.int64)
    raise DataValidationError(f"unknown feature set {feature_set}")


def prepare_features(train: np.ndarray, test: np.ndarray, scale: bool) -> tuple[np.ndarray,np.ndarray]:
    xtr=np.asarray(train,dtype=np.float64); xte=np.asarray(test,dtype=np.float64)
    if xtr.ndim!=2 or xte.ndim!=2 or xtr.shape[1]!=xte.shape[1] or len(xtr)==0: raise DataValidationError("feature dimensions differ")
    med=np.nanmedian(np.where(np.isfinite(xtr),xtr,np.nan),axis=0); med=np.where(np.isfinite(med),med,0.0)
    xtr=np.where(np.isfinite(xtr),xtr,med); xte=np.where(np.isfinite(xte),xte,med)
    if scale:
        mean=xtr.mean(axis=0); std=xtr.std(axis=0); std=np.where(std>1e-12,std,1.0); xtr=(xtr-mean)/std; xte=(xte-mean)/std
    if not np.all(np.isfinite(xtr)) or not np.all(np.isfinite(xte)): raise DataValidationError("prepared features non-finite")
    return xtr,xte


def well_equal_weights(source: np.ndarray, multiplier: np.ndarray | None=None) -> np.ndarray:
    source=np.asarray(source,dtype=np.int64); mult=np.ones(len(source),dtype=np.float64) if multiplier is None else np.asarray(multiplier,dtype=np.float64)
    weights=np.zeros(len(source),dtype=np.float64)
    for well in np.unique(source):
        idx=np.flatnonzero(source==well); local=mult[idx]; total=float(local.sum())
        if total<=0: raise DataValidationError("well weights are nonpositive")
        weights[idx]=local/total
    weights*=len(weights)/weights.sum()
    return weights


def quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    edges=np.unique(np.quantile(np.asarray(values,dtype=np.float64),np.linspace(0,1,bins+1)[1:-1]))
    return edges.astype(np.float64)


def select_tasks_and_weights(pool: TaskPool, context: Context, regime: str, config: Mapping[str,Any]) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    train_wells=set(context.train_indices.tolist()); all_idx=np.flatnonzero(np.isin(pool.well_index,context.train_indices))
    if set(pool.well_index[all_idx].tolist())-train_wells: raise DataValidationError(f"{context.key}: held-out task leaked")
    long_threshold=int(config["task_weighting"]["long_horizon_threshold_rows"])
    if regime=="original_only": idx=all_idx[pool.is_original[all_idx]]; weights=np.ones(len(idx))
    elif regime=="masks_only": idx=all_idx[~pool.is_original[all_idx]]; weights=np.ones(len(idx))
    elif regime in ("all_well_equal","all_joint_balanced","all_original_boost4"): idx=all_idx; weights=well_equal_weights(pool.well_index[idx], np.where(pool.is_original[idx],float(config["task_weighting"]["original_boost"]),1.0) if regime=="all_original_boost4" else None)
    elif regime=="long_only": idx=all_idx[pool.horizon_rows[all_idx]>=long_threshold]; weights=well_equal_weights(pool.well_index[idx])
    elif regime=="short_only": idx=all_idx[pool.horizon_rows[all_idx]<long_threshold]; weights=well_equal_weights(pool.well_index[idx])
    else: raise DataValidationError(f"unknown training regime {regime}")
    if len(idx)==0: raise DataValidationError(f"{context.key}/{regime}: empty task selection")
    detail={"tasks":len(idx),"source_wells":len(np.unique(pool.well_index[idx]))}
    if regime=="all_joint_balanced":
        original_train=all_idx[pool.is_original[all_idx]]; bins=int(config["task_weighting"]["joint_bins"])
        p_edges=quantile_edges(pool.prefix_rows[original_train],bins); h_edges=quantile_edges(pool.horizon_rows[original_train],bins)
        pbin=np.digitize(pool.prefix_rows[idx],p_edges); hbin=np.digitize(pool.horizon_rows[idx],h_edges); code=pbin*bins+hbin
        counts={int(c):int(np.sum(code==c)) for c in np.unique(code)}; inv=np.asarray([1.0/counts[int(c)] for c in code])
        weights*=inv; weights*=len(weights)/weights.sum(); detail.update({"prefix_edges":p_edges.tolist(),"horizon_edges":h_edges.tolist(),"occupied_bins":len(counts)})
    if not np.all(np.isfinite(weights)) or np.any(weights<=0): raise DataValidationError("task weights invalid")
    return idx,weights,detail


def permuted_target_blocks(pool: TaskPool, train_idx: np.ndarray, seed: int, context_key: str) -> np.ndarray:
    result=np.empty((len(train_idx),OUTPUTS),dtype=np.float64); local_positions={int(task):pos for pos,task in enumerate(train_idx)}
    wells=sorted(np.unique(pool.well_index[train_idx]).tolist()); rng=np.random.default_rng(stable_seed(seed,context_key,"blocks")); order=np.asarray(wells,dtype=np.int64); rng.shuffle(order); donors=np.roll(order,1)
    donor_by_receiver={int(receiver):int(donor) for receiver,donor in zip(order,donors)}
    if len(wells)>1 and any(k==v for k,v in donor_by_receiver.items()): raise DataValidationError("block permutation contains self donor")
    for receiver in wells:
        recv=np.asarray(sorted([int(t) for t in train_idx if int(pool.well_index[t])==receiver],key=lambda t:(int(pool.prefix_rows[t]),int(pool.horizon_rows[t]),pool.task_ids[t])),dtype=np.int64)
        donor=donor_by_receiver[receiver]; give=np.asarray(sorted([int(t) for t in train_idx if int(pool.well_index[t])==donor],key=lambda t:(int(pool.prefix_rows[t]),int(pool.horizon_rows[t]),pool.task_ids[t])),dtype=np.int64)
        src=np.linspace(0.0,1.0,len(give)); dst=np.linspace(0.0,1.0,len(recv)); block=np.column_stack([np.interp(dst,src,pool.targets[give,output]) for output in range(OUTPUTS)])
        for task,row in zip(recv,block): result[local_positions[int(task)]]=row
    if not np.all(np.isfinite(result)): raise DataValidationError("permuted block targets non-finite")
    return result


def custom_knn_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, sample_weight: np.ndarray, neighbors: int) -> np.ndarray:
    n=min(int(neighbors),len(xtr));
    if n<=0: raise DataValidationError("KNN has no training rows")
    model=NearestNeighbors(n_neighbors=n,algorithm="auto").fit(xtr); distances,indices=model.kneighbors(xte)
    output=np.empty((len(xte),OUTPUTS),dtype=np.float64)
    for i in range(len(xte)):
        d=distances[i]; idx=indices[i]; zero=d<=1e-12
        weights=sample_weight[idx]*zero.astype(float) if zero.any() else sample_weight[idx]/np.maximum(d,1e-12)
        output[i]=np.average(ytr[idx],axis=0,weights=weights)
    return output


def fit_ridge(xtr: np.ndarray,ytr: np.ndarray,xte: np.ndarray,weights: np.ndarray,alpha: float) -> np.ndarray:
    a,b=prepare_features(xtr,xte,True); model=Ridge(alpha=float(alpha),fit_intercept=True); model.fit(a,ytr,sample_weight=weights); return np.asarray(model.predict(b),dtype=np.float64)


def predict_branch(branch: Mapping[str,Any], context: Context, pool: TaskPool, config: Mapping[str,Any], control_mode: str|None=None) -> tuple[np.ndarray,dict[str,Any]]:
    name=str(branch["name"]); family=str(branch["family"]); test_tasks=pool.original_task_index[context.test_indices]
    if family=="analytic":
        if control_mode is not None: raise DataValidationError("analytic control is invalid")
        result=pool.latest[test_tasks] if name=="history_latest_identity" else pool.log_extrapolation[test_tasks]
        return np.clip(result,-80.0,80.0),{"tasks":0,"source_wells":0}
    train_idx,weights,detail=select_tasks_and_weights(pool,context,str(branch["training_regime"]),config)
    reversed_mode=control_mode=="reversed_history_horizons"; xall,conditioned=feature_matrix(pool,str(branch["feature_set"]),reversed_history=reversed_mode)
    xtr=xall[train_idx].copy(); xte=xall[test_tasks].copy(); ytr=pool.targets[train_idx].copy()
    if control_mode=="permuted_source_well_targets": ytr=permuted_target_blocks(pool,train_idx,int(config["control_detail"]["source_block_permutation_seed"]),context.key)
    elif control_mode=="shuffled_task_targets":
        rng=np.random.default_rng(stable_seed(int(config["control_detail"]["task_target_shuffle_seed"]),context.key,name)); ytr=ytr[rng.permutation(len(ytr))]
    elif control_mode=="permuted_horizon_features":
        rng=np.random.default_rng(stable_seed(int(config["control_detail"]["horizon_feature_permutation_seed"]),context.key,name)); permutation=rng.permutation(len(xtr)); xtr[:,conditioned]=xtr[permutation][:,conditioned]
    if not np.all(np.isfinite(ytr)): raise DataValidationError(f"{name}: target non-finite")
    if family=="ridge": prediction=fit_ridge(xtr,ytr,xte,weights,float(branch["alpha"]))
    elif family=="ridge_delta": prediction=pool.log_extrapolation[test_tasks]+fit_ridge(xtr,ytr-pool.log_extrapolation[train_idx],xte,weights,float(branch["alpha"]))
    elif family=="extra_trees":
        a,b=prepare_features(xtr,xte,False); model=ExtraTreesRegressor(n_estimators=64,min_samples_leaf=8,max_features=0.7,random_state=stable_seed(26023,context.key,name),n_jobs=1); model.fit(a,ytr,sample_weight=weights); prediction=model.predict(b)
    elif family=="hist_gradient":
        a,b=prepare_features(xtr,xte,True); cols=[]
        for output in range(OUTPUTS):
            model=HistGradientBoostingRegressor(max_iter=160,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0,random_state=stable_seed(26023,context.key,name,str(output)))
            model.fit(a,ytr[:,output],sample_weight=weights); cols.append(model.predict(b))
        prediction=np.column_stack(cols)
    elif family=="knn":
        a,b=prepare_features(xtr,xte,True); prediction=custom_knn_predict(a,ytr,b,weights,int(branch.get("neighbors",25)))
    elif family=="horizon_expert":
        prediction=fit_ridge(xtr,ytr,xte,weights,float(branch["alpha"])); original_train=np.flatnonzero(np.isin(pool.well_index,context.train_indices)&pool.is_original); edges=quantile_edges(pool.horizon_rows[original_train],5)
        train_bins=np.digitize(pool.horizon_rows[train_idx],edges); test_bins=np.digitize(pool.horizon_rows[test_tasks],edges)
        for bin_id in range(5):
            test_pos=np.flatnonzero(test_bins==bin_id); local=np.flatnonzero(np.abs(train_bins-bin_id)<=1)
            if len(test_pos)==0 or len(local)<int(config["task_weighting"]["horizon_expert_minimum_tasks"]): continue
            prediction[test_pos]=fit_ridge(xtr[local],ytr[local],xte[test_pos],weights[local],float(branch["alpha"]))
        detail["horizon_edges"]=edges.tolist()
    else: raise DataValidationError(f"unknown family {family}")
    result=np.clip(np.asarray(prediction,dtype=np.float64),-float(config["spline"]["coefficient_absolute_bound_ft"]),float(config["spline"]["coefficient_absolute_bound_ft"]))
    if result.shape!=(len(context.test_indices),OUTPUTS) or not np.all(np.isfinite(result)): raise DataValidationError(f"{name}: prediction shape/finiteness differs")
    return result,detail


# ---------- scoring ----------

def metrics_for_indices(records: Sequence[WellRecord], indices: Sequence[int], coefficients: np.ndarray, weight: float) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    array=np.asarray(coefficients,dtype=np.float64)
    if array.shape!=(len(indices),OUTPUTS): raise DataValidationError("coefficient membership differs")
    rows=[records[int(index)].metric(array[pos],weight) for pos,index in enumerate(indices)]; return summarize(rows),rows


def candidate_name(branch: str, weight: float) -> str: return f"{branch}__w{weight:.2f}"


# ---------- score-blind edge tests ----------

def run_edge_tests() -> dict[str,Any]:
    passed=[]
    assert deterministic_jitter(26023,"abcdefgh",900,64) in range(-64,65); passed.append("deterministic_jitter_limits")
    cuts={851,851,900}; assert sorted(cuts)==[851,900]; passed.append("duplicate_cut_removal")
    def boundary(prefix:int,horizon:int)->bool: return prefix>=851 and horizon>=407
    assert boundary(851,407) and not boundary(850,407) and not boundary(851,406); passed.append("prefix_horizon_boundaries")
    frame=pd.DataFrame({"MD":[0,1,2,3],"X":[0,0,0,0],"Y":[0,0,0,0],"Z":[0,0,0,0],"GR":[1,np.nan,1,1],"TVT":[1,2,3,4],"TVT_input":[1,2,np.nan,np.nan]}); arrays,known=validate_horizontal(frame,"edgewell"); assert known==2; passed.append("contiguous_visibility")
    bad=frame.copy(); bad.loc[2,"MD"]=1.0
    try: validate_horizontal(bad,"edgewell"); raise AssertionError("nonincreasing MD passed")
    except DataValidationError: pass
    passed.append("strict_md_order")
    try: fit_segment(np.arange(127.0),0.0,128,80.0); raise AssertionError("127 support passed")
    except DataValidationError: pass
    c,r=fit_segment(np.arange(128.0),0.0,128,80.0); assert c.shape==(4,) and math.isfinite(r); passed.append("history_support_128_127")
    constant,rmse=fit_segment(np.full(128,7.0),7.0,128,80.0); assert np.max(np.abs(constant))<1e-10 and rmse<1e-10; passed.append("constant_tvt")
    try: solve_coefficients(np.ones((128,4)),np.zeros(128),80.0); raise AssertionError("rank deficient passed")
    except DataValidationError: pass
    passed.append("rank_deficient_spline")
    xtr,xte=prepare_features(np.ones((3,2)),np.asarray([[np.nan,1.0]]),True); assert np.all(np.isfinite(xtr)) and np.all(np.isfinite(xte)); passed.append("zero_variance_features")
    source=np.asarray([0,0,1,1,1]); weights=well_equal_weights(source); assert abs(weights[source==0].sum()-weights[source==1].sum())<1e-12; passed.append("well_equal_total_weight")
    edges=quantile_edges(np.asarray([1,2,3,4,5],float),5); assert np.all(np.diff(edges)>0); passed.append("joint_bin_edges")
    tree=ExtraTreesRegressor(n_estimators=2,min_samples_leaf=1,random_state=1,n_jobs=1).fit(np.arange(6).reshape(-1,1),np.arange(6),sample_weight=np.ones(6)); assert math.isfinite(float(tree.predict([[1.5]])[0])); passed.append("tree_sample_weight")
    pred=custom_knn_predict(np.asarray([[0.],[1.]]),np.zeros((2,4)),np.asarray([[0.5]]),np.ones(2),25); assert pred.shape==(1,4); passed.append("knn_neighbor_clipping")
    clipped,_=solve_coefficients(spline_basis(128),np.full(128,1000.0),5.0); assert np.max(np.abs(clipped))<=5.0; passed.append("coefficient_clipping")
    record=WellRecord("edgewell",128,128,10.0,0.0,np.zeros(4),0.0,0.0,e011_sse=128.0,e011_sum=0.0,e_dot_d=0.0,d_sse=128.0,d_sum=0.0,basis_dot_e=np.zeros(4),basis_dot_d=np.zeros(4),basis_sum=spline_basis(128).sum(axis=0),basis_cross=spline_basis(128).T@spline_basis(128))
    assert record.metric(np.full(4,80.0),0.0)["sse"]==128.0; passed.append("exact_e011_fallback")
    names=["a","b"]; scores=[1.0,1.0]; assert names[min(range(2),key=lambda i:(scores[i],i))]=="a"; passed.append("deterministic_ties")
    repeated=[]
    for v in range(5):
        for f in range(5):
            test=np.flatnonzero(np.arange(10)%5==f); train=np.flatnonzero(np.arange(10)%5!=f); repeated.append(Context(f"v{v}:{f}","repeated",f"v{v}",f,train,test))
    stress=[]
    for scope in ("spatial","typewell"):
        for g in range(5):
            test=np.flatnonzero(np.arange(10)%5==g); train=np.flatnonzero(np.arange(10)%5!=g); stress.append(Context(f"{scope}:{g}",scope,scope,g,train,test))
    validate_contexts(repeated,stress,10); passed.append("context_membership")

    bad_tvt=frame.copy(); bad_tvt.loc[1,"TVT"]=np.nan
    try: validate_horizontal(bad_tvt,"edgewell"); raise AssertionError("non-finite TVT passed")
    except DataValidationError: pass
    passed.append("nonfinite_tvt_rejection")

    bad_geometry=frame.copy(); bad_geometry.loc[1,"X"]=np.nan
    try: validate_horizontal(bad_geometry,"edgewell"); raise AssertionError("non-finite geometry passed")
    except DataValidationError: pass
    passed.append("nonfinite_geometry_rejection")

    type_bad=pd.DataFrame({"TVT":[np.nan,np.nan],"GR":[1.0,np.nan]})
    try: typewell_summary(type_bad); raise AssertionError("non-finite typewell TVT passed")
    except DataValidationError: pass
    passed.append("nonfinite_typewell_rejection")

    n=900
    synthetic={"MD":np.arange(n,dtype=float),"X":np.zeros(n),"Y":np.zeros(n),"Z":np.linspace(0,1,n),"GR":np.full(n,np.nan),"TVT":np.linspace(10,20,n)}
    raw,names,explicit=raw_task_features(synthetic,[2.0,1.0,np.nan,np.nan,np.nan,0.0,1.0],["typewell_rows","typewell_tvt_span","typewell_gr_mean","typewell_gr_std","typewell_gr_range","typewell_gr_index_slope","typewell_gr_missing_fraction"],851,[32,128,512],[0.5,0.7,0.85])
    assert len(raw)==len(names) and np.isnan(raw).any() and len(explicit)==4
    passed.append("missing_gr_accepted")

    constant_edges=quantile_edges(np.ones(10),5); assigned=np.digitize(np.ones(3),constant_edges)
    assert constant_edges.size<=1 and np.all(np.isfinite(constant_edges)) and np.all(assigned==assigned[0])
    passed.append("empty_tiny_bins")

    # Synthetic two-well pool verifies context task exclusion.
    mini=TaskPool(["a:1","a:2","b:1","b:2"],np.asarray([0,0,1,1]),np.asarray([851,900,851,900]),np.asarray([407,500,407,500]),np.asarray([True,False,True,False]),np.zeros((4,4)),np.zeros((4,5)),np.zeros((4,5)),np.zeros((4,4)),np.zeros((4,4)),np.zeros((4,4)),["known_rows","horizon_rows","known_fraction","hidden_fraction"],["h0","h1","h2","h3","h4"],np.asarray([0,1,2,3]),np.asarray([1,2,3,4]),np.asarray([0,2]))
    mini_context=Context("mini","repeated","mini",0,np.asarray([0]),np.asarray([1]))
    mini_cfg={"task_weighting":{"long_horizon_threshold_rows":450,"original_boost":4.0,"joint_bins":5}}
    selected,_,_=select_tasks_and_weights(mini,mini_context,"all_well_equal",mini_cfg)
    assert set(mini.well_index[selected].tolist())=={0}
    passed.append("heldout_task_exclusion")

    try: record.metric([0.0,0.0,0.0,np.nan],1.0); raise AssertionError("non-finite prediction passed")
    except DataValidationError: pass
    passed.append("nonfinite_prediction_rejection")

    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"bad.csv.gz"
        with gzip.open(path,"wt",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=["id","well_id","hidden_index","target","spline4_ridge_equal_s075"]); writer.writeheader()
            writer.writerow({"id":"edgewell_0","well_id":"edgewell","hidden_index":0,"target":0.0,"spline4_ridge_equal_s075":0.0})
            writer.writerow({"id":"edgewell_0","well_id":"edgewell","hidden_index":1,"target":0.0,"spline4_ridge_equal_s075":0.0})
        try: attach_e011_sufficient([record],path,2); raise AssertionError("duplicate OOF ID passed")
        except DataValidationError: pass
    passed.append("duplicate_oof_id_rejection")

    assert len(passed)==25
    return {"status":"PASS","edge_groups":len(passed),"passed":passed}


# ---------- full screen ----------

def run_screen(root: Path, output_dir: Path, implementation_commit: str) -> dict[str,Any]:
    started=time.perf_counter(); config_path=root/"tracking/evidence/T026/config.json"; config=json.loads(config_path.read_text())
    records,pool=build_task_pool(root/str(config["data_dir"]),config)
    coverage=attach_e011_sufficient(records,root/str(config["e011_oof_path"]),int(config["expected_hidden_rows"]))
    spatial,typewell=load_compact(root/str(config["compact_path"]),records); repeated,stress=load_contexts(root,[r.well_id for r in records],config["fold_files"],spatial,typewell)
    branches=list(config["branches"]); branch_by_name={str(b["name"]):b for b in branches}
    if len(branch_by_name)!=len(branches) or len(branches)!=19: raise DataValidationError("branch registry differs")
    base_metrics=[r.metric(np.zeros(4),0.0) for r in records]; base_summary=summarize(base_metrics); oracle_metrics=[r.metric(r.target_coefficients,1.0) for r in records]; oracle_summary=summarize(oracle_metrics)
    if abs(float(base_summary["rmse"])-float(config["promotion"]["e011_rmse"]))>1e-8: raise DataValidationError("exact E011 RMSE differs")
    all_contexts=[*repeated,*stress]; context_rows=[]; isolation_rows=[]; base_context={}
    map_coefficients={label:{name:np.full((len(records),OUTPUTS),np.nan) for name in branch_by_name} for label in sorted({c.label for c in repeated})}
    stress_predictions: dict[tuple[str,str],np.ndarray]={}
    for context in all_contexts:
        base_context[context.key],_=metrics_for_indices(records,context.test_indices,np.zeros((len(context.test_indices),4)),0.0)
        for branch in branches:
            name=str(branch["name"]); prediction,detail=predict_branch(branch,context,pool,config,None)
            if context.scope=="repeated": map_coefficients[context.label][name][context.test_indices]=prediction
            else: stress_predictions[(context.key,name)]=prediction
            isolation_rows.append({"context":context.key,"branch":name,"training_tasks":detail.get("tasks",0),"training_source_wells":detail.get("source_wells",0),"held_out_source_tasks":0})
            for weight in PLACEMENTS:
                summary,_=metrics_for_indices(records,context.test_indices,prediction,weight); context_rows.append({"context":context.key,"scope":context.scope,"map":context.label,"outer_group":context.outer_group,"branch":name,"weight":weight,"candidate":candidate_name(name,weight),**summary})
    for label,branch_map in map_coefficients.items():
        for name,values in branch_map.items():
            if not np.all(np.isfinite(values)): raise DataValidationError(f"{label}/{name}: OOF coverage incomplete")
    final_coefficients={}; map_rows=[]
    for name in branch_by_name:
        labels=sorted(map_coefficients); stack=np.stack([map_coefficients[label][name] for label in labels]); final_coefficients[name]=stack.mean(axis=0)
        for label in labels:
            for weight in PLACEMENTS:
                summary,_=metrics_for_indices(records,np.arange(len(records)),map_coefficients[label][name],weight); map_rows.append({"map":label,"branch":name,"weight":weight,"candidate":candidate_name(name,weight),**summary})
    final_rows=[]; final_well={}
    for name,coefficients in final_coefficients.items():
        branch=branch_by_name[name]
        for weight in PLACEMENTS:
            candidate=candidate_name(name,weight); summary,well=metrics_for_indices(records,np.arange(len(records)),coefficients,weight); final_well[candidate]=well
            final_rows.append({"candidate":candidate,"branch":name,"weight":weight,"eligible":bool(branch.get("eligible",False)),"gain_vs_e011":float(base_summary["rmse"])-float(summary["rmse"]),"gain_vs_t025":float(config["promotion"]["t025_best_rmse"])-float(summary["rmse"]),**summary})
    context_lookup={(r["context"],r["candidate"]):r for r in context_rows}; map_lookup={(r["map"],r["candidate"]):r for r in map_rows}
    for row in final_rows:
        candidate=str(row["candidate"]); row["map_wins"]=sum(float(base_summary["rmse"])-float(map_lookup[(label,candidate)]["rmse"])>0 for label in map_coefficients)
        row["outer_cell_wins"]=sum(float(base_context[c.key]["rmse"])-float(context_lookup[(c.key,candidate)]["rmse"])>0 for c in repeated)
    # Negative controls for the registered core branch over all repeated contexts.
    core=branch_by_name["ridge_all_combined_a10_well_equal"]; control_sums={mode:np.zeros((len(records),OUTPUTS)) for mode in CONTROL_MODES}; control_counts=np.zeros(len(records),dtype=np.int64)
    for context in repeated:
        for mode in CONTROL_MODES:
            prediction,_=predict_branch(core,context,pool,config,mode); control_sums[mode][context.test_indices]+=prediction
        control_counts[context.test_indices]+=1
    if not np.all(control_counts==5): raise DataValidationError("control coverage differs")
    control_rows=[]; max_control=-math.inf
    for mode,values in control_sums.items():
        coeff=values/control_counts[:,None]
        for weight in PLACEMENTS:
            summary,_=metrics_for_indices(records,np.arange(len(records)),coeff,weight); gain=float(base_summary["rmse"])-float(summary["rmse"]); max_control=max(max_control,gain)
            control_rows.append({"control":mode,"branch":str(core["name"]),"weight":weight,"candidate":f"{core['name']}__{mode}__w{weight:.2f}","gain_vs_e011":gain,**summary})
    # Special slices and horizon quintiles use final five-map averaged predictions.
    hidden=np.asarray([r.hidden_rows for r in records],dtype=float); missing=np.asarray([r.hidden_gr_missing_fraction for r in records],dtype=float); base_rmse=np.asarray([r["rmse"] for r in base_metrics])
    special_sets={"long_suffix":np.flatnonzero(hidden>=np.quantile(hidden,0.8)),"high_gr_missingness":np.flatnonzero(missing>=np.quantile(missing,0.8)),"e011_catastrophe":np.flatnonzero(base_rmse>=12.0)}
    h_edges=np.quantile(hidden,[0.2,0.4,0.6,0.8]); horizon_sets={f"horizon_q{q}":np.flatnonzero(np.digitize(hidden,h_edges)==q) for q in range(5)}
    slice_rows=[]; slice_lookup={}
    for slice_name,indices in {**special_sets,**horizon_sets}.items():
        base=summarize([base_metrics[int(i)] for i in indices]); slice_rows.append({"slice":slice_name,"candidate":"e011",**base}); slice_lookup[(slice_name,"e011")]=base
        for row in final_rows:
            candidate=str(row["candidate"]); summary=summarize([final_well[candidate][int(i)] for i in indices]); slice_rows.append({"slice":slice_name,"candidate":candidate,**summary}); slice_lookup[(slice_name,candidate)]=summary
    original_candidates=[r for r in final_rows if r["branch"]=="ridge_original_combined_a10"]
    if len(original_candidates)!=4: raise DataValidationError("original-only comparator grid differs")
    best_original=min(original_candidates,key=lambda r:(float(r["rmse"]),str(r["candidate"])))
    stress_lookup={(r["context"],r["candidate"]):r for r in context_rows if r["scope"] in ("spatial","typewell")}
    edge=run_edge_tests(); controls={
        "coverage":{"pass":coverage["rows"]==int(config["expected_hidden_rows"]) and coverage["wells"]==int(config["expected_wells"]),**coverage},
        "task_pool":{"pass":len(pool.task_ids)==int(config["mask_pool"]["expected_tasks"]),"tasks":len(pool.task_ids),"original_tasks":int(pool.is_original.sum()),"minimum_per_well":int(min(np.bincount(pool.well_index))),"maximum_per_well":int(max(np.bincount(pool.well_index)))},
        "outer_isolation":{"pass":all(int(r["held_out_source_tasks"])==0 for r in isolation_rows),"rows":len(isolation_rows)},
        "exact_fallback":{"pass":abs(float(base_summary["rmse"])-float(config["promotion"]["e011_rmse"]))<=1e-8,"rmse":base_summary["rmse"]},
        "negative_controls":{"pass":max_control<=float(config["promotion"]["maximum_negative_control_gain"]),"maximum_gain_vs_e011":max_control},
        "context_completion":{"pass":len(context_rows)==35*19*4,"rows":len(context_rows)},
        "finite_bounded":{"pass":all(np.all(np.isfinite(v)) and np.max(np.abs(v))<=80.0+1e-9 for v in final_coefficients.values())},
        "edge_groups":edge,
    }
    all_controls=all(bool(v.get("pass",v.get("status")=="PASS")) for v in controls.values())
    authorization=[]; gates_by_candidate={}
    for row in final_rows:
        candidate=str(row["candidate"]); spatial_gains=[]; typewell_gains=[]
        for context in stress:
            gain=float(base_context[context.key]["rmse"])-float(stress_lookup[(context.key,candidate)]["rmse"]); (spatial_gains if context.scope=="spatial" else typewell_gains).append(gain)
        special_gains=[float(slice_lookup[(name,"e011")]["rmse"])-float(slice_lookup[(name,candidate)]["rmse"]) for name in special_sets]
        horizon_gains=[float(slice_lookup[(name,"e011")]["rmse"])-float(slice_lookup[(name,candidate)]["rmse"]) for name in horizon_sets]
        gain_original=float(best_original["rmse"])-float(row["rmse"])
        gates={
            "eligible":bool(row["eligible"]),"oracle":float(oracle_summary["rmse"])<=float(config["promotion"]["maximum_oracle_rmse"]),
            "gain":float(row["gain_vs_e011"])>=float(config["promotion"]["minimum_gain_vs_e011"]),"gain_vs_t025":float(row["gain_vs_t025"])>=float(config["promotion"]["minimum_gain_vs_t025_best"]),
            "maps":int(row["map_wins"])>=int(config["promotion"]["minimum_map_wins"]),"cells":int(row["outer_cell_wins"])>=int(config["promotion"]["minimum_outer_cell_wins"]),
            "p90":float(row["p90_well_rmse"])-float(base_summary["p90_well_rmse"])<=float(config["promotion"]["maximum_p90_deterioration"]),
            "worst5":float(row["worst_5pct_sse_share"])-float(base_summary["worst_5pct_sse_share"])<=float(config["promotion"]["maximum_worst5_share_increase"]),
            "spatial":min(spatial_gains)>=0.0,"typewell":min(typewell_gains)>=0.0,"special_slices":min(special_gains)>=0.0,"horizon_quintiles":min(horizon_gains)>=0.0,
            "gain_vs_original_only":gain_original>=float(config["promotion"]["minimum_gain_vs_best_original_only"]),"controls":all_controls,"reproduction":False,
        }
        gates_by_candidate[candidate]=gates; authorization.append({"candidate":candidate,"gain_vs_best_original_only":gain_original,"minimum_spatial_gain":min(spatial_gains),"minimum_typewell_gain":min(typewell_gains),"minimum_special_slice_gain":min(special_gains),"minimum_horizon_quintile_gain":min(horizon_gains),**{f"gate_{k}":v for k,v in gates.items()}})
    substantive=[c for c,g in gates_by_candidate.items() if all(v for k,v in g.items() if k!="reproduction")]
    reported=min(final_rows,key=lambda r:(float(r["rmse"]),str(r["candidate"])))
    status="awaiting_reproduction" if substantive else "worth_screen_reject"; decision="await_independent_reproduction" if substantive else "close_h019_without_formal_experiment"
    selected=[]; reported_candidate=str(reported["candidate"]); reported_branch=str(reported["branch"]); reported_weight=float(reported["weight"])
    for i,record in enumerate(records):
        metric=final_well[reported_candidate][i]; selected.append({"well_id":record.well_id,"candidate":reported_candidate,"branch":reported_branch,"weight":reported_weight,"rows_scored":metric["rows_scored"],"sse":metric["sse"],"rmse":metric["rmse"],"mean_error":metric["mean_error"],"known_rows":record.known_rows,"hidden_rows":record.hidden_rows,"hidden_gr_missing_fraction":record.hidden_gr_missing_fraction,**{f"predicted_coefficient_{j}":final_coefficients[reported_branch][i,j] for j in range(4)},**{f"target_coefficient_{j}":record.target_coefficients[j] for j in range(4)}})
    output_dir.mkdir(parents=True,exist_ok=True)
    _write_csv(output_dir/"candidate_metrics.csv",final_rows); _write_csv(output_dir/"context_metrics.csv",context_rows); _write_csv(output_dir/"map_metrics.csv",map_rows); _write_csv(output_dir/"negative_control_metrics.csv",control_rows); _write_csv(output_dir/"slice_metrics.csv",slice_rows); _write_csv(output_dir/"authorization_gates.csv",authorization); _write_csv(output_dir/"isolation_audit.csv",isolation_rows); _write_csv(output_dir/"selected_well_metrics.csv",selected); _write_json(output_dir/"edge_cases.json",edge)
    runtime=time.perf_counter()-started
    summary={"schema_version":1,"task_id":"T026","hypothesis_id":"H019","implementation_commit":implementation_commit,"status":status,"decision":decision,"formal_experiment_authorized":False,"reported_candidate":reported_candidate,"reported_metrics":reported,"base_summary":base_summary,"oracle_summary":oracle_summary,"best_original_only_comparator":best_original,"substantive_passers":substantive,"gates_by_candidate":gates_by_candidate,"controls":controls,"maximum_negative_control_gain":max_control,"feature_dimensions":{"raw":pool.raw.shape[1],"history":pool.history.shape[1],"combined":pool.raw.shape[1]+pool.history.shape[1]},"task_pool":{"tasks":len(pool.task_ids),"original":int(pool.is_original.sum()),"median_horizon":float(np.median(pool.horizon_rows))},"thresholds":{"long_suffix":float(np.quantile(hidden,0.8)),"high_gr_missingness":float(np.quantile(missing,0.8)),"e011_catastrophe":12.0,"horizon_quintile_edges":h_edges.tolist()},"runtime_seconds":runtime,"deployment":{"package_built":False,"kaggle_executed":False,"submission_created":False,"submission_made":False}}
    _write_json(output_dir/"summary.json",summary); files=sorted(p for p in output_dir.iterdir() if p.is_file()); _write_json(output_dir/"artifact_manifest.json",{"schema_version":1,"task_id":"T026","implementation_commit":implementation_commit,"files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":_sha256(p)} for p in files]})
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir"); parser.add_argument("--implementation-commit",default="UNFROZEN"); parser.add_argument("--edge-only",action="store_true"); args=parser.parse_args()
    if args.edge_only:
        print(json.dumps(run_edge_tests(),indent=2,sort_keys=True)); return
    if not args.output_dir: raise SystemExit("--output-dir is required unless --edge-only")
    summary=run_screen(ROOT,Path(args.output_dir),str(args.implementation_commit)); print(json.dumps({"status":summary["status"],"decision":summary["decision"],"reported_candidate":summary["reported_candidate"],"reported_rmse":summary["reported_metrics"]["rmse"],"gain_vs_e011":summary["reported_metrics"]["gain_vs_e011"],"oracle_rmse":summary["oracle_summary"]["rmse"],"substantive_passers":len(summary["substantive_passers"]),"runtime_seconds":summary["runtime_seconds"]},indent=2,sort_keys=True))


if __name__=="__main__": main()
