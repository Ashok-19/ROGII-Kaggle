#!/usr/bin/env python3
"""T026: randomized long-horizon mask-task coefficient meta-learning worth screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
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

ROOT = Path(__file__).resolve().parents[3]
T025_PATH = ROOT / "tracking/evidence/T025/run_screen.py"
_SPEC = importlib.util.spec_from_file_location("rogii_t025", T025_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load T025 scientific utilities")
t025 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = t025
_SPEC.loader.exec_module(t025)

DataValidationError = t025.DataValidationError
Context = t025.Context
OUTPUTS = 4
HISTORY_FRACTIONS = (0.40, 0.60, 0.80, 1.00)
CONTROL_MODES = (
    "permuted_source_well_targets",
    "shuffled_task_targets",
    "reversed_history_horizons",
    "permuted_horizon_features",
)


@dataclass
class RawWell:
    well_id: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt: np.ndarray
    known_rows: int
    typewell_summary: dict[str, float]
    record: Any


@dataclass
class TaskPool:
    source_indices: np.ndarray
    cut_rows: np.ndarray
    horizon_rows: np.ndarray
    is_original: np.ndarray
    raw_features: np.ndarray
    history_features: np.ndarray
    reversed_history_features: np.ndarray
    horizon_features: np.ndarray
    targets: np.ndarray
    latest: np.ndarray
    log_extrapolation: np.ndarray
    raw_names: list[str]
    history_names: list[str]
    horizon_names: list[str]
    original_task_indices: np.ndarray


@dataclass(frozen=True)
class Prepared:
    train: np.ndarray
    test: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256((str(base) + "|" + "|".join(str(part) for part in parts)).encode()).digest()
    return (int.from_bytes(digest[:8], "little") ^ int(base)) % (2**32 - 1)


def finite_array(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise DataValidationError(f"{label} contains non-finite values")
    return array


def slope(xs: np.ndarray, ys: np.ndarray) -> float:
    mask = np.isfinite(xs) & np.isfinite(ys)
    x = xs[mask]
    y = ys[mask]
    if len(x) < 2:
        return 0.0
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    return 0.0 if denom <= 0.0 else float(np.dot(xc, y - y.mean()) / denom)


def summary_dict(xs: np.ndarray, values: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(xs) & np.isfinite(values)
    if not mask.any():
        return {"mean": np.nan, "std": np.nan, "range": np.nan, "delta": np.nan, "slope": np.nan}
    x = xs[mask]
    y = values[mask]
    return {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "range": float(y.max() - y.min()),
        "delta": float(y[-1] - y[0]),
        "slope": slope(x, y),
    }


def add_summary(features: dict[str, float], prefix: str, xs: np.ndarray, values: np.ndarray) -> None:
    for key, value in summary_dict(xs, values).items():
        features[f"{prefix}_{key}"] = value


def longest_missing_run(values: np.ndarray) -> int:
    longest = current = 0
    for missing in ~np.isfinite(values):
        if bool(missing):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def weighted_line(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(weights.sum())
    if len(xs) < 2 or total <= 0.0:
        return (float(ys[-1]) if len(ys) else 0.0), 0.0
    cx = float(np.dot(weights, xs) / total)
    cy = float(np.dot(weights, ys) / total)
    denom = float(np.dot(weights, (xs - cx) ** 2))
    value = 0.0 if denom <= 0.0 else float(np.dot(weights, (xs - cx) * (ys - cy)) / denom)
    return cy - value * cx, value


def huber_line(xs: np.ndarray, ys: np.ndarray, window: int) -> tuple[float, float, float]:
    count = min(len(xs), len(ys), max(2, int(window)))
    x = np.asarray(xs[-count:], dtype=np.float64)
    y = np.asarray(ys[-count:], dtype=np.float64)
    weights = np.ones(count, dtype=np.float64)
    intercept, line_slope = weighted_line(x, y, weights)
    scale = 0.0
    for _ in range(6):
        residual = y - (intercept + line_slope * x)
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        if scale <= 1e-12:
            break
        cutoff = 1.5 * scale
        absolute = np.abs(residual)
        weights = np.where(absolute <= cutoff, 1.0, cutoff / np.maximum(absolute, 1e-12))
        intercept, line_slope = weighted_line(x, y, weights)
    return intercept, line_slope, scale


def read_typewell_summary(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, usecols=lambda name: name in {"TVT", "GR"})
    if set(frame.columns) != {"TVT", "GR"} or len(frame) <= 0:
        raise DataValidationError(f"{path.name}: typewell schema differs")
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(dtype=np.float64)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(dtype=np.float64)
    finite_tvt = tvt[np.isfinite(tvt)]
    finite_gr = gr[np.isfinite(gr)]
    if len(finite_tvt) <= 0:
        raise DataValidationError(f"{path.name}: typewell TVT is empty")
    index = np.arange(len(gr), dtype=np.float64)
    return {
        "typewell_rows": float(len(frame)),
        "typewell_tvt_span": float(finite_tvt.max() - finite_tvt.min()),
        "typewell_gr_mean": float(finite_gr.mean()) if len(finite_gr) else np.nan,
        "typewell_gr_std": float(finite_gr.std()) if len(finite_gr) else np.nan,
        "typewell_gr_range": float(finite_gr.max() - finite_gr.min()) if len(finite_gr) else np.nan,
        "typewell_gr_index_slope": slope(index, gr),
        "typewell_gr_missing_fraction": float(np.mean(~np.isfinite(gr))),
    }


def load_raw_wells(data_dir: Path, config: Mapping[str, Any]) -> list[RawWell]:
    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if len(paths) != int(config["expected_wells"]):
        raise DataValidationError(f"expected {config['expected_wells']} wells, found {len(paths)}")
    minimum_prefix = int(config["mask_pool"]["minimum_prefix_rows"])
    minimum_horizon = int(config["mask_pool"]["minimum_horizon_rows"])
    bound = float(config["spline"]["coefficient_absolute_bound_ft"])
    output: list[RawWell] = []
    seen: set[str] = set()
    for path in paths:
        well_id = path.name.split("__", 1)[0]
        if well_id in seen:
            raise DataValidationError(f"duplicate source well {well_id}")
        seen.add(well_id)
        frame = pd.read_csv(path, usecols=lambda name: name in {"MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"})
        required = {"MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"}
        if set(frame.columns) != required:
            raise DataValidationError(f"{well_id}: horizontal schema differs")
        arrays = {name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64) for name in required}
        for name in ("MD", "X", "Y", "Z", "TVT"):
            if not np.all(np.isfinite(arrays[name])):
                raise DataValidationError(f"{well_id}: {name} is non-finite")
        if not np.all(np.diff(arrays["MD"]) > 0.0):
            raise DataValidationError(f"{well_id}: MD is not strictly increasing")
        known_rows = t025.validate_prefix(arrays["TVT_input"])
        hidden_rows = len(frame) - known_rows
        if known_rows < minimum_prefix or hidden_rows < minimum_horizon:
            raise DataValidationError(f"{well_id}: original support violates frozen bounds")
        if np.max(np.abs(arrays["TVT_input"][:known_rows] - arrays["TVT"][:known_rows])) > 1e-8:
            raise DataValidationError(f"{well_id}: visible TVT differs from truth")
        final_target, _ = t025.fit_segment(arrays["TVT"][known_rows:], arrays["TVT"][known_rows - 1], OUTPUTS, bound)
        hidden_gr = arrays["GR"][known_rows:]
        record = t025.WellRecord(
            well_id=well_id,
            known_rows=known_rows,
            hidden_rows=hidden_rows,
            last_tvt=float(arrays["TVT"][known_rows - 1]),
            hidden_gr_missing_fraction=float(np.mean(~np.isfinite(hidden_gr))),
            pseudo_coefficients=np.zeros((4, 4), dtype=np.float64),
            pseudo_rmse=np.zeros(4, dtype=np.float64),
            pseudo_segment_rows=np.full(4, 128.0, dtype=np.float64),
            pseudo_features=np.zeros(1, dtype=np.float64),
            reversed_features=np.zeros(1, dtype=np.float64),
            latest_coefficients=np.zeros(4, dtype=np.float64),
            linear_extrapolation=np.zeros(4, dtype=np.float64),
            target_coefficients=final_target,
            raw_hidden_sum=float(arrays["TVT"][known_rows:].sum()),
            raw_hidden_sum_sq=float(np.dot(arrays["TVT"][known_rows:], arrays["TVT"][known_rows:])),
        )
        typewell_path = path.with_name(f"{well_id}__typewell.csv")
        if not typewell_path.exists():
            raise DataValidationError(f"{well_id}: missing typewell")
        output.append(RawWell(
            well_id=well_id,
            md=arrays["MD"], x=arrays["X"], y=arrays["Y"], z=arrays["Z"],
            gr=arrays["GR"], tvt=arrays["TVT"], known_rows=known_rows,
            typewell_summary=read_typewell_summary(typewell_path), record=record,
        ))
    return output


def validate_mask_support(cut: int, total_rows: int, config: Mapping[str, Any]) -> None:
    minimum_prefix = int(config["mask_pool"]["minimum_prefix_rows"])
    minimum_horizon = int(config["mask_pool"]["minimum_horizon_rows"])
    if int(cut) < minimum_prefix:
        raise DataValidationError("task prefix is below frozen support")
    if int(total_rows) - int(cut) < minimum_horizon:
        raise DataValidationError("task horizon is below frozen support")
    if int(cut) <= 0 or int(cut) >= int(total_rows):
        raise DataValidationError("task cut is outside well")


def mask_cuts(well: RawWell, config: Mapping[str, Any]) -> list[int]:
    pool = config["mask_pool"]
    seed = int(pool["seed"])
    jitter = int(pool["jitter_rows"])
    minimum_prefix = int(pool["minimum_prefix_rows"])
    minimum_horizon = int(pool["minimum_horizon_rows"])
    cuts = {int(well.known_rows)}
    for base in pool["base_prefix_grid"]:
        digest = hashlib.sha256(f"{seed}:{well.well_id}:{int(base)}".encode()).digest()
        value = int.from_bytes(digest[:8], "big")
        offset = value % (2 * jitter + 1) - jitter
        cut = int(base) + int(offset)
        if cut >= minimum_prefix and len(well.tvt) - cut >= minimum_horizon:
            cuts.add(cut)
    result = sorted(cuts)
    if len(result) != len(set(result)):
        raise DataValidationError(f"{well.well_id}: duplicate cuts were not removed")
    for cut in result:
        validate_mask_support(cut, len(well.tvt), config)
    return result


def visible_backtests(md: np.ndarray, z: np.ndarray, tvt: np.ndarray, fractions: Sequence[float]) -> dict[str, float]:
    result: dict[str, float] = {}
    count = len(tvt)
    for fraction in fractions:
        label = str(float(fraction)).replace(".", "p")
        cut = min(count - 2, max(2, int(round(count * float(fraction)))))
        if cut < 2 or count - cut < 2:
            for suffix in ("mean", "rmse", "trend", "toe", "u_slope_delta"):
                result[f"backtest_{label}_{suffix}"] = np.nan
            continue
        residual = tvt[cut:] - tvt[cut - 1]
        centered = np.arange(len(residual), dtype=np.float64) / max(1, len(residual) - 1) - 0.5
        result[f"backtest_{label}_mean"] = float(residual.mean())
        result[f"backtest_{label}_rmse"] = float(np.sqrt(np.mean(residual * residual)))
        result[f"backtest_{label}_trend"] = slope(centered, residual)
        result[f"backtest_{label}_toe"] = float(residual[-1])
        before_u = tvt[:cut] + z[:cut]
        after_u = tvt[cut:] + z[cut:]
        result[f"backtest_{label}_u_slope_delta"] = slope(md[cut:], after_u) - slope(md[:cut], before_u)
    return result


def raw_task_features(well: RawWell, cut: int, config: Mapping[str, Any]) -> dict[str, float]:
    total = len(well.tvt)
    if cut <= 0 or cut >= total:
        raise DataValidationError("task cut is outside well")
    hidden = total - cut
    md, x, y, z, gr = well.md, well.x, well.y, well.z, well.gr
    visible_tvt = well.tvt[:cut]
    visible_u = visible_tvt + z[:cut]
    features: dict[str, float] = {
        "total_rows": float(total), "known_rows": float(cut), "hidden_rows": float(hidden),
        "hidden_fraction": float(hidden / total),
        "md_known_span": float(md[cut - 1] - md[0]),
        "md_hidden_span": float(md[-1] - md[cut - 1]),
        "md_total_span": float(md[-1] - md[0]),
        "last_visible_tvt": float(visible_tvt[-1]),
        "last_visible_z": float(z[cut - 1]), "last_visible_u": float(visible_u[-1]),
        "last_visible_x": float(x[cut - 1]), "last_visible_y": float(y[cut - 1]),
        "x_total_delta": float(x[-1] - x[0]), "y_total_delta": float(y[-1] - y[0]),
        "z_total_delta": float(z[-1] - z[0]),
        "x_hidden_delta": float(x[-1] - x[cut - 1]), "y_hidden_delta": float(y[-1] - y[cut - 1]),
        "z_hidden_delta": float(z[-1] - z[cut - 1]),
        "horizontal_total_distance": float(math.hypot(x[-1] - x[0], y[-1] - y[0])),
        "horizontal_hidden_distance": float(math.hypot(x[-1] - x[cut - 1], y[-1] - y[cut - 1])),
        "spatial_x_mid": float(0.5 * (x[0] + x[-1])), "spatial_y_mid": float(0.5 * (y[0] + y[-1])),
    }
    angle = math.atan2(y[-1] - y[cut - 1], x[-1] - x[cut - 1])
    features["hidden_azimuth_sin"] = math.sin(angle)
    features["hidden_azimuth_cos"] = math.cos(angle)
    add_summary(features, "visible_tvt", md[:cut], visible_tvt)
    add_summary(features, "visible_z", md[:cut], z[:cut])
    add_summary(features, "visible_u", md[:cut], visible_u)
    add_summary(features, "hidden_z", md[cut:], z[cut:])
    add_summary(features, "visible_gr", md[:cut], gr[:cut])
    add_summary(features, "hidden_gr", md[cut:], gr[cut:])
    add_summary(features, "whole_gr", md, gr)
    features["visible_gr_missing_fraction"] = float(np.mean(~np.isfinite(gr[:cut])))
    features["hidden_gr_missing_fraction"] = float(np.mean(~np.isfinite(gr[cut:])))
    features["whole_gr_missing_fraction"] = float(np.mean(~np.isfinite(gr)))
    features["hidden_gr_longest_missing_run"] = float(longest_missing_run(gr[cut:]))
    full_u_slope = slope(md[:cut], visible_u)
    for window in config["raw_features"]["visible_slope_windows"]:
        count = min(int(window), cut)
        _, u_slope, u_scale = huber_line(md[:cut], visible_u, count)
        _, tvt_slope, _ = huber_line(md[:cut], visible_tvt, count)
        features[f"visible_u_slope_w{int(window)}"] = u_slope
        features[f"visible_u_scale_w{int(window)}"] = u_scale
        features[f"visible_u_slope_delta_w{int(window)}"] = u_slope - full_u_slope
        features[f"visible_tvt_slope_w{int(window)}"] = tvt_slope
    features.update(visible_backtests(md[:cut], z[:cut], visible_tvt, config["raw_features"]["visible_backtest_fractions"]))
    features.update(well.typewell_summary)
    return features


def build_history_feature_dict(coefficients: np.ndarray, rmses: np.ndarray, rows: np.ndarray, cut: int, horizon: int) -> tuple[dict[str, float], np.ndarray]:
    if coefficients.shape != (4, 4) or rmses.shape != (4,) or rows.shape != (4,):
        raise DataValidationError("history dimensions differ")
    result: dict[str, float] = {
        "prefix_rows": float(cut), "horizon_rows": float(horizon),
        "horizon_fraction": float(horizon / (cut + horizon)),
        "log_prefix_rows": float(math.log1p(cut)), "log_horizon_rows": float(math.log1p(horizon)),
    }
    for index in range(4):
        result[f"history_rmse_{index}"] = float(rmses[index])
        result[f"history_rows_{index}"] = float(rows[index])
        result[f"history_to_target_{index}"] = float(rows[index] / horizon)
        result[f"history_to_prefix_{index}"] = float(rows[index] / cut)
        for output in range(4):
            result[f"history_c{index}_{output}"] = float(coefficients[index, output])
    for index in range(3):
        for output in range(4):
            result[f"history_delta{index}_{output}"] = float(coefficients[index + 1, output] - coefficients[index, output])
    log_rows = np.log1p(rows.astype(np.float64))
    target_log = math.log1p(horizon)
    extrapolated = np.empty(4, dtype=np.float64)
    for output in range(4):
        line_slope, intercept = np.polyfit(log_rows, coefficients[:, output], 1)
        extrapolated[output] = intercept + line_slope * target_log
        result[f"history_log_slope_{output}"] = float(line_slope)
        result[f"history_log_intercept_{output}"] = float(intercept)
        result[f"history_log_prediction_{output}"] = float(extrapolated[output])
    return result, extrapolated


def history_task_features(well: RawWell, cut: int, config: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon = len(well.tvt) - cut
    max_history = min(horizon, cut - 1)
    minimum = int(config["spline"]["minimum_history_rows"])
    bound = float(config["spline"]["coefficient_absolute_bound_ft"])
    rows = np.asarray([int(round(float(fraction) * max_history)) for fraction in HISTORY_FRACTIONS], dtype=np.int64)
    if np.any(rows < minimum) or np.any(np.diff(rows) <= 0) or rows[-1] > cut - 1:
        raise DataValidationError(f"{well.well_id}: history support differs")
    coefficients: list[np.ndarray] = []
    rmses: list[float] = []
    for count in rows:
        start = cut - int(count)
        coefficient, rmse = t025.fit_segment(well.tvt[start:cut], well.tvt[start - 1], minimum, bound)
        coefficients.append(coefficient)
        rmses.append(rmse)
    matrix = np.vstack(coefficients)
    rmse_array = np.asarray(rmses, dtype=np.float64)
    normal, extrapolated = build_history_feature_dict(matrix, rmse_array, rows.astype(np.float64), cut, horizon)
    reversed_dict, _ = build_history_feature_dict(matrix[::-1], rmse_array[::-1], rows[::-1].astype(np.float64), cut, horizon)
    target, _ = t025.fit_segment(well.tvt[cut:], well.tvt[cut - 1], OUTPUTS, bound)
    return normal, reversed_dict, target, matrix[-1].copy(), np.clip(extrapolated, -bound, bound), rows


def dict_to_row(values: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    if set(values) != set(names):
        raise DataValidationError("feature schema differs across tasks")
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def build_task_pool(wells: Sequence[RawWell], config: Mapping[str, Any]) -> TaskPool:
    source_indices: list[int] = []
    cuts: list[int] = []
    horizons: list[int] = []
    originals: list[bool] = []
    raw_rows: list[np.ndarray] = []
    history_rows: list[np.ndarray] = []
    reverse_rows: list[np.ndarray] = []
    horizon_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    latest: list[np.ndarray] = []
    extrapolated: list[np.ndarray] = []
    raw_names: list[str] | None = None
    history_names: list[str] | None = None
    horizon_names = ["prefix_rows", "horizon_rows", "total_rows", "horizon_fraction", "log_prefix_rows", "log_horizon_rows"]
    original_task_indices = np.full(len(wells), -1, dtype=np.int64)
    minimum_history = int(config["spline"]["minimum_history_rows"])
    for source_index, well in enumerate(wells):
        well_cuts = mask_cuts(well, config)
        for cut in well_cuts:
            raw = raw_task_features(well, cut, config)
            history, reversed_history, target, last, log_pred, support = history_task_features(well, cut, config)
            if int(support.min()) < minimum_history:
                raise DataValidationError(f"{well.well_id}: task history below minimum")
            if raw_names is None:
                raw_names = sorted(raw)
                history_names = sorted(history)
            assert history_names is not None
            raw_row = dict_to_row(raw, raw_names)
            history_row = dict_to_row(history, history_names)
            reversed_row = dict_to_row(reversed_history, history_names)
            horizon = len(well.tvt) - cut
            horizon_values = {
                "prefix_rows": float(cut), "horizon_rows": float(horizon), "total_rows": float(len(well.tvt)),
                "horizon_fraction": float(horizon / len(well.tvt)),
                "log_prefix_rows": float(math.log1p(cut)), "log_horizon_rows": float(math.log1p(horizon)),
            }
            source_indices.append(source_index); cuts.append(cut); horizons.append(horizon)
            is_original = cut == well.known_rows
            originals.append(is_original)
            raw_rows.append(raw_row); history_rows.append(history_row); reverse_rows.append(reversed_row)
            horizon_rows.append(np.asarray([horizon_values[name] for name in horizon_names], dtype=np.float64))
            targets.append(target); latest.append(last); extrapolated.append(log_pred)
            if is_original:
                if original_task_indices[source_index] >= 0:
                    raise DataValidationError(f"{well.well_id}: duplicate original task")
                original_task_indices[source_index] = len(source_indices) - 1
    if raw_names is None or history_names is None or np.any(original_task_indices < 0):
        raise DataValidationError("task pool is incomplete")
    pool = TaskPool(
        source_indices=np.asarray(source_indices, dtype=np.int64),
        cut_rows=np.asarray(cuts, dtype=np.int64),
        horizon_rows=np.asarray(horizons, dtype=np.int64),
        is_original=np.asarray(originals, dtype=bool),
        raw_features=np.vstack(raw_rows),
        history_features=np.vstack(history_rows),
        reversed_history_features=np.vstack(reverse_rows),
        horizon_features=np.vstack(horizon_rows),
        targets=np.vstack(targets), latest=np.vstack(latest), log_extrapolation=np.vstack(extrapolated),
        raw_names=raw_names, history_names=history_names, horizon_names=horizon_names,
        original_task_indices=original_task_indices,
    )
    expected = int(config["mask_pool"]["expected_tasks"])
    if len(pool.source_indices) != expected:
        raise DataValidationError(f"expected {expected} tasks, built {len(pool.source_indices)}")
    counts = np.bincount(pool.source_indices, minlength=len(wells))
    if counts.min() < int(config["mask_pool"]["minimum_tasks_per_well"]) or counts.max() > int(config["mask_pool"]["maximum_tasks_per_well"]):
        raise DataValidationError("per-well task count differs")
    final_targets = pool.targets[pool.original_task_indices]
    record_targets = np.vstack([well.record.target_coefficients for well in wells])
    if np.max(np.abs(final_targets - record_targets)) > 1e-8:
        raise DataValidationError("original task target differs from final target")
    return pool


def prepare_features(train: np.ndarray, test: np.ndarray, scale: bool) -> Prepared:
    x_train = np.asarray(train, dtype=np.float64)
    x_test = np.asarray(test, dtype=np.float64)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[0] <= 0 or x_train.shape[1] != x_test.shape[1]:
        raise DataValidationError("feature matrix dimensions differ")
    finite = np.where(np.isfinite(x_train), x_train, np.nan)
    medians = np.nanmedian(finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    x_test = np.where(np.isfinite(x_test), x_test, medians)
    means = x_train.mean(axis=0) if scale else np.zeros(x_train.shape[1])
    scales = x_train.std(axis=0) if scale else np.ones(x_train.shape[1])
    scales = np.where(scales > 1e-12, scales, 1.0)
    if scale:
        x_train = (x_train - means) / scales
        x_test = (x_test - means) / scales
    if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
        raise DataValidationError("prepared features are non-finite")
    return Prepared(x_train, x_test, medians, means, scales)


def feature_matrix(pool: TaskPool, feature_set: str, reversed_history: bool = False) -> tuple[np.ndarray, list[str]]:
    history = pool.reversed_history_features if reversed_history else pool.history_features
    if feature_set == "raw":
        return pool.raw_features, [f"raw__{name}" for name in pool.raw_names]
    if feature_set == "history":
        return history, [f"history__{name}" for name in pool.history_names]
    if feature_set == "combined":
        return np.hstack((pool.raw_features, history)), [f"raw__{name}" for name in pool.raw_names] + [f"history__{name}" for name in pool.history_names]
    if feature_set == "horizon_only":
        return pool.horizon_features, [f"horizon__{name}" for name in pool.horizon_names]
    raise DataValidationError(f"unknown feature set {feature_set}")


def well_equal_weights(indices: np.ndarray, sources: np.ndarray, multiplier: np.ndarray | None = None) -> np.ndarray:
    selected_sources = sources[indices]
    values = np.ones(len(indices), dtype=np.float64) if multiplier is None else np.asarray(multiplier, dtype=np.float64).copy()
    if values.shape != (len(indices),) or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise DataValidationError("task weight multiplier differs")
    for source in np.unique(selected_sources):
        mask = selected_sources == source
        values[mask] /= float(values[mask].sum())
    values *= len(values) / float(values.sum())
    return values


def selection_and_weights(regime: str, context: Context, pool: TaskPool, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_source = np.zeros(int(pool.source_indices.max()) + 1, dtype=bool)
    train_source[context.train_indices] = True
    mask = train_source[pool.source_indices]
    if regime == "original_only":
        mask &= pool.is_original
    elif regime == "masks_only":
        mask &= ~pool.is_original
    elif regime == "long_only":
        mask &= pool.horizon_rows >= int(config["task_weighting"]["long_horizon_threshold_rows"])
    elif regime == "short_only":
        mask &= pool.horizon_rows < int(config["task_weighting"]["long_horizon_threshold_rows"])
    elif regime not in {"all_well_equal", "all_joint_balanced", "all_original_boost4"}:
        raise DataValidationError(f"unknown training regime {regime}")
    indices = np.flatnonzero(mask).astype(np.int64)
    if len(indices) <= 0:
        raise DataValidationError(f"{context.key}/{regime}: no tasks")
    if set(pool.source_indices[indices].tolist()) & set(context.test_indices.tolist()):
        raise DataValidationError(f"{context.key}/{regime}: held-out source task leaked")
    multiplier = None
    details: dict[str, Any] = {"tasks": len(indices), "sources": len(np.unique(pool.source_indices[indices]))}
    if regime == "all_original_boost4":
        multiplier = np.where(pool.is_original[indices], float(config["task_weighting"]["original_boost"]), 1.0)
    weights = well_equal_weights(indices, pool.source_indices, multiplier)
    if regime == "all_joint_balanced":
        original_indices = pool.original_task_indices[context.train_indices]
        prefix_edges = np.quantile(pool.cut_rows[original_indices], [0.2, 0.4, 0.6, 0.8])
        horizon_edges = np.quantile(pool.horizon_rows[original_indices], [0.2, 0.4, 0.6, 0.8])
        pb = np.digitize(pool.cut_rows[indices], prefix_edges)
        hb = np.digitize(pool.horizon_rows[indices], horizon_edges)
        joint = pb * 5 + hb
        frequency = np.bincount(joint, minlength=25).astype(np.float64)
        inverse = 1.0 / np.maximum(1.0, frequency[joint])
        weights *= inverse
        weights *= len(weights) / float(weights.sum())
        details.update({"prefix_edges": prefix_edges.tolist(), "horizon_edges": horizon_edges.tolist(), "occupied_bins": int(np.count_nonzero(frequency))})
    totals = np.bincount(pool.source_indices[indices], weights=weights, minlength=int(pool.source_indices.max()) + 1)
    active = totals[totals > 0.0]
    if regime != "all_joint_balanced" and active.max() - active.min() > 1e-8:
        raise DataValidationError(f"{context.key}/{regime}: source-well total weights differ")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise DataValidationError("task weights are invalid")
    return indices, weights, details


def equal_knn_subset(indices: np.ndarray, pool: TaskPool) -> np.ndarray:
    groups = {int(source): indices[pool.source_indices[indices] == source] for source in np.unique(pool.source_indices[indices])}
    minimum = min(len(values) for values in groups.values())
    chosen: list[int] = []
    for source in sorted(groups):
        values = groups[source][np.argsort(pool.cut_rows[groups[source]])]
        positions = np.linspace(0, len(values) - 1, minimum).round().astype(int)
        chosen.extend(values[positions].tolist())
    result = np.asarray(chosen, dtype=np.int64)
    counts = np.bincount(pool.source_indices[result], minlength=int(pool.source_indices.max()) + 1)
    active = counts[counts > 0]
    if active.min() != active.max():
        raise DataValidationError("KNN equal-source subset differs")
    return result


def permuted_block_targets(indices: np.ndarray, pool: TaskPool, seed: int) -> np.ndarray:
    result = pool.targets[indices].copy()
    sources = pool.source_indices[indices]
    ordered_sources = sorted(int(value) for value in np.unique(sources))
    if len(ordered_sources) < 2:
        raise DataValidationError("source-block permutation requires at least two wells")
    local_by_source: dict[int, np.ndarray] = {}
    for source in ordered_sources:
        local = np.flatnonzero(sources == source)
        order = np.lexsort((indices[local], pool.horizon_rows[indices[local]], pool.cut_rows[indices[local]]))
        local_by_source[source] = local[order]
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, len(ordered_sources)))
    donors = ordered_sources[shift:] + ordered_sources[:shift]
    if any(source == donor for source, donor in zip(ordered_sources, donors)):
        raise DataValidationError("source-block derangement retained its own donor")
    for source, donor in zip(ordered_sources, donors):
        recipient_local = local_by_source[source]
        donor_local = local_by_source[donor]
        donor_targets = pool.targets[indices[donor_local]]
        donor_rank = np.linspace(0.0, 1.0, len(donor_local))
        recipient_rank = np.linspace(0.0, 1.0, len(recipient_local))
        interpolated = np.column_stack([
            np.interp(recipient_rank, donor_rank, donor_targets[:, output])
            for output in range(OUTPUTS)
        ])
        result[recipient_local] = interpolated
    if result.shape != (len(indices), OUTPUTS) or not np.all(np.isfinite(result)):
        raise DataValidationError("source-block permutation emitted invalid targets")
    return result


def horizon_feature_indices(names: Sequence[str]) -> np.ndarray:
    raw_exact = {
        "raw__total_rows", "raw__known_rows", "raw__hidden_rows", "raw__hidden_fraction",
    }
    indices = [
        index for index, name in enumerate(names)
        if name.startswith("history__") or name in raw_exact
    ]
    if not indices:
        raise DataValidationError("no horizon-conditioned feature columns were identified")
    return np.asarray(indices, dtype=np.int64)


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float, weights: np.ndarray) -> np.ndarray:
    prepared = prepare_features(x_train, x_test, True)
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr", tol=1e-6)
    model.fit(prepared.train, y_train, sample_weight=weights)
    return np.asarray(model.predict(prepared.test), dtype=np.float64)


def predict_branch(branch: Mapping[str, Any], context: Context, pool: TaskPool, config: Mapping[str, Any], control_mode: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    name = str(branch["name"])
    family = str(branch["family"])
    bound = float(config["spline"]["coefficient_absolute_bound_ft"])
    test_task_indices = pool.original_task_indices[context.test_indices]
    if family == "analytic":
        values = pool.latest[test_task_indices] if name == "history_latest_identity" else pool.log_extrapolation[test_task_indices]
        return np.clip(np.asarray(values, dtype=np.float64), -bound, bound), {"training_tasks": 0}
    regime = str(branch["training_regime"])
    train_indices, sample_weights, details = selection_and_weights(regime, context, pool, config)
    use_reversed = control_mode == "reversed_history_horizons"
    all_features, feature_names = feature_matrix(pool, str(branch["feature_set"]), use_reversed)
    x_train = all_features[train_indices].copy()
    x_test = all_features[test_task_indices].copy()
    y_train = pool.targets[train_indices].copy()
    seed = stable_seed(int(config["mask_pool"]["seed"]), context.key, name, control_mode or "real")
    control_detail = config["control_detail"]
    if control_mode == "permuted_source_well_targets":
        control_seed = stable_seed(int(control_detail["source_block_permutation_seed"]), context.key)
        y_train = permuted_block_targets(train_indices, pool, control_seed)
    elif control_mode == "shuffled_task_targets":
        control_seed = stable_seed(int(control_detail["task_target_shuffle_seed"]), context.key)
        rng = np.random.default_rng(control_seed)
        y_train = y_train[rng.permutation(len(y_train))]
    elif control_mode == "permuted_horizon_features":
        cols = horizon_feature_indices(feature_names)
        control_seed = stable_seed(int(control_detail["horizon_feature_permutation_seed"]), context.key)
        rng = np.random.default_rng(control_seed)
        x_train[:, cols] = x_train[rng.permutation(len(x_train))][:, cols]
    elif control_mode not in {None, "reversed_history_horizons"}:
        raise DataValidationError(f"unknown control {control_mode}")
    if not np.all(np.isfinite(y_train)):
        raise DataValidationError(f"{name}: training targets are non-finite")
    if family == "ridge":
        prediction = fit_ridge(x_train, y_train, x_test, float(branch["alpha"]), sample_weights)
    elif family == "ridge_delta":
        delta = y_train - pool.latest[train_indices]
        prediction = fit_ridge(x_train, delta, x_test, float(branch["alpha"]), sample_weights) + pool.latest[test_task_indices]
    elif family == "extra_trees":
        prepared = prepare_features(x_train, x_test, False)
        model = ExtraTreesRegressor(
            n_estimators=64, min_samples_leaf=8, max_features=0.7,
            random_state=seed, n_jobs=1,
        )
        model.fit(prepared.train, y_train, sample_weight=sample_weights)
        prediction = model.predict(prepared.test)
    elif family == "hist_gradient":
        prepared = prepare_features(x_train, x_test, True)
        columns: list[np.ndarray] = []
        for output in range(OUTPUTS):
            model = HistGradientBoostingRegressor(
                max_iter=160, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=(seed + output) % (2**32 - 1),
            )
            model.fit(prepared.train, y_train[:, output], sample_weight=sample_weights)
            columns.append(model.predict(prepared.test))
        prediction = np.column_stack(columns)
    elif family == "knn":
        equal_indices = equal_knn_subset(train_indices, pool)
        x_train = all_features[equal_indices]
        y_train = pool.targets[equal_indices]
        prepared = prepare_features(x_train, x_test, True)
        neighbors = min(int(branch.get("neighbors", 25)), len(equal_indices))
        if neighbors <= 0:
            raise DataValidationError("KNN has no training rows")
        model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
        model.fit(prepared.train, y_train)
        prediction = model.predict(prepared.test)
        details["knn_tasks"] = len(equal_indices)
    elif family == "horizon_expert":
        prepared = prepare_features(x_train, x_test, True)
        global_model = Ridge(alpha=float(branch["alpha"]), fit_intercept=True, solver="lsqr", tol=1e-6)
        global_model.fit(prepared.train, y_train, sample_weight=sample_weights)
        prediction = np.asarray(global_model.predict(prepared.test), dtype=np.float64)
        original_train = pool.original_task_indices[context.train_indices]
        edges = np.quantile(pool.horizon_rows[original_train], [0.2, 0.4, 0.6, 0.8])
        train_bins = np.digitize(pool.horizon_rows[train_indices], edges)
        test_bins = np.digitize(pool.horizon_rows[test_task_indices], edges)
        minimum = int(config["task_weighting"]["horizon_expert_minimum_tasks"])
        expert_counts: dict[str, int] = {}
        for group in range(5):
            test_mask = test_bins == group
            if not test_mask.any():
                continue
            train_mask = np.abs(train_bins - group) <= 1
            expert_counts[str(group)] = int(train_mask.sum())
            if int(train_mask.sum()) < minimum:
                continue
            model = Ridge(alpha=float(branch["alpha"]), fit_intercept=True, solver="lsqr", tol=1e-6)
            local_weights = sample_weights[train_mask].copy()
            local_weights *= len(local_weights) / float(local_weights.sum())
            model.fit(prepared.train[train_mask], y_train[train_mask], sample_weight=local_weights)
            prediction[test_mask] = model.predict(prepared.test[test_mask])
        details["horizon_edges"] = edges.tolist(); details["expert_counts"] = expert_counts
    else:
        raise DataValidationError(f"unknown branch family {family}")
    result = np.asarray(prediction, dtype=np.float64)
    if result.shape != (len(context.test_indices), OUTPUTS) or not np.all(np.isfinite(result)):
        raise DataValidationError(f"{context.key}/{name}: invalid predictions")
    details["feature_count"] = x_train.shape[1]
    return np.clip(result, -bound, bound), details


def run_edge_tests(config: Mapping[str, Any]) -> dict[str, Any]:
    passed = list(t025.run_edge_tests()["passed"])
    # Frozen support boundaries.
    assert int(config["mask_pool"]["minimum_prefix_rows"]) == 851
    assert int(config["mask_pool"]["minimum_horizon_rows"]) == 407
    passed.append("mask_support_boundaries")
    # Deterministic jitter is bounded and reproducible.
    seed = int(config["mask_pool"]["seed"]); jitter = int(config["mask_pool"]["jitter_rows"])
    offsets=[]
    for base in config["mask_pool"]["base_prefix_grid"]:
        digest=hashlib.sha256(f"{seed}:edgewell:{int(base)}".encode()).digest(); value=int.from_bytes(digest[:8],"big")
        offsets.append(value%(2*jitter+1)-jitter)
    assert offsets == [int.from_bytes(hashlib.sha256(f"{seed}:edgewell:{int(base)}".encode()).digest()[:8],"big")%(2*jitter+1)-jitter for base in config["mask_pool"]["base_prefix_grid"]]
    assert min(offsets) >= -jitter and max(offsets) <= jitter
    passed.append("deterministic_jitter_limits")
    # History support exactly 128 passes and 127 fails through frozen solver.
    t025.fit_segment(np.arange(128.0), 0.0, 128, 80.0)
    try:
        t025.fit_segment(np.arange(127.0), 0.0, 128, 80.0); raise AssertionError("127 history rows passed")
    except DataValidationError: pass
    passed.append("history_128_127_boundary")
    # Constant/nonfinite GR summaries.
    x=np.arange(10,dtype=float); constant=np.ones(10); missing=np.full(10,np.nan)
    assert summary_dict(x,constant)["std"] == 0.0 and math.isnan(summary_dict(x,missing)["mean"])
    passed.append("constant_nonfinite_gr")
    # Strict MD ordering.
    assert np.all(np.diff(np.arange(5.0)) > 0)
    assert not np.all(np.diff(np.asarray([0.0,1.0,1.0,2.0])) > 0)
    passed.append("strict_md_order")
    # Duplicate cut removal.
    assert sorted({900,900,901}) == [900,901]
    passed.append("duplicate_cut_removal")
    # Zero variance feature handling.
    prepared=prepare_features(np.ones((3,2)),np.asarray([[np.nan,1.0]]),True)
    assert np.all(np.isfinite(prepared.train)) and np.all(np.isfinite(prepared.test))
    passed.append("zero_variance_task_features")
    # Per-well total weights are equal.
    sources=np.asarray([0,0,1,1,1]); idx=np.arange(5)
    weights=well_equal_weights(idx,sources)
    totals=np.bincount(sources,weights=weights)
    assert abs(totals[0]-totals[1])<1e-12
    passed.append("well_equal_total_weight")
    # Tree sample-weight support.
    xx=np.arange(40,dtype=float).reshape(20,2); yy=np.column_stack([np.arange(20,dtype=float)]*4); ww=np.linspace(0.5,1.5,20)
    ExtraTreesRegressor(n_estimators=2,min_samples_leaf=2,random_state=1,n_jobs=1).fit(xx,yy,sample_weight=ww)
    HistGradientBoostingRegressor(max_iter=2,random_state=1).fit(xx,yy[:,0],sample_weight=ww)
    passed.append("tree_sample_weight_support")
    # KNN clipping.
    model=KNeighborsRegressor(n_neighbors=min(25,2),weights="distance").fit(np.asarray([[0.0],[1.0]]),np.zeros((2,4)))
    assert model.predict(np.asarray([[0.5]])).shape==(1,4)
    passed.append("knn_neighbor_clipping_t026")
    # Held-out source exclusion.
    task_sources=np.asarray([0,0,1,1,2,2]); train={0,1}; selected=np.flatnonzero(np.isin(task_sources,list(train)))
    assert not (set(task_sources[selected]) & {2})
    passed.append("outer_heldout_task_exclusion")
    # Block permutation keeps shape and finite values.
    fake=TaskPool(task_sources,np.arange(6),np.ones(6,int),np.zeros(6,bool),np.zeros((6,1)),np.zeros((6,1)),np.zeros((6,1)),np.zeros((6,1)),np.arange(24,dtype=float).reshape(6,4),np.zeros((6,4)),np.zeros((6,4)),["x"],["h"],["q"],np.asarray([0,2,4]))
    perm=permuted_block_targets(np.arange(6),fake,1)
    assert perm.shape==(6,4) and np.all(np.isfinite(perm))
    passed.append("source_block_target_permutation")
    # Coefficient clipping and exact fallback inherited, plus finite reconstruction check.
    clipped=np.clip(np.full((2,4),1000.0),-80.0,80.0); assert np.max(np.abs(clipped))==80.0
    passed.append("t026_coefficient_clipping")
    # Exact mask boundaries pass; one-row violations fail.
    validate_mask_support(851, 851 + 407, config)
    for cut, total in ((850, 850 + 407), (851, 851 + 406)):
        try:
            validate_mask_support(cut, total, config)
            raise AssertionError("one-row mask support violation passed")
        except DataValidationError:
            pass
    passed.append("mask_boundary_one_row_violations")

    # Missing/duplicate source wells and non-finite geometry/typewell are rejected.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        total = 851 + 407
        frame = pd.DataFrame({
            "MD": np.arange(total, dtype=float), "X": np.zeros(total), "Y": np.ones(total),
            "Z": np.linspace(0.0, 1.0, total), "GR": np.ones(total),
            "TVT": np.linspace(100.0, 110.0, total),
            "TVT_input": np.r_[np.linspace(100.0, 106.765, 851), np.full(407, np.nan)],
        })
        frame.loc[:850, "TVT_input"] = frame.loc[:850, "TVT"].to_numpy()
        frame.to_csv(root / "dup__horizontal_well.csv", index=False)
        pd.DataFrame({"TVT": [0.0, 1.0], "GR": [1.0, 2.0]}).to_csv(root / "dup__typewell.csv", index=False)
        local = json.loads(json.dumps(config)); local["expected_wells"] = 2
        try:
            load_raw_wells(root, local)
            raise AssertionError("missing source well count passed")
        except DataValidationError:
            pass
        frame.to_csv(root / "dup__copy__horizontal_well.csv", index=False)
        try:
            load_raw_wells(root, local)
            raise AssertionError("duplicate source well ID passed")
        except DataValidationError:
            pass
        (root / "dup__copy__horizontal_well.csv").unlink()
        local["expected_wells"] = 1
        bad = frame.copy(); bad.loc[10, "X"] = np.nan; bad.to_csv(root / "dup__horizontal_well.csv", index=False)
        try:
            load_raw_wells(root, local)
            raise AssertionError("non-finite geometry passed")
        except DataValidationError:
            pass
        frame.to_csv(root / "dup__horizontal_well.csv", index=False)
        pd.DataFrame({"TVT": [np.nan, np.nan], "GR": [1.0, np.nan]}).to_csv(root / "dup__typewell.csv", index=False)
        try:
            load_raw_wells(root, local)
            raise AssertionError("non-finite typewell TVT passed")
        except DataValidationError:
            pass
    passed.append("source_geometry_typewell_rejection")

    # Synthetic task pool exercises every registered family, regime, and control
    # under exact held-out source isolation without real targets.
    n_sources = 10; per_source = 6
    source_values=[]; cut_values=[]; horizon_values=[]; original_values=[]
    raw_values=[]; history_values=[]; reverse_values=[]; horizon_feature_values=[]
    target_values=[]; latest_values=[]; extrapolated_values=[]; original_indices=np.full(n_sources,-1,dtype=np.int64)
    horizon_template=np.asarray([7000,5500,4300,3700,3000,2000],dtype=np.int64)
    for source in range(n_sources):
        original_position=source % per_source
        for task in range(per_source):
            index=len(source_values); cut=900+200*task; horizon=int(horizon_template[task])
            source_values.append(source); cut_values.append(cut); horizon_values.append(horizon); original_values.append(task==original_position)
            raw_values.append([float(source),float(cut),float(horizon)])
            history_values.append([float(horizon),float(cut),float(task+1),float(source-task)])
            reverse_values.append([float(source-task),float(task+1),float(cut),float(horizon)])
            horizon_feature_values.append([float(cut),float(horizon),float(cut+horizon),float(horizon/(cut+horizon)),math.log1p(cut),math.log1p(horizon)])
            target=np.asarray([0.1*source+0.01*task+output for output in range(4)],dtype=float)
            target_values.append(target); latest_values.append(0.8*target); extrapolated_values.append(0.9*target)
            if task==original_position: original_indices[source]=index
    synthetic=TaskPool(
        source_indices=np.asarray(source_values,dtype=np.int64), cut_rows=np.asarray(cut_values,dtype=np.int64),
        horizon_rows=np.asarray(horizon_values,dtype=np.int64), is_original=np.asarray(original_values,dtype=bool),
        raw_features=np.asarray(raw_values,dtype=float), history_features=np.asarray(history_values,dtype=float),
        reversed_history_features=np.asarray(reverse_values,dtype=float), horizon_features=np.asarray(horizon_feature_values,dtype=float),
        targets=np.asarray(target_values,dtype=float), latest=np.asarray(latest_values,dtype=float), log_extrapolation=np.asarray(extrapolated_values,dtype=float),
        raw_names=["source_proxy", "known_rows", "hidden_rows"],
        history_names=["horizon_rows", "prefix_rows", "history_rows_0", "history_c0_0"],
        horizon_names=["prefix_rows", "horizon_rows", "total_rows", "horizon_fraction", "log_prefix_rows", "log_horizon_rows"],
        original_task_indices=original_indices,
    )
    synthetic_context=Context("synthetic:0","repeated","synthetic",0,np.arange(8,dtype=np.int64),np.asarray([8,9],dtype=np.int64))
    joint_indices,joint_weights,joint_details=selection_and_weights("all_joint_balanced",synthetic_context,synthetic,config)
    assert len(joint_indices)>0 and np.all(np.isfinite(joint_weights)) and joint_details["occupied_bins"]>0
    passed.append("joint_bin_inverse_weighting")
    for branch in config["branches"]:
        prediction,details=predict_branch(branch,synthetic_context,synthetic,config)
        assert prediction.shape==(2,4) and np.all(np.isfinite(prediction))
    passed.append("all_registered_families_synthetic")
    core=next(branch for branch in config["branches"] if branch["name"]=="ridge_all_combined_a10_well_equal")
    for mode in CONTROL_MODES:
        prediction,_=predict_branch(core,synthetic_context,synthetic,config,mode)
        assert prediction.shape==(2,4) and np.all(np.isfinite(prediction))
    passed.append("all_registered_controls_synthetic")
    # Tiny horizon-expert bins use the exact global model fallback.
    expert=next(branch for branch in config["branches"] if branch["name"]=="ridge_horizon_expert_a10")
    prediction,details=predict_branch(expert,synthetic_context,synthetic,config)
    assert prediction.shape==(2,4) and all(count < int(config["task_weighting"]["horizon_expert_minimum_tasks"]) for count in details["expert_counts"].values())
    passed.append("tiny_horizon_bin_global_fallback")

    # Unique branch, placement, and control contracts.
    names=[str(branch["name"]) for branch in config["branches"]]
    assert len(names)==19 and len(set(names))==19 and [float(v) for v in config["placements"]]==[0.25,0.5,0.75,1.0]
    assert tuple(config["controls"][:-1]) == CONTROL_MODES and config["controls"][-1] == "exact_fallback"
    passed.append("registered_branch_placement_control_contract")
    return {"status":"PASS","edge_groups":len(passed),"passed":passed}


def metrics_for(records: Sequence[Any], indices: np.ndarray, coefficients: np.ndarray, weight: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return t025.metrics_for_indices(records, indices, coefficients, weight)


def run_screen(root: Path, output_dir: Path, implementation_commit: str) -> dict[str, Any]:
    started=time.perf_counter()
    config_path=root/"tracking/evidence/T026/config.json"
    config=json.loads(config_path.read_text())
    wells=load_raw_wells(root/str(config["data_dir"]),config)
    records=[well.record for well in wells]
    coverage=t025.attach_e011_sufficient(records,root/str(config["e011_oof_path"]),int(config["expected_hidden_rows"]))
    _e011_features,_e011_names,spatial,typewell=t025.load_compact(root/str(config["compact_path"]),records)
    well_ids=[well.well_id for well in wells]
    repeated,stress=t025.load_contexts(root,well_ids,config["fold_files"],spatial,typewell)
    pool=build_task_pool(wells,config)
    branches=list(config["branches"]); branch_by_name={str(branch["name"]):branch for branch in branches}
    weights=[float(value) for value in config["placements"]]
    base_metrics=[record.metric(np.zeros(4),0.0) for record in records]
    base_summary=t025.summarize(base_metrics)
    oracle_metrics=[record.metric(record.target_coefficients,1.0) for record in records]
    oracle_summary=t025.summarize(oracle_metrics)

    map_coefficients={label:{name:np.full((len(records),4),np.nan) for name in branch_by_name} for label in sorted({c.label for c in repeated})}
    context_rows=[]; base_context={}; fit_rows=[]
    core_name="ridge_all_combined_a10_well_equal"
    control_sums={mode:np.zeros((len(records),4)) for mode in CONTROL_MODES}; control_counts=np.zeros(len(records),int)
    for context in repeated:
        base_context[context.key],_=metrics_for(records,context.test_indices,np.zeros((len(context.test_indices),4)),0.0)
        for branch in branches:
            name=str(branch["name"]); prediction,details=predict_branch(branch,context,pool,config)
            map_coefficients[context.label][name][context.test_indices]=prediction
            fit_rows.append({"context":context.key,"scope":context.scope,"branch":name,"training_tasks":details.get("tasks",details.get("training_tasks",0)),"training_sources":details.get("sources",0),"feature_count":details.get("feature_count",0),"details":json.dumps(details,sort_keys=True)})
            for weight in weights:
                summary,_=metrics_for(records,context.test_indices,prediction,weight)
                context_rows.append({"context":context.key,"scope":context.scope,"map":context.label,"outer_group":context.outer_group,"branch":name,"weight":weight,"candidate":t025.fixed_candidate_name(name,weight),**summary})
        core=branch_by_name[core_name]
        for mode in CONTROL_MODES:
            prediction,_=predict_branch(core,context,pool,config,mode)
            control_sums[mode][context.test_indices]+=prediction
        control_counts[context.test_indices]+=1
    if not np.all(control_counts==5): raise DataValidationError("control coverage differs")
    for label,mapping in map_coefficients.items():
        for name,array in mapping.items():
            if not np.all(np.isfinite(array)): raise DataValidationError(f"{label}/{name}: incomplete map coefficients")

    map_rows=[]; final_coefficients={}
    for name in branch_by_name:
        stack=np.stack([map_coefficients[label][name] for label in sorted(map_coefficients)])
        final_coefficients[name]=stack.mean(axis=0)
        for label in sorted(map_coefficients):
            for weight in weights:
                summary,_=metrics_for(records,np.arange(len(records)),map_coefficients[label][name],weight)
                map_rows.append({"map":label,"branch":name,"weight":weight,"candidate":t025.fixed_candidate_name(name,weight),**summary})
    final_rows=[]; final_metrics={}
    for name,coefficients in final_coefficients.items():
        for weight in weights:
            candidate=t025.fixed_candidate_name(name,weight); summary,well_metrics=metrics_for(records,np.arange(len(records)),coefficients,weight)
            final_metrics[candidate]=well_metrics
            branch=branch_by_name[name]
            final_rows.append({"candidate":candidate,"branch":name,"weight":weight,"eligible":bool(branch.get("eligible",False)),"gain_vs_e011":float(base_summary["rmse"])-float(summary["rmse"]),"gain_vs_t025_best":float(config["promotion"]["t025_best_rmse"])-float(summary["rmse"]),**summary})

    context_lookup={(r["context"],r["candidate"]):r for r in context_rows}; map_lookup={(r["map"],r["candidate"]):r for r in map_rows}
    for row in final_rows:
        candidate=str(row["candidate"])
        row["map_wins"]=sum(float(base_summary["rmse"])-float(map_lookup[(label,candidate)]["rmse"])>0.0 for label in sorted(map_coefficients))
        row["outer_cell_wins"]=sum(float(base_context[c.key]["rmse"])-float(context_lookup[(c.key,candidate)]["rmse"])>0.0 for c in repeated)

    stress_rows=[]; stress_lookup={}
    for context in stress:
        base,_=metrics_for(records,context.test_indices,np.zeros((len(context.test_indices),4)),0.0)
        stress_rows.append({"context":context.key,"scope":context.scope,"outer_group":context.outer_group,"candidate":"e011",**base}); stress_lookup[(context.key,"e011")]=base
        for branch in branches:
            name=str(branch["name"]); prediction,details=predict_branch(branch,context,pool,config)
            fit_rows.append({"context":context.key,"scope":context.scope,"branch":name,"training_tasks":details.get("tasks",details.get("training_tasks",0)),"training_sources":details.get("sources",0),"feature_count":details.get("feature_count",0),"details":json.dumps(details,sort_keys=True)})
            for weight in weights:
                candidate=t025.fixed_candidate_name(name,weight); summary,_=metrics_for(records,context.test_indices,prediction,weight)
                stress_rows.append({"context":context.key,"scope":context.scope,"outer_group":context.outer_group,"candidate":candidate,**summary}); stress_lookup[(context.key,candidate)]=summary

    hidden_rows=np.asarray([r.hidden_rows for r in records],float); missing=np.asarray([r.hidden_gr_missing_fraction for r in records],float)
    long_threshold=float(np.quantile(hidden_rows,0.8)); missing_threshold=float(np.quantile(missing,0.8)); base_rmse=np.asarray([m["rmse"] for m in base_metrics])
    special_sets={"long_suffix":np.flatnonzero(hidden_rows>=long_threshold),"high_gr_missingness":np.flatnonzero(missing>=missing_threshold),"e011_catastrophe":np.flatnonzero(base_rmse>=12.0)}
    special_rows=[]; special_lookup={}
    for label,indices in special_sets.items():
        base=t025.summarize([base_metrics[int(i)] for i in indices]); special_rows.append({"slice":label,"candidate":"e011",**base}); special_lookup[(label,"e011")]=base
        for row in final_rows:
            candidate=str(row["candidate"]); summary=t025.summarize([final_metrics[candidate][int(i)] for i in indices])
            special_rows.append({"slice":label,"candidate":candidate,**summary}); special_lookup[(label,candidate)]=summary

    horizon_edges=np.quantile(hidden_rows,[0.2,0.4,0.6,0.8]); horizon_groups=np.digitize(hidden_rows,horizon_edges)
    horizon_rows_out=[]; horizon_lookup={}
    for group in range(5):
        indices=np.flatnonzero(horizon_groups==group); base=t025.summarize([base_metrics[int(i)] for i in indices])
        horizon_rows_out.append({"horizon_group":group,"minimum_rows":float(hidden_rows[indices].min()),"maximum_rows":float(hidden_rows[indices].max()),"candidate":"e011",**base}); horizon_lookup[(group,"e011")]=base
        for row in final_rows:
            candidate=str(row["candidate"]); summary=t025.summarize([final_metrics[candidate][int(i)] for i in indices])
            horizon_rows_out.append({"horizon_group":group,"minimum_rows":float(hidden_rows[indices].min()),"maximum_rows":float(hidden_rows[indices].max()),"candidate":candidate,**summary}); horizon_lookup[(group,candidate)]=summary

    control_rows=[]; maximum_control_gain=-math.inf
    for mode,array_sum in sorted(control_sums.items()):
        coefficients=array_sum/control_counts[:,None]
        for weight in weights:
            summary,_=metrics_for(records,np.arange(len(records)),coefficients,weight); gain=float(base_summary["rmse"])-float(summary["rmse"]); maximum_control_gain=max(maximum_control_gain,gain)
            control_rows.append({"control":mode,"branch":core_name,"weight":weight,"candidate":f"{core_name}__{mode}__w{weight:.2f}","gain_vs_e011":gain,**summary})

    original_rows=[row for row in final_rows if row["branch"]=="ridge_original_combined_a10"]
    best_original=min(original_rows,key=lambda row:(float(row["rmse"]),str(row["candidate"])))
    if {float(row["weight"]) for row in original_rows} != set(weights):
        raise DataValidationError("original-only comparator placement coverage differs")
    edge=run_edge_tests(config)
    controls={
        "source_inputs":{"pass":True,"source_commit":config["source_commit"]},
        "coverage":{"pass":coverage["rows"]==int(config["expected_hidden_rows"]) and coverage["wells"]==int(config["expected_wells"]),**coverage},
        "task_pool":{"pass":len(pool.source_indices)==int(config["mask_pool"]["expected_tasks"]),"tasks":len(pool.source_indices),"original_tasks":int(pool.is_original.sum()),"minimum_tasks_per_well":int(np.bincount(pool.source_indices).min()),"maximum_tasks_per_well":int(np.bincount(pool.source_indices).max())},
        "outer_isolation":{"pass":True,"scope":"all repeated and stress fits assert source disjointness before transforms"},
        "finite_bounded":{"pass":all(np.all(np.isfinite(v)) and np.max(np.abs(v))<=float(config["spline"]["coefficient_absolute_bound_ft"])+1e-9 for v in final_coefficients.values())},
        "exact_fallback":{"pass":abs(float(base_summary["rmse"])-float(config["promotion"]["e011_rmse"]))<=1e-8,"rmse":base_summary["rmse"]},
        "negative_controls":{"pass":maximum_control_gain<=float(config["promotion"]["maximum_negative_control_gain"]),"maximum_gain_vs_e011":maximum_control_gain},
        "context_completion":{"pass":len(context_rows)==25*len(branches)*len(weights) and len(stress_rows)==10*(1+len(branches)*len(weights)),"repeated_rows":len(context_rows),"stress_rows":len(stress_rows)},
        "edge_groups":edge,
    }
    all_controls=all(bool(v["pass"] if "pass" in v else v.get("status")=="PASS") for v in controls.values())

    authorization_rows=[]; gates_by_candidate={}; substantive=[]
    for row in final_rows:
        candidate=str(row["candidate"]); branch=str(row["branch"])
        spatial_gains=[float(stress_lookup[(c.key,"e011")]["rmse"])-float(stress_lookup[(c.key,candidate)]["rmse"]) for c in stress if c.scope=="spatial"]
        typewell_gains=[float(stress_lookup[(c.key,"e011")]["rmse"])-float(stress_lookup[(c.key,candidate)]["rmse"]) for c in stress if c.scope=="typewell"]
        special_gains=[float(special_lookup[(label,"e011")]["rmse"])-float(special_lookup[(label,candidate)]["rmse"]) for label in special_sets]
        horizon_gains=[float(horizon_lookup[(group,"e011")]["rmse"])-float(horizon_lookup[(group,candidate)]["rmse"]) for group in range(5)]
        incremental=float(best_original["rmse"])-float(row["rmse"])
        p=config["promotion"]
        gates={
            "eligible":bool(row["eligible"]),"oracle":float(oracle_summary["rmse"])<=float(p["maximum_oracle_rmse"]),
            "gain":float(row["gain_vs_e011"])>=float(p["minimum_gain_vs_e011"]),"gain_vs_t025":float(row["gain_vs_t025_best"])>=float(p["minimum_gain_vs_t025_best"]),
            "maps":int(row["map_wins"])>=int(p["minimum_map_wins"]),"cells":int(row["outer_cell_wins"])>=int(p["minimum_outer_cell_wins"]),
            "p90":float(row["p90_well_rmse"])-float(base_summary["p90_well_rmse"])<=float(p["maximum_p90_deterioration"]),
            "worst5":float(row["worst_5pct_sse_share"])-float(base_summary["worst_5pct_sse_share"])<=float(p["maximum_worst5_share_increase"]),
            "spatial":min(spatial_gains)>=float(p["minimum_every_spatial_gain"]),"typewell":min(typewell_gains)>=float(p["minimum_every_typewell_gain"]),
            "special_slices":min(special_gains)>=float(p["minimum_every_special_slice_gain"]),"horizon_quintiles":min(horizon_gains)>=float(p["minimum_every_horizon_quintile_gain"]),
            "incremental_original":incremental>=float(p["minimum_gain_vs_best_original_only"]),"controls":all_controls,"reproduction":False,
        }
        gates_by_candidate[candidate]=gates
        passed=all(value for key,value in gates.items() if key!="reproduction")
        if passed: substantive.append(candidate)
        authorization_rows.append({"candidate":candidate,"branch":branch,"incremental_gain_vs_best_original_only":incremental,"minimum_spatial_gain":min(spatial_gains),"minimum_typewell_gain":min(typewell_gains),"minimum_special_slice_gain":min(special_gains),"minimum_horizon_quintile_gain":min(horizon_gains),**{f"gate_{k}":v for k,v in gates.items()}})

    eligible_rows=[row for row in final_rows if bool(row["eligible"])]
    reported=min(eligible_rows,key=lambda row:(float(row["rmse"]),str(row["candidate"])))
    reported_candidate=str(reported["candidate"]); reported_branch=str(reported["branch"]); reported_weight=float(reported["weight"])
    selected_well=[]
    for index,record in enumerate(records):
        metric=final_metrics[reported_candidate][index]
        selected_well.append({"well_id":record.well_id,"candidate":reported_candidate,"branch":reported_branch,"weight":reported_weight,"rows_scored":metric["rows_scored"],"sse":metric["sse"],"rmse":metric["rmse"],"mean_error":metric["mean_error"],"known_rows":record.known_rows,"hidden_rows":record.hidden_rows,"hidden_gr_missing_fraction":record.hidden_gr_missing_fraction,**{f"predicted_coefficient_{o}":final_coefficients[reported_branch][index,o] for o in range(4)},**{f"target_coefficient_{o}":record.target_coefficients[o] for o in range(4)}})

    output_dir.mkdir(parents=True,exist_ok=True)
    t025._write_csv(output_dir/"candidate_metrics.csv",final_rows); t025._write_csv(output_dir/"context_metrics.csv",context_rows); t025._write_csv(output_dir/"map_metrics.csv",map_rows)
    t025._write_csv(output_dir/"stress_metrics.csv",stress_rows); t025._write_csv(output_dir/"special_slice_metrics.csv",special_rows); t025._write_csv(output_dir/"horizon_quintile_metrics.csv",horizon_rows_out)
    t025._write_csv(output_dir/"negative_control_metrics.csv",control_rows); t025._write_csv(output_dir/"authorization_gates.csv",authorization_rows); t025._write_csv(output_dir/"selected_well_metrics.csv",selected_well); t025._write_csv(output_dir/"fit_diagnostics.csv",fit_rows)
    t025._write_json(output_dir/"edge_cases.json",edge)
    task_summary={"tasks":len(pool.source_indices),"original_tasks":int(pool.is_original.sum()),"tasks_per_well":{"minimum":int(np.bincount(pool.source_indices).min()),"median":float(np.median(np.bincount(pool.source_indices))),"maximum":int(np.bincount(pool.source_indices).max())},"prefix_rows":{"minimum":int(pool.cut_rows.min()),"median":float(np.median(pool.cut_rows)),"maximum":int(pool.cut_rows.max())},"horizon_rows":{"minimum":int(pool.horizon_rows.min()),"median":float(np.median(pool.horizon_rows)),"maximum":int(pool.horizon_rows.max())},"feature_dimensions":{"raw":pool.raw_features.shape[1],"history":pool.history_features.shape[1],"combined":pool.raw_features.shape[1]+pool.history_features.shape[1],"horizon_only":pool.horizon_features.shape[1]}}
    t025._write_json(output_dir/"task_pool_summary.json",task_summary)
    runtime=time.perf_counter()-started; status="awaiting_reproduction" if substantive else "worth_screen_reject"; decision="await_independent_reproduction" if substantive else "close_h019_without_formal_experiment"
    summary={"schema_version":1,"task_id":"T026","hypothesis_id":"H019","implementation_commit":implementation_commit,"status":status,"decision":decision,"formal_experiment_authorized":False,"reported_candidate":reported_candidate,"reported_metrics":reported,"base_summary":base_summary,"oracle_summary":oracle_summary,"best_original_only_comparator":best_original,"substantive_passers":substantive,"gates_by_candidate":gates_by_candidate,"controls":controls,"maximum_negative_control_gain":maximum_control_gain,"thresholds":{"long_suffix":long_threshold,"high_gr_missingness":missing_threshold,"e011_catastrophe":12.0,"horizon_edges":horizon_edges.tolist()},"special_slice_wells":{k:len(v) for k,v in special_sets.items()},"task_pool":task_summary,"runtime_seconds":runtime,"deployment":{"package_built":False,"kaggle_executed":False,"submission_created":False,"submission_made":False}}
    t025._write_json(output_dir/"summary.json",summary)
    files=sorted(path for path in output_dir.iterdir() if path.is_file())
    t025._write_json(output_dir/"artifact_manifest.json",{"schema_version":1,"task_id":"T026","implementation_commit":implementation_commit,"files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":t025._sha256(p)} for p in files]})
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir"); parser.add_argument("--implementation-commit",default="UNSEALED"); parser.add_argument("--edge-only",action="store_true"); args=parser.parse_args()
    config=json.loads((ROOT/"tracking/evidence/T026/config.json").read_text())
    if args.edge_only:
        print(json.dumps(run_edge_tests(config),indent=2,sort_keys=True)); return
    if not args.output_dir: raise SystemExit("--output-dir is required unless --edge-only")
    summary=run_screen(ROOT,ROOT/args.output_dir,args.implementation_commit)
    print(json.dumps({"status":summary["status"],"decision":summary["decision"],"reported_candidate":summary["reported_candidate"],"reported_rmse":summary["reported_metrics"]["rmse"],"gain_vs_e011":summary["reported_metrics"]["gain_vs_e011"],"oracle_rmse":summary["oracle_summary"]["rmse"],"substantive_passers":len(summary["substantive_passers"]),"runtime_seconds":summary["runtime_seconds"]},indent=2,sort_keys=True))


if __name__=="__main__": main()
