#!/usr/bin/env python3
"""T031: raw-sequence latent formation-displacement identifiability gate.

This script is intentionally limited to the preregistered auxiliary target. It never
emits or scores a competition TVT prediction. All model choices, controls, splits,
and decision thresholds are frozen in tracking/evidence/T031/config.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
LOCAL_TORCH = ROOT / "scratch" / "_t031_deps"
if LOCAL_TORCH.exists():
    sys.path.insert(0, str(LOCAL_TORCH))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - surfaced as a visible run failure
    raise RuntimeError(
        "T031 requires PyTorch. Install torch into scratch/_t031_deps or the active environment."
    ) from exc

CONFIG_PATH = ROOT / "tracking" / "evidence" / "T031" / "config.json"
DEFAULT_OUTPUT = ROOT / "scratch" / "agents" / "t031-state-gate-20260724" / "main"

H_CHANNELS = [
    "global_position",
    "md_relative_scaled",
    "x_relative_scaled",
    "y_relative_scaled",
    "z_relative_scaled",
    "dz_dmd_scaled",
    "gr_robust_z",
    "gr_finite_mask",
    "visible_tvt_relative_100ft",
    "visible_u_relative_100ft",
    "visible_tvt_mask",
    "boundary_side",
    "visible_gr_coverage",
    "hidden_gr_coverage",
]
T_CHANNELS = [
    "typewell_position",
    "typewell_tvt_relative_500ft",
    "typewell_gr_robust_z",
    "typewell_gr_finite_mask",
    "typewell_gr_gradient",
]
COORD_CHANNELS = [0, 1, 2, 3, 4, 5, 11]
GR_CHANNELS = [6, 7]
VISIBLE_TVT_CHANNELS = [8, 9, 10]


@dataclass(frozen=True)
class Context:
    map_index: int
    outer_fold: int
    inner_fold: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    outer_train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def context_id(self) -> str:
        return f"map{self.map_index}_fold{self.outer_fold}"


@dataclass
class Tensors:
    well_ids: np.ndarray
    horizontal: np.ndarray
    typewell: np.ndarray
    summary_features: np.ndarray
    coordinate_flat: np.ndarray
    target: np.ndarray
    legacy_spatial: np.ndarray
    legacy_typewell: np.ndarray
    legal_kmeans: np.ndarray
    spatial_2d_kmeans: np.ndarray
    fold_assignments: list[np.ndarray]
    tensor_manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "mae_ft": float("nan"), "rmse_ft": float("nan"), "count": int(mask.sum())}
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    if float(np.std(yt)) <= 1e-12 or float(np.std(yp)) <= 1e-12:
        pearson = 0.0
    else:
        pearson = float(pearsonr(yt, yp).statistic)
    if float(np.std(yt)) <= 1e-12 or float(np.std(yp)) <= 1e-12:
        spearman = 0.0
    else:
        spearman = float(spearmanr(yt, yp).statistic)
    err = yp - yt
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mae_ft": float(np.mean(np.abs(err))),
        "rmse_ft": float(np.sqrt(np.mean(err * err))),
        "count": int(mask.sum()),
    }


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return center, scale


def interpolate_finite(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros_like(positions, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, len(values), dtype=float)
    if finite.sum() == 1:
        return np.full_like(positions, float(values[finite][0]), dtype=np.float32)
    return np.interp(positions, xp[finite], values[finite]).astype(np.float32)


def nearest_mask(mask: np.ndarray, positions: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=float)
    if len(mask) == 1:
        return np.full_like(positions, mask[0], dtype=np.float32)
    idx = np.rint(positions * (len(mask) - 1)).astype(int)
    idx = np.clip(idx, 0, len(mask) - 1)
    return mask[idx].astype(np.float32)


def segment_resample(values: np.ndarray, bins: int, *, is_mask: bool = False) -> np.ndarray:
    positions = (np.arange(bins, dtype=float) + 0.5) / bins
    return nearest_mask(values, positions) if is_mask else interpolate_finite(values, positions)


def gradient_safe(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(values) < 2:
        return np.zeros_like(values, dtype=float)
    safe = values.copy()
    finite = np.isfinite(safe)
    if finite.sum() == 0:
        return np.zeros_like(values, dtype=float)
    if finite.sum() == 1:
        safe[:] = safe[finite][0]
    else:
        safe[~finite] = np.interp(x[~finite], x[finite], safe[finite])
    dx = np.gradient(x)
    dx[np.abs(dx) < 1e-9] = 1.0
    return np.gradient(safe) / dx


def build_horizontal_tensor(frame: pd.DataFrame, visible_bins: int, hidden_bins: int) -> tuple[np.ndarray, dict[str, float]]:
    required = {"MD", "X", "Y", "Z", "GR", "TVT_input"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"horizontal file missing columns: {sorted(missing)}")
    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(float)
    x = pd.to_numeric(frame["X"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
    finite_tvt = np.isfinite(tvt)
    if finite_tvt.all() or not finite_tvt.any():
        raise ValueError("TVT_input must have a visible prefix and hidden suffix")
    cut = int(np.flatnonzero(~finite_tvt)[0])
    if not finite_tvt[:cut].all() or finite_tvt[cut:].any():
        raise ValueError("TVT_input boundary is not contiguous")
    if cut < 2 or len(frame) - cut < 2:
        raise ValueError("insufficient visible or hidden rows")
    boundary = cut - 1
    md0, x0, y0, z0, tvt0 = md[boundary], x[boundary], y[boundary], z[boundary], tvt[boundary]
    md_span = max(float(md[-1] - md[0]), 1.0)
    horizontal_span = max(float(np.hypot(x[-1] - x[0], y[-1] - y[0])), 1.0)
    z_span = max(float(np.nanmax(np.abs(z - z0))), 1.0)
    gr_center, gr_scale = robust_center_scale(gr)
    gr_z = np.where(np.isfinite(gr), (gr - gr_center) / gr_scale, 0.0)
    gr_z = np.clip(gr_z, -8.0, 8.0)
    gr_mask = np.isfinite(gr).astype(float)
    dz = gradient_safe(z, md)
    dz_scale = max(float(np.nanpercentile(np.abs(dz[np.isfinite(dz)]), 95.0)) if np.isfinite(dz).any() else 1.0, 1e-4)
    tvt_rel = np.where(finite_tvt, (tvt - tvt0) / 100.0, 0.0)
    u_rel = np.where(finite_tvt, ((tvt + z) - (tvt0 + z0)) / 100.0, 0.0)
    global_pos = np.linspace(-1.0, 1.0, len(frame), dtype=float)
    boundary_side = np.where(np.arange(len(frame)) < cut, -1.0, 1.0)
    visible_cov = float(np.mean(gr_mask[:cut]))
    hidden_cov = float(np.mean(gr_mask[cut:]))
    raw = np.column_stack([
        global_pos,
        (md - md0) / md_span,
        (x - x0) / horizontal_span,
        (y - y0) / horizontal_span,
        (z - z0) / z_span,
        np.clip(dz / dz_scale, -8.0, 8.0),
        gr_z,
        gr_mask,
        tvt_rel,
        u_rel,
        finite_tvt.astype(float),
        boundary_side,
        np.full(len(frame), visible_cov),
        np.full(len(frame), hidden_cov),
    ])
    out = np.zeros((visible_bins + hidden_bins, raw.shape[1]), dtype=np.float32)
    for channel in range(raw.shape[1]):
        is_mask = channel in {7, 10, 11, 12, 13}
        out[:visible_bins, channel] = segment_resample(raw[:cut, channel], visible_bins, is_mask=is_mask)
        out[visible_bins:, channel] = segment_resample(raw[cut:, channel], hidden_bins, is_mask=is_mask)
    stats = {
        "rows": int(len(frame)),
        "visible_rows": int(cut),
        "hidden_rows": int(len(frame) - cut),
        "visible_gr_coverage": visible_cov,
        "hidden_gr_coverage": hidden_cov,
        "last_visible_tvt": float(tvt0),
    }
    return out, stats


def build_typewell_tensor(frame: pd.DataFrame, bins: int, last_visible_tvt: float) -> np.ndarray:
    required = {"TVT", "GR"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"typewell file missing columns: {sorted(missing)}")
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float)
    finite_tvt = np.isfinite(tvt)
    if finite_tvt.sum() < 2:
        raise ValueError("typewell TVT has fewer than two finite rows")
    order = np.argsort(np.where(finite_tvt, tvt, np.inf), kind="mergesort")
    tvt = tvt[order]
    gr = gr[order]
    pos = np.linspace(-1.0, 1.0, len(tvt), dtype=float)
    gr_center, gr_scale = robust_center_scale(gr)
    gr_z = np.where(np.isfinite(gr), (gr - gr_center) / gr_scale, 0.0)
    gr_z = np.clip(gr_z, -8.0, 8.0)
    mask = np.isfinite(gr).astype(float)
    grad = gradient_safe(gr_z, np.where(np.isfinite(tvt), tvt, np.arange(len(tvt), dtype=float)))
    grad_scale = max(float(np.nanpercentile(np.abs(grad[np.isfinite(grad)]), 95.0)) if np.isfinite(grad).any() else 1.0, 1e-4)
    raw = np.column_stack([
        pos,
        (np.where(np.isfinite(tvt), tvt, last_visible_tvt) - last_visible_tvt) / 500.0,
        gr_z,
        mask,
        np.clip(grad / grad_scale, -8.0, 8.0),
    ])
    out = np.zeros((bins, raw.shape[1]), dtype=np.float32)
    for channel in range(raw.shape[1]):
        out[:, channel] = segment_resample(raw[:, channel], bins, is_mask=(channel == 3))
    return out


def build_groups(feature_table: pd.DataFrame, config: dict[str, Any], well_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indexed = feature_table.set_index("well_id").loc[well_ids]
    group_cfg = config["validation"]["new_target_free_stress"]
    outputs: list[np.ndarray] = []
    for name in ["legal_covariate_kmeans", "spatial_2d_kmeans"]:
        entry = group_cfg[name]
        matrix = indexed[entry["features"]].to_numpy(float)
        medians = np.nanmedian(matrix, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        matrix = StandardScaler().fit_transform(matrix)
        model = KMeans(
            n_clusters=int(entry["groups"]),
            n_init=int(entry["n_init"]),
            random_state=int(entry["random_state"]),
        )
        outputs.append(model.fit_predict(matrix).astype(np.int8))
    return outputs[0], outputs[1]


def load_tensors(config: dict[str, Any], output_dir: Path, *, force_rebuild: bool = False) -> Tensors:
    cache_path = output_dir / "tensors_v1.npz"
    manifest_path = output_dir / "tensor_manifest.json"
    if cache_path.exists() and manifest_path.exists() and not force_rebuild:
        data = np.load(cache_path, allow_pickle=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return Tensors(
            well_ids=data["well_ids"],
            horizontal=data["horizontal"],
            typewell=data["typewell"],
            summary_features=data["summary_features"],
            coordinate_flat=data["coordinate_flat"],
            target=data["target"],
            legacy_spatial=data["legacy_spatial"],
            legacy_typewell=data["legacy_typewell"],
            legal_kmeans=data["legal_kmeans"],
            spatial_2d_kmeans=data["spatial_2d_kmeans"],
            fold_assignments=[data[f"fold_{i}"] for i in range(5)],
            tensor_manifest=manifest,
        )

    started = time.time()
    compact_path = ROOT / config["inputs"]["e011_compact"]
    compact = np.load(compact_path, allow_pickle=False)
    well_ids = compact["well_ids"].astype(str)
    if len(well_ids) != int(config["expected_wells"]):
        raise ValueError("unexpected well count")
    summary_features = compact["features"].astype(np.float32)
    legacy_spatial = compact["spatial_assignment"].astype(np.int8)
    legacy_typewell = compact["typewell_assignment"].astype(np.int8)

    feature_path = ROOT / config["inputs"]["e003_feature_table"]
    feature_table = pd.read_csv(feature_path)
    feature_table["well_id"] = feature_table["well_id"].astype(str).str.zfill(8)
    target_columns = config["target"]["columns"]
    target_frame = feature_table.set_index("well_id").loc[well_ids, target_columns]
    finite_counts = np.isfinite(target_frame.to_numpy(float)).sum(axis=1)
    if int(finite_counts.min()) < 5:
        raise ValueError("target has fewer than five finite components for a well")
    target = np.nanmean(target_frame.to_numpy(float), axis=1).astype(np.float64)
    if sha256_array(target.astype("<f8")) != config["target"]["sha256_float64_well_order"]:
        raise ValueError("target hash mismatch")

    visible_bins = int(config["sequence_grid"]["visible_prefix_bins"])
    hidden_bins = int(config["sequence_grid"]["post_boundary_covariate_bins"])
    type_bins = int(config["sequence_grid"]["typewell_bins"])
    horizontal = np.zeros((len(well_ids), visible_bins + hidden_bins, len(H_CHANNELS)), dtype=np.float32)
    typewell = np.zeros((len(well_ids), type_bins, len(T_CHANNELS)), dtype=np.float32)
    well_stats: list[dict[str, Any]] = []
    for i, well_id in enumerate(well_ids):
        h_path = ROOT / "data" / "train" / f"{well_id}__horizontal_well.csv"
        t_path = ROOT / "data" / "train" / f"{well_id}__typewell.csv"
        h_tensor, stats = build_horizontal_tensor(pd.read_csv(h_path), visible_bins, hidden_bins)
        t_tensor = build_typewell_tensor(pd.read_csv(t_path), type_bins, stats["last_visible_tvt"])
        horizontal[i] = h_tensor
        typewell[i] = t_tensor
        stats["well_id"] = well_id
        well_stats.append(stats)

    absolute_coord = feature_table.set_index("well_id").loc[well_ids, [
        "last_visible_x", "last_visible_y", "last_visible_z", "spatial_x_mid", "spatial_y_mid",
        "x_total_delta", "y_total_delta", "z_total_delta", "md_total_span", "md_hidden_span",
    ]].to_numpy(float)
    coord_sequence = horizontal[:, :, COORD_CHANNELS].reshape(len(well_ids), -1)
    coordinate_flat = np.concatenate([coord_sequence, absolute_coord], axis=1).astype(np.float32)
    legal_kmeans, spatial_2d_kmeans = build_groups(feature_table, config, well_ids)

    fold_assignments: list[np.ndarray] = []
    for fold_path in config["inputs"]["fold_files"]:
        payload = json.loads((ROOT / fold_path).read_text(encoding="utf-8"))
        assignments = payload["assignments"]
        fold_assignments.append(np.array([int(assignments[w]) for w in well_ids], dtype=np.int8))

    arrays: dict[str, np.ndarray] = {
        "well_ids": well_ids,
        "horizontal": horizontal,
        "typewell": typewell,
        "summary_features": summary_features,
        "coordinate_flat": coordinate_flat,
        "target": target,
        "legacy_spatial": legacy_spatial,
        "legacy_typewell": legacy_typewell,
        "legal_kmeans": legal_kmeans,
        "spatial_2d_kmeans": spatial_2d_kmeans,
    }
    for i, folds in enumerate(fold_assignments):
        arrays[f"fold_{i}"] = folds
    np.savez_compressed(cache_path, **arrays)
    manifest = {
        "schema_version": 1,
        "created_at_epoch": time.time(),
        "seconds": time.time() - started,
        "well_count": len(well_ids),
        "horizontal_shape": list(horizontal.shape),
        "typewell_shape": list(typewell.shape),
        "summary_shape": list(summary_features.shape),
        "coordinate_shape": list(coordinate_flat.shape),
        "horizontal_channels": H_CHANNELS,
        "typewell_channels": T_CHANNELS,
        "target_sha256": sha256_array(target.astype("<f8")),
        "horizontal_sha256": sha256_array(horizontal),
        "typewell_sha256": sha256_array(typewell),
        "cache_sha256": sha256_file(cache_path),
        "input_files": {
            str(compact_path.relative_to(ROOT)): sha256_file(compact_path),
            str(feature_path.relative_to(ROOT)): sha256_file(feature_path),
        },
        "well_stats": {
            "visible_rows_min": min(int(r["visible_rows"]) for r in well_stats),
            "visible_rows_max": max(int(r["visible_rows"]) for r in well_stats),
            "hidden_rows_min": min(int(r["hidden_rows"]) for r in well_stats),
            "hidden_rows_max": max(int(r["hidden_rows"]) for r in well_stats),
            "hidden_gr_coverage_min": min(float(r["hidden_gr_coverage"]) for r in well_stats),
            "hidden_gr_coverage_median": float(np.median([r["hidden_gr_coverage"] for r in well_stats])),
        },
    }
    json_dump(manifest_path, manifest)
    return Tensors(
        well_ids=well_ids,
        horizontal=horizontal,
        typewell=typewell,
        summary_features=summary_features,
        coordinate_flat=coordinate_flat,
        target=target,
        legacy_spatial=legacy_spatial,
        legacy_typewell=legacy_typewell,
        legal_kmeans=legal_kmeans,
        spatial_2d_kmeans=spatial_2d_kmeans,
        fold_assignments=fold_assignments,
        tensor_manifest=manifest,
    )


def make_contexts(tensors: Tensors) -> list[Context]:
    contexts: list[Context] = []
    for map_index, assignment in enumerate(tensors.fold_assignments):
        for outer_fold in range(5):
            inner_fold = (outer_fold + 1) % 5
            test_idx = np.flatnonzero(assignment == outer_fold)
            valid_idx = np.flatnonzero(assignment == inner_fold)
            train_idx = np.flatnonzero((assignment != outer_fold) & (assignment != inner_fold))
            outer_train_idx = np.flatnonzero(assignment != outer_fold)
            if not len(test_idx) or not len(valid_idx) or not len(train_idx):
                raise ValueError("empty context partition")
            if set(test_idx) & set(outer_train_idx):
                raise ValueError("outer leakage")
            contexts.append(Context(map_index, outer_fold, inner_fold, train_idx, valid_idx, outer_train_idx, test_idx))
    return contexts


def impute_scale_fit(matrix: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = matrix[indices].astype(float)
    med = np.nanmedian(train, axis=0)
    med[~np.isfinite(med)] = 0.0
    filled = np.where(np.isfinite(train), train, med)
    scale = np.std(filled, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return med, scale


def impute_scale_apply(matrix: np.ndarray, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(matrix), matrix, med)
    return ((filled - med) / scale).astype(np.float32)


def ridge_context(
    matrix: np.ndarray,
    y: np.ndarray,
    context: Context,
    alphas: list[float],
) -> tuple[np.ndarray, float, dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    med, scale = impute_scale_fit(matrix, context.train_idx)
    scaled = impute_scale_apply(matrix, med, scale)
    best: tuple[float, float] | None = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(scaled[context.train_idx], y[context.train_idx])
        pred = model.predict(scaled[context.valid_idx])
        mae = float(np.mean(np.abs(pred - y[context.valid_idx])))
        key = (mae, float(alpha))
        if best is None or key < best:
            best = key
    assert best is not None
    selected_alpha = best[1]
    final_med, final_scale = impute_scale_fit(matrix, context.outer_train_idx)
    final_scaled = impute_scale_apply(matrix, final_med, final_scale)
    final = Ridge(alpha=selected_alpha)
    final.fit(final_scaled[context.outer_train_idx], y[context.outer_train_idx])
    prediction = final.predict(final_scaled[context.test_idx]).astype(float)
    metadata = {"alpha": selected_alpha, "inner_mae_ft": best[0]}
    return prediction, selected_alpha, metadata, (final_med, final_scale)


def ridge_predict_with_fixed(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> np.ndarray:
    med = np.nanmedian(train_matrix, axis=0)
    med[~np.isfinite(med)] = 0.0
    train = np.where(np.isfinite(train_matrix), train_matrix, med)
    test = np.where(np.isfinite(test_matrix), test_matrix, med)
    scale = np.std(train, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    train = (train - med) / scale
    test = (test - med) / scale
    model = Ridge(alpha=float(alpha))
    model.fit(train, y_train)
    return model.predict(test).astype(float)


class SeparableResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, kernel: int, dilations: list[int], stacks: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Conv1d(in_channels, width, 1)
        blocks: list[nn.Module] = []
        for _ in range(stacks):
            for dilation in dilations:
                blocks.append(SeparableResidualBlock(width, kernel, dilation, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.attention = nn.Conv1d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.blocks(self.input(x))
        mean_pool = x.mean(dim=2)
        weights = torch.softmax(self.attention(x).squeeze(1), dim=1)
        attn_pool = torch.sum(x * weights.unsqueeze(1), dim=2)
        return torch.cat([mean_pool, attn_pool], dim=1)


class StateTCN(nn.Module):
    def __init__(self, horizontal_channels: int, typewell_channels: int, use_typewell: bool, config: dict[str, Any]) -> None:
        super().__init__()
        branch = next(item for item in config["branches"] if item["name"] == "state_tcn")
        width = int(branch["channels"])
        self.use_typewell = use_typewell
        self.horizontal = SequenceEncoder(
            horizontal_channels,
            width,
            int(branch["kernel"]),
            [int(v) for v in branch["dilations"]],
            int(branch["residual_stacks"]),
            float(branch["dropout"]),
        )
        if use_typewell:
            self.typewell = SequenceEncoder(typewell_channels, width, int(branch["kernel"]), [1, 2, 4], 1, float(branch["dropout"]))
            head_in = 4 * width
        else:
            self.typewell = None
            head_in = 2 * width
        self.head = nn.Sequential(nn.Linear(head_in, width), nn.GELU(), nn.Dropout(float(branch["dropout"])), nn.Linear(width, 1))

    def forward(self, horizontal: torch.Tensor, typewell: torch.Tensor) -> torch.Tensor:
        features = [self.horizontal(horizontal)]
        if self.use_typewell:
            assert self.typewell is not None
            features.append(self.typewell(typewell))
        return self.head(torch.cat(features, dim=1)).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def sequence_channel_scaler(horizontal: np.ndarray, typewell: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = horizontal[indices].astype(np.float64)
    t = typewell[indices].astype(np.float64)
    h_mean = h.mean(axis=(0, 1))
    h_std = h.std(axis=(0, 1))
    t_mean = t.mean(axis=(0, 1))
    t_std = t.std(axis=(0, 1))
    h_std[h_std < 1e-6] = 1.0
    t_std[t_std < 1e-6] = 1.0
    return h_mean.astype(np.float32), h_std.astype(np.float32), t_mean.astype(np.float32), t_std.astype(np.float32)


def scale_sequences(horizontal: np.ndarray, typewell: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    h_mean, h_std, t_mean, t_std = params
    return ((horizontal - h_mean) / h_std).astype(np.float32), ((typewell - t_mean) / t_std).astype(np.float32)


def fit_tcn_once(
    horizontal: np.ndarray,
    typewell: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray | None,
    config: dict[str, Any],
    *,
    use_typewell: bool,
    seed: int,
    epochs: int | None = None,
) -> tuple[StateTCN, int, float]:
    set_seed(seed)
    training = config["training"]
    model = StateTCN(horizontal.shape[2], typewell.shape[2], use_typewell, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    y_mean = float(np.mean(y[train_idx]))
    y_std = float(np.std(y[train_idx]))
    if y_std < 1e-6:
        y_std = 1.0
    train_ds = TensorDataset(
        torch.from_numpy(horizontal[train_idx]),
        torch.from_numpy(typewell[train_idx]),
        torch.from_numpy(((y[train_idx] - y_mean) / y_std).astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=int(training["batch_size"]), shuffle=True, generator=generator, num_workers=0)
    max_epochs = int(epochs if epochs is not None else training["maximum_epochs"])
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = max_epochs
    best_mae = float("inf")
    patience = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for h_batch, t_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(h_batch, t_batch)
            loss = torch.mean(torch.abs(pred - y_batch))
            loss.backward()
            optimizer.step()
        if valid_idx is not None:
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(horizontal[valid_idx]), torch.from_numpy(typewell[valid_idx])).numpy() * y_std + y_mean
            mae = float(np.mean(np.abs(pred - y[valid_idx])))
            if mae < best_mae - 1e-8:
                best_mae = mae
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= int(training["early_stop_patience"]):
                    break
    if valid_idx is not None and best_state is not None:
        model.load_state_dict(best_state)
    model._target_mean = y_mean  # type: ignore[attr-defined]
    model._target_std = y_std  # type: ignore[attr-defined]
    return model, best_epoch, best_mae


def predict_tcn(model: StateTCN, horizontal: np.ndarray, typewell: np.ndarray, indices: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(horizontal[indices]), torch.from_numpy(typewell[indices])).numpy()
    return pred.astype(float) * float(model._target_std) + float(model._target_mean)  # type: ignore[attr-defined]


def perturb_sequences(horizontal: np.ndarray, typewell: np.ndarray, mode: str, visible_bins: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    h = horizontal.copy()
    t = typewell.copy()
    if mode == "reverse_typewell":
        t = t[:, ::-1, :].copy()
    elif mode == "circular_shift_horizontal_gr":
        shift = 53
        for channel in GR_CHANNELS:
            h[:, :, channel] = np.roll(h[:, :, channel], shift=shift, axis=1)
    elif mode == "remove_visible_tvt":
        h[:, :, VISIBLE_TVT_CHANNELS] = 0.0
    elif mode == "matched_noise_post_boundary":
        rng = np.random.default_rng(seed)
        channels = [1, 2, 3, 4, 5, 6, 7]
        for channel in channels:
            values = h[:, visible_bins:, channel]
            mean = float(values.mean())
            std = float(values.std())
            if std < 1e-6:
                std = 1.0
            h[:, visible_bins:, channel] = rng.normal(mean, std, size=values.shape).astype(np.float32)
    else:
        raise ValueError(f"unknown perturbation: {mode}")
    return h, t


def flattened_raw(horizontal: np.ndarray, typewell: np.ndarray) -> np.ndarray:
    return np.concatenate([horizontal.reshape(len(horizontal), -1), typewell.reshape(len(typewell), -1)], axis=1).astype(np.float32)


def branch_metric_rows(branch_predictions: dict[str, np.ndarray], y: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch, repeated in branch_predictions.items():
        average = np.mean(repeated, axis=0)
        metric = finite_metrics(y, average)
        rows.append({"branch": branch, "scope": "repeated_map_average", **metric})
        for map_index in range(repeated.shape[0]):
            rows.append({"branch": branch, "scope": f"map_{map_index}", **finite_metrics(y, repeated[map_index])})
    return rows


def group_metric_rows(branch_predictions: dict[str, np.ndarray], tensors: Tensors) -> list[dict[str, Any]]:
    systems = {
        "legacy_spatial": tensors.legacy_spatial,
        "legacy_typewell": tensors.legacy_typewell,
        "legal_covariate_kmeans": tensors.legal_kmeans,
        "spatial_2d_kmeans": tensors.spatial_2d_kmeans,
    }
    rows: list[dict[str, Any]] = []
    for branch, repeated in branch_predictions.items():
        average = np.mean(repeated, axis=0)
        for system, groups in systems.items():
            for group in sorted(set(int(v) for v in groups)):
                idx = np.flatnonzero(groups == group)
                rows.append({"branch": branch, "group_system": system, "group": group, **finite_metrics(tensors.target[idx], average[idx])})
    return rows


def run(config: dict[str, Any], output_dir: Path, *, build_only: bool = False, force_rebuild: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tensors = load_tensors(config, output_dir, force_rebuild=force_rebuild)
    if build_only:
        return {"status": "BUILD_ONLY_PASS", "tensor_manifest": tensors.tensor_manifest}

    contexts = make_contexts(tensors)
    n_wells = len(tensors.well_ids)
    branch_names = [item["name"] for item in config["branches"]]
    predictions = {name: np.full((5, n_wells), np.nan, dtype=float) for name in branch_names}
    control_prediction_maps = {name: np.full((5, n_wells), np.nan, dtype=float) for name in [
        "reverse_typewell", "circular_shift_horizontal_gr", "remove_visible_tvt", "matched_noise_post_boundary"
    ]}
    context_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    tcn_selected_epochs: dict[tuple[str, str], int] = {}
    raw_matrix = flattened_raw(tensors.horizontal, tensors.typewell)
    alphas = [10.0, 100.0, 1000.0]
    visible_bins = int(config["sequence_grid"]["visible_prefix_bins"])

    for context in contexts:
        map_index = context.map_index
        test = context.test_idx
        outer_y = tensors.target[context.outer_train_idx]
        predictions["mean_target"][map_index, test] = float(np.mean(outer_y))

        for name, matrix in [("summary_ridge", tensors.summary_features), ("raw_binned_ridge", raw_matrix), ("coordinate_only_control", tensors.coordinate_flat)]:
            pred, alpha, metadata, _ = ridge_context(matrix, tensors.target, context, alphas)
            predictions[name][map_index, test] = pred
            hyper_rows.append({"context": context.context_id, "branch": name, **metadata})
            context_rows.append({"context": context.context_id, "branch": name, **finite_metrics(tensors.target[test], pred)})

        for branch_index, branch_name in enumerate(["state_tcn", "state_tcn_no_typewell", "state_tcn_no_hidden_gr"]):
            use_typewell = branch_name != "state_tcn_no_typewell"
            h_all = tensors.horizontal.copy()
            t_all = tensors.typewell.copy()
            if branch_name == "state_tcn_no_hidden_gr":
                h_all[:, visible_bins:, GR_CHANNELS] = 0.0
            scale_params = sequence_channel_scaler(h_all, t_all, context.train_idx)
            h_scaled, t_scaled = scale_sequences(h_all, t_all, scale_params)
            seed = int(config["training"]["seed"]) + map_index * 100 + context.outer_fold * 10 + branch_index
            _, selected_epoch, inner_mae = fit_tcn_once(
                h_scaled, t_scaled, tensors.target, context.train_idx, context.valid_idx, config,
                use_typewell=use_typewell, seed=seed,
            )
            final_scale = sequence_channel_scaler(h_all, t_all, context.outer_train_idx)
            h_final, t_final = scale_sequences(h_all, t_all, final_scale)
            final_model, _, _ = fit_tcn_once(
                h_final, t_final, tensors.target, context.outer_train_idx, None, config,
                use_typewell=use_typewell, seed=seed, epochs=selected_epoch,
            )
            pred = predict_tcn(final_model, h_final, t_final, test)
            predictions[branch_name][map_index, test] = pred
            tcn_selected_epochs[(context.context_id, branch_name)] = selected_epoch
            hyper_rows.append({"context": context.context_id, "branch": branch_name, "selected_epoch": selected_epoch, "inner_mae_ft": inner_mae, "seed": seed})
            context_rows.append({"context": context.context_id, "branch": branch_name, **finite_metrics(tensors.target[test], pred)})

            if branch_name == "state_tcn":
                for control_index, mode in enumerate(control_prediction_maps):
                    pert_h, pert_t = perturb_sequences(h_all, t_all, mode, visible_bins, seed + 1000 + control_index)
                    pert_h, pert_t = scale_sequences(pert_h, pert_t, final_scale)
                    control_prediction_maps[mode][map_index, test] = predict_tcn(final_model, pert_h, pert_t, test)

    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"incomplete predictions for {name}")
    for name, values in control_prediction_maps.items():
        if not np.isfinite(values).all():
            raise ValueError(f"incomplete control predictions for {name}")

    branch_rows = branch_metric_rows(predictions, tensors.target)
    group_rows = group_metric_rows(predictions, tensors)
    average_metrics = {row["branch"]: row for row in branch_rows if row["scope"] == "repeated_map_average"}
    eligible = ["raw_binned_ridge", "state_tcn", "state_tcn_no_typewell", "state_tcn_no_hidden_gr"]
    selected_branch = sorted(eligible, key=lambda name: (-float(average_metrics[name]["pearson"]), float(average_metrics[name]["mae_ft"]), name))[0]

    permuted = np.full((5, n_wells), np.nan, dtype=float)
    duplicate_raw = np.full((5, n_wells), np.nan, dtype=float)
    for context in contexts:
        map_index = context.map_index
        test = context.test_idx
        seed = int(config["training"]["seed"]) + map_index * 100 + context.outer_fold * 10 + 9000
        rng = np.random.default_rng(seed)
        permuted_y = tensors.target.copy()
        permuted_y[context.outer_train_idx] = permuted_y[rng.permutation(context.outer_train_idx)]
        if selected_branch == "raw_binned_ridge":
            selected_alpha = next(float(r["alpha"]) for r in hyper_rows if r["context"] == context.context_id and r["branch"] == selected_branch)
            permuted[map_index, test] = ridge_predict_with_fixed(
                raw_matrix[context.outer_train_idx], raw_matrix[test], permuted_y[context.outer_train_idx], selected_alpha,
            )
        else:
            use_typewell = selected_branch != "state_tcn_no_typewell"
            h_all = tensors.horizontal.copy()
            t_all = tensors.typewell.copy()
            if selected_branch == "state_tcn_no_hidden_gr":
                h_all[:, visible_bins:, GR_CHANNELS] = 0.0
            scale_params = sequence_channel_scaler(h_all, t_all, context.outer_train_idx)
            h_scaled, t_scaled = scale_sequences(h_all, t_all, scale_params)
            selected_epoch = tcn_selected_epochs[(context.context_id, selected_branch)]
            model, _, _ = fit_tcn_once(
                h_scaled, t_scaled, permuted_y, context.outer_train_idx, None, config,
                use_typewell=use_typewell, seed=seed, epochs=selected_epoch,
            )
            permuted[map_index, test] = predict_tcn(model, h_scaled, t_scaled, test)

        selected_alpha = next(float(r["alpha"]) for r in hyper_rows if r["context"] == context.context_id and r["branch"] == "raw_binned_ridge")
        duplicate_count = max(1, len(context.outer_train_idx) // 10)
        duplicate_idx = np.sort(context.outer_train_idx)[:duplicate_count]
        train_idx = np.concatenate([context.outer_train_idx, duplicate_idx])
        duplicate_raw[map_index, test] = ridge_predict_with_fixed(raw_matrix[train_idx], raw_matrix[test], tensors.target[train_idx], selected_alpha)

    control_rows: list[dict[str, Any]] = []
    selected_main = np.mean(predictions[selected_branch], axis=0)
    main_metric = finite_metrics(tensors.target, selected_main)
    perm_metric = finite_metrics(tensors.target, np.mean(permuted, axis=0))
    control_rows.append({"control": "outer_training_target_permutation", "branch": selected_branch, **perm_metric})
    for name, values in control_prediction_maps.items():
        metric = finite_metrics(tensors.target, np.mean(values, axis=0))
        destruction = 1.0 - abs(float(metric["pearson"])) / max(abs(float(average_metrics["state_tcn"]["pearson"])), 1e-12)
        control_rows.append({"control": name, "branch": "state_tcn", "signal_destruction_fraction": destruction, **metric})
    duplicate_metric = finite_metrics(tensors.target, np.mean(duplicate_raw, axis=0))
    duplicate_delta = float(np.max(np.abs(np.mean(duplicate_raw, axis=0) - np.mean(predictions["raw_binned_ridge"], axis=0))))
    control_rows.append({"control": "duplicate_wells_without_inverse_weight", "branch": "raw_binned_ridge", "maximum_prediction_delta_ft": duplicate_delta, **duplicate_metric})

    summary_metric = average_metrics["summary_ridge"]
    coordinate_metric = average_metrics["coordinate_only_control"]
    selected_group_rows = [row for row in group_rows if row["branch"] == selected_branch]
    legacy_floor = min(float(row["pearson"]) for row in selected_group_rows if row["group_system"] in {"legacy_spatial", "legacy_typewell"})
    new_floor = min(float(row["pearson"]) for row in selected_group_rows if row["group_system"] in {"legal_covariate_kmeans", "spatial_2d_kmeans"})
    destructive = [row for row in control_rows if row["control"] in {"reverse_typewell", "circular_shift_horizontal_gr", "remove_visible_tvt", "matched_noise_post_boundary"}]
    min_destruction = min(float(row.get("signal_destruction_fraction", float("nan"))) for row in destructive)
    gate_cfg = config["go_gate"]
    gates = {
        "minimum_pearson": float(main_metric["pearson"]) >= float(gate_cfg["minimum_pearson"]),
        "minimum_spearman": float(main_metric["spearman"]) >= float(gate_cfg["minimum_spearman"]),
        "maximum_mae_ft": float(main_metric["mae_ft"]) <= float(gate_cfg["maximum_mae_ft"]),
        "minimum_every_legacy_group_pearson": legacy_floor >= float(gate_cfg["minimum_every_legacy_group_pearson"]),
        "minimum_every_new_group_pearson": new_floor >= float(gate_cfg["minimum_every_new_group_pearson"]),
        "minimum_pearson_advantage_over_summary_ridge": float(main_metric["pearson"]) - float(summary_metric["pearson"]) >= float(gate_cfg["minimum_pearson_advantage_over_summary_ridge"]),
        "minimum_mae_advantage_over_summary_ridge_ft": float(summary_metric["mae_ft"]) - float(main_metric["mae_ft"]) >= float(gate_cfg["minimum_mae_advantage_over_summary_ridge_ft"]),
        "minimum_pearson_advantage_over_coordinate_control": float(main_metric["pearson"]) - float(coordinate_metric["pearson"]) >= float(gate_cfg["minimum_pearson_advantage_over_coordinate_control"]),
        "maximum_permuted_abs_pearson": abs(float(perm_metric["pearson"])) <= float(gate_cfg["maximum_permuted_abs_pearson"]),
        "minimum_control_gain_destruction_fraction": min_destruction >= float(gate_cfg["minimum_control_gain_destruction_fraction"]),
    }
    stop_cfg = config["stop_gate"]
    hard_stop = float(main_metric["pearson"]) < float(stop_cfg["maximum_pearson_for_hard_stop"]) or float(main_metric["mae_ft"]) > float(stop_cfg["minimum_mae_ft_for_hard_stop"])
    if all(gates.values()):
        decision = "GO_WRITE_T033_PREREGISTRATION_ONLY"
    elif hard_stop:
        decision = "HARD_STOP_CLOSE_COMPACT_RAW_SEQUENCE_STATE_RECOVERY"
    else:
        decision = "RESEARCH_ONLY_CLOSE_WITHOUT_JOINT_TVT"

    prediction_rows: list[dict[str, Any]] = []
    for i, well_id in enumerate(tensors.well_ids):
        row: dict[str, Any] = {
            "well_id": well_id,
            "target": float(tensors.target[i]),
            "legacy_spatial": int(tensors.legacy_spatial[i]),
            "legacy_typewell": int(tensors.legacy_typewell[i]),
            "legal_covariate_kmeans": int(tensors.legal_kmeans[i]),
            "spatial_2d_kmeans": int(tensors.spatial_2d_kmeans[i]),
        }
        for name, values in predictions.items():
            row[name] = float(np.mean(values[:, i]))
        row["selected_branch_prediction"] = float(selected_main[i])
        row["permuted_target_prediction"] = float(np.mean(permuted[:, i]))
        prediction_rows.append(row)

    implementation = {
        "schema_version": 1,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "threads": 2,
        "architecture": "depthwise-separable symmetric-padded TCN; 64 channels; kernel 5; registered dilations and two stacks; masked-fixed-bin mean plus learned attention pooling",
        "inner_validation": "same immutable map, deterministic fold (outer+1)%5",
        "refit": "full outer-training refit for the selected epoch",
        "branch_selection": "highest repeated-map-average Pearson, then lowest MAE, then branch name among raw_binned_ridge and three TCN branches",
        "primary_gate_aggregation": "mean of five whole-well OOF map predictions",
        "control_signal_destruction": "1 - abs(control Pearson)/abs(state_tcn Pearson)",
    }
    json_dump(output_dir / "implementation_receipt.json", implementation)
    write_csv(output_dir / "branch_metrics.csv", branch_rows)
    write_csv(output_dir / "context_metrics.csv", context_rows)
    write_csv(output_dir / "group_metrics.csv", group_rows)
    write_csv(output_dir / "selected_hyperparameters.csv", hyper_rows)
    write_csv(output_dir / "control_metrics.csv", control_rows)
    write_csv(output_dir / "predictions.csv", prediction_rows)

    summary = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "COMPLETE",
        "decision": decision,
        "selected_branch": selected_branch,
        "selected_metrics": main_metric,
        "summary_ridge_metrics": {k: summary_metric[k] for k in ["pearson", "spearman", "mae_ft", "rmse_ft", "count"]},
        "coordinate_control_metrics": {k: coordinate_metric[k] for k in ["pearson", "spearman", "mae_ft", "rmse_ft", "count"]},
        "legacy_group_pearson_floor": legacy_floor,
        "new_group_pearson_floor": new_floor,
        "minimum_destructive_control_signal_destruction_fraction": min_destruction,
        "permuted_target_metrics": perm_metric,
        "gates": gates,
        "hard_stop": hard_stop,
        "runtime_seconds": time.time() - started,
        "wells": n_wells,
        "contexts": len(contexts),
        "target_sha256": sha256_array(tensors.target.astype("<f8")),
    }
    json_dump(output_dir / "summary.json", summary)

    edge_cases = {
        "status": "PASS",
        "checks": {
            "all_branch_predictions_finite": all(np.isfinite(v).all() for v in predictions.values()),
            "all_control_predictions_finite": all(np.isfinite(v).all() for v in control_prediction_maps.values()) and np.isfinite(permuted).all(),
            "five_predictions_per_well": all(v.shape == (5, n_wells) for v in predictions.values()),
            "target_hash": sha256_array(tensors.target.astype("<f8")) == config["target"]["sha256_float64_well_order"],
            "context_count": len(contexts) == int(config["validation"]["repeated_contexts"]),
            "no_partition_overlap": all(not (set(c.test_idx) & set(c.outer_train_idx)) for c in contexts),
            "group_counts": all(len(set(map(int, groups))) == 5 for groups in [tensors.legacy_spatial, tensors.legacy_typewell, tensors.legal_kmeans, tensors.spatial_2d_kmeans]),
            "tensor_shapes": tensors.horizontal.shape == (773, 256, len(H_CHANNELS)) and tensors.typewell.shape == (773, 192, len(T_CHANNELS)),
        },
    }
    edge_cases["status"] = "PASS" if all(edge_cases["checks"].values()) else "FAIL"
    json_dump(output_dir / "edge_cases.json", edge_cases)

    report = [
        "# T031 Result — Raw-sequence latent formation-displacement identifiability gate",
        "",
        f"Decision: **{decision}**",
        "",
        f"Selected legal raw-sequence branch: `{selected_branch}`",
        f"- Pearson: {main_metric['pearson']:.6f}",
        f"- Spearman: {main_metric['spearman']:.6f}",
        f"- MAE: {main_metric['mae_ft']:.6f} ft",
        f"- RMSE: {main_metric['rmse_ft']:.6f} ft",
        f"- Legacy group Pearson floor: {legacy_floor:.6f}",
        f"- New target-free group Pearson floor: {new_floor:.6f}",
        "",
        "## Comparators",
        "",
        f"- Summary ridge Pearson/MAE: {summary_metric['pearson']:.6f} / {summary_metric['mae_ft']:.6f} ft",
        f"- Coordinate-only Pearson/MAE: {coordinate_metric['pearson']:.6f} / {coordinate_metric['mae_ft']:.6f} ft",
        f"- Permuted-target Pearson: {perm_metric['pearson']:.6f}",
        "",
        "## Gate outcomes",
        "",
    ]
    report.extend([f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in gates.items()])
    report.extend([
        "",
        "This task is an auxiliary-state gate only. It does not authorize or emit a competition prediction.",
    ])
    (output_dir / "RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest_files = [
        "implementation_receipt.json", "tensor_manifest.json", "summary.json", "branch_metrics.csv",
        "context_metrics.csv", "group_metrics.csv", "selected_hyperparameters.csv", "control_metrics.csv",
        "predictions.csv", "edge_cases.json", "RESULT.md",
    ]
    manifest = {
        "schema_version": 1,
        "files": [
            {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in manifest_files
        ],
    }
    json_dump(output_dir / "artifact_manifest.json", manifest)
    return summary


def synthetic_self_test(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(123)
    n = 32
    h = np.random.default_rng(1).normal(size=(n, 256, len(H_CHANNELS))).astype(np.float32)
    t = np.random.default_rng(2).normal(size=(n, 192, len(T_CHANNELS))).astype(np.float32)
    y = np.linspace(-2.0, 2.0, n)
    model, epoch, _ = fit_tcn_once(h, t, y, np.arange(24), np.arange(24, 32), config, use_typewell=True, seed=123, epochs=2)
    pred = predict_tcn(model, h, t, np.arange(4))
    return {
        "status": "PASS" if pred.shape == (4,) and np.isfinite(pred).all() and epoch >= 1 else "FAIL",
        "prediction_shape": list(pred.shape),
        "selected_epoch": epoch,
        "finite": bool(np.isfinite(pred).all()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--synthetic-self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.synthetic_self_test:
        print(json.dumps(synthetic_self_test(config), indent=2, sort_keys=True))
        return
    result = run(config, args.output_dir, build_only=args.build_only, force_rebuild=args.force_rebuild)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
