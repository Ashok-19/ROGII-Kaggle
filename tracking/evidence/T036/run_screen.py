#!/usr/bin/env python3
"""T036 candidate-path-conditioned GR evidence ranking.

Candidate construction and features are legal at inference. Hidden TVT is used
only for outer-training labels, inner selection, oracle diagnostics, and held-out
scoring. No package, Kaggle execution, or submission is emitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rogii_validation.gr_path import _calibration, read_well  # noqa: E402

CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = ROOT / "scratch/agents/t036-path-ranker-20260725/main"
T033_PATH = ROOT / "tracking/evidence/T033/run_screen.py"
T031_TENSOR = ROOT / "scratch/agents/t031-state-gate-20260724/main/tensors_v1.npz"
GRID = 128
CENTER_COUNT = 32
ACTION_CAP = 160.0
EVIDENCE_FEATURES = 22
BRANCHES = (
    "robust_alignment",
    "pointwise_ridge",
    "pointwise_hgb",
    "pointwise_extra_trees",
    "pairwise_logistic",
)
CONTROL_BRANCHES = (
    "reversed_typewell_gr",
    "circular_shift_hidden_gr",
    "action_prior_only",
    "shuffled_within_well_training_losses",
    "sign_flipped_selected_action",
)
ALL_REPORTED = BRANCHES + ("nested_best",) + CONTROL_BRANCHES + ("candidate_oracle",)


@dataclass(frozen=True)
class Context:
    map_index: int
    fold: int
    key: str
    train: np.ndarray
    test: np.ndarray
    inner_train: np.ndarray
    inner_valid: np.ndarray


@dataclass(frozen=True)
class EvidenceWell:
    well_id: str
    eligible: bool
    horizontal: np.ndarray
    hidden_positions: np.ndarray
    base_samples: np.ndarray
    type_tvt: np.ndarray
    type_gr: np.ndarray
    hidden_gr_coverage: float
    log_hidden_rows: float
    normalized_hidden_md_span: float
    typewell_tvt_span: float
    e011_support_fraction: float


@dataclass(frozen=True)
class Metric:
    rows: int
    sse: float
    datum_sse: float
    trend_sse: float
    shape_sse: float
    rmse: float


@dataclass
class FittedModel:
    family: str
    model: Any | None
    scaler: StandardScaler | None


@dataclass(frozen=True)
class Selection:
    branch: str
    params: dict[str, Any]
    output_rule: dict[str, Any]
    inner_rmse: float


def load_t033() -> Any:
    spec = importlib.util.spec_from_file_location("t036_t033", T033_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load T033 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_hash_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)


def error_metric(error: np.ndarray) -> Metric:
    e = np.asarray(error, dtype=np.float64)
    n = int(e.size)
    if n <= 0 or not np.isfinite(e).all():
        raise ValueError("metric requires finite nonempty error")
    x = np.arange(n, dtype=np.float64)
    sse = float(e @ e)
    total = float(e.sum())
    datum = total * total / n
    xc = x - x.mean()
    denominator = float(xc @ xc)
    trend = float(xc @ e) ** 2 / denominator if denominator > 0.0 else 0.0
    shape = sse - datum - trend
    tolerance = max(1e-9, sse * 1e-12)
    if shape < 0.0 and abs(shape) <= tolerance:
        shape = 0.0
    if shape < 0.0:
        raise ArithmeticError("negative shape SSE")
    return Metric(n, sse, datum, trend, shape, math.sqrt(sse / n))


def summarize(metrics: Iterable[Metric]) -> dict[str, float | int]:
    values = list(metrics)
    rows = sum(item.rows for item in values)
    sse = sum(item.sse for item in values)
    datum = sum(item.datum_sse for item in values)
    trend = sum(item.trend_sse for item in values)
    shape = sum(item.shape_sse for item in values)
    return {
        "rows": int(rows),
        "wells": int(len(values)),
        "sse": float(sse),
        "rmse": float(math.sqrt(sse / rows)),
        "datum_sse": float(datum),
        "trend_sse": float(trend),
        "shape_sse": float(shape),
    }


def quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def tail_stats(metrics: Sequence[Metric]) -> dict[str, float]:
    rmses = np.asarray([metric.rmse for metric in metrics], dtype=np.float64)
    sses = np.asarray([metric.sse for metric in metrics], dtype=np.float64)
    count = max(1, int(math.ceil(0.05 * len(metrics))))
    worst = np.argsort(sses, kind="mergesort")[-count:]
    return {
        "p90_well_rmse": float(np.quantile(rmses, 0.9)),
        "worst5_sse_share": float(sses[worst].sum() / max(float(sses.sum()), 1e-300)),
    }


def build_contexts(wells: Sequence[Any], fold_maps: Sequence[dict[str, int]]) -> list[Context]:
    contexts: list[Context] = []
    for map_index, assignment in enumerate(fold_maps):
        folds = np.asarray([assignment[well.well_id] for well in wells], dtype=int)
        for fold in range(5):
            test = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            inner_fold = (fold + 1) % 5
            inner_valid = np.flatnonzero(folds == inner_fold)
            inner_train = np.flatnonzero((folds != fold) & (folds != inner_fold))
            if set(test) & set(train) or set(inner_valid) & set(inner_train):
                raise ValueError("context overlap")
            contexts.append(Context(map_index, fold, f"v{map_index + 1}_f{fold}", train, test, inner_train, inner_valid))
    if len(contexts) != 25:
        raise ValueError("expected 25 contexts")
    return contexts


def build_evidence(wells: Sequence[Any], maximum_samples: int) -> list[EvidenceWell]:
    output: list[EvidenceWell] = []
    for source in wells:
        raw = read_well(
            ROOT / "data/train" / f"{source.well_id}__horizontal_well.csv",
            ROOT / "data/train" / f"{source.well_id}__typewell.csv",
            require_truth=True,
        )
        if raw.hidden_rows != source.rows:
            raise ValueError(f"{source.well_id}: hidden row mismatch")
        finite_abs = np.asarray(
            [index for index in range(raw.known_rows, len(raw.md)) if raw.gr[index] is not None],
            dtype=np.int64,
        )
        if finite_abs.size > maximum_samples:
            positions = np.unique(np.rint(np.linspace(0, finite_abs.size - 1, maximum_samples)).astype(int))
            finite_abs = finite_abs[positions]
        calibration = _calibration(raw, raw.typewell, 24)
        eligible = calibration is not None and finite_abs.size >= 24
        if eligible:
            hidden_index = finite_abs - raw.known_rows
            horizontal = np.asarray([float(raw.gr[int(index)]) for index in finite_abs], dtype=np.float64)
            assert calibration is not None
            horizontal = float(calibration["sign"]) * (horizontal - float(calibration["h_mean"])) / max(float(calibration["h_std"]), 1e-9)
            positions = hidden_index / max(1, raw.hidden_rows - 1) * (GRID - 1)
            base_samples = source.e011[hidden_index]
            type_tvt = np.asarray(raw.typewell.tvt, dtype=np.float64)
            type_gr = (np.asarray(raw.typewell.gr, dtype=np.float64) - float(calibration["r_mean"])) / max(float(calibration["r_std"]), 1e-9)
        else:
            horizontal = np.empty(0, dtype=np.float64)
            positions = np.empty(0, dtype=np.float64)
            base_samples = np.empty(0, dtype=np.float64)
            type_tvt = np.asarray(raw.typewell.tvt, dtype=np.float64)
            type_gr = np.zeros(len(type_tvt), dtype=np.float64)
        hidden_md = np.asarray(raw.md[raw.known_rows :], dtype=np.float64)
        total_md_span = max(float(raw.md[-1] - raw.md[0]), 1e-9)
        e011_support = float(np.mean((source.e011 >= type_tvt[0]) & (source.e011 <= type_tvt[-1])))
        output.append(EvidenceWell(
            well_id=source.well_id,
            eligible=bool(eligible),
            horizontal=horizontal,
            hidden_positions=positions,
            base_samples=base_samples,
            type_tvt=type_tvt,
            type_gr=type_gr,
            hidden_gr_coverage=float(raw.hidden_gr_coverage),
            log_hidden_rows=float(math.log1p(raw.hidden_rows)),
            normalized_hidden_md_span=float((hidden_md[-1] - hidden_md[0]) / total_md_span) if hidden_md.size > 1 else 0.0,
            typewell_tvt_span=float(type_tvt[-1] - type_tvt[0]),
            e011_support_fraction=e011_support,
        ))
    return output


def candidate_actions(profiles: np.ndarray, train: np.ndarray, map_index: int, fold: int) -> np.ndarray:
    model = KMeans(
        n_clusters=CENTER_COUNT,
        random_state=33000 + 100 * map_index + 10 * fold + CENTER_COUNT,
        n_init=16,
        max_iter=300,
        algorithm="lloyd",
    )
    model.fit(profiles[train])
    centers = np.clip(model.cluster_centers_, -ACTION_CAP, ACTION_CAP)
    blocks = [centers]
    for weight in (0.25, 0.5, 0.75):
        blocks.append(np.asarray([
            (1.0 - weight) * centers[left] + weight * centers[right]
            for left in range(CENTER_COUNT)
            for right in range(left + 1, CENTER_COUNT)
        ], dtype=np.float64))
    nonzero = np.vstack(blocks)
    if nonzero.shape != (1520, GRID) or np.unique(np.round(nonzero, 12), axis=0).shape[0] != 1520:
        raise ValueError("candidate dictionary is not exactly 1520 unique actions")
    actions = np.vstack([nonzero, np.zeros((1, GRID), dtype=np.float64)])
    if actions.shape != (1521, GRID) or not np.isfinite(actions).all():
        raise ValueError("candidate action contract failed")
    return actions


def action_prior_features(actions: np.ndarray) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, GRID, dtype=np.float64)
    slope = actions @ t / float(t @ t)
    return np.column_stack([
        actions[:, 0],
        actions.mean(axis=1),
        actions[:, -1],
        slope,
        np.sqrt(np.mean(actions * actions, axis=1)),
        np.mean(np.abs(np.diff(actions, axis=1)), axis=1),
        np.sqrt(np.mean(np.diff(actions, n=2, axis=1) ** 2, axis=1)),
        np.max(np.abs(actions), axis=1),
    ])


def masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = mask.sum(axis=1)
    return np.divide((values * mask).sum(axis=1), np.maximum(counts, 1), out=np.zeros(values.shape[0]), where=counts > 0)


def masked_correlation(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = mask.sum(axis=1).astype(np.float64)
    sx = (mask * x).sum(axis=1)
    sy = (mask * y).sum(axis=1)
    sxx = (mask * x * x).sum(axis=1)
    syy = (mask * y * y).sum(axis=1)
    sxy = (mask * x * y).sum(axis=1)
    covariance = sxy - sx * sy / np.maximum(counts, 1.0)
    vx = sxx - sx * sx / np.maximum(counts, 1.0)
    vy = syy - sy * sy / np.maximum(counts, 1.0)
    denominator = np.sqrt(np.maximum(vx * vy, 1e-18))
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=(counts >= 3) & (vx > 1e-12) & (vy > 1e-12))


def huber(values: np.ndarray, delta: float = 1.5) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute <= delta, 0.5 * absolute * absolute, delta * (absolute - 0.5 * delta))


def candidate_features(evidence: EvidenceWell, actions: np.ndarray, mode: str = "main") -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    priors = action_prior_features(actions)
    count = len(actions)
    if not evidence.eligible:
        evidence_block = np.zeros((count, EVIDENCE_FEATURES), dtype=np.float64)
        path_block = np.zeros((count, 4), dtype=np.float64)
    else:
        positions = evidence.hidden_positions
        left = np.floor(positions).astype(np.int64)
        right = np.minimum(left + 1, GRID - 1)
        wr = positions - left
        wl = 1.0 - wr
        action_samples = actions[:, left] * wl[None, :] + actions[:, right] * wr[None, :]
        candidate_tvt = evidence.base_samples[None, :] + action_samples
        type_gr = evidence.type_gr[::-1] if mode == "reversed_typewell_gr" else evidence.type_gr
        horizontal = evidence.horizontal.copy()
        if mode == "circular_shift_hidden_gr" and horizontal.size:
            horizontal = np.roll(horizontal, max(1, horizontal.size // 3))
        indices = np.searchsorted(evidence.type_tvt, candidate_tvt, side="left")
        valid = (candidate_tvt >= evidence.type_tvt[0]) & (candidate_tvt <= evidence.type_tvt[-1])
        high = np.clip(indices, 1, len(evidence.type_tvt) - 1)
        low = high - 1
        span = evidence.type_tvt[high] - evidence.type_tvt[low]
        fraction = np.divide(candidate_tvt - evidence.type_tvt[low], span, out=np.zeros_like(candidate_tvt), where=span > 0.0)
        reference = type_gr[low] * (1.0 - fraction) + type_gr[high] * fraction
        reference = np.where(valid, reference, 0.0)
        horizontal_matrix = np.broadcast_to(horizontal[None, :], reference.shape)
        difference = horizontal_matrix - reference
        support = valid.mean(axis=1)
        mismatch_huber = masked_mean(huber(difference), valid)
        mismatch_mae = masked_mean(np.abs(difference), valid)
        mismatch_rmse = np.sqrt(masked_mean(difference * difference, valid))
        mismatch_bias = masked_mean(difference, valid)
        mismatch_pearson = masked_correlation(horizontal_matrix, reference, valid)
        dot = masked_mean(horizontal_matrix * reference, valid)
        xnorm = np.sqrt(masked_mean(horizontal_matrix * horizontal_matrix, valid))
        ynorm = np.sqrt(masked_mean(reference * reference, valid))
        cosine = np.divide(dot, np.maximum(xnorm * ynorm, 1e-12))
        ref_mean = masked_mean(reference, valid)
        ref_var = masked_mean((reference - ref_mean[:, None]) ** 2, valid)
        ref_std = np.sqrt(np.maximum(ref_var, 0.0))
        hstd = float(np.std(horizontal))
        std_ratio = ref_std / max(hstd, 1e-9)
        pair = valid[:, :-1] & valid[:, 1:]
        dh = np.diff(horizontal)[None, :]
        dr = np.diff(reference, axis=1)
        derivative_difference = dh - dr
        derivative_huber = masked_mean(huber(derivative_difference), pair)
        derivative_pearson = masked_correlation(np.broadcast_to(dh, dr.shape), dr, pair)
        derivative_sign = masked_mean((np.sign(dh) == np.sign(dr)).astype(np.float64), pair)
        if reference.shape[1] >= 3:
            triple = pair[:, :-1] & pair[:, 1:]
            d2h = np.diff(horizontal, n=2)[None, :]
            d2r = np.diff(reference, n=2, axis=1)
            second_huber = masked_mean(huber(d2h - d2r), triple)
        else:
            second_huber = np.zeros(count, dtype=np.float64)
        segment_huber: list[np.ndarray] = []
        segment_corr: list[np.ndarray] = []
        for segment in range(4):
            start = segment * horizontal.size // 4
            end = (segment + 1) * horizontal.size // 4
            local_mask = valid[:, start:end]
            local_h = np.broadcast_to(horizontal[None, start:end], local_mask.shape)
            local_r = reference[:, start:end]
            segment_huber.append(masked_mean(huber(local_h - local_r), local_mask))
            segment_corr.append(masked_correlation(local_h, local_r, local_mask))
        evidence_block = np.column_stack([
            support,
            valid.sum(axis=1) / max(1, horizontal.size),
            mismatch_huber,
            mismatch_mae,
            mismatch_rmse,
            mismatch_bias,
            mismatch_pearson,
            cosine,
            ref_std,
            std_ratio,
            derivative_huber,
            derivative_pearson,
            derivative_sign,
            second_huber,
            *segment_huber,
            *segment_corr,
        ])
        type_span = max(float(evidence.type_tvt[-1] - evidence.type_tvt[0]), 1e-9)
        path_block = np.column_stack([
            (candidate_tvt[:, 0] - evidence.type_tvt[0]) / type_span,
            (candidate_tvt[:, -1] - evidence.type_tvt[0]) / type_span,
            (candidate_tvt < evidence.type_tvt[0]).mean(axis=1),
            (candidate_tvt > evidence.type_tvt[-1]).mean(axis=1),
        ])
    constants = np.tile(np.asarray([
        evidence.hidden_gr_coverage,
        evidence.log_hidden_rows,
        evidence.normalized_hidden_md_span,
        evidence.typewell_tvt_span,
        evidence.e011_support_fraction,
    ], dtype=np.float64), (count, 1))
    features = np.column_stack([evidence_block, priors, path_block, constants])
    if mode == "action_prior_only":
        features[:, :EVIDENCE_FEATURES] = 0.0
    if features.shape[1] != 39 or not np.isfinite(features).all():
        raise ValueError(f"candidate feature contract failed: {features.shape}")
    return features.astype(np.float32)


def candidate_sses(t033: Any, well: Any, actions: np.ndarray) -> np.ndarray:
    values = np.asarray(t033.profile_sse(well, actions), dtype=np.float64)
    if values.shape != (len(actions),) or not np.isfinite(values).all():
        raise ValueError("candidate SSE contract failed")
    return values


def deterministic_remaining(well_id: str, context_key: str, candidates: Iterable[int]) -> list[int]:
    return sorted(candidates, key=lambda index: (stable_hash_int(f"{context_key}|{well_id}|{index}"), index))


def pointwise_indices(well_id: str, context_key: str, sses: np.ndarray) -> np.ndarray:
    order = np.argsort(sses, kind="mergesort")
    chosen: list[int] = [len(sses) - 1]
    chosen.extend(map(int, order[:8]))
    chosen.extend(map(int, order[-8:]))
    middle = len(order) // 2
    chosen.extend(map(int, order[middle - 4 : middle + 4]))
    unique: list[int] = []
    seen: set[int] = set()
    for index in chosen:
        if index not in seen:
            unique.append(index)
            seen.add(index)
    remaining = [index for index in range(len(sses)) if index not in seen]
    for index in deterministic_remaining(well_id, context_key, remaining):
        unique.append(index)
        if len(unique) == 64:
            break
    if len(unique) != 64:
        raise ValueError("pointwise sampling did not yield 64 candidates")
    return np.asarray(unique, dtype=np.int64)


def shuffled_sses(well_id: str, context_key: str, sses: np.ndarray) -> np.ndarray:
    shift = 1 + stable_hash_int(f"shuffle|{context_key}|{well_id}") % (len(sses) - 1)
    return np.roll(sses, int(shift))


def build_pointwise_dataset(
    t033: Any,
    wells: Sequence[Any],
    evidence: Sequence[EvidenceWell],
    well_indices: np.ndarray,
    actions: np.ndarray,
    context_key: str,
    mode: str,
    shuffle_targets: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for index in well_indices:
        ev = evidence[int(index)]
        if not ev.eligible:
            continue
        sses = candidate_sses(t033, wells[int(index)], actions)
        target_sses = shuffled_sses(ev.well_id, context_key, sses) if shuffle_targets else sses
        selected = pointwise_indices(ev.well_id, context_key, target_sses)
        x_rows.append(candidate_features(ev, actions[selected], mode=mode))
        baseline = max(float(target_sses[-1] / wells[int(index)].rows), 1e-12)
        y_rows.append(np.log((target_sses[selected] / wells[int(index)].rows + 1e-6) / (baseline + 1e-6)))
    if not x_rows:
        raise ValueError("pointwise dataset is empty")
    x = np.vstack(x_rows).astype(np.float64)
    y = np.concatenate(y_rows).astype(np.float64)
    return x, y


def pair_indices(well_id: str, context_key: str, sses: np.ndarray) -> list[tuple[int, int]]:
    order = np.argsort(sses, kind="mergesort")
    pairs: list[tuple[int, int]] = [(int(order[r]), int(order[-1 - r])) for r in range(32)]
    zero = len(sses) - 1
    remaining = [index for index in range(len(sses) - 1)]
    hashed = deterministic_remaining(well_id, f"pair|{context_key}", remaining)[:32]
    pairs.extend((zero, int(index)) for index in hashed)
    return pairs


def build_pairwise_dataset(
    t033: Any,
    wells: Sequence[Any],
    evidence: Sequence[EvidenceWell],
    well_indices: np.ndarray,
    actions: np.ndarray,
    context_key: str,
    mode: str,
    shuffle_targets: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for index in well_indices:
        ev = evidence[int(index)]
        if not ev.eligible:
            continue
        original = candidate_sses(t033, wells[int(index)], actions)
        sses = shuffled_sses(ev.well_id, context_key, original) if shuffle_targets else original
        pairs = pair_indices(ev.well_id, context_key, sses)
        unique = sorted({candidate for pair in pairs for candidate in pair})
        feature_values = candidate_features(ev, actions[unique], mode=mode).astype(np.float64)
        lookup = {candidate: feature_values[position] for position, candidate in enumerate(unique)}
        local_x: list[np.ndarray] = []
        local_y: list[float] = []
        for first, second in pairs:
            if abs(float(sses[first] - sses[second])) <= 1e-12:
                continue
            better, worse = (first, second) if sses[first] < sses[second] else (second, first)
            difference = lookup[better] - lookup[worse]
            local_x.extend([difference, -difference])
            local_y.extend([1.0, 0.0])
        if local_x:
            x_rows.append(np.vstack(local_x))
            y_rows.append(np.asarray(local_y, dtype=np.float64))
    if not x_rows:
        raise ValueError("pairwise dataset is empty")
    return np.vstack(x_rows), np.concatenate(y_rows)


def model_parameters(config: dict[str, Any], branch: str) -> list[dict[str, Any]]:
    item = next(value for value in config["branches"] if value["name"] == branch)
    if branch == "robust_alignment":
        return [{}]
    if branch == "pointwise_ridge":
        return [{"alpha": float(alpha)} for alpha in item["alphas"]]
    if branch == "pointwise_hgb":
        return [
            {"max_leaf_nodes": int(leaves), "l2_regularization": float(l2)}
            for leaves in item["max_leaf_nodes"]
            for l2 in item["l2_regularization"]
        ]
    if branch == "pointwise_extra_trees":
        return [
            {"max_features": float(max_features), "min_samples_leaf": int(leaf)}
            for max_features in item["max_features"]
            for leaf in item["min_samples_leaf"]
        ]
    if branch == "pairwise_logistic":
        return [{"C": float(value)} for value in item["C"]]
    raise ValueError(branch)


def output_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [{"kind": "hard", "top_k": 1, "temperature": None}]
    for top_k in config["output_rules"]["soft_top_k"]:
        for temperature in config["output_rules"]["temperatures"]:
            rules.append({"kind": "soft", "top_k": int(top_k), "temperature": float(temperature)})
    return rules


def fit_model(branch: str, params: dict[str, Any], x: np.ndarray, y: np.ndarray, seed: int, config: dict[str, Any]) -> FittedModel:
    if branch == "robust_alignment":
        return FittedModel(branch, None, None)
    if branch == "pointwise_ridge":
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=params["alpha"]).fit(scaler.transform(x), y)
        return FittedModel(branch, model, scaler)
    if branch == "pointwise_hgb":
        item = next(value for value in config["branches"] if value["name"] == branch)
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=params["max_leaf_nodes"],
            l2_regularization=params["l2_regularization"],
            learning_rate=float(item["learning_rate"]),
            max_iter=int(item["max_iter"]),
            min_samples_leaf=int(item["min_samples_leaf"]),
            random_state=seed,
        ).fit(x, y)
        return FittedModel(branch, model, None)
    if branch == "pointwise_extra_trees":
        item = next(value for value in config["branches"] if value["name"] == branch)
        model = ExtraTreesRegressor(
            n_estimators=int(item["n_estimators"]),
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=int(item["n_jobs"]),
        ).fit(x, y)
        return FittedModel(branch, model, None)
    if branch == "pairwise_logistic":
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=params["C"], max_iter=500, fit_intercept=False, random_state=seed).fit(scaler.transform(x), y)
        return FittedModel(branch, model, scaler)
    raise ValueError(branch)


def score_model(fitted: FittedModel, features: np.ndarray) -> np.ndarray:
    if fitted.family == "robust_alignment":
        return features[:, 2].astype(np.float64) + 0.25 * features[:, 10] + 2.0 * (1.0 - features[:, 0])
    transformed = fitted.scaler.transform(features) if fitted.scaler is not None else features
    if fitted.family == "pairwise_logistic":
        return -np.asarray(fitted.model.decision_function(transformed), dtype=np.float64)
    return np.asarray(fitted.model.predict(transformed), dtype=np.float64)


def action_tie_keys(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(1, actions.shape[1] + 1, dtype=np.float64)
    first_weights = np.sin(positions * 0.7548776662466927)
    second_weights = np.cos(positions * 0.5698402909980532)
    return actions @ first_weights, actions @ second_weights


def selected_action(scores: np.ndarray, actions: np.ndarray, rule: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(actions),) or not np.isfinite(scores).all():
        raise ValueError("candidate scores must be finite and aligned with actions")
    tie_first, tie_second = action_tie_keys(np.asarray(actions, dtype=np.float64))
    order = np.lexsort((tie_second, tie_first, scores))
    if rule["kind"] == "hard":
        return actions[int(order[0])].copy(), order[:16]
    top = order[: int(rule["top_k"])]
    q25, q75 = np.quantile(scores, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-6)
    normalized = (scores[top] - float(scores[top].min())) / scale
    weights = np.exp(-normalized / float(rule["temperature"]))
    weights /= max(float(weights.sum()), 1e-300)
    profile = weights @ actions[top]
    return np.clip(profile, -ACTION_CAP, ACTION_CAP), order[:16]


def validation_cache(evidence: Sequence[EvidenceWell], indices: np.ndarray, actions: np.ndarray, mode: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    blocks: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    position = 0
    for index in indices:
        item = evidence[int(index)]
        if not item.eligible:
            offsets.append((position, position))
            continue
        block = candidate_features(item, actions, mode=mode)
        blocks.append(block)
        offsets.append((position, position + len(actions)))
        position += len(actions)
    matrix = np.vstack(blocks) if blocks else np.empty((0, 39), dtype=np.float32)
    return matrix, offsets


def evaluate_scores(
    t033: Any,
    wells: Sequence[Any],
    indices: np.ndarray,
    actions: np.ndarray,
    all_scores: np.ndarray,
    offsets: list[tuple[int, int]],
    rule: dict[str, Any],
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    total_sse = 0.0
    profiles: list[np.ndarray] = []
    top_orders: list[np.ndarray] = []
    for local, well_index in enumerate(indices):
        well = wells[int(well_index)]
        if offsets[local][1] <= offsets[local][0]:
            profile = actions[-1].copy()
            top = np.asarray([len(actions) - 1], dtype=int)
        else:
            start, end = offsets[local]
            profile, top = selected_action(all_scores[start:end], actions, rule)
        total_sse += float(t033.profile_sse(well, profile))
        profiles.append(profile)
        top_orders.append(top)
    rows = sum(wells[int(index)].rows for index in indices)
    return math.sqrt(total_sse / rows), profiles, top_orders


def select_families(
    config: dict[str, Any],
    t033: Any,
    wells: Sequence[Any],
    evidence: Sequence[EvidenceWell],
    context: Context,
    actions: np.ndarray,
    mode: str = "main",
    shuffle_targets: bool = False,
) -> tuple[dict[str, Selection], dict[str, Any]]:
    dataset_key = context.key + "|inner|" + mode + ("|shuffled" if shuffle_targets else "")
    x_point, y_point = build_pointwise_dataset(
        t033, wells, evidence, context.inner_train, actions, dataset_key, mode, shuffle_targets
    )
    x_pair, y_pair = build_pairwise_dataset(
        t033, wells, evidence, context.inner_train, actions, dataset_key, mode, shuffle_targets
    )
    x_valid, offsets = validation_cache(evidence, context.inner_valid, actions, mode)
    rules = output_rules(config)
    selected: dict[str, Selection] = {}
    diagnostics: dict[str, Any] = {
        "pointwise_rows": len(y_point),
        "pairwise_rows": len(y_pair),
        "validation_candidates": len(x_valid),
        "mode": mode,
        "shuffle_targets": bool(shuffle_targets),
    }
    for branch in BRANCHES:
        branch_best: tuple[float, int, int, dict[str, Any], dict[str, Any]] | None = None
        for parameter_index, params in enumerate(model_parameters(config, branch)):
            if branch == "pairwise_logistic":
                fitted = fit_model(branch, params, x_pair, y_pair, 36000 + 100 * context.map_index + 10 * context.fold, config)
            elif branch == "robust_alignment":
                fitted = fit_model(branch, params, np.zeros((1, 39)), np.zeros(1), 0, config)
            else:
                fitted = fit_model(branch, params, x_point, y_point, 36000 + 100 * context.map_index + 10 * context.fold, config)
            scores = score_model(fitted, x_valid)
            for rule_index, rule in enumerate(rules):
                rmse, _, _ = evaluate_scores(t033, wells, context.inner_valid, actions, scores, offsets, rule)
                candidate = (rmse, parameter_index, rule_index, params, rule)
                if branch_best is None or candidate[:3] < branch_best[:3]:
                    branch_best = candidate
        assert branch_best is not None
        selected[branch] = Selection(branch, branch_best[3], branch_best[4], float(branch_best[0]))
    nested = min(selected.values(), key=lambda item: (item.inner_rmse, BRANCHES.index(item.branch)))
    diagnostics["nested_best_branch"] = nested.branch
    diagnostics["nested_best_inner_rmse"] = nested.inner_rmse
    return selected, diagnostics


def fit_family_for_outer(
    config: dict[str, Any],
    t033: Any,
    wells: Sequence[Any],
    evidence: Sequence[EvidenceWell],
    context: Context,
    actions: np.ndarray,
    selection: Selection,
    mode: str = "main",
    shuffle_targets: bool = False,
) -> FittedModel:
    if selection.branch == "robust_alignment":
        return fit_model(selection.branch, selection.params, np.zeros((1, 39)), np.zeros(1), 0, config)
    if selection.branch == "pairwise_logistic":
        x, y = build_pairwise_dataset(t033, wells, evidence, context.train, actions, context.key + "|outer", mode, shuffle_targets)
    else:
        x, y = build_pointwise_dataset(t033, wells, evidence, context.train, actions, context.key + "|outer", mode, shuffle_targets)
    return fit_model(selection.branch, selection.params, x, y, 37000 + 100 * context.map_index + 10 * context.fold, config)


def group_assignments(wells: Sequence[Any], spatial: np.ndarray, typewell: np.ndarray) -> dict[str, np.ndarray]:
    tensor = np.load(T031_TENSOR, allow_pickle=False)
    tensor_ids = tensor["well_ids"].astype(str)
    well_ids = np.asarray([well.well_id for well in wells], dtype=str)
    if not np.array_equal(tensor_ids, well_ids):
        raise ValueError("T031 group order mismatch")
    row_order = np.argsort(np.asarray([well.rows for well in wells]), kind="mergesort")
    horizon = np.empty(len(wells), dtype=int)
    for rank, index in enumerate(row_order):
        horizon[index] = min(4, int(rank * 5 / len(wells)))
    return {
        "legacy_spatial": spatial.astype(int),
        "legacy_typewell": typewell.astype(int),
        "legal_covariate_kmeans": tensor["legal_kmeans"].astype(int),
        "spatial_2d_kmeans": tensor["spatial_2d_kmeans"].astype(int),
        "horizon_quintile": horizon,
    }


def run(config: dict[str, Any], output_dir: Path, preflight_only: bool = False) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    t033 = load_t033()
    wells, spatial, typewell, fold_maps = t033.load_wells()
    if len(wells) != int(config["expected_wells"]):
        raise ValueError("unexpected well count")
    profiles = np.stack([well.profile for well in wells])
    contexts = build_contexts(wells, fold_maps)
    evidence = build_evidence(wells, int(config["path_evidence"]["maximum_hidden_gr_samples"]))
    groups = group_assignments(wells, spatial, typewell)
    baseline_metrics = [error_metric(well.e011 - well.truth) for well in wells]
    baseline = summarize(baseline_metrics)
    baseline_tail = tail_stats(baseline_metrics)
    hidden_rows = np.asarray([well.rows for well in wells], dtype=np.float64)
    hidden_missingness = np.asarray(
        [1.0 - item.hidden_gr_coverage for item in evidence], dtype=np.float64
    )
    baseline_well_rmse = np.asarray(
        [metric.rmse for metric in baseline_metrics], dtype=np.float64
    )
    special_slices = {
        "long_suffix": hidden_rows >= np.quantile(hidden_rows, 0.8),
        "high_hidden_gr_missingness": hidden_missingness
        >= np.quantile(hidden_missingness, 0.8),
        "e011_catastrophe": baseline_well_rmse
        >= np.quantile(baseline_well_rmse, 0.9),
    }

    first = contexts[0]
    first_actions = candidate_actions(profiles, first.train, first.map_index, first.fold)
    first_feature = candidate_features(evidence[int(first.test[0])], first_actions)
    preflight = {
        "status": "PASS",
        "wells": len(wells),
        "contexts": len(contexts),
        "eligible_evidence_wells": int(sum(item.eligible for item in evidence)),
        "candidate_shape": list(first_actions.shape),
        "feature_shape": list(first_feature.shape),
        "registered_feature_names": len(config["path_evidence"]["feature_names"]),
        "candidate_finite": bool(np.isfinite(first_actions).all()),
        "feature_finite": bool(np.isfinite(first_feature).all()),
        "baseline_rmse": baseline["rmse"],
        "input_hashes": {
            "t033_runner": sha256_file(T033_PATH),
            "t031_tensor": sha256_file(T031_TENSOR),
            "config": sha256_file(CONFIG_PATH),
        },
    }
    if preflight_only:
        return preflight

    action_sums = {name: np.zeros((len(wells), GRID), dtype=np.float64) for name in ALL_REPORTED}
    map_actions = {name: np.zeros((5, len(wells), GRID), dtype=np.float64) for name in ALL_REPORTED}
    context_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    all_feature_finite = True
    same_well_excluded = True

    for context in contexts:
        context_baseline = summarize(baseline_metrics[int(index)] for index in context.test)
        context_base_rmse = float(context_baseline["rmse"])
        inner_actions = candidate_actions(
            profiles, context.inner_train, context.map_index, (context.fold + 1) % 5
        )
        selections, selection_diagnostics = select_families(
            config, t033, wells, evidence, context, inner_actions, mode="main", shuffle_targets=False
        )
        outer_actions = candidate_actions(profiles, context.train, context.map_index, context.fold)
        test_features, test_offsets = validation_cache(evidence, context.test, outer_actions, "main")
        all_feature_finite = all_feature_finite and bool(np.isfinite(test_features).all())
        family_profiles: dict[str, list[np.ndarray]] = {}
        family_top: dict[str, list[np.ndarray]] = {}
        for branch in BRANCHES:
            selection = selections[branch]
            fitted = fit_family_for_outer(
                config, t033, wells, evidence, context, outer_actions, selection
            )
            scores = score_model(fitted, test_features)
            rmse, local_profiles, local_top = evaluate_scores(
                t033, wells, context.test, outer_actions, scores, test_offsets, selection.output_rule
            )
            family_profiles[branch] = local_profiles
            family_top[branch] = local_top
            context_rows.append({
                "context": context.key,
                "branch": branch,
                "wells": len(context.test),
                "baseline_rmse": context_base_rmse,
                "rmse": rmse,
                "gain_vs_e011": context_base_rmse - rmse,
            })
            hyper_rows.append({
                "context": context.key,
                "branch": branch,
                "inner_rmse": selection.inner_rmse,
                "params": json.dumps(selection.params, sort_keys=True),
                "output_rule": json.dumps(selection.output_rule, sort_keys=True),
                **selection_diagnostics,
            })
        nested_selection = min(
            selections.values(), key=lambda item: (item.inner_rmse, BRANCHES.index(item.branch))
        )
        nested_profiles = family_profiles[nested_selection.branch]
        nested_top = family_top[nested_selection.branch]
        nested_metrics = [
            error_metric(
                wells[int(index)].e011
                + t033.interpolate_profile(profile, wells[int(index)].rows)
                - wells[int(index)].truth
            )
            for index, profile in zip(context.test, nested_profiles)
        ]
        nested_summary = summarize(nested_metrics)
        context_rows.append({
            "context": context.key,
            "branch": "nested_best",
            "wells": len(context.test),
            "baseline_rmse": context_base_rmse,
            "rmse": nested_summary["rmse"],
            "gain_vs_e011": context_base_rmse - float(nested_summary["rmse"]),
            "selected_family": nested_selection.branch,
        })

        control_profiles: dict[str, list[np.ndarray]] = {}
        control_specs = {
            "reversed_typewell_gr": ("reversed_typewell_gr", False),
            "circular_shift_hidden_gr": ("circular_shift_hidden_gr", False),
            "action_prior_only": ("action_prior_only", False),
            "shuffled_within_well_training_losses": ("main", True),
        }
        for control, (mode, shuffle) in control_specs.items():
            control_selections, control_diagnostics = select_families(
                config,
                t033,
                wells,
                evidence,
                context,
                inner_actions,
                mode=mode,
                shuffle_targets=shuffle,
            )
            control_selection = min(
                control_selections.values(),
                key=lambda item: (item.inner_rmse, BRANCHES.index(item.branch)),
            )
            fitted = fit_family_for_outer(
                config,
                t033,
                wells,
                evidence,
                context,
                outer_actions,
                control_selection,
                mode=mode,
                shuffle_targets=shuffle,
            )
            control_features, control_offsets = validation_cache(
                evidence, context.test, outer_actions, mode
            )
            scores = score_model(fitted, control_features)
            rmse, local_profiles, _ = evaluate_scores(
                t033,
                wells,
                context.test,
                outer_actions,
                scores,
                control_offsets,
                control_selection.output_rule,
            )
            control_profiles[control] = local_profiles
            context_rows.append({
                "context": context.key,
                "branch": control,
                "wells": len(context.test),
                "baseline_rmse": context_base_rmse,
                "rmse": rmse,
                "gain_vs_e011": context_base_rmse - rmse,
                "selected_family": control_selection.branch,
            })
            hyper_rows.append({
                "context": context.key,
                "branch": control,
                "inner_rmse": control_selection.inner_rmse,
                "selected_family": control_selection.branch,
                "params": json.dumps(control_selection.params, sort_keys=True),
                "output_rule": json.dumps(control_selection.output_rule, sort_keys=True),
                **control_diagnostics,
            })
        control_profiles["sign_flipped_selected_action"] = [
            -profile for profile in nested_profiles
        ]
        sign_metrics = [
            error_metric(
                wells[int(index)].e011
                + t033.interpolate_profile(profile, wells[int(index)].rows)
                - wells[int(index)].truth
            )
            for index, profile in zip(
                context.test, control_profiles["sign_flipped_selected_action"]
            )
        ]
        sign_summary = summarize(sign_metrics)
        context_rows.append({
            "context": context.key,
            "branch": "sign_flipped_selected_action",
            "wells": len(context.test),
            "baseline_rmse": context_base_rmse,
            "rmse": sign_summary["rmse"],
            "gain_vs_e011": context_base_rmse - float(sign_summary["rmse"]),
        })

        oracle_profiles: list[np.ndarray] = []
        oracle_metrics: list[Metric] = []
        for local, index in enumerate(context.test):
            well = wells[int(index)]
            sses = candidate_sses(t033, well, outer_actions)
            oracle_index = int(np.argmin(sses[:-1]))
            oracle_profile = outer_actions[oracle_index].copy()
            oracle_profiles.append(oracle_profile)
            oracle_metrics.append(
                error_metric(
                    well.e011
                    + t033.interpolate_profile(oracle_profile, well.rows)
                    - well.truth
                )
            )
            top = nested_top[local]
            selection_rows.append({
                "context": context.key,
                "well_id": well.well_id,
                "selected_family": nested_selection.branch,
                "oracle_candidate": oracle_index,
                "top1_hit": int(len(top) > 0 and int(top[0]) == oracle_index),
                "top16_recall": int(oracle_index in set(map(int, top[:16]))),
                "selected_sse": float(t033.profile_sse(well, nested_profiles[local])),
                "oracle_sse": float(sses[oracle_index]),
                "regret_sse": float(
                    t033.profile_sse(well, nested_profiles[local]) - sses[oracle_index]
                ),
            })
        oracle_summary = summarize(oracle_metrics)
        context_rows.append({
            "context": context.key,
            "branch": "candidate_oracle",
            "wells": len(context.test),
            "baseline_rmse": context_base_rmse,
            "rmse": oracle_summary["rmse"],
            "gain_vs_e011": context_base_rmse - float(oracle_summary["rmse"]),
        })

        for local, index in enumerate(context.test):
            q = int(index)
            for branch in BRANCHES:
                profile = family_profiles[branch][local]
                action_sums[branch][q] += profile
                map_actions[branch][context.map_index, q] = profile
            action_sums["nested_best"][q] += nested_profiles[local]
            map_actions["nested_best"][context.map_index, q] = nested_profiles[local]
            for control in CONTROL_BRANCHES:
                profile = control_profiles[control][local]
                action_sums[control][q] += profile
                map_actions[control][context.map_index, q] = profile
            action_sums["candidate_oracle"][q] += oracle_profiles[local]
            map_actions["candidate_oracle"][context.map_index, q] = oracle_profiles[local]
        membership_rows.append({
            "context": context.key,
            "map": context.map_index + 1,
            "fold": context.fold,
            "train_wells": len(context.train),
            "test_wells": len(context.test),
            "inner_train_wells": len(context.inner_train),
            "inner_valid_wells": len(context.inner_valid),
            "same_well_excluded": not bool(set(context.train) & set(context.test)),
            "selected_family": nested_selection.branch,
        })
        same_well_excluded = same_well_excluded and not bool(
            set(context.train) & set(context.test)
        )

    branch_metrics: dict[str, list[Metric]] = {}
    averaged_actions: dict[str, np.ndarray] = {}
    branch_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    special_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    for branch in ALL_REPORTED:
        actions = np.clip(action_sums[branch] / 5.0, -ACTION_CAP, ACTION_CAP)
        averaged_actions[branch] = actions
        metrics: list[Metric] = []
        for index, well in enumerate(wells):
            prediction = well.e011 + t033.interpolate_profile(actions[index], well.rows)
            metrics.append(error_metric(prediction - well.truth))
        branch_metrics[branch] = metrics
        summary = summarize(metrics)
        tails = tail_stats(metrics)
        positive = float(np.mean([metric.sse < baseline_metrics[index].sse - 1e-9 for index, metric in enumerate(metrics)]))
        context_subset = [row for row in context_rows if row["branch"] == branch]
        branch_rows.append({
            "branch": branch,
            "rmse": summary["rmse"],
            "gain_vs_e011": baseline["rmse"] - summary["rmse"],
            "oracle_gain_retention": (baseline["rmse"] - summary["rmse"]) / max(baseline["rmse"] - float(config["candidate_bank"]["oracle_reference_rmse"]), 1e-12),
            "positive_well_fraction": positive,
            "map_wins": 0,
            "outer_cell_wins": int(sum(float(row["gain_vs_e011"]) > 0.0 for row in context_subset)),
            "p90_well_rmse": tails["p90_well_rmse"],
            "worst5_sse_share": tails["worst5_sse_share"],
            "datum_sse": summary["datum_sse"],
            "trend_sse": summary["trend_sse"],
            "shape_sse": summary["shape_sse"],
        })
        for system, assignment in groups.items():
            for group in range(5):
                ids = np.flatnonzero(assignment == group)
                candidate = summarize(metrics[int(index)] for index in ids)
                base_group = summarize(baseline_metrics[int(index)] for index in ids)
                group_rows.append({
                    "branch": branch,
                    "group_system": system,
                    "group": group,
                    "wells": len(ids),
                    "baseline_rmse": base_group["rmse"],
                    "candidate_rmse": candidate["rmse"],
                    "gain_vs_e011": base_group["rmse"] - candidate["rmse"],
                })
        for slice_name, mask in special_slices.items():
            ids = np.flatnonzero(mask)
            candidate_slice = summarize(metrics[int(index)] for index in ids)
            baseline_slice = summarize(baseline_metrics[int(index)] for index in ids)
            special_rows.append({
                "branch": branch,
                "slice": slice_name,
                "wells": len(ids),
                "baseline_rmse": baseline_slice["rmse"],
                "candidate_rmse": candidate_slice["rmse"],
                "gain_vs_e011": baseline_slice["rmse"] - candidate_slice["rmse"],
            })
        for map_index in range(5):
            map_metrics_local: list[Metric] = []
            for index, well in enumerate(wells):
                prediction = well.e011 + t033.interpolate_profile(map_actions[branch][map_index, index], well.rows)
                map_metrics_local.append(error_metric(prediction - well.truth))
            map_summary = summarize(map_metrics_local)
            map_rows.append({"branch": branch, "map": map_index + 1, "rmse": map_summary["rmse"], "gain_vs_e011": baseline["rmse"] - map_summary["rmse"]})

    map_frame = pd.DataFrame(map_rows)
    for row in branch_rows:
        row["map_wins"] = int((map_frame[map_frame["branch"] == row["branch"]]["gain_vs_e011"] > 0.0).sum())
    branch_frame = pd.DataFrame(branch_rows).set_index("branch")
    main = branch_frame.loc["nested_best"].to_dict()
    oracle_rmse = float(branch_frame.loc["candidate_oracle", "rmse"])
    oracle_identity = abs(oracle_rmse - float(config["candidate_bank"]["oracle_reference_rmse"])) <= float(config["candidate_bank"]["oracle_tolerance"])
    legacy = pd.DataFrame(group_rows)
    legacy_main = legacy[(legacy["branch"] == "nested_best") & legacy["group_system"].isin(["legacy_spatial", "legacy_typewell"])]
    prior_rmse = float(branch_frame.loc["action_prior_only", "rmse"])
    path_advantage = prior_rmse - float(main["rmse"])
    destructive = [float(branch_frame.loc[name, "gain_vs_e011"]) for name in CONTROL_BRANCHES[:-1]]
    destructive_fail = all(value <= float(config["controls"]["maximum_destructive_control_gain_vs_e011"]) for value in destructive)
    tail_ok = (
        float(main["p90_well_rmse"]) - baseline_tail["p90_well_rmse"] <= float(config["go_gate"]["maximum_p90_deterioration"]) + 1e-12
        and float(main["worst5_sse_share"]) - baseline_tail["worst5_sse_share"] <= float(config["go_gate"]["maximum_worst5_sse_share_increase"]) + 1e-12
    )
    breakthrough = {
        "maximum_legal_rmse": float(main["rmse"]) <= float(config["breakthrough_gate"]["maximum_legal_rmse"]),
        "minimum_map_wins": int(main["map_wins"]) >= int(config["breakthrough_gate"]["minimum_map_wins"]),
        "positive_every_legacy_group": bool((legacy_main["gain_vs_e011"] > 0.0).all()),
        "tail_gates": tail_ok,
        "destructive_controls_fail": destructive_fail,
    }
    go = {
        "maximum_legal_rmse": float(main["rmse"]) <= float(config["go_gate"]["maximum_legal_rmse"]),
        "minimum_oracle_gain_retention": float(main["oracle_gain_retention"]) >= float(config["go_gate"]["minimum_oracle_gain_retention"]),
        "minimum_map_wins": int(main["map_wins"]) >= int(config["go_gate"]["minimum_map_wins"]),
        "minimum_outer_cell_wins": int(main["outer_cell_wins"]) >= int(config["go_gate"]["minimum_outer_cell_wins"]),
        "positive_every_legacy_group": bool((legacy_main["gain_vs_e011"] > 0.0).all()),
        "path_conditioned_advantage": path_advantage >= float(config["go_gate"]["minimum_path_conditioned_advantage_vs_action_prior_rmse"]),
        "tail_gates": tail_ok,
        "destructive_controls_fail": destructive_fail,
    }
    stop = (
        float(main["rmse"]) > float(config["stop_gate"]["minimum_rmse_for_stop"])
        or float(main["oracle_gain_retention"]) < float(config["stop_gate"]["maximum_oracle_gain_retention_for_stop"])
        or path_advantage <= 0.0
    )
    if all(breakthrough.values()):
        decision = "BREAKTHROUGH_SUB5_PATH_CONDITIONED_RANKER"
    elif all(go.values()):
        decision = "GO_PATH_CONDITIONED_EVIDENCE"
    elif stop:
        decision = "STOP_CLOSE_PATH_CONDITIONED_RANKING"
    else:
        decision = "RESEARCH_ONLY_PATH_CONDITIONED_RANKING"

    selected_metrics = branch_metrics["nested_best"]
    for index, well in enumerate(wells):
        well_rows.append({
            "well_id": well.well_id,
            "rows": well.rows,
            "baseline_rmse": baseline_metrics[index].rmse,
            "candidate_rmse": selected_metrics[index].rmse,
            "baseline_sse": baseline_metrics[index].sse,
            "candidate_sse": selected_metrics[index].sse,
            **{name: int(groups[name][index]) for name in groups},
        })

    permutation = np.arange(len(first_actions), dtype=np.int64)[::-1]
    permuted_features = candidate_features(
        evidence[int(first.test[0])], first_actions[permutation]
    )
    robust_scores = score_model(
        FittedModel("robust_alignment", None, None), first_feature
    )
    robust_profile, _ = selected_action(
        robust_scores, first_actions, {"kind": "hard", "top_k": 1, "temperature": None}
    )
    permuted_profile, _ = selected_action(
        robust_scores[permutation],
        first_actions[permutation],
        {"kind": "hard", "top_k": 1, "temperature": None},
    )
    duplicate_actions = np.vstack([first_actions, first_actions[[0]]])
    duplicate_features = candidate_features(
        evidence[int(first.test[0])], duplicate_actions
    )
    deduplicated_count = np.unique(np.round(duplicate_actions, 12), axis=0).shape[0]
    ineligible = EvidenceWell(
        "ineligible",
        False,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    fallback_matrix, fallback_offsets = validation_cache(
        [ineligible], np.asarray([0], dtype=int), first_actions, "main"
    )
    _, fallback_profiles, _ = evaluate_scores(
        t033,
        wells,
        np.asarray([0], dtype=int),
        first_actions,
        np.empty(0, dtype=np.float64),
        fallback_offsets,
        {"kind": "hard", "top_k": 1, "temperature": None},
    )
    edge_checks = {
        "wells_773": len(wells) == 773,
        "hidden_rows_exact": sum(well.rows for well in wells) == int(config["expected_hidden_rows"]),
        "contexts_25": len(contexts) == 25,
        "candidates_1521": first_actions.shape == (1521, GRID),
        "features_39": first_feature.shape == (1521, 39),
        "all_features_finite": all_feature_finite and bool(np.isfinite(first_feature).all()),
        "same_well_exclusion": same_well_excluded and all(bool(row["same_well_excluded"]) for row in membership_rows),
        "candidate_oracle_identity": oracle_identity,
        "exact_zero_fallback": bool(np.array_equal(first_actions[-1], np.zeros(GRID))),
        "action_cap": max(float(np.max(np.abs(value))) for value in averaged_actions.values()) <= ACTION_CAP + 1e-9,
        "candidate_order_feature_identity": bool(
            np.allclose(permuted_features, first_feature[permutation], atol=1e-7, rtol=0.0)
        ),
        "candidate_order_selection_identity": bool(
            np.allclose(robust_profile, permuted_profile, atol=1e-12, rtol=0.0)
        ),
        "duplicate_action_feature_identity": bool(
            np.allclose(duplicate_features[-1], duplicate_features[0], atol=1e-7, rtol=0.0)
        ),
        "duplicate_action_deduplication": deduplicated_count == len(first_actions),
        "ineligible_feature_matrix_empty": fallback_matrix.shape == (0, 39),
        "ineligible_exact_zero_fallback": bool(
            np.array_equal(fallback_profiles[0], first_actions[-1])
        ),
        "branch_rows_complete": len(branch_rows) == len(ALL_REPORTED),
        "context_rows_complete": len(context_rows) == 25 * len(ALL_REPORTED),
        "hyperparameter_rows_complete": len(hyper_rows) == 25 * (len(BRANCHES) + 4),
        "group_rows_complete": len(group_rows) == len(ALL_REPORTED) * 25,
        "special_slice_rows_complete": len(special_rows) == len(ALL_REPORTED) * 3,
        "map_rows_complete": len(map_rows) == len(ALL_REPORTED) * 5,
        "membership_rows_25": len(membership_rows) == 25,
        "selection_rows_complete": len(selection_rows) == 5 * len(wells),
        "all_metrics_finite": bool(np.isfinite(pd.DataFrame(branch_rows).select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    edge_status = "PASS" if all(edge_checks.values()) else "FAIL"
    selection_frame = pd.DataFrame(selection_rows)
    ranking = {
        "top1_hit_fraction": float(selection_frame["top1_hit"].mean()),
        "top16_recall_fraction": float(selection_frame["top16_recall"].mean()),
        "mean_regret_sse": float(selection_frame["regret_sse"].mean()),
    }
    implementation = {
        "schema_version": 1,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "t033_script_sha256": sha256_file(T033_PATH),
        "t031_tensor_sha256": sha256_file(T031_TENSOR),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "threads": 2,
        "candidate_dictionary": "32 centers plus unique 0.25/0.50/0.75 pair mixtures and zero action",
        "feature_contract": config["path_evidence"]["feature_names"],
        "selection": "one inner fold per outer context; actual-row TVT RMSE",
    }
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "decision": decision,
        "baseline": {**baseline, **baseline_tail},
        "candidate_oracle_rmse": oracle_rmse,
        "candidate_oracle_definition": "best of 1520 nonzero T033 pair-dictionary actions; exact zero remains a legal fallback only",
        "candidate_oracle_identity": oracle_identity,
        "best_legal_branch": "nested_best",
        "best_legal_metrics": main,
        "path_conditioned_advantage_vs_action_prior_rmse": path_advantage,
        "ranking": ranking,
        "breakthrough_gates": breakthrough,
        "go_gates": go,
        "stop_gate": bool(stop),
        "destructive_control_gains": {name: float(branch_frame.loc[name, "gain_vs_e011"]) for name in CONTROL_BRANCHES[:-1]},
        "special_slices": {
            name: {
                "wells": int(mask.sum()),
                "nested_best_gain": float(
                    next(
                        row["gain_vs_e011"]
                        for row in special_rows
                        if row["branch"] == "nested_best" and row["slice"] == name
                    )
                ),
            }
            for name, mask in special_slices.items()
        },
        "edge_status": edge_status,
        "runtime_seconds": time.time() - started,
        "wells": len(wells),
        "contexts": len(contexts),
        "candidates": len(first_actions),
    }

    write_csv(output_dir / "branch_metrics.csv", branch_rows)
    write_csv(output_dir / "context_metrics.csv", context_rows)
    write_csv(output_dir / "group_metrics.csv", group_rows)
    write_csv(output_dir / "special_slice_metrics.csv", special_rows)
    write_csv(output_dir / "map_metrics.csv", map_rows)
    write_csv(output_dir / "hyperparameters.csv", hyper_rows)
    write_csv(output_dir / "selection_metrics.csv", selection_rows)
    write_csv(output_dir / "membership_audit.csv", membership_rows)
    write_csv(output_dir / "selected_well_metrics.csv", well_rows)
    dump_json(output_dir / "implementation_receipt.json", implementation)
    dump_json(output_dir / "edge_cases.json", {"schema_version": 1, "status": edge_status, "checks": edge_checks})
    dump_json(output_dir / "summary.json", summary)
    report = [
        "# T036 Result — Candidate-path-conditioned GR evidence ranking",
        "",
        f"Decision: **{decision}**",
        "",
        f"E011 baseline RMSE: {baseline['rmse']:.12f}",
        f"Candidate oracle RMSE: {oracle_rmse:.12f}",
        f"Nested legal ranker RMSE: {float(main['rmse']):.12f}",
        f"Gain versus E011: {float(main['gain_vs_e011']):.12f}",
        f"Oracle-gain retention: {float(main['oracle_gain_retention']):.6%}",
        f"Path-conditioned advantage versus action-prior control: {path_advantage:.12f} RMSE",
        f"Map/cell wins: {int(main['map_wins'])}/5 and {int(main['outer_cell_wins'])}/25",
        "",
        "## Breakthrough gates",
    ]
    report.extend(f"- {'PASS' if value else 'FAIL'} — {key}" for key, value in breakthrough.items())
    report.extend(["", "## GO gates"])
    report.extend(f"- {'PASS' if value else 'FAIL'} — {key}" for key, value in go.items())
    report.extend(["", "This is a legal diagnostic only. It does not authorize a Kaggle run or submission."])
    (output_dir / "RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    artifact_names = [
        "RESULT.md", "branch_metrics.csv", "context_metrics.csv", "group_metrics.csv", "special_slice_metrics.csv", "map_metrics.csv",
        "hyperparameters.csv", "selection_metrics.csv", "membership_audit.csv", "selected_well_metrics.csv",
        "implementation_receipt.json", "edge_cases.json", "summary.json",
    ]
    dump_json(output_dir / "artifact_manifest.json", {
        "schema_version": 1,
        "files": [{"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)} for name in artifact_names],
    })
    return summary


def synthetic_self_test() -> dict[str, Any]:
    actions = np.zeros((1521, GRID), dtype=np.float64)
    actions[:1520] = np.linspace(-2.0, 3.0, GRID)[None, :] * np.linspace(0.1, 1.0, 1520)[:, None]
    evidence = EvidenceWell(
        "synthetic", True, np.sin(np.linspace(0.0, 8.0, 64)), np.linspace(0.0, GRID - 1, 64),
        np.linspace(1000.0, 1050.0, 64), np.linspace(900.0, 1100.0, 500),
        np.sin(np.linspace(0.0, 20.0, 500)), 0.5, math.log1p(500), 0.5, 200.0, 1.0,
    )
    features = candidate_features(evidence, actions)
    prior = candidate_features(evidence, actions, mode="action_prior_only")
    robust_fitted = fit_model("robust_alignment", {}, np.zeros((1, 39)), np.zeros(1), 0, {"branches": []})
    robust_scores = score_model(robust_fitted, features)
    sses = np.linspace(1000.0, 2000.0, 1521)
    indices = pointwise_indices("synthetic", "v1_f0", sses)
    rule = {"kind": "soft", "top_k": 8, "temperature": 0.5}
    scores = features[:, 2]
    profile, order = selected_action(scores, actions, rule)
    permutation = np.arange(len(actions))[::-1]
    perm_profile, _ = selected_action(scores[permutation], actions[permutation], rule)
    ineligible = EvidenceWell(
        "ineligible", False, np.empty(0), np.empty(0), np.empty(0),
        np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]),
        0.0, 0.0, 0.0, 1.0, 0.0,
    )
    ineligible_features = candidate_features(ineligible, actions)
    fallback_matrix, fallback_offsets = validation_cache(
        [ineligible], np.asarray([0], dtype=int), actions, "main"
    )
    checks = {
        "feature_shape": features.shape == (1521, 39),
        "features_finite": bool(np.isfinite(features).all()),
        "score_model_field_contract": robust_scores.shape == (1521,) and bool(np.isfinite(robust_scores).all()),
        "prior_evidence_zero": bool(np.all(prior[:, :EVIDENCE_FEATURES] == 0.0)),
        "pointwise_sample_64": len(indices) == 64 and len(set(map(int, indices))) == 64,
        "soft_profile_finite": bool(np.isfinite(profile).all()),
        "top16": len(order) == 16,
        "candidate_permutation_invariance": bool(np.allclose(profile, perm_profile, atol=1e-12, rtol=0.0)),
        "zero_action_exact": bool(np.array_equal(actions[-1], np.zeros(GRID))),
        "ineligible_features_finite": bool(np.isfinite(ineligible_features).all()),
        "ineligible_evidence_zero": bool(
            np.all(ineligible_features[:, :EVIDENCE_FEATURES] == 0.0)
        ),
        "ineligible_cache_empty": fallback_matrix.shape == (0, 39)
        and fallback_offsets == [(0, 0)],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--synthetic-self-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.synthetic_self_test:
        print(json.dumps(synthetic_self_test(), indent=2, sort_keys=True, default=json_default))
        return
    result = run(config, args.output_dir, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
