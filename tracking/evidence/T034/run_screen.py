#!/usr/bin/env python3
"""T034 legal residual-action identifiability and privileged-geology mediation gate.

The experiment predicts the frozen seven-dimensional spline7 correction target.
Legal branches use only the frozen E011 summaries and T031 legal raw tensors.
The privileged formation-offset branch is diagnostic and never promotion-eligible.
No deployment package, Kaggle execution, or submission is emitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = ROOT / "scratch/agents/t034-identifiability-20260724/main"
COMPACT_PATH = ROOT / "artifacts/E011/compact_stats_v1.npz"
OOF_PATH = ROOT / "artifacts/E011/oof_predictions.csv.gz"
FEATURE_PATH = ROOT / "experiments/E003/results/legal_features.csv"
T031_CACHE = ROOT / "scratch/agents/t031-state-gate-20260724/main/tensors_v1.npz"
T031_BUILDER = ROOT / "tracking/evidence/T031/run_screen.py"
KNOTS = np.linspace(1.0 / 7.0, 1.0, 7)
PRIVILEGED_COLUMNS = [
    "ancc_hidden_offset_change_to_z",
    "astnl_hidden_offset_change_to_z",
    "astnu_hidden_offset_change_to_z",
    "buda_hidden_offset_change_to_z",
    "egfdl_hidden_offset_change_to_z",
    "egfdu_hidden_offset_change_to_z",
]
LEGAL_BRANCHES = [
    "mean_coefficients",
    "summary_ridge",
    "summary_pls",
    "summary_knn",
    "raw_binned_ridge",
    "summary_plus_raw_ridge",
]


@dataclass(frozen=True)
class Context:
    map_index: int
    fold: int
    key: str
    outer_train: np.ndarray
    inner_train: np.ndarray
    inner_valid: np.ndarray
    outer_test: np.ndarray


@dataclass(frozen=True)
class DataBundle:
    well_ids: np.ndarray
    target: np.ndarray
    summary: np.ndarray
    raw: np.ndarray
    privileged: np.ndarray
    folds: list[np.ndarray]
    groups: dict[str, np.ndarray]
    horizon_group: np.ndarray
    rows: np.ndarray
    e011_sse: np.ndarray
    e006_base_sse: np.ndarray
    cross: np.ndarray
    gram: np.ndarray
    total_rows: int
    baseline_rmse: float
    oracle_rmse: float
    input_hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


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


def ensure_t031_cache() -> None:
    if T031_CACHE.is_file():
        return
    subprocess.run(
        [sys.executable, str(T031_BUILDER), "--build-only"],
        cwd=ROOT,
        check=True,
    )
    if not T031_CACHE.is_file():
        raise FileNotFoundError("T031 legal tensor cache was not created")


def interpolation_basis(rows: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, rows)
    basis = np.empty((rows, 7), dtype=np.float64)
    xp = np.concatenate([[0.0], KNOTS])
    for j in range(7):
        values = np.zeros(8, dtype=np.float64)
        values[j + 1] = 1.0
        basis[:, j] = np.interp(s, xp, values)
    return basis


def load_folds(well_ids: np.ndarray) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    expected = set(map(str, well_ids.tolist()))
    for version in range(1, 6):
        payload = json.loads((ROOT / f"folds/v{version}.json").read_text(encoding="utf-8"))
        assignments = {str(k): int(v) for k, v in payload["assignments"].items()}
        if set(assignments) != expected or set(assignments.values()) != set(range(5)):
            raise ValueError(f"fold map v{version} is incomplete")
        output.append(np.asarray([assignments[str(w)] for w in well_ids], dtype=np.int8))
    return output


def build_contexts(folds: list[np.ndarray]) -> list[Context]:
    contexts: list[Context] = []
    for map_index, assignment in enumerate(folds):
        for fold in range(5):
            outer_test = np.flatnonzero(assignment == fold)
            outer_train = np.flatnonzero(assignment != fold)
            valid_fold = (fold + 1) % 5
            inner_valid = np.flatnonzero(assignment == valid_fold)
            inner_train = np.flatnonzero((assignment != fold) & (assignment != valid_fold))
            if (
                outer_test.size == 0
                or outer_train.size == 0
                or inner_train.size == 0
                or inner_valid.size == 0
                or np.intersect1d(outer_test, outer_train).size
                or np.intersect1d(inner_train, inner_valid).size
            ):
                raise ValueError("invalid nested whole-well context")
            contexts.append(Context(
                map_index=map_index,
                fold=fold,
                key=f"v{map_index + 1}_f{fold}",
                outer_train=outer_train,
                inner_train=inner_train,
                inner_valid=inner_valid,
                outer_test=outer_test,
            ))
    if len(contexts) != 25:
        raise ValueError("expected 25 contexts")
    return contexts


def load_data() -> DataBundle:
    ensure_t031_cache()
    compact = np.load(COMPACT_PATH, allow_pickle=False)
    raw_cache = np.load(T031_CACHE, allow_pickle=False)
    well_ids = compact["well_ids"].astype(str)
    if len(well_ids) != 773 or len(set(well_ids.tolist())) != 773:
        raise ValueError("expected 773 unique wells")
    if not np.array_equal(raw_cache["well_ids"].astype(str), well_ids):
        raise ValueError("T031 tensor order differs from E011 compact order")
    target = compact["spline7__target_coefficients"].astype(np.float64)
    summary = compact["features"].astype(np.float64)
    horizontal = raw_cache["horizontal"].astype(np.float64).reshape(len(well_ids), -1)
    typewell = raw_cache["typewell"].astype(np.float64).reshape(len(well_ids), -1)
    raw = np.concatenate([horizontal, typewell], axis=1)
    if target.shape != (773, 7) or summary.shape != (773, 142) or raw.shape[0] != 773:
        raise ValueError("T034 feature/target shape contract failed")
    if not np.isfinite(target).all() or not np.isfinite(raw).all():
        raise ValueError("target or raw legal tensors contain non-finite values")

    feature_frame = pd.read_csv(FEATURE_PATH, dtype={"well_id": str}).set_index("well_id").loc[well_ids]
    values = feature_frame[PRIVILEGED_COLUMNS].to_numpy(np.float64)
    masks = np.isfinite(values).astype(np.float64)
    privileged = np.concatenate([values, masks], axis=1)
    if np.sum(masks, axis=1).min() < 5:
        raise ValueError("privileged formation target has fewer than five finite components")

    folds = load_folds(well_ids)
    rows = compact["rows"].astype(np.int32)
    row_order = np.argsort(rows, kind="mergesort")
    horizon_group = np.empty(len(well_ids), dtype=np.int8)
    for rank, index in enumerate(row_order):
        horizon_group[index] = min(4, int(rank * 5 / len(well_ids)))
    groups = {
        "legacy_spatial": compact["spatial_assignment"].astype(np.int8),
        "legacy_typewell": compact["typewell_assignment"].astype(np.int8),
        "legal_covariate_kmeans": raw_cache["legal_kmeans"].astype(np.int8),
        "spatial_2d_kmeans": raw_cache["spatial_2d_kmeans"].astype(np.int8),
        "horizon_quintile": horizon_group,
    }
    if any(set(map(int, np.unique(v))) != set(range(5)) for v in groups.values()):
        raise ValueError("one or more stress systems does not contain five groups")

    frame = pd.read_csv(
        OOF_PATH,
        usecols=[
            "well_id", "hidden_index", "target", "e006_nested_fusion",
            "spline4_ridge_equal_s075",
        ],
        dtype={"well_id": str},
    ).sort_values(["well_id", "hidden_index"], kind="mergesort")
    grouped = {str(w): g for w, g in frame.groupby("well_id", sort=False)}
    if set(grouped) != set(well_ids.tolist()):
        raise ValueError("OOF well identities differ from compact artifact")
    e011_sse = np.empty(773, dtype=np.float64)
    e006_base_sse = np.empty(773, dtype=np.float64)
    cross = np.empty((773, 7), dtype=np.float64)
    gram = np.empty((773, 7, 7), dtype=np.float64)
    for i, well_id in enumerate(well_ids):
        g = grouped[str(well_id)]
        hidden_index = g["hidden_index"].to_numpy(int)
        if len(g) != int(rows[i]) or not np.array_equal(hidden_index, np.arange(len(g))):
            raise ValueError(f"{well_id}: OOF hidden row contract failed")
        truth = g["target"].to_numpy(np.float64)
        e006 = g["e006_nested_fusion"].to_numpy(np.float64)
        e011 = g["spline4_ridge_equal_s075"].to_numpy(np.float64)
        if not np.isfinite(np.column_stack([truth, e006, e011])).all():
            raise ValueError(f"{well_id}: OOF contains non-finite values")
        base_error = e006 - truth
        basis = interpolation_basis(len(g))
        e011_sse[i] = float(np.dot(e011 - truth, e011 - truth))
        e006_base_sse[i] = float(np.dot(base_error, base_error))
        cross[i] = basis.T @ base_error
        gram[i] = basis.T @ basis
    total_rows = int(rows.sum())
    baseline_rmse = math.sqrt(float(e011_sse.sum()) / total_rows)
    oracle_sse = coefficient_sse(target, e006_base_sse, cross, gram)
    oracle_rmse = math.sqrt(float(oracle_sse.sum()) / total_rows)
    if abs(baseline_rmse - 12.550756295689673) > 1e-9:
        raise ValueError("E011 baseline identity failed")
    if abs(oracle_rmse - 2.625141070954217) > 1e-9:
        raise ValueError("spline7 oracle identity failed")
    return DataBundle(
        well_ids=well_ids,
        target=target,
        summary=summary,
        raw=raw,
        privileged=privileged,
        folds=folds,
        groups=groups,
        horizon_group=horizon_group,
        rows=rows,
        e011_sse=e011_sse,
        e006_base_sse=e006_base_sse,
        cross=cross,
        gram=gram,
        total_rows=total_rows,
        baseline_rmse=baseline_rmse,
        oracle_rmse=oracle_rmse,
        input_hashes={
            str(COMPACT_PATH.relative_to(ROOT)): sha256_file(COMPACT_PATH),
            str(OOF_PATH.relative_to(ROOT)): sha256_file(OOF_PATH),
            str(FEATURE_PATH.relative_to(ROOT)): sha256_file(FEATURE_PATH),
            str(T031_CACHE.relative_to(ROOT)): sha256_file(T031_CACHE),
        },
    )


def coefficient_sse(
    coefficients: np.ndarray,
    base_sse: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    c = np.asarray(coefficients, dtype=np.float64)
    return np.maximum(
        base_sse
        + 2.0 * np.einsum("ij,ij->i", c, cross)
        + np.einsum("ij,ijk,ik->i", c, gram, c),
        0.0,
    )


def tvt_rmse(data: DataBundle, indices: np.ndarray, coefficients: np.ndarray) -> float:
    sse = coefficient_sse(
        coefficients,
        data.e006_base_sse[indices],
        data.cross[indices],
        data.gram[indices],
    )
    return math.sqrt(float(sse.sum()) / int(data.rows[indices].sum()))


def fit_imputer_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(x), x, median)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return median, mean, std


def transform(x: np.ndarray, prep: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    median, mean, std = prep
    filled = np.where(np.isfinite(x), x, median)
    return np.clip((filled - mean) / std, -12.0, 12.0)


def target_feature_order(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    prep = fit_imputer_scaler(x)
    xz = transform(x, prep)
    yc = y - y.mean(axis=0)
    ystd = yc.std(axis=0)
    ystd = np.where(ystd > 1e-10, ystd, 1.0)
    yz = yc / ystd
    score = np.sum((xz.T @ yz / max(1, len(xz) - 1)) ** 2, axis=1)
    return np.lexsort((np.arange(x.shape[1]), -score))


def fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    selected: np.ndarray | None = None,
) -> np.ndarray:
    if selected is not None:
        x_train = x_train[:, selected]
        x_test = x_test[:, selected]
    prep = fit_imputer_scaler(x_train)
    train = transform(x_train, prep)
    test = transform(x_test, prep)
    model = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
        solver="lsqr",
        tol=1e-6,
        max_iter=10000,
    )
    model.fit(train, y_train)
    prediction = np.asarray(model.predict(test), dtype=np.float64)
    if prediction.shape != (len(x_test), 7) or not np.isfinite(prediction).all():
        raise ValueError("ridge emitted invalid coefficient predictions")
    return prediction


def fit_pls_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    components: int,
) -> np.ndarray:
    prep = fit_imputer_scaler(x_train)
    train = transform(x_train, prep)
    test = transform(x_test, prep)
    model = PLSRegression(
        n_components=int(components),
        scale=False,
        max_iter=500,
        tol=1e-7,
    )
    model.fit(train, y_train)
    prediction = np.asarray(model.predict(test), dtype=np.float64)
    if prediction.shape != (len(x_test), 7) or not np.isfinite(prediction).all():
        raise ValueError("PLS emitted invalid coefficient predictions")
    return prediction


def fit_knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    neighbors: int,
    selected: np.ndarray,
) -> np.ndarray:
    train_x = x_train[:, selected]
    test_x = x_test[:, selected]
    prep = fit_imputer_scaler(train_x)
    train = transform(train_x, prep)
    test = transform(test_x, prep)
    model = KNeighborsRegressor(
        n_neighbors=int(neighbors),
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=1,
    )
    model.fit(train, y_train)
    prediction = np.asarray(model.predict(test), dtype=np.float64)
    if prediction.shape != (len(x_test), 7) or not np.isfinite(prediction).all():
        raise ValueError("kNN emitted invalid coefficient predictions")
    return prediction


def select_summary_ridge(
    data: DataBundle,
    context: Context,
    feature_counts: list[int],
    alphas: list[float],
) -> tuple[int, float, float]:
    order = target_feature_order(data.summary[context.inner_train], data.target[context.inner_train])
    candidates: list[tuple[float, int, float]] = []
    for count in feature_counts:
        selected = order[: int(count)]
        for alpha in alphas:
            pred = fit_ridge_predict(
                data.summary[context.inner_train], data.target[context.inner_train],
                data.summary[context.inner_valid], float(alpha), selected,
            )
            score = tvt_rmse(data, context.inner_valid, pred)
            candidates.append((score, int(count), float(alpha)))
    score, count, alpha = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    return count, alpha, score


def select_pls(
    data: DataBundle,
    context: Context,
    components: list[int],
) -> tuple[int, float]:
    candidates: list[tuple[float, int]] = []
    for component in components:
        pred = fit_pls_predict(
            data.summary[context.inner_train], data.target[context.inner_train],
            data.summary[context.inner_valid], int(component),
        )
        candidates.append((tvt_rmse(data, context.inner_valid, pred), int(component)))
    score, component = min(candidates, key=lambda x: (x[0], x[1]))
    return component, score


def select_knn(
    data: DataBundle,
    context: Context,
    feature_counts: list[int],
    neighbors: list[int],
) -> tuple[int, int, float]:
    order = target_feature_order(data.summary[context.inner_train], data.target[context.inner_train])
    candidates: list[tuple[float, int, int]] = []
    for count in feature_counts:
        selected = order[: int(count)]
        for k in neighbors:
            pred = fit_knn_predict(
                data.summary[context.inner_train], data.target[context.inner_train],
                data.summary[context.inner_valid], int(k), selected,
            )
            candidates.append((tvt_rmse(data, context.inner_valid, pred), int(count), int(k)))
    score, count, k = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    return count, k, score


def select_ridge_alpha(
    data: DataBundle,
    context: Context,
    x: np.ndarray,
    alphas: list[float],
) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    for alpha in alphas:
        pred = fit_ridge_predict(
            x[context.inner_train], data.target[context.inner_train],
            x[context.inner_valid], float(alpha), None,
        )
        candidates.append((tvt_rmse(data, context.inner_valid, pred), float(alpha)))
    score, alpha = min(candidates, key=lambda x: (x[0], x[1]))
    return alpha, score


def select_privileged_alpha(
    data: DataBundle,
    context: Context,
    alphas: list[float],
) -> tuple[float, float]:
    return select_ridge_alpha(data, context, data.privileged, alphas)


def deterministic_derangement(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        perm = rng.permutation(values)
        if np.all(perm != values):
            return perm
    return np.roll(values, 1)


def shuffled_control_predict(
    data: DataBundle,
    context: Context,
    feature_counts: list[int],
    alphas: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    outer = context.outer_train
    permuted_indices = deterministic_derangement(
        outer,
        34000 + context.map_index * 10 + context.fold,
    )
    mapping = {int(src): int(dst) for src, dst in zip(outer, permuted_indices)}
    pseudo_target = np.stack([data.target[mapping[int(i)]] for i in outer])
    local = {int(index): pos for pos, index in enumerate(outer)}
    inner_train_positions = np.asarray([local[int(i)] for i in context.inner_train], dtype=int)
    inner_valid_positions = np.asarray([local[int(i)] for i in context.inner_valid], dtype=int)
    x_outer = data.summary[outer]
    order = target_feature_order(
        x_outer[inner_train_positions],
        pseudo_target[inner_train_positions],
    )
    candidates: list[tuple[float, int, float]] = []
    for count in feature_counts:
        selected = order[: int(count)]
        for alpha in alphas:
            pred = fit_ridge_predict(
                x_outer[inner_train_positions], pseudo_target[inner_train_positions],
                x_outer[inner_valid_positions], float(alpha), selected,
            )
            error = pred - pseudo_target[inner_valid_positions]
            score = float(np.sqrt(np.mean(error * error)))
            candidates.append((score, int(count), float(alpha)))
    score, count, alpha = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    outer_order = target_feature_order(x_outer, pseudo_target)
    selected = outer_order[:count]
    prediction = fit_ridge_predict(
        x_outer, pseudo_target, data.summary[context.outer_test], alpha, selected,
    )
    return prediction, {
        "feature_count": count,
        "alpha": alpha,
        "inner_pseudo_coefficient_rmse": score,
    }


def coefficient_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = pred - y
    denominator = float(np.sum((y - y.mean(axis=0)) ** 2))
    multivariate_r2 = 1.0 - float(np.sum(error * error)) / denominator
    pearsons: list[float] = []
    for j in range(y.shape[1]):
        if np.std(pred[:, j]) <= 1e-12 or np.std(y[:, j]) <= 1e-12:
            pearsons.append(0.0)
        else:
            pearsons.append(float(pearsonr(y[:, j], pred[:, j]).statistic))
    return {
        "coefficient_rmse": float(np.sqrt(np.mean(error * error))),
        "multivariate_r2": multivariate_r2,
        "coefficient_pearsons": pearsons,
        "median_coefficient_pearson": float(np.median(pearsons)),
    }


def model_summary(data: DataBundle, prediction: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    per_well_sse = coefficient_sse(
        prediction, data.e006_base_sse, data.cross, data.gram,
    )
    rmse = math.sqrt(float(per_well_sse.sum()) / data.total_rows)
    gain = data.baseline_rmse - rmse
    oracle_gain = data.baseline_rmse - data.oracle_rmse
    metrics = coefficient_metrics(data.target, prediction)
    metrics.update({
        "tvt_rmse": rmse,
        "gain_vs_e011": gain,
        "oracle_gain_retention": gain / oracle_gain,
        "positive_well_fraction": float(np.mean(per_well_sse < data.e011_sse - 1e-9)),
    })
    return metrics, per_well_sse


def pairwise_squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    value = (
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * (a @ b.T)
    )
    return np.maximum(value, 0.0)


def neighbor_enrichment(
    data: DataBundle,
    contexts: list[Context],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spaces = {
        "summary_142": data.summary,
        "raw_binned": data.raw,
    }
    accum: dict[tuple[str, int], list[float]] = {
        (space, int(k)): [0.0, 0.0, 0.0]
        for space in spaces
        for k in config["neighborhood_enrichment"]["nearest_k"]
    }
    for context in contexts:
        library = context.outer_train
        query = context.outer_test
        target_prep = fit_imputer_scaler(data.target[library])
        target_library = transform(data.target[library], target_prep)
        target_query = transform(data.target[query], target_prep)
        target_distance = np.sqrt(pairwise_squared(target_query, target_library))
        for space_name, x in spaces.items():
            prep = fit_imputer_scaler(x[library])
            library_x = transform(x[library], prep)
            query_x = transform(x[query], prep)
            distance = pairwise_squared(query_x, library_x)
            order = np.argsort(distance, axis=1, kind="mergesort")
            for k_value in config["neighborhood_enrichment"]["nearest_k"]:
                k = int(k_value)
                nearest_sum = 0.0
                random_sum = 0.0
                count = 0
                for local_q, global_q in enumerate(query):
                    nearest_local = order[local_q, :k]
                    nearest_sum += float(target_distance[local_q, nearest_local].sum())
                    candidates_local = np.flatnonzero(
                        data.horizon_group[library] == data.horizon_group[int(global_q)]
                    )
                    if candidates_local.size < k:
                        candidates_local = np.arange(len(library), dtype=int)
                    rng = np.random.default_rng(
                        34100
                        + context.map_index * 100000
                        + context.fold * 10000
                        + int(global_q) * 100
                        + k
                    )
                    random_local = rng.choice(candidates_local, size=k, replace=False)
                    random_sum += float(target_distance[local_q, random_local].sum())
                    count += k
                key = (space_name, k)
                accum[key][0] += nearest_sum
                accum[key][1] += random_sum
                accum[key][2] += count
    for (space_name, k), (nearest_sum, random_sum, count) in sorted(accum.items()):
        nearest_mean = nearest_sum / count
        random_mean = random_sum / count
        rows.append({
            "space": space_name,
            "neighbors": k,
            "nearest_action_distance": nearest_mean,
            "matched_random_action_distance": random_mean,
            "distance_reduction_fraction": 1.0 - nearest_mean / random_mean,
            "pair_count": int(count),
        })
    return rows


def run(config: dict[str, Any], output_dir: Path, preflight_only: bool = False) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    contexts = build_contexts(data.folds)
    if preflight_only:
        return {
            "status": "PREFLIGHT_PASS",
            "wells": len(data.well_ids),
            "target_shape": list(data.target.shape),
            "summary_shape": list(data.summary.shape),
            "raw_shape": list(data.raw.shape),
            "privileged_shape": list(data.privileged.shape),
            "contexts": len(contexts),
            "baseline_rmse": data.baseline_rmse,
            "oracle_rmse": data.oracle_rmse,
            "input_hashes": data.input_hashes,
        }

    predictions = {
        name: np.zeros((5, len(data.well_ids), 7), dtype=np.float64)
        for name in LEGAL_BRANCHES + ["privileged_geology_ridge", "shuffled_target_control"]
    }
    filled = {
        name: np.zeros((5, len(data.well_ids)), dtype=bool)
        for name in predictions
    }
    hyper_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    branch_config = {item["name"]: item for item in config["branches"]}
    combined = np.concatenate([data.summary, data.raw], axis=1)

    for context in contexts:
        test = context.outer_test
        map_index = context.map_index

        mean_pred = np.repeat(
            data.target[context.outer_train].mean(axis=0, keepdims=True),
            len(test), axis=0,
        )
        predictions["mean_coefficients"][map_index, test] = mean_pred
        filled["mean_coefficients"][map_index, test] = True
        hyper_rows.append({
            "context": context.key,
            "branch": "mean_coefficients",
            "parameter": "outer_training_mean",
        })

        item = branch_config["summary_ridge"]
        count, alpha, inner_score = select_summary_ridge(
            data, context, item["feature_counts"], item["alphas"],
        )
        outer_order = target_feature_order(
            data.summary[context.outer_train], data.target[context.outer_train],
        )
        pred = fit_ridge_predict(
            data.summary[context.outer_train], data.target[context.outer_train],
            data.summary[test], alpha, outer_order[:count],
        )
        predictions["summary_ridge"][map_index, test] = pred
        filled["summary_ridge"][map_index, test] = True
        hyper_rows.append({
            "context": context.key, "branch": "summary_ridge",
            "feature_count": count, "alpha": alpha, "inner_tvt_rmse": inner_score,
        })

        item = branch_config["summary_pls"]
        components, inner_score = select_pls(data, context, item["components"])
        pred = fit_pls_predict(
            data.summary[context.outer_train], data.target[context.outer_train],
            data.summary[test], components,
        )
        predictions["summary_pls"][map_index, test] = pred
        filled["summary_pls"][map_index, test] = True
        hyper_rows.append({
            "context": context.key, "branch": "summary_pls",
            "components": components, "inner_tvt_rmse": inner_score,
        })

        item = branch_config["summary_knn"]
        count, neighbors, inner_score = select_knn(
            data, context, item["feature_counts"], item["neighbors"],
        )
        outer_order = target_feature_order(
            data.summary[context.outer_train], data.target[context.outer_train],
        )
        pred = fit_knn_predict(
            data.summary[context.outer_train], data.target[context.outer_train],
            data.summary[test], neighbors, outer_order[:count],
        )
        predictions["summary_knn"][map_index, test] = pred
        filled["summary_knn"][map_index, test] = True
        hyper_rows.append({
            "context": context.key, "branch": "summary_knn",
            "feature_count": count, "neighbors": neighbors,
            "inner_tvt_rmse": inner_score,
        })

        item = branch_config["raw_binned_ridge"]
        alpha, inner_score = select_ridge_alpha(
            data, context, data.raw, item["alphas"],
        )
        pred = fit_ridge_predict(
            data.raw[context.outer_train], data.target[context.outer_train],
            data.raw[test], alpha,
        )
        predictions["raw_binned_ridge"][map_index, test] = pred
        filled["raw_binned_ridge"][map_index, test] = True
        hyper_rows.append({
            "context": context.key, "branch": "raw_binned_ridge",
            "alpha": alpha, "inner_tvt_rmse": inner_score,
        })

        item = branch_config["summary_plus_raw_ridge"]
        alpha, inner_score = select_ridge_alpha(
            data, context, combined, item["alphas"],
        )
        pred = fit_ridge_predict(
            combined[context.outer_train], data.target[context.outer_train],
            combined[test], alpha,
        )
        predictions["summary_plus_raw_ridge"][map_index, test] = pred
        filled["summary_plus_raw_ridge"][map_index, test] = True
        hyper_rows.append({
            "context": context.key, "branch": "summary_plus_raw_ridge",
            "alpha": alpha, "inner_tvt_rmse": inner_score,
        })

        item = branch_config["privileged_geology_ridge"]
        alpha, inner_score = select_privileged_alpha(
            data, context, item["alphas"],
        )
        pred = fit_ridge_predict(
            data.privileged[context.outer_train], data.target[context.outer_train],
            data.privileged[test], alpha,
        )
        predictions["privileged_geology_ridge"][map_index, test] = pred
        filled["privileged_geology_ridge"][map_index, test] = True
        hyper_rows.append({
            "context": context.key, "branch": "privileged_geology_ridge",
            "alpha": alpha, "inner_tvt_rmse": inner_score,
        })

        item = branch_config["summary_ridge"]
        pred, params = shuffled_control_predict(
            data, context, item["feature_counts"], item["alphas"],
        )
        predictions["shuffled_target_control"][map_index, test] = pred
        filled["shuffled_target_control"][map_index, test] = True
        hyper_rows.append({
            "context": context.key, "branch": "shuffled_target_control", **params,
        })

        for branch in predictions:
            context_prediction = predictions[branch][map_index, test]
            score = tvt_rmse(data, test, context_prediction)
            context_rows.append({
                "context": context.key,
                "branch": branch,
                "wells": len(test),
                "tvt_rmse": score,
            })

    if not all(mask.all() for mask in filled.values()):
        raise AssertionError("one or more OOF coefficient predictions are missing")

    averaged = {name: value.mean(axis=0) for name, value in predictions.items()}
    summaries: dict[str, dict[str, Any]] = {}
    well_sse: dict[str, np.ndarray] = {}
    for name, prediction in averaged.items():
        summaries[name], well_sse[name] = model_summary(data, prediction)

    best_legal = min(
        LEGAL_BRANCHES,
        key=lambda name: (float(summaries[name]["tvt_rmse"]), name),
    )
    sign_flipped = -averaged[best_legal]
    averaged["sign_flipped_control"] = sign_flipped
    summaries["sign_flipped_control"], well_sse["sign_flipped_control"] = model_summary(
        data, sign_flipped,
    )

    group_rows: list[dict[str, Any]] = []
    for branch in list(averaged):
        for system, assignment in data.groups.items():
            for group in range(5):
                indices = np.flatnonzero(assignment == group)
                candidate_rmse = math.sqrt(
                    float(well_sse[branch][indices].sum()) / int(data.rows[indices].sum())
                )
                baseline_rmse = math.sqrt(
                    float(data.e011_sse[indices].sum()) / int(data.rows[indices].sum())
                )
                group_rows.append({
                    "branch": branch,
                    "group_system": system,
                    "group": group,
                    "wells": len(indices),
                    "baseline_rmse": baseline_rmse,
                    "candidate_rmse": candidate_rmse,
                    "gain_vs_e011": baseline_rmse - candidate_rmse,
                })

    branch_rows: list[dict[str, Any]] = []
    for name in list(averaged):
        row = {
            "branch": name,
            "eligible": name in LEGAL_BRANCHES,
            **{k: v for k, v in summaries[name].items() if k != "coefficient_pearsons"},
        }
        for j, value in enumerate(summaries[name]["coefficient_pearsons"]):
            row[f"coefficient_{j + 1}_pearson"] = value
        branch_rows.append(row)

    enrichment_rows = neighbor_enrichment(data, contexts, config)
    best_enrichment = max(
        enrichment_rows,
        key=lambda row: (float(row["distance_reduction_fraction"]), row["space"], row["neighbors"]),
    )
    best_metrics = summaries[best_legal]
    best_legacy_groups = [
        row for row in group_rows
        if row["branch"] == best_legal
        and row["group_system"] in {"legacy_spatial", "legacy_typewell"}
    ]
    negative_control_gain = max(
        float(summaries["shuffled_target_control"]["gain_vs_e011"]),
        float(summaries["sign_flipped_control"]["gain_vs_e011"]),
    )
    gates = {
        "maximum_best_legal_tvt_rmse": (
            float(best_metrics["tvt_rmse"])
            <= float(config["go_gate"]["maximum_best_legal_tvt_rmse"])
        ),
        "minimum_oracle_gain_retention": (
            float(best_metrics["oracle_gain_retention"])
            >= float(config["go_gate"]["minimum_oracle_gain_retention"])
        ),
        "minimum_multivariate_coefficient_r2": (
            float(best_metrics["multivariate_r2"])
            >= float(config["go_gate"]["minimum_multivariate_coefficient_r2"])
        ),
        "minimum_median_coefficient_pearson": (
            float(best_metrics["median_coefficient_pearson"])
            >= float(config["go_gate"]["minimum_median_coefficient_pearson"])
        ),
        "minimum_nearest_neighbor_distance_reduction_fraction": (
            float(best_enrichment["distance_reduction_fraction"])
            >= float(config["go_gate"]["minimum_nearest_neighbor_distance_reduction_fraction"])
        ),
        "positive_every_legacy_group": all(
            float(row["gain_vs_e011"]) > 0.0 for row in best_legacy_groups
        ),
        "negative_controls_below_gain_cap": (
            negative_control_gain
            <= float(config["go_gate"]["maximum_shuffled_or_signflipped_gain_vs_e011"])
        ),
    }
    privileged_rmse = float(summaries["privileged_geology_ridge"]["tvt_rmse"])
    strong_mediation = privileged_rmse <= 5.5 and not all(gates.values())
    weak_mediation = privileged_rmse > 8.0
    legal_min_rmse = min(float(summaries[name]["tvt_rmse"]) for name in LEGAL_BRANCHES)
    legal_max_r2 = max(float(summaries[name]["multivariate_r2"]) for name in LEGAL_BRANCHES)
    stop = legal_min_rmse > 10.0 or legal_max_r2 < 0.25
    if all(gates.values()):
        decision = "GO_LEGAL_ACTION_IDENTIFIABILITY"
    elif strong_mediation:
        decision = "STOP_LEGAL_FAIL_STRONG_PRIVILEGED_GEOLOGY_MEDIATION"
    elif stop:
        decision = "STOP_CLOSE_CURRENT_LEGAL_ACTION_IDENTIFICATION"
    else:
        decision = "RESEARCH_ONLY_PARTIAL_ACTION_IDENTIFIABILITY"

    selected_well_rows: list[dict[str, Any]] = []
    for i, well_id in enumerate(data.well_ids):
        row: dict[str, Any] = {
            "well_id": str(well_id),
            "rows": int(data.rows[i]),
            "baseline_sse": float(data.e011_sse[i]),
            "candidate_sse": float(well_sse[best_legal][i]),
            "baseline_rmse": math.sqrt(float(data.e011_sse[i]) / int(data.rows[i])),
            "candidate_rmse": math.sqrt(float(well_sse[best_legal][i]) / int(data.rows[i])),
        }
        for j in range(7):
            row[f"target_coefficient_{j + 1}"] = float(data.target[i, j])
            row[f"predicted_coefficient_{j + 1}"] = float(averaged[best_legal][i, j])
        selected_well_rows.append(row)

    edge_checks = {
        "wells_773": len(data.well_ids) == 773,
        "contexts_25": len(contexts) == 25,
        "all_oof_predictions_filled": all(mask.all() for mask in filled.values()),
        "all_predictions_finite": all(np.isfinite(value).all() for value in averaged.values()),
        "target_shape_773x7": data.target.shape == (773, 7),
        "summary_shape_773x142": data.summary.shape == (773, 142),
        "stress_systems_complete": len(data.groups) == 5 and all(
            set(map(int, np.unique(value))) == set(range(5))
            for value in data.groups.values()
        ),
        "baseline_identity": abs(data.baseline_rmse - 12.550756295689673) <= 1e-9,
        "oracle_identity": abs(data.oracle_rmse - 2.625141070954217) <= 1e-9,
        "privileged_ineligible": "privileged_geology_ridge" not in LEGAL_BRANCHES,
        "controls_ineligible": all(
            name not in LEGAL_BRANCHES
            for name in ["shuffled_target_control", "sign_flipped_control"]
        ),
        "best_legal_is_eligible": best_legal in LEGAL_BRANCHES,
        "coefficient_metric_finite": all(
            np.isfinite([
                summaries[name]["coefficient_rmse"],
                summaries[name]["multivariate_r2"],
                summaries[name]["median_coefficient_pearson"],
            ]).all()
            for name in summaries
        ),
        "neighbor_enrichment_complete": len(enrichment_rows) == 8,
        "same_well_isolation": all(
            not np.intersect1d(context.outer_train, context.outer_test).size
            for context in contexts
        ),
    }
    edge_status = "PASS" if all(edge_checks.values()) else "FAIL"

    implementation = {
        "schema_version": 1,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "threads": 2,
        "target": "frozen spline7 coefficients relative to E006",
        "scoring": "exact actual-row TVT SSE from per-well quadratic sufficient statistics",
        "nested_selection": "inner validation fold=(outer fold+1)%5; select by inner actual-row TVT RMSE",
        "feature_ranking": "split-local sum of squared standardized feature/coefficient correlations",
        "aggregation": "average five map-specific OOF coefficient predictions per well",
        "ridge_solver": "LSQR",
        "knn": "Euclidean distance with inverse-distance weighting",
        "shuffled_control": "deterministic outer-training target derangement selected only against pseudo targets",
        "sign_flipped_control": "negative of the selected best legal coefficient prediction",
        "neighbor_enrichment": "outer-fold standardized legal nearest neighbors versus deterministic suffix-length-quintile matched random pairs",
        "privileged_rule": "six train-only hidden formation offsets plus masks; diagnostic only and never eligible",
        "input_hashes": data.input_hashes,
    }
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "decision": decision,
        "baseline_rmse": data.baseline_rmse,
        "oracle_rmse": data.oracle_rmse,
        "best_legal_branch": best_legal,
        "best_legal_metrics": best_metrics,
        "best_neighbor_enrichment": best_enrichment,
        "privileged_geology_metrics": summaries["privileged_geology_ridge"],
        "privileged_interpretation": {
            "strong_mediation": strong_mediation,
            "weak_mediation": weak_mediation,
        },
        "negative_control_gain_max": negative_control_gain,
        "gates": gates,
        "stop_gate": stop,
        "edge_status": edge_status,
        "contexts": len(contexts),
        "wells": len(data.well_ids),
        "runtime_seconds": time.time() - started,
    }

    write_csv(output_dir / "branch_metrics.csv", branch_rows)
    write_csv(output_dir / "context_metrics.csv", context_rows)
    write_csv(output_dir / "group_metrics.csv", group_rows)
    write_csv(output_dir / "hyperparameters.csv", hyper_rows)
    write_csv(output_dir / "neighbor_enrichment.csv", enrichment_rows)
    write_csv(output_dir / "selected_well_metrics.csv", selected_well_rows)
    dump_json(output_dir / "implementation_receipt.json", implementation)
    dump_json(output_dir / "edge_cases.json", {
        "schema_version": 1,
        "status": edge_status,
        "checks": edge_checks,
    })
    dump_json(output_dir / "summary.json", summary)

    report = [
        "# T034 Result — Legal residual-action identifiability and privileged-geology mediation gate",
        "",
        f"Decision: **{decision}**",
        "",
        f"Best legal branch: `{best_legal}`",
        f"- TVT RMSE: {best_metrics['tvt_rmse']:.12f}",
        f"- Gain versus E011: {best_metrics['gain_vs_e011']:.12f}",
        f"- Oracle-gain retention: {best_metrics['oracle_gain_retention']:.6%}",
        f"- Multivariate coefficient R2: {best_metrics['multivariate_r2']:.6f}",
        f"- Median coefficient Pearson: {best_metrics['median_coefficient_pearson']:.6f}",
        f"- Positive wells: {best_metrics['positive_well_fraction']:.6%}",
        "",
        f"Best legal-neighbor enrichment: `{best_enrichment['space']}`, k={best_enrichment['neighbors']}",
        f"- Action-distance reduction: {best_enrichment['distance_reduction_fraction']:.6%}",
        "",
        "Privileged geology diagnostic:",
        f"- TVT RMSE: {summaries['privileged_geology_ridge']['tvt_rmse']:.12f}",
        f"- Multivariate coefficient R2: {summaries['privileged_geology_ridge']['multivariate_r2']:.6f}",
        "",
        "## Gates",
    ]
    report.extend(
        f"- {'PASS' if value else 'FAIL'} — {name}"
        for name, value in gates.items()
    )
    report.extend([
        "",
        "T034 is an identifiability diagnostic. The privileged branch is nondeployable, and no package, Kaggle run, or submission is authorized automatically.",
    ])
    (output_dir / "RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    artifact_names = [
        "implementation_receipt.json", "summary.json", "edge_cases.json",
        "branch_metrics.csv", "context_metrics.csv", "group_metrics.csv",
        "hyperparameters.csv", "neighbor_enrichment.csv",
        "selected_well_metrics.csv", "RESULT.md",
    ]
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "name": name,
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in artifact_names
        ],
    }
    dump_json(output_dir / "artifact_manifest.json", manifest)
    return summary


def synthetic_self_test() -> dict[str, Any]:
    rng = np.random.default_rng(34)
    n = 80
    p = 24
    x = rng.normal(size=(n, p))
    coefficient = rng.normal(size=(p, 7))
    y = x @ coefficient + rng.normal(scale=0.05, size=(n, 7))
    selected = target_feature_order(x[:60], y[:60])[:16]
    pred = fit_ridge_predict(x[:60], y[:60], x[60:], 10.0, selected)
    metrics = coefficient_metrics(y[60:], pred)
    perm = deterministic_derangement(np.arange(60), 34)
    a = rng.normal(size=(12, 5))
    distance = pairwise_squared(a, a)
    checks = {
        "ridge_shape": pred.shape == (20, 7),
        "ridge_finite": bool(np.isfinite(pred).all()),
        "feature_order_unique": len(set(selected.tolist())) == len(selected),
        "derangement": bool(np.all(perm != np.arange(60))),
        "distance_symmetric": bool(np.allclose(distance, distance.T, atol=1e-12)),
        "distance_diagonal_zero": bool(np.allclose(np.diag(distance), 0.0, atol=1e-12)),
        "coefficient_r2_positive": float(metrics["multivariate_r2"]) > 0.5,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--synthetic-self-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.synthetic_self_test:
        print(json.dumps(synthetic_self_test(), indent=2, sort_keys=True, default=json_default))
        return
    result = run(config, args.output_dir, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
