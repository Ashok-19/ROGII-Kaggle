#!/usr/bin/env python3
"""T035: legal identification of the actual T033 rank-8 PCA action coordinates."""
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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = ROOT / "scratch/agents/t035-pca-identifiability-20260724/main"
T033_SCRIPT = ROOT / "tracking/evidence/T033/run_screen.py"
COMPACT_PATH = ROOT / "artifacts/E011/compact_stats_v1.npz"
FEATURE_PATH = ROOT / "experiments/E003/results/legal_features.csv"
T031_CACHE = ROOT / "scratch/agents/t031-state-gate-20260724/main/tensors_v1.npz"
GRID = 128
RANK = 8
CAP = 160.0
PRIVILEGED_COLUMNS = [
    "ancc_hidden_offset_change_to_z",
    "astnl_hidden_offset_change_to_z",
    "astnu_hidden_offset_change_to_z",
    "buda_hidden_offset_change_to_z",
    "egfdl_hidden_offset_change_to_z",
    "egfdu_hidden_offset_change_to_z",
]
LEGAL_BRANCHES = [
    "mean_coordinates",
    "summary_ridge",
    "summary_extra_trees",
    "summary_hist_gradient_boosting",
    "summary_mlp",
    "raw_pca_ridge",
    "raw_pca_extra_trees",
    "summary_plus_raw_pca_ridge",
]
RUN_BRANCHES = LEGAL_BRANCHES + [
    "privileged_geology_extra_trees",
    "shuffled_coordinate_control",
]
ALL_BRANCHES = RUN_BRANCHES + ["sign_flipped_control"]


def load_t033():
    spec = importlib.util.spec_from_file_location("t035_t033_reference", T033_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load T033 reference implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T033 = load_t033()


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
class FeatureBundle:
    well_ids: np.ndarray
    summary: np.ndarray
    raw: np.ndarray
    privileged: np.ndarray
    groups: dict[str, np.ndarray]
    folds: list[np.ndarray]


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


def load_features(wells: list[Any], assignments: list[dict[str, int]]) -> FeatureBundle:
    if not T031_CACHE.is_file():
        raise FileNotFoundError("missing frozen T031 legal tensor cache")
    compact = np.load(COMPACT_PATH, allow_pickle=False)
    cache = np.load(T031_CACHE, allow_pickle=False)
    well_ids = compact["well_ids"].astype(str)
    expected = np.asarray([well.well_id for well in wells], dtype=str)
    if not np.array_equal(well_ids, expected):
        raise ValueError("E011 and T033 well orders differ")
    if not np.array_equal(cache["well_ids"].astype(str), expected):
        raise ValueError("T031 and T033 well orders differ")
    summary = compact["features"].astype(np.float64)
    raw = np.concatenate(
        [
            cache["horizontal"].astype(np.float64).reshape(len(wells), -1),
            cache["typewell"].astype(np.float64).reshape(len(wells), -1),
        ],
        axis=1,
    )
    frame = pd.read_csv(FEATURE_PATH, dtype={"well_id": str}).set_index("well_id").loc[well_ids]
    privileged_values = frame[PRIVILEGED_COLUMNS].to_numpy(np.float64)
    privileged_masks = np.isfinite(privileged_values).astype(np.float64)
    privileged = np.concatenate([privileged_values, privileged_masks], axis=1)
    rows = np.asarray([well.rows for well in wells], dtype=int)
    order = np.argsort(rows, kind="mergesort")
    horizon = np.empty(len(wells), dtype=np.int8)
    for rank, index in enumerate(order):
        horizon[index] = min(4, int(rank * 5 / len(wells)))
    groups = {
        "legacy_spatial": compact["spatial_assignment"].astype(np.int8),
        "legacy_typewell": compact["typewell_assignment"].astype(np.int8),
        "legal_covariate_kmeans": cache["legal_kmeans"].astype(np.int8),
        "spatial_2d_kmeans": cache["spatial_2d_kmeans"].astype(np.int8),
        "horizon_quintile": horizon,
    }
    folds = [
        np.asarray([assignment[str(well_id)] for well_id in well_ids], dtype=np.int8)
        for assignment in assignments
    ]
    if summary.shape != (773, 142) or raw.shape != (773, 4544) or privileged.shape != (773, 12):
        raise ValueError("T035 feature shape contract failed")
    if not np.isfinite(raw).all() or np.sum(privileged_masks, axis=1).min() < 5:
        raise ValueError("T035 legal/privileged feature contract failed")
    if any(set(map(int, np.unique(value))) != set(range(5)) for value in groups.values()):
        raise ValueError("T035 stress groups are incomplete")
    return FeatureBundle(well_ids, summary, raw, privileged, groups, folds)


def build_contexts(folds: list[np.ndarray]) -> list[Context]:
    contexts: list[Context] = []
    for map_index, assignment in enumerate(folds):
        for fold in range(5):
            outer_test = np.flatnonzero(assignment == fold)
            outer_train = np.flatnonzero(assignment != fold)
            inner_valid = np.flatnonzero(assignment == ((fold + 1) % 5))
            inner_train = np.flatnonzero(
                (assignment != fold) & (assignment != ((fold + 1) % 5))
            )
            if (
                outer_test.size == 0
                or outer_train.size == 0
                or inner_train.size == 0
                or inner_valid.size == 0
                or np.intersect1d(outer_train, outer_test).size
                or np.intersect1d(inner_train, inner_valid).size
            ):
                raise ValueError("invalid nested whole-well context")
            contexts.append(
                Context(
                    map_index,
                    fold,
                    f"v{map_index + 1}_f{fold}",
                    outer_train,
                    inner_train,
                    inner_valid,
                    outer_test,
                )
            )
    if len(contexts) != 25:
        raise ValueError("T035 requires 25 contexts")
    return contexts


def fit_action_basis(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = PCA(n_components=RANK, svd_solver="full")
    model.fit(profiles)
    mean = np.asarray(model.mean_, dtype=np.float64)
    components = np.asarray(model.components_, dtype=np.float64)
    if mean.shape != (GRID,) or components.shape != (RANK, GRID):
        raise ValueError("rank-8 action basis shape failure")
    return mean, components


def exact_coordinates(well: Any, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    v = components[:RANK]
    vg = v * well.diag[None, :]
    vg[:, :-1] += v[:, 1:] * well.off[None, :]
    vg[:, 1:] += v[:, :-1] * well.off[None, :]
    gram = vg @ v.T
    gmean = well.diag * mean
    gmean[:-1] += well.off * mean[1:]
    gmean[1:] += well.off * mean[:-1]
    rhs = v @ (well.q - gmean)
    coordinate = np.linalg.lstsq(gram + np.eye(RANK) * 1e-10, rhs, rcond=None)[0]
    if coordinate.shape != (RANK,) or not np.isfinite(coordinate).all():
        raise ValueError("invalid exact PCA coordinate")
    return coordinate


def coordinates_for(
    indices: np.ndarray,
    wells: list[Any],
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return np.stack([exact_coordinates(wells[int(i)], mean, components) for i in indices])


def profiles_from_coordinates(
    mean: np.ndarray,
    components: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    return np.clip(mean[None, :] + np.asarray(coordinates) @ components, -CAP, CAP)


def profile_rmse(wells: list[Any], indices: np.ndarray, profiles: np.ndarray) -> float:
    sse = 0.0
    rows = 0
    for local, index in enumerate(indices):
        well = wells[int(index)]
        sse += float(T033.profile_sse(well, profiles[local]))
        rows += int(well.rows)
    return math.sqrt(sse / rows)


def fit_preprocessor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    xz = transform(x, fit_preprocessor(x))
    yc = y - y.mean(axis=0)
    ystd = yc.std(axis=0)
    ystd = np.where(ystd > 1e-10, ystd, 1.0)
    score = np.sum((xz.T @ (yc / ystd) / max(1, len(xz) - 1)) ** 2, axis=1)
    return np.lexsort((np.arange(x.shape[1]), -score))


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    selected: np.ndarray | None = None,
) -> np.ndarray:
    if selected is not None:
        x_train = x_train[:, selected]
        x_test = x_test[:, selected]
    prep = fit_preprocessor(x_train)
    model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-6, max_iter=10000)
    model.fit(transform(x_train, prep), y_train)
    prediction = np.asarray(model.predict(transform(x_test, prep)), dtype=np.float64)
    return validate_prediction(prediction, len(x_test), "ridge")


def extra_trees_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    n_estimators: int,
    max_features: float,
    min_samples_leaf: int,
    seed: int,
) -> np.ndarray:
    prep = fit_preprocessor(x_train)
    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        max_features=float(max_features),
        min_samples_leaf=int(min_samples_leaf),
        bootstrap=False,
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(transform(x_train, prep), y_train)
    prediction = np.asarray(model.predict(transform(x_test, prep)), dtype=np.float64)
    return validate_prediction(prediction, len(x_test), "extra_trees")


def hist_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    max_leaf_nodes: int,
    l2: float,
    learning_rate: float,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    prep = fit_preprocessor(x_train)
    train = transform(x_train, prep)
    test = transform(x_test, prep)
    output = []
    for dimension in range(RANK):
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=int(max_leaf_nodes),
            l2_regularization=float(l2),
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            early_stopping=False,
            random_state=int(seed + dimension),
        )
        model.fit(train, y_train[:, dimension])
        output.append(model.predict(test))
    return validate_prediction(np.column_stack(output), len(x_test), "hist_gradient_boosting")


def mlp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    hidden: tuple[int, ...],
    alpha: float,
    max_iter: int,
    early_stopping: bool,
    seed: int,
) -> np.ndarray:
    prep = fit_preprocessor(x_train)
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        batch_size=32,
        learning_rate_init=5e-4,
        max_iter=int(max_iter),
        early_stopping=bool(early_stopping),
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=int(seed),
    )
    model.fit(transform(x_train, prep), y_train)
    prediction = np.asarray(model.predict(transform(x_test, prep)), dtype=np.float64)
    return validate_prediction(prediction, len(x_test), "mlp")


def validate_prediction(prediction: np.ndarray, rows: int, name: str) -> np.ndarray:
    value = np.asarray(prediction, dtype=np.float64)
    if value.shape != (rows, RANK) or not np.isfinite(value).all():
        raise ValueError(f"{name} emitted invalid coordinates")
    return value


def raw_pca_scores(
    x_train: np.ndarray,
    x_test: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    prep = fit_preprocessor(x_train)
    train = transform(x_train, prep)
    test = transform(x_test, prep)
    count = min(int(components), train.shape[0] - 1, train.shape[1])
    model = PCA(n_components=count, svd_solver="randomized", random_state=int(seed))
    train_scores = model.fit_transform(train)
    test_scores = model.transform(test)
    if not np.isfinite(train_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("raw legal PCA produced non-finite scores")
    return train_scores, test_scores


def coordinate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    error = y_pred - y_true
    denominator = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    r2 = 1.0 - float(np.sum(error * error)) / denominator
    correlations: list[float] = []
    for dimension in range(RANK):
        if np.std(y_true[:, dimension]) <= 1e-12 or np.std(y_pred[:, dimension]) <= 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(
                float(pearsonr(y_true[:, dimension], y_pred[:, dimension]).statistic)
            )
    return {
        "coordinate_rmse": float(np.sqrt(np.mean(error * error))),
        "multivariate_coordinate_r2": r2,
        "coordinate_pearsons": correlations,
        "median_coordinate_pearson": float(np.median(correlations)),
    }


def deterministic_derangement(length: int, seed: int) -> np.ndarray:
    original = np.arange(length)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        permutation = rng.permutation(original)
        if np.all(permutation != original):
            return permutation
    return np.roll(original, 1)


def inner_score(
    wells: list[Any],
    valid: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    return profile_rmse(
        wells,
        valid,
        profiles_from_coordinates(mean, components, coordinates),
    )


def best(candidates: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    return min(candidates, key=lambda value: tuple(value))


def score_complete_profiles(
    wells: list[Any],
    profiles: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    metrics = []
    per_well = np.empty(len(wells), dtype=np.float64)
    for index, well in enumerate(wells):
        sufficient_sse = float(T033.profile_sse(well, profiles[index]))
        error = T033.interpolate_profile(profiles[index], well.rows) - well.residual
        metric = T033.error_metric(error)
        if abs(metric.sse - sufficient_sse) > max(1e-6, sufficient_sse * 1e-11):
            raise ArithmeticError("actual-row sufficient-statistic mismatch")
        metrics.append(metric)
        per_well[index] = sufficient_sse
    return T033.summarize(metrics), per_well


def run(config: dict[str, Any], output_dir: Path, preflight_only: bool = False) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    wells, _, _, assignments = T033.load_wells()
    features = load_features(wells, assignments)
    contexts = build_contexts(features.folds)
    residual_profiles = np.stack([well.profile for well in wells])
    baseline_metrics = [T033.error_metric(well.e011 - well.truth) for well in wells]
    baseline = T033.summarize(baseline_metrics)
    baseline_well_sse = np.asarray([metric.sse for metric in baseline_metrics])
    rows_per_well = np.asarray([well.rows for well in wells], dtype=int)
    if preflight_only:
        return {
            "status": "PREFLIGHT_PASS",
            "wells": len(wells),
            "contexts": len(contexts),
            "profiles_shape": list(residual_profiles.shape),
            "summary_shape": list(features.summary.shape),
            "raw_shape": list(features.raw.shape),
            "privileged_shape": list(features.privileged.shape),
            "baseline_rmse": baseline["rmse"],
            "t033_script_sha256": sha256_file(T033_SCRIPT),
            "t031_tensor_sha256": sha256_file(T031_CACHE),
        }

    branch_config = {item["name"]: item for item in config["branches"]}
    profile_maps = {
        name: np.zeros((5, len(wells), GRID), dtype=np.float64)
        for name in RUN_BRANCHES
    }
    coordinate_maps = {
        name: np.zeros((5, len(wells), RANK), dtype=np.float64)
        for name in RUN_BRANCHES
    }
    true_coordinate_maps = np.zeros((5, len(wells), RANK), dtype=np.float64)
    basis_mean_maps = np.zeros((5, len(wells), GRID), dtype=np.float64)
    basis_component_maps = np.zeros((5, len(wells), RANK, GRID), dtype=np.float64)
    oracle_maps = np.zeros((5, len(wells), GRID), dtype=np.float64)
    filled = {name: np.zeros((5, len(wells)), dtype=bool) for name in RUN_BRANCHES}
    oracle_filled = np.zeros((5, len(wells)), dtype=bool)
    context_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    max_raw_components = max(
        max(branch_config["raw_pca_ridge"]["components"]),
        max(branch_config["raw_pca_extra_trees"]["components"]),
        max(branch_config["summary_plus_raw_pca_ridge"]["raw_components"]),
    )

    for context in contexts:
        seed = 35000 + context.map_index * 1000 + context.fold * 100
        inner_mean, inner_components = fit_action_basis(residual_profiles[context.inner_train])
        inner_y_train = coordinates_for(context.inner_train, wells, inner_mean, inner_components)
        inner_y_valid = coordinates_for(context.inner_valid, wells, inner_mean, inner_components)
        outer_mean, outer_components = fit_action_basis(residual_profiles[context.outer_train])
        outer_y_train = coordinates_for(context.outer_train, wells, outer_mean, outer_components)
        outer_y_test = coordinates_for(context.outer_test, wells, outer_mean, outer_components)
        oracle_profile = profiles_from_coordinates(outer_mean, outer_components, outer_y_test)
        oracle_maps[context.map_index, context.outer_test] = oracle_profile
        oracle_filled[context.map_index, context.outer_test] = True
        true_coordinate_maps[context.map_index, context.outer_test] = outer_y_test
        basis_mean_maps[context.map_index, context.outer_test] = outer_mean
        basis_component_maps[context.map_index, context.outer_test] = outer_components
        membership_rows.append(
            {
                "context": context.key,
                "map": context.map_index + 1,
                "fold": context.fold,
                "outer_train_wells": len(context.outer_train),
                "inner_train_wells": len(context.inner_train),
                "inner_valid_wells": len(context.inner_valid),
                "outer_test_wells": len(context.outer_test),
                "same_well_excluded": not bool(
                    np.intersect1d(context.outer_train, context.outer_test).size
                ),
            }
        )

        inner_raw_train, inner_raw_valid = raw_pca_scores(
            features.raw[context.inner_train],
            features.raw[context.inner_valid],
            max_raw_components,
            seed + 1,
        )
        outer_raw_train, outer_raw_test = raw_pca_scores(
            features.raw[context.outer_train],
            features.raw[context.outer_test],
            max_raw_components,
            seed + 2,
        )

        def record(name: str, prediction: np.ndarray, params: dict[str, Any]) -> None:
            prediction = validate_prediction(prediction, len(context.outer_test), name)
            profile = profiles_from_coordinates(outer_mean, outer_components, prediction)
            profile_maps[name][context.map_index, context.outer_test] = profile
            coordinate_maps[name][context.map_index, context.outer_test] = prediction
            filled[name][context.map_index, context.outer_test] = True
            context_rows.append(
                {
                    "context": context.key,
                    "branch": name,
                    "wells": len(context.outer_test),
                    "tvt_rmse": profile_rmse(wells, context.outer_test, profile),
                    **coordinate_metrics(outer_y_test, prediction),
                }
            )
            hyper_rows.append({"context": context.key, "branch": name, **params})

        record(
            "mean_coordinates",
            np.repeat(outer_y_train.mean(axis=0, keepdims=True), len(context.outer_test), axis=0),
            {"parameter": "outer_training_mean"},
        )

        item = branch_config["summary_ridge"]
        inner_order = target_feature_order(features.summary[context.inner_train], inner_y_train)
        candidates = []
        for count in item["feature_counts"]:
            for alpha in item["alphas"]:
                prediction = ridge_predict(
                    features.summary[context.inner_train],
                    inner_y_train,
                    features.summary[context.inner_valid],
                    alpha,
                    inner_order[: int(count)],
                )
                candidates.append(
                    (
                        inner_score(wells, context.inner_valid, inner_mean, inner_components, prediction),
                        int(count),
                        float(alpha),
                    )
                )
        selected_score, count, alpha = best(candidates)
        outer_order = target_feature_order(features.summary[context.outer_train], outer_y_train)
        record(
            "summary_ridge",
            ridge_predict(
                features.summary[context.outer_train],
                outer_y_train,
                features.summary[context.outer_test],
                alpha,
                outer_order[:count],
            ),
            {"feature_count": count, "alpha": alpha, "inner_tvt_rmse": selected_score},
        )

        item = branch_config["summary_extra_trees"]
        candidates = []
        candidate_index = 0
        for max_features in item["max_features"]:
            for leaf in item["min_samples_leaf"]:
                prediction = extra_trees_predict(
                    features.summary[context.inner_train],
                    inner_y_train,
                    features.summary[context.inner_valid],
                    n_estimators=item["n_estimators"],
                    max_features=max_features,
                    min_samples_leaf=leaf,
                    seed=seed + 100 + candidate_index,
                )
                candidates.append(
                    (
                        inner_score(wells, context.inner_valid, inner_mean, inner_components, prediction),
                        float(max_features),
                        int(leaf),
                    )
                )
                candidate_index += 1
        selected_score, max_features, leaf = best(candidates)
        record(
            "summary_extra_trees",
            extra_trees_predict(
                features.summary[context.outer_train],
                outer_y_train,
                features.summary[context.outer_test],
                n_estimators=item["n_estimators"],
                max_features=max_features,
                min_samples_leaf=leaf,
                seed=seed + 199,
            ),
            {
                "max_features": max_features,
                "min_samples_leaf": leaf,
                "inner_tvt_rmse": selected_score,
            },
        )

        item = branch_config["summary_hist_gradient_boosting"]
        candidates = []
        candidate_index = 0
        for leaves in item["max_leaf_nodes"]:
            for l2 in item["l2_regularization"]:
                prediction = hist_predict(
                    features.summary[context.inner_train],
                    inner_y_train,
                    features.summary[context.inner_valid],
                    max_leaf_nodes=leaves,
                    l2=l2,
                    learning_rate=item["learning_rate"],
                    max_iter=item["max_iter"],
                    seed=seed + 200 + candidate_index * 10,
                )
                candidates.append(
                    (
                        inner_score(wells, context.inner_valid, inner_mean, inner_components, prediction),
                        int(leaves),
                        float(l2),
                    )
                )
                candidate_index += 1
        selected_score, leaves, l2 = best(candidates)
        record(
            "summary_hist_gradient_boosting",
            hist_predict(
                features.summary[context.outer_train],
                outer_y_train,
                features.summary[context.outer_test],
                max_leaf_nodes=leaves,
                l2=l2,
                learning_rate=item["learning_rate"],
                max_iter=item["max_iter"],
                seed=seed + 299,
            ),
            {
                "max_leaf_nodes": leaves,
                "l2_regularization": l2,
                "inner_tvt_rmse": selected_score,
            },
        )

        item = branch_config["summary_mlp"]
        candidates = []
        candidate_index = 0
        for hidden_values in item["hidden_layer_sizes"]:
            hidden = tuple(int(value) for value in hidden_values)
            for alpha_value in item["alphas"]:
                prediction = mlp_predict(
                    features.summary[context.inner_train],
                    inner_y_train,
                    features.summary[context.inner_valid],
                    hidden=hidden,
                    alpha=alpha_value,
                    max_iter=item["max_iter"],
                    early_stopping=item["early_stopping"],
                    seed=seed + 300 + candidate_index,
                )
                candidates.append(
                    (
                        inner_score(wells, context.inner_valid, inner_mean, inner_components, prediction),
                        hidden,
                        float(alpha_value),
                    )
                )
                candidate_index += 1
        selected_score, hidden, alpha_value = best(candidates)
        record(
            "summary_mlp",
            mlp_predict(
                features.summary[context.outer_train],
                outer_y_train,
                features.summary[context.outer_test],
                hidden=hidden,
                alpha=alpha_value,
                max_iter=item["max_iter"],
                early_stopping=item["early_stopping"],
                seed=seed + 399,
            ),
            {
                "hidden": "x".join(map(str, hidden)),
                "alpha": alpha_value,
                "inner_tvt_rmse": selected_score,
            },
        )

        item = branch_config["raw_pca_ridge"]
        candidates = []
        for components_count in item["components"]:
            for alpha_value in item["alphas"]:
                prediction = ridge_predict(
                    inner_raw_train[:, : int(components_count)],
                    inner_y_train,
                    inner_raw_valid[:, : int(components_count)],
                    alpha_value,
                )
                candidates.append(
                    (
                        inner_score(wells, context.inner_valid, inner_mean, inner_components, prediction),
                        int(components_count),
                        float(alpha_value),
                    )
                )
        selected_score, components_count, alpha_value = best(candidates)
        record(
            "raw_pca_ridge",
            ridge_predict(
                outer_raw_train[:, :components_count],
                outer_y_train,
                outer_raw_test[:, :components_count],
                alpha_value,
            ),
            {
                "raw_components": components_count,
                "alpha": alpha_value,
                "inner_tvt_rmse": selected_score,
            },
        )

        item = branch_config["raw_pca_extra_trees"]
        candidates = []
        candidate_index = 0
        for components_count in item["components"]:
            for max_features in item["max_features"]:
                for leaf in item["min_samples_leaf"]:
                    prediction = extra_trees_predict(
                        inner_raw_train[:, : int(components_count)],
                        inner_y_train,
                        inner_raw_valid[:, : int(components_count)],
                        n_estimators=item["n_estimators"],
                        max_features=max_features,
                        min_samples_leaf=leaf,
                        seed=seed + 400 + candidate_index,
                    )
                    candidates.append(
                        (
                            inner_score(wells, context.inner_valid, inner_mean, inner_components, prediction),
                            int(components_count),
                            float(max_features),
                            int(leaf),
                        )
                    )
                    candidate_index += 1
        selected_score, components_count, max_features, leaf = best(candidates)
        record(
            "raw_pca_extra_trees",
            extra_trees_predict(
                outer_raw_train[:, :components_count],
                outer_y_train,
                outer_raw_test[:, :components_count],
                n_estimators=item["n_estimators"],
                max_features=max_features,
                min_samples_leaf=leaf,
                seed=seed + 499,
            ),
            {
                "raw_components": components_count,
                "max_features": max_features,
                "min_samples_leaf": leaf,
                "inner_tvt_rmse": selected_score,
            },
        )

        item = branch_config["summary_plus_raw_pca_ridge"]
        candidates = []
        for components_count in item["raw_components"]:
            inner_combined_train = np.concatenate(
                [features.summary[context.inner_train], inner_raw_train[:, : int(components_count)]],
                axis=1,
            )
            inner_combined_valid = np.concatenate(
                [features.summary[context.inner_valid], inner_raw_valid[:, : int(components_count)]],
                axis=1,
            )
            for alpha_value in item["alphas"]:
                prediction = ridge_predict(
                    inner_combined_train,
                    inner_y_train,
                    inner_combined_valid,
                    alpha_value,
                )
                candidates.append(
                    (
                        inner_score(wells, context.inner_valid, inner_mean, inner_components, prediction),
                        int(components_count),
                        float(alpha_value),
                    )
                )
        selected_score, components_count, alpha_value = best(candidates)
        record(
            "summary_plus_raw_pca_ridge",
            ridge_predict(
                np.concatenate(
                    [features.summary[context.outer_train], outer_raw_train[:, :components_count]],
                    axis=1,
                ),
                outer_y_train,
                np.concatenate(
                    [features.summary[context.outer_test], outer_raw_test[:, :components_count]],
                    axis=1,
                ),
                alpha_value,
            ),
            {
                "raw_components": components_count,
                "alpha": alpha_value,
                "inner_tvt_rmse": selected_score,
            },
        )

        item = branch_config["privileged_geology_extra_trees"]
        candidates = []
        candidate_index = 0
        for max_features in item["max_features"]:
            for leaf in item["min_samples_leaf"]:
                prediction = extra_trees_predict(
                    features.privileged[context.inner_train],
                    inner_y_train,
                    features.privileged[context.inner_valid],
                    n_estimators=item["n_estimators"],
                    max_features=max_features,
                    min_samples_leaf=leaf,
                    seed=seed + 500 + candidate_index,
                )
                candidates.append(
                    (
                        inner_score(wells, context.inner_valid, inner_mean, inner_components, prediction),
                        float(max_features),
                        int(leaf),
                    )
                )
                candidate_index += 1
        selected_score, max_features, leaf = best(candidates)
        record(
            "privileged_geology_extra_trees",
            extra_trees_predict(
                features.privileged[context.outer_train],
                outer_y_train,
                features.privileged[context.outer_test],
                n_estimators=item["n_estimators"],
                max_features=max_features,
                min_samples_leaf=leaf,
                seed=seed + 599,
            ),
            {
                "max_features": max_features,
                "min_samples_leaf": leaf,
                "inner_tvt_rmse": selected_score,
            },
        )

        # Destructive control: independently derange training and validation coordinates.
        pseudo_inner_train = inner_y_train[
            deterministic_derangement(len(inner_y_train), seed + 600)
        ]
        pseudo_inner_valid = inner_y_valid[
            deterministic_derangement(len(inner_y_valid), seed + 601)
        ]
        item = branch_config["summary_ridge"]
        inner_order = target_feature_order(
            features.summary[context.inner_train], pseudo_inner_train
        )
        candidates = []
        for count in item["feature_counts"]:
            for alpha_value in item["alphas"]:
                prediction = ridge_predict(
                    features.summary[context.inner_train],
                    pseudo_inner_train,
                    features.summary[context.inner_valid],
                    alpha_value,
                    inner_order[: int(count)],
                )
                candidates.append(
                    (
                        float(np.sqrt(np.mean((prediction - pseudo_inner_valid) ** 2))),
                        int(count),
                        float(alpha_value),
                    )
                )
        pseudo_score, count, alpha_value = best(candidates)
        pseudo_outer_train = outer_y_train[
            deterministic_derangement(len(outer_y_train), seed + 602)
        ]
        outer_order = target_feature_order(
            features.summary[context.outer_train], pseudo_outer_train
        )
        record(
            "shuffled_coordinate_control",
            ridge_predict(
                features.summary[context.outer_train],
                pseudo_outer_train,
                features.summary[context.outer_test],
                alpha_value,
                outer_order[:count],
            ),
            {
                "feature_count": count,
                "alpha": alpha_value,
                "inner_pseudo_coordinate_rmse": pseudo_score,
            },
        )

    if not oracle_filled.all() or not all(mask.all() for mask in filled.values()):
        raise AssertionError("T035 OOF profile coverage is incomplete")

    averaged_profiles = {
        name: np.clip(value.mean(axis=0), -CAP, CAP)
        for name, value in profile_maps.items()
    }
    averaged_oracle = np.clip(oracle_maps.mean(axis=0), -CAP, CAP)
    oracle_summary, oracle_well_sse = score_complete_profiles(wells, averaged_oracle)
    if abs(oracle_summary["rmse"] - 2.161328942895813) > 1e-9:
        raise ValueError("T035 does not reproduce the frozen T033 rank-8 oracle")

    summaries: dict[str, dict[str, Any]] = {}
    well_sse: dict[str, np.ndarray] = {}
    for name, profile in averaged_profiles.items():
        summary, per_well = score_complete_profiles(wells, profile)
        true_coordinates = true_coordinate_maps.reshape(-1, RANK)
        predicted_coordinates = coordinate_maps[name].reshape(-1, RANK)
        summary.update(coordinate_metrics(true_coordinates, predicted_coordinates))
        summary["gain_vs_e011"] = baseline["rmse"] - summary["rmse"]
        summary["oracle_gain_retention"] = summary["gain_vs_e011"] / (
            baseline["rmse"] - oracle_summary["rmse"]
        )
        summary["positive_well_fraction"] = float(
            np.mean(per_well < baseline_well_sse - 1e-9)
        )
        summaries[name] = summary
        well_sse[name] = per_well

    best_legal = min(
        LEGAL_BRANCHES,
        key=lambda name: (float(summaries[name]["rmse"]), name),
    )
    sign_coordinate_maps = -coordinate_maps[best_legal]
    sign_profile_maps = np.empty_like(profile_maps[best_legal])
    for map_index in range(5):
        for well_index in range(len(wells)):
            sign_profile_maps[map_index, well_index] = np.clip(
                basis_mean_maps[map_index, well_index]
                + sign_coordinate_maps[map_index, well_index]
                @ basis_component_maps[map_index, well_index],
                -CAP,
                CAP,
            )
    averaged_profiles["sign_flipped_control"] = np.clip(
        sign_profile_maps.mean(axis=0), -CAP, CAP
    )
    sign_summary, sign_well_sse = score_complete_profiles(
        wells, averaged_profiles["sign_flipped_control"]
    )
    sign_summary.update(
        coordinate_metrics(
            true_coordinate_maps.reshape(-1, RANK),
            sign_coordinate_maps.reshape(-1, RANK),
        )
    )
    sign_summary["gain_vs_e011"] = baseline["rmse"] - sign_summary["rmse"]
    sign_summary["oracle_gain_retention"] = sign_summary["gain_vs_e011"] / (
        baseline["rmse"] - oracle_summary["rmse"]
    )
    sign_summary["positive_well_fraction"] = float(
        np.mean(sign_well_sse < baseline_well_sse - 1e-9)
    )
    summaries["sign_flipped_control"] = sign_summary
    well_sse["sign_flipped_control"] = sign_well_sse

    branch_rows: list[dict[str, Any]] = []
    for name in ALL_BRANCHES:
        summary = summaries[name]
        row = {
            "branch": name,
            "eligible": name in LEGAL_BRANCHES,
            **{key: value for key, value in summary.items() if key != "coordinate_pearsons"},
        }
        for dimension, value in enumerate(summary["coordinate_pearsons"], start=1):
            row[f"coordinate_{dimension}_pearson"] = value
        branch_rows.append(row)

    group_rows: list[dict[str, Any]] = []
    for name in ALL_BRANCHES:
        for system, assignment in features.groups.items():
            for group in range(5):
                indices = np.flatnonzero(assignment == group)
                candidate_rmse = math.sqrt(
                    float(well_sse[name][indices].sum())
                    / int(rows_per_well[indices].sum())
                )
                baseline_rmse = math.sqrt(
                    float(baseline_well_sse[indices].sum())
                    / int(rows_per_well[indices].sum())
                )
                group_rows.append(
                    {
                        "branch": name,
                        "group_system": system,
                        "group": group,
                        "wells": len(indices),
                        "baseline_rmse": baseline_rmse,
                        "candidate_rmse": candidate_rmse,
                        "gain_vs_e011": baseline_rmse - candidate_rmse,
                    }
                )

    map_rows: list[dict[str, Any]] = []
    profile_maps["sign_flipped_control"] = sign_profile_maps
    for name in ALL_BRANCHES:
        for map_index in range(5):
            map_summary, _ = score_complete_profiles(wells, profile_maps[name][map_index])
            map_rows.append(
                {
                    "branch": name,
                    "map": map_index + 1,
                    "rmse": map_summary["rmse"],
                    "gain_vs_e011": baseline["rmse"] - map_summary["rmse"],
                }
            )

    best_metrics = summaries[best_legal]
    legacy_rows = [
        row
        for row in group_rows
        if row["branch"] == best_legal
        and row["group_system"] in {"legacy_spatial", "legacy_typewell"}
    ]
    maximum_negative_gain = max(
        summaries["shuffled_coordinate_control"]["gain_vs_e011"],
        summaries["sign_flipped_control"]["gain_vs_e011"],
    )
    gates = {
        "maximum_best_legal_tvt_rmse": best_metrics["rmse"]
        <= config["go_gate"]["maximum_best_legal_tvt_rmse"],
        "minimum_oracle_gain_retention": best_metrics["oracle_gain_retention"]
        >= config["go_gate"]["minimum_oracle_gain_retention"],
        "minimum_multivariate_coordinate_r2": best_metrics["multivariate_coordinate_r2"]
        >= config["go_gate"]["minimum_multivariate_coordinate_r2"],
        "minimum_median_coordinate_pearson": best_metrics["median_coordinate_pearson"]
        >= config["go_gate"]["minimum_median_coordinate_pearson"],
        "minimum_positive_well_fraction": best_metrics["positive_well_fraction"]
        >= config["go_gate"]["minimum_positive_well_fraction"],
        "positive_every_legacy_group": all(
            float(row["gain_vs_e011"]) > 0.0 for row in legacy_rows
        ),
        "negative_controls_below_gain_cap": maximum_negative_gain
        <= config["go_gate"]["maximum_negative_control_gain_vs_e011"],
    }
    go = all(gates.values())
    breakthrough = go and best_metrics["rmse"] <= config["breakthrough_gate"][
        "maximum_best_legal_tvt_rmse"
    ]
    privileged_rmse = summaries["privileged_geology_extra_trees"]["rmse"]
    privileged_mediation = (
        not go
        and privileged_rmse
        <= config["privileged_mediation"]["maximum_privileged_tvt_rmse"]
    )
    stop = (
        best_metrics["rmse"]
        > config["stop_gate"]["minimum_best_legal_tvt_rmse_for_stop"]
        or best_metrics["multivariate_coordinate_r2"]
        < config["stop_gate"]["maximum_best_legal_coordinate_r2_for_stop"]
    )
    if breakthrough:
        decision = "BREAKTHROUGH_LEGAL_PCA_COORDINATES"
    elif go:
        decision = "GO_LEGAL_PCA_COORDINATES"
    elif privileged_mediation:
        decision = "STOP_LEGAL_FAIL_STRONG_PRIVILEGED_MEDIATION"
    elif stop:
        decision = "STOP_CLOSE_COMPACT_MANIFOLD_IDENTIFICATION"
    else:
        decision = "RESEARCH_ONLY_PARTIAL_PCA_COORDINATE_SIGNAL"

    well_rows: list[dict[str, Any]] = []
    for index, well in enumerate(wells):
        row: dict[str, Any] = {
            "well_id": well.well_id,
            "rows": well.rows,
            "baseline_sse": baseline_well_sse[index],
            "candidate_sse": well_sse[best_legal][index],
            "baseline_rmse": math.sqrt(baseline_well_sse[index] / well.rows),
            "candidate_rmse": math.sqrt(well_sse[best_legal][index] / well.rows),
        }
        for dimension in range(RANK):
            row[f"mean_true_coordinate_{dimension + 1}"] = float(
                true_coordinate_maps[:, index, dimension].mean()
            )
            row[f"mean_predicted_coordinate_{dimension + 1}"] = float(
                coordinate_maps[best_legal][:, index, dimension].mean()
            )
        well_rows.append(row)

    edge_checks = {
        "wells_773": len(wells) == 773,
        "contexts_25": len(contexts) == 25,
        "rank8": true_coordinate_maps.shape == (5, 773, RANK),
        "all_oof_predictions_filled": all(mask.all() for mask in filled.values()),
        "oracle_profiles_filled": oracle_filled.all(),
        "all_profiles_finite": all(
            np.isfinite(profile).all() for profile in averaged_profiles.values()
        ),
        "all_profiles_capped": all(
            float(np.max(np.abs(profile))) <= CAP + 1e-9
            for profile in averaged_profiles.values()
        ),
        "t033_rank8_oracle_identity": abs(
            oracle_summary["rmse"] - 2.161328942895813
        )
        <= 1e-9,
        "e011_identity": abs(baseline["rmse"] - 12.550756295689673) <= 1e-9,
        "branch_count_11": len(branch_rows) == 11,
        "stress_systems_complete": len(features.groups) == 5
        and all(set(map(int, np.unique(value))) == set(range(5)) for value in features.groups.values()),
        "same_well_exclusion": all(row["same_well_excluded"] for row in membership_rows),
        "controls_ineligible": all(
            name not in LEGAL_BRANCHES
            for name in [
                "privileged_geology_extra_trees",
                "shuffled_coordinate_control",
                "sign_flipped_control",
            ]
        ),
        "best_legal_eligible": best_legal in LEGAL_BRANCHES,
    }
    edge_status = "PASS" if all(edge_checks.values()) else "FAIL"
    implementation = {
        "schema_version": 1,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "t033_script_sha256": sha256_file(T033_SCRIPT),
        "t031_tensor_sha256": sha256_file(T031_CACHE),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "threads": 2,
        "target": "partition-local T033 rank-8 actual-row least-squares PCA coordinates",
        "selection": "inner fold=(outer+1)%5 by exact actual-row TVT RMSE",
        "aggregation": "reconstruct context action profiles and average five maps per well",
        "controls": "shuffled coordinate targets and sign-flipped selected coordinates",
        "privileged": "six train-only formation offsets plus masks, ineligible",
    }
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "decision": decision,
        "baseline": baseline,
        "rank8_oracle": oracle_summary,
        "best_legal_branch": best_legal,
        "best_legal_metrics": best_metrics,
        "privileged_metrics": summaries["privileged_geology_extra_trees"],
        "negative_control_gain_max": maximum_negative_gain,
        "gates": gates,
        "breakthrough": breakthrough,
        "privileged_mediation": privileged_mediation,
        "stop_gate": stop,
        "edge_status": edge_status,
        "contexts": len(contexts),
        "wells": len(wells),
        "runtime_seconds": time.time() - started,
    }

    write_csv(output_dir / "branch_metrics.csv", branch_rows)
    write_csv(output_dir / "context_metrics.csv", context_rows)
    write_csv(output_dir / "group_metrics.csv", group_rows)
    write_csv(output_dir / "map_metrics.csv", map_rows)
    write_csv(output_dir / "hyperparameters.csv", hyper_rows)
    write_csv(output_dir / "membership_audit.csv", membership_rows)
    write_csv(output_dir / "selected_well_metrics.csv", well_rows)
    dump_json(output_dir / "implementation_receipt.json", implementation)
    dump_json(
        output_dir / "edge_cases.json",
        {"schema_version": 1, "status": edge_status, "checks": edge_checks},
    )
    dump_json(output_dir / "summary.json", summary)
    report = [
        "# T035 Result — Actual rank-8 PCA coordinate identifiability",
        "",
        f"Decision: **{decision}**",
        "",
        f"Best legal branch: `{best_legal}`",
        f"- TVT RMSE: {best_metrics['rmse']:.12f}",
        f"- Gain versus E011: {best_metrics['gain_vs_e011']:.12f}",
        f"- Rank-8 oracle gain retention: {best_metrics['oracle_gain_retention']:.6%}",
        f"- Coordinate R2: {best_metrics['multivariate_coordinate_r2']:.6f}",
        f"- Median coordinate Pearson: {best_metrics['median_coordinate_pearson']:.6f}",
        f"- Positive wells: {best_metrics['positive_well_fraction']:.6%}",
        "",
        "Privileged geology diagnostic:",
        f"- TVT RMSE: {summaries['privileged_geology_extra_trees']['rmse']:.12f}",
        f"- Coordinate R2: {summaries['privileged_geology_extra_trees']['multivariate_coordinate_r2']:.6f}",
        "",
        "## Gates",
    ]
    report.extend(
        f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in gates.items()
    )
    report.extend(
        [
            "",
            "T035 is the final bounded manifold-identifiability gate. No deployment, Kaggle run, or submission is automatic.",
        ]
    )
    (output_dir / "RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    artifact_names = [
        "RESULT.md",
        "branch_metrics.csv",
        "context_metrics.csv",
        "edge_cases.json",
        "group_metrics.csv",
        "hyperparameters.csv",
        "implementation_receipt.json",
        "map_metrics.csv",
        "membership_audit.csv",
        "selected_well_metrics.csv",
        "summary.json",
    ]
    dump_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": 1,
            "files": [
                {
                    "name": name,
                    "bytes": (output_dir / name).stat().st_size,
                    "sha256": sha256_file(output_dir / name),
                }
                for name in artifact_names
            ],
        },
    )
    return summary


def synthetic_self_test() -> dict[str, Any]:
    rng = np.random.default_rng(35)
    n = 100
    profiles = rng.normal(size=(n, GRID))
    mean, components = fit_action_basis(profiles[:80])
    coordinates = rng.normal(size=(20, RANK))
    reconstructed = profiles_from_coordinates(mean, components, coordinates)
    x = rng.normal(size=(100, 30))
    y = x[:, :8] + rng.normal(scale=0.1, size=(100, 8))
    prediction = ridge_predict(x[:80], y[:80], x[80:], 10.0)
    metrics = coordinate_metrics(y[80:], prediction)
    permutation = deterministic_derangement(80, 35)
    checks = {
        "basis_shape": mean.shape == (GRID,) and components.shape == (RANK, GRID),
        "reconstruction_shape": reconstructed.shape == (20, GRID),
        "reconstruction_cap": float(np.max(np.abs(reconstructed))) <= CAP,
        "ridge_shape": prediction.shape == (20, RANK),
        "ridge_finite": bool(np.isfinite(prediction).all()),
        "derangement": bool(np.all(permutation != np.arange(80))),
        "coordinate_r2_positive": metrics["multivariate_coordinate_r2"] > 0.5,
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
