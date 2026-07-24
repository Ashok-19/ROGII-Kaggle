#!/usr/bin/env python3
"""T033 cross-fitted residual action-manifold capacity diagnostic.

Hidden labels are used only for held-out oracle fitting/selection and scoring.
No legal selector, competition prediction, deployment package, or submission is emitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "tracking/evidence/T033/config.json"
DEFAULT_OUTPUT = ROOT / "scratch/agents/t033-manifold-20260724/main"
GRID = 128
CAP = 160.0


@dataclass(frozen=True)
class WellData:
    well_id: str
    rows: int
    truth: np.ndarray
    e011: np.ndarray
    e006: np.ndarray
    residual: np.ndarray
    profile: np.ndarray
    q: np.ndarray
    diag: np.ndarray
    off: np.ndarray
    residual_sq: float


@dataclass(frozen=True)
class Metric:
    rows: int
    sse: float
    datum_sse: float
    trend_sse: float
    shape_sse: float
    rmse: float


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


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


def interpolation_sufficient(residual: np.ndarray, grid: int = GRID) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = residual.size
    q = np.zeros(grid, dtype=np.float64)
    diag = np.zeros(grid, dtype=np.float64)
    off = np.zeros(grid - 1, dtype=np.float64)
    if n == 1:
        q[0] = residual[0]
        diag[0] = 1.0
        return q, diag, off
    positions = np.linspace(0.0, grid - 1.0, n)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, grid - 1)
    wr = positions - left
    wl = 1.0 - wr
    np.add.at(q, left, wl * residual)
    np.add.at(q, right, wr * residual)
    np.add.at(diag, left, wl * wl)
    np.add.at(diag, right, wr * wr)
    mask = right > left
    np.add.at(off, left[mask], wl[mask] * wr[mask])
    return q, diag, off


def interpolate_profile(profile: np.ndarray, rows: int) -> np.ndarray:
    if rows == 1:
        return np.asarray([float(profile[0])], dtype=np.float64)
    return np.interp(np.linspace(0.0, GRID - 1.0, rows), np.arange(GRID), profile)


def profile_sse(well: WellData, profiles: np.ndarray) -> np.ndarray:
    p = np.asarray(profiles, dtype=np.float64)
    one = p.ndim == 1
    if one:
        p = p[None, :]
    values = (
        well.residual_sq
        - 2.0 * (p @ well.q)
        + (p * p) @ well.diag
        + 2.0 * (p[:, :-1] * p[:, 1:]) @ well.off
    )
    values = np.maximum(values, 0.0)
    return values[0] if one else values


def error_metric(error: np.ndarray) -> Metric:
    e = np.asarray(error, dtype=np.float64)
    n = e.size
    x = np.arange(n, dtype=np.float64)
    sse = float(e @ e)
    sum_e = float(e.sum())
    datum = sum_e * sum_e / n
    xc = x - x.mean()
    denom = float(xc @ xc)
    slope_num = float(xc @ e)
    trend = slope_num * slope_num / denom if denom > 0.0 else 0.0
    shape = sse - datum - trend
    tol = max(1e-9, sse * 1e-12)
    if shape < 0.0 and abs(shape) <= tol:
        shape = 0.0
    if shape < 0.0:
        raise ArithmeticError(f"negative shape SSE {shape}")
    return Metric(n, sse, datum, trend, shape, math.sqrt(sse / n))


def summarize(metrics: Iterable[Metric]) -> dict[str, float]:
    values = list(metrics)
    rows = sum(m.rows for m in values)
    sse = sum(m.sse for m in values)
    datum = sum(m.datum_sse for m in values)
    trend = sum(m.trend_sse for m in values)
    shape = sum(m.shape_sse for m in values)
    return {
        "rows": rows,
        "wells": len(values),
        "sse": sse,
        "rmse": math.sqrt(sse / rows),
        "datum_sse": datum,
        "trend_sse": trend,
        "shape_sse": shape,
    }


def load_wells() -> tuple[list[WellData], np.ndarray, np.ndarray, list[dict[str, int]]]:
    path = ROOT / "artifacts/E011/oof_predictions.csv.gz"
    columns = [
        "well_id",
        "hidden_index",
        "target",
        "e006_nested_fusion",
        "spline4_ridge_equal_s075",
    ]
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    frame = frame.sort_values(["well_id", "hidden_index"], kind="mergesort").reset_index(drop=True)
    compact = np.load(ROOT / "artifacts/E011/compact_stats_v1.npz", allow_pickle=False)
    well_order = [str(x) for x in compact["well_ids"]]
    expected_rows = compact["rows"].astype(int)
    if len(well_order) != 773 or len(set(well_order)) != 773:
        raise ValueError("T033 requires exactly 773 unique wells")
    groups = {str(well_id): group for well_id, group in frame.groupby("well_id", sort=False)}
    if set(groups) != set(well_order):
        raise ValueError("OOF well IDs do not exactly match the compact artifact")
    wells: list[WellData] = []
    for index, well_id in enumerate(well_order):
        g = groups[well_id]
        hidden_index = g["hidden_index"].to_numpy(int)
        if len(g) != int(expected_rows[index]):
            raise ValueError(f"{well_id}: hidden row count differs from compact artifact")
        if not np.array_equal(hidden_index, np.arange(len(g), dtype=int)):
            raise ValueError(f"{well_id}: hidden_index is not contiguous from zero")
        truth = g["target"].to_numpy(np.float64)
        e006 = g["e006_nested_fusion"].to_numpy(np.float64)
        e011 = g["spline4_ridge_equal_s075"].to_numpy(np.float64)
        if not np.isfinite(np.column_stack([truth, e006, e011])).all():
            raise ValueError(f"{well_id}: OOF path contains non-finite values")
        residual = truth - e011
        q, diag, off = interpolation_sufficient(residual)
        profile = np.interp(
            np.linspace(0.0, residual.size - 1.0, GRID),
            np.arange(residual.size),
            residual,
        )
        wells.append(WellData(
            well_id=well_id,
            rows=residual.size,
            truth=truth,
            e011=e011,
            e006=e006,
            residual=residual,
            profile=np.clip(profile, -CAP, CAP),
            q=q,
            diag=diag,
            off=off,
            residual_sq=float(residual @ residual),
        ))
    assignments: list[dict[str, int]] = []
    for version in range(1, 6):
        payload = json.loads((ROOT / f"folds/v{version}.json").read_text(encoding="utf-8"))
        assignment = {str(k): int(v) for k, v in payload["assignments"].items()}
        if set(assignment) != set(well_order) or set(assignment.values()) != set(range(5)):
            raise ValueError(f"fold map v{version} does not cover all wells and folds")
        assignments.append(assignment)
    return wells, compact["spatial_assignment"].astype(int), compact["typewell_assignment"].astype(int), assignments


def branch_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in config["branches"]:
        name = item["name"]
        if name in {"e011_zero_action", "spline7_oracle_reference"}:
            continue
        if name == "outer_pca_oracle":
            for rank in item["ranks"]:
                specs.append({"name": f"pca_r{rank:02d}", "family": "pca", "size": int(rank), "compact": rank <= 8, "expanded": rank <= 16})
        elif name == "outer_kmeans_single_oracle":
            for size in item["dictionary_sizes"]:
                specs.append({"name": f"kmeans_single_k{size:03d}", "family": "kmeans_single", "size": int(size), "compact": size <= 32, "expanded": size <= 64})
        elif name == "outer_kmeans_pair_oracle":
            for size in item["dictionary_sizes"]:
                specs.append({"name": f"kmeans_pair_k{size:03d}", "family": "kmeans_pair", "size": int(size), "compact": size <= 16, "expanded": size <= 32})
        elif name == "outer_random_source_oracle":
            for seed in item["seeds"]:
                for size in item["dictionary_sizes"]:
                    specs.append({"name": f"random_source_s{seed}_k{size:03d}", "family": "random_source", "size": int(size), "seed": int(seed), "compact": False, "expanded": size <= 64})
        elif name == "smooth_gaussian_dictionary_control":
            for seed in item["seeds"]:
                for size in item["dictionary_sizes"]:
                    specs.append({"name": f"smooth_gaussian_s{seed}_k{size:03d}", "family": "smooth_gaussian", "size": int(size), "seed": int(seed), "compact": False, "expanded": size <= 64})
    return specs


def pca_oracle_profile(well: WellData, mean: np.ndarray, components: np.ndarray, rank: int) -> np.ndarray:
    v = components[:rank]
    # G is tridiagonal. Form V G V^T without a dense 128x128 matrix.
    vg = v * well.diag[None, :]
    vg[:, :-1] += v[:, 1:] * well.off[None, :]
    vg[:, 1:] += v[:, :-1] * well.off[None, :]
    gram = vg @ v.T
    gmean = well.diag * mean
    gmean[:-1] += well.off * mean[1:]
    gmean[1:] += well.off * mean[:-1]
    rhs = v @ (well.q - gmean)
    coef = np.linalg.lstsq(gram + np.eye(rank) * 1e-10, rhs, rcond=None)[0]
    return np.clip(mean + coef @ v, -CAP, CAP)


def best_profile(well: WellData, candidates: np.ndarray) -> np.ndarray:
    scores = profile_sse(well, candidates)
    index = int(np.argmin(scores))
    return np.asarray(candidates[index], dtype=np.float64)


def pair_candidates(centers: np.ndarray, weights: list[float]) -> np.ndarray:
    profiles: list[np.ndarray] = []
    k = centers.shape[0]
    for i in range(k):
        for j in range(i, k):
            for weight in weights:
                profiles.append((1.0 - weight) * centers[i] + weight * centers[j])
    return np.clip(np.asarray(profiles, dtype=np.float64), -CAP, CAP)


def smooth_gaussian_dictionary(train_profiles: np.ndarray, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=min(32, train_profiles.shape[0] - 1, GRID), svd_solver="full")
    pca.fit(train_profiles)
    z = rng.normal(size=(size, pca.n_components_))
    samples = pca.mean_[None, :] + (z * np.sqrt(np.maximum(pca.explained_variance_, 1e-12))[None, :]) @ pca.components_
    # Suppress high-frequency synthetic artifacts while retaining covariance scale.
    kernel = np.asarray([0.25, 0.5, 0.25])
    smoothed = np.empty_like(samples)
    for i, row in enumerate(samples):
        smoothed[i] = np.convolve(np.pad(row, (1, 1), mode="edge"), kernel, mode="valid")
    return np.clip(smoothed, -CAP, CAP)


def spline7_reference(wells: list[WellData]) -> tuple[list[Metric], list[np.ndarray]]:
    compact = np.load(ROOT / "artifacts/E011/compact_stats_v1.npz", allow_pickle=False)
    coeff = compact["spline7__target_coefficients"].astype(np.float64)
    metrics: list[Metric] = []
    paths: list[np.ndarray] = []
    knots = np.linspace(1.0 / 7.0, 1.0, 7)
    xp = np.concatenate([[0.0], knots])
    for i, well in enumerate(wells):
        s = np.linspace(0.0, 1.0, well.rows)
        correction = np.interp(s, xp, np.concatenate([[0.0], coeff[i]]))
        prediction = well.e006 + correction
        paths.append(prediction)
        metrics.append(error_metric(prediction - well.truth))
    return metrics, paths


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    wells, spatial, typewell, fold_maps = load_wells()
    n = len(wells)
    profiles = np.stack([w.profile for w in wells])
    specs = branch_specs(config)
    names = [spec["name"] for spec in specs]
    sums = {name: np.zeros((n, GRID), dtype=np.float64) for name in names}
    map_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    same_well_violations = 0
    deterministic_dictionary_delta = 0.0
    random_dictionary_deterministic = True

    kmeans_sizes = sorted({spec["size"] for spec in specs if spec["family"] in {"kmeans_single", "kmeans_pair"}})
    pca_max = max(spec["size"] for spec in specs if spec["family"] == "pca")
    pair_weights = next(item["weights"] for item in config["branches"] if item["name"] == "outer_kmeans_pair_oracle")

    for map_index, assignment in enumerate(fold_maps):
        folds = np.asarray([assignment[w.well_id] for w in wells], dtype=int)
        for fold in range(5):
            test_idx = np.flatnonzero(folds == fold)
            train_idx = np.flatnonzero(folds != fold)
            train_profiles = profiles[train_idx]
            pca = PCA(n_components=pca_max, svd_solver="full")
            pca.fit(train_profiles)
            centers_by_size: dict[int, np.ndarray] = {}
            pair_by_size: dict[int, np.ndarray] = {}
            for size in kmeans_sizes:
                model = KMeans(n_clusters=size, random_state=33000 + 100 * map_index + 10 * fold + size, n_init=16, max_iter=300, algorithm="lloyd")
                model.fit(train_profiles)
                centers = np.clip(model.cluster_centers_, -CAP, CAP)
                centers_by_size[size] = centers
                if any(spec["family"] == "kmeans_pair" and spec["size"] == size for spec in specs):
                    pair_by_size[size] = pair_candidates(centers, pair_weights)
            if map_index == 0 and fold == 0:
                check_size = kmeans_sizes[0]
                check_seed = 33000 + check_size
                repeat_model = KMeans(
                    n_clusters=check_size,
                    random_state=check_seed,
                    n_init=16,
                    max_iter=300,
                    algorithm="lloyd",
                )
                repeat_model.fit(train_profiles)
                repeat_centers = np.clip(repeat_model.cluster_centers_, -CAP, CAP)
                deterministic_dictionary_delta = max(
                    deterministic_dictionary_delta,
                    float(np.max(np.abs(centers_by_size[check_size] - repeat_centers))),
                )
            random_by_key: dict[tuple[int, int], np.ndarray] = {}
            for spec in specs:
                if spec["family"] == "random_source":
                    key = (spec["seed"], spec["size"])
                    rng = np.random.default_rng(spec["seed"] + 1000 * map_index + 100 * fold)
                    order = rng.permutation(train_idx)
                    random_by_key[key] = profiles[order[: spec["size"]]]
            if map_index == 0 and fold == 0:
                check_seed = 3301
                first_order = np.random.default_rng(check_seed).permutation(train_idx)
                second_order = np.random.default_rng(check_seed).permutation(train_idx)
                random_dictionary_deterministic = bool(np.array_equal(first_order, second_order))
            gaussian_by_key: dict[tuple[int, int], np.ndarray] = {}
            for spec in specs:
                if spec["family"] == "smooth_gaussian":
                    key = (spec["seed"], spec["size"])
                    gaussian_by_key[key] = smooth_gaussian_dictionary(
                        train_profiles,
                        spec["size"],
                        spec["seed"] + 1000 * map_index + 100 * fold,
                    )
            if map_index == 0 and fold == 0:
                gaussian_seed = 3311
                first_gaussian = smooth_gaussian_dictionary(train_profiles, 16, gaussian_seed)
                second_gaussian = smooth_gaussian_dictionary(train_profiles, 16, gaussian_seed)
                deterministic_dictionary_delta = max(
                    deterministic_dictionary_delta,
                    float(np.max(np.abs(first_gaussian - second_gaussian))),
                )

            for q in test_idx:
                well = wells[int(q)]
                for spec in specs:
                    family = spec["family"]
                    if family == "pca":
                        action = pca_oracle_profile(well, pca.mean_, pca.components_, spec["size"])
                    elif family == "kmeans_single":
                        action = best_profile(well, centers_by_size[spec["size"]])
                    elif family == "kmeans_pair":
                        action = best_profile(well, pair_by_size[spec["size"]])
                    elif family == "random_source":
                        candidates = random_by_key[(spec["seed"], spec["size"])]
                        action = best_profile(well, candidates)
                        # Candidate rows are drawn only from train_idx, which excludes q by construction.
                        if int(q) in set(map(int, train_idx)):
                            same_well_violations += 1
                    elif family == "smooth_gaussian":
                        action = best_profile(well, gaussian_by_key[(spec["seed"], spec["size"])] )
                    else:
                        raise AssertionError(family)
                    sums[spec["name"]][q] += action
                membership_rows.append({
                    "map": map_index + 1,
                    "fold": fold,
                    "well_id": well.well_id,
                    "train_wells": len(train_idx),
                    "test_wells": len(test_idx),
                    "same_well_excluded": int(q) not in set(map(int, train_idx)),
                })

        # Score each completed map independently for stability diagnostics.
        for spec in specs:
            name = spec["name"]
            map_metrics: list[Metric] = []
            for i, well in enumerate(wells):
                profile = sums[name][i]  # exactly one action so far for this map plus previous maps
                # Isolate the current map by tracking average delta from prior maps.
                # Recompute as cumulative-map score; final decision uses five-map average only.
                profile = profile / float(map_index + 1)
                prediction = well.e011 + interpolate_profile(profile, well.rows)
                map_metrics.append(error_metric(prediction - well.truth))
            summary = summarize(map_metrics)
            map_rows.append({"map": map_index + 1, "branch": name, "cumulative_rmse": summary["rmse"]})

    baseline_metrics = [error_metric(w.e011 - w.truth) for w in wells]
    baseline = summarize(baseline_metrics)
    spline_metrics, _ = spline7_reference(wells)
    spline_summary = summarize(spline_metrics)

    def pooled_decomposition_matches(value: dict[str, float]) -> bool:
        delta = abs(value["sse"] - value["datum_sse"] - value["trend_sse"] - value["shape_sse"])
        return delta <= max(1e-6, abs(value["sse"]) * 1e-12)

    pooled_decomposition_consistent = (
        pooled_decomposition_matches(baseline)
        and pooled_decomposition_matches(spline_summary)
    )
    row_order = np.argsort(np.asarray([well.rows for well in wells]), kind="mergesort")
    horizon_group = np.empty(n, dtype=int)
    for rank, index in enumerate(row_order):
        horizon_group[index] = min(4, int(rank * 5 / n))

    config_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    branch_metrics: dict[str, list[Metric]] = {}
    branch_profiles: dict[str, np.ndarray] = {}
    all_predictions_finite = True
    for spec in specs:
        name = spec["name"]
        averaged = np.clip(sums[name] / 5.0, -CAP, CAP)
        branch_profiles[name] = averaged
        metrics: list[Metric] = []
        positive = 0
        for i, well in enumerate(wells):
            prediction = well.e011 + interpolate_profile(averaged[i], well.rows)
            all_predictions_finite = all_predictions_finite and bool(np.isfinite(prediction).all())
            metric = error_metric(prediction - well.truth)
            metrics.append(metric)
            positive += int(metric.sse < baseline_metrics[i].sse - 1e-9)
        branch_metrics[name] = metrics
        summary = summarize(metrics)
        pooled_decomposition_consistent = (
            pooled_decomposition_consistent and pooled_decomposition_matches(summary)
        )
        legacy_group_rmses: list[float] = []
        horizon_group_rmses: list[float] = []
        for system, assignment, include_in_legacy_gate in [
            ("legacy_spatial", spatial, True),
            ("legacy_typewell", typewell, True),
            ("horizon_quintile", horizon_group, False),
        ]:
            for group in range(5):
                ids = np.flatnonzero(assignment == group)
                candidate = summarize(metrics[int(i)] for i in ids)
                base_group = summarize(baseline_metrics[int(i)] for i in ids)
                if include_in_legacy_gate:
                    legacy_group_rmses.append(candidate["rmse"])
                else:
                    horizon_group_rmses.append(candidate["rmse"])
                group_rows.append({
                    "branch": name,
                    "group_system": system,
                    "group": group,
                    "wells": len(ids),
                    "rmse": candidate["rmse"],
                    "gain_vs_e011": base_group["rmse"] - candidate["rmse"],
                })
        config_rows.append({
            **spec,
            "rmse": summary["rmse"],
            "gain_vs_e011": baseline["rmse"] - summary["rmse"],
            "shape_sse_reduction_fraction": (baseline["shape_sse"] - summary["shape_sse"]) / baseline["shape_sse"],
            "positive_well_fraction": positive / n,
            "maximum_legacy_group_rmse": max(legacy_group_rmses),
            "maximum_horizon_group_rmse": max(horizon_group_rmses),
        })

    compact_rows = [row for row in config_rows if bool(row["compact"])]
    best_compact = min(compact_rows, key=lambda row: (float(row["rmse"]), str(row["name"])))
    expanded_rows = [row for row in config_rows if bool(row["expanded"])]
    best_expanded = min(expanded_rows, key=lambda row: (float(row["rmse"]), str(row["name"])))
    gates = {
        "maximum_compact_oracle_rmse": float(best_compact["rmse"]) <= float(config["go_gate"]["maximum_compact_oracle_rmse"]),
        "minimum_shape_sse_reduction_fraction": float(best_compact["shape_sse_reduction_fraction"]) >= float(config["go_gate"]["minimum_shape_sse_reduction_fraction"]),
        "maximum_every_legacy_group_rmse": float(best_compact["maximum_legacy_group_rmse"]) <= float(config["go_gate"]["maximum_every_legacy_group_rmse"]),
        "minimum_positive_well_fraction": float(best_compact["positive_well_fraction"]) >= float(config["go_gate"]["minimum_positive_well_fraction"]),
        "same_well_exclusion": same_well_violations == 0,
    }
    if all(gates.values()):
        decision = "GO_COMPACT_ACTION_MANIFOLD"
    elif float(best_compact["rmse"]) <= 6.0:
        decision = "RESEARCH_ONLY_COMPACT_MANIFOLD"
    elif float(best_expanded["rmse"]) <= 6.0:
        decision = "RESEARCH_ONLY_NONCOMPACT_MANIFOLD"
    else:
        decision = "STOP_CLOSE_COMPACT_ACTION_MANIFOLD"

    best_name = str(best_compact["name"])
    best_metrics = branch_metrics[best_name]
    for i, well in enumerate(wells):
        well_rows.append({
            "well_id": well.well_id,
            "rows": well.rows,
            "legacy_spatial": int(spatial[i]),
            "legacy_typewell": int(typewell[i]),
            "horizon_quintile": int(horizon_group[i]),
            "baseline_rmse": baseline_metrics[i].rmse,
            "candidate_rmse": best_metrics[i].rmse,
            "baseline_sse": baseline_metrics[i].sse,
            "candidate_sse": best_metrics[i].sse,
        })

    duplicate_delta = 0.0
    sample_well = wells[0]
    sample_profiles = profiles[1:10]
    duplicated = np.concatenate([sample_profiles, sample_profiles[[0]]], axis=0)
    duplicate_delta = abs(float(np.min(profile_sse(sample_well, sample_profiles))) - float(np.min(profile_sse(sample_well, duplicated))))
    zero_delta = max(float(np.max(np.abs((well.e011 + interpolate_profile(np.zeros(GRID), well.rows)) - well.e011))) for well in wells)
    all_profiles_finite = all(np.isfinite(value).all() for value in branch_profiles.values())
    maximum_action = max(float(np.max(np.abs(value))) for value in branch_profiles.values())
    edge = {
        "status": "PASS",
        "checks": {
            "wells_773": n == 773,
            "contexts_25": len(membership_rows) == 25 * n,
            "five_map_actions_every_branch": all(np.isfinite(sums[name]).all() for name in names),
            "same_well_exclusion": same_well_violations == 0 and all(bool(row["same_well_excluded"]) for row in membership_rows),
            "duplicate_action_invariance": duplicate_delta <= 1e-9,
            "zero_action_exact_e011": zero_delta <= 1e-12,
            "finite_profiles": all_profiles_finite,
            "finite_reconstructions": all_predictions_finite,
            "action_cap": maximum_action <= CAP + 1e-9,
            "deterministic_dictionary_initialization": (
                deterministic_dictionary_delta <= 1e-12 and random_dictionary_deterministic
            ),
            "pooled_sse_decomposition_consistency": pooled_decomposition_consistent,
            "horizon_groups_complete": (
                set(map(int, np.unique(horizon_group))) == set(range(5))
                and len(group_rows) == len(specs) * 15
            ),
            "spline7_reference": abs(spline_summary["rmse"] - float(next(x["expected_rmse"] for x in config["branches"] if x["name"] == "spline7_oracle_reference"))) <= 1e-9,
            "branch_count": len(config_rows) == len(specs),
        },
    }
    edge["status"] = "PASS" if all(edge["checks"].values()) else "FAIL"

    implementation = {
        "schema_version": 1,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "python": sys.version,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "threads": 2,
        "profile_grid": GRID,
        "exact_scoring": "tridiagonal interpolation sufficient statistics reproduce actual-row SSE for every candidate profile",
        "aggregation": "five outer-map oracle action profiles averaged per well",
        "same_well_rule": "all empirical actions fitted or selected from outer-training wells only",
        "validation_completion": "five horizon groups emitted; deterministic KMeans/random/Gaussian initialization and pooled SSE decomposition asserted",
    }
    summary = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "COMPLETE",
        "decision": decision,
        "baseline": baseline,
        "spline7_oracle_reference": spline_summary,
        "best_compact": best_compact,
        "best_expanded": best_expanded,
        "gates": gates,
        "edge_status": edge["status"],
        "branches": len(config_rows),
        "contexts": 25,
        "runtime_seconds": time.time() - started,
    }

    write_csv(output_dir / "config_metrics.csv", config_rows)
    write_csv(output_dir / "group_metrics.csv", group_rows)
    write_csv(output_dir / "map_metrics.csv", map_rows)
    write_csv(output_dir / "membership_audit.csv", membership_rows)
    write_csv(output_dir / "selected_well_metrics.csv", well_rows)
    json_dump(output_dir / "implementation_receipt.json", implementation)
    json_dump(output_dir / "edge_cases.json", edge)
    json_dump(output_dir / "summary.json", summary)
    report = [
        "# T033 Result — Cross-fitted residual action-manifold capacity gate",
        "",
        f"Decision: **{decision}**",
        "",
        f"E011 baseline RMSE: {baseline['rmse']:.12f}",
        f"Existing spline7 oracle reference: {spline_summary['rmse']:.12f}",
        f"Best compact branch: `{best_compact['name']}`",
        f"- RMSE: {best_compact['rmse']:.12f}",
        f"- Shape-SSE reduction: {best_compact['shape_sse_reduction_fraction']:.6%}",
        f"- Positive wells: {best_compact['positive_well_fraction']:.6%}",
        f"- Maximum legacy group RMSE: {best_compact['maximum_legacy_group_rmse']:.6f}",
        f"- Maximum horizon-group RMSE: {best_compact['maximum_horizon_group_rmse']:.6f}",
        "",
        "## Gates",
    ]
    report.extend([f"- {'PASS' if value else 'FAIL'} — {key}" for key, value in gates.items()])
    report.extend(["", "This is a hidden-label capacity diagnostic only. It does not authorize a legal selector or competition prediction."])
    (output_dir / "RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    files = [
        "implementation_receipt.json", "summary.json", "edge_cases.json", "config_metrics.csv",
        "group_metrics.csv", "map_metrics.csv", "membership_audit.csv", "selected_well_metrics.csv", "RESULT.md",
    ]
    manifest = {"schema_version": 1, "files": [{"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)} for name in files]}
    json_dump(output_dir / "artifact_manifest.json", manifest)
    return summary


def synthetic_self_test() -> dict[str, Any]:
    rng = np.random.default_rng(33)
    residual = rng.normal(size=257)
    q, diag, off = interpolation_sufficient(residual)
    profile = rng.normal(size=GRID)
    well = WellData("synthetic", residual.size, residual, np.zeros_like(residual), np.zeros_like(residual), residual, np.zeros(GRID), q, diag, off, float(residual @ residual))
    direct = float(np.sum((residual - interpolate_profile(profile, residual.size)) ** 2))
    sufficient = float(profile_sse(well, profile))
    candidates = np.stack([profile, profile * 0.5, np.zeros(GRID)])
    duplicate = np.concatenate([candidates, candidates[[0]]])
    pair = pair_candidates(candidates[:2], [0.0, 0.5, 1.0])
    checks = {
        "sse_exact": abs(direct - sufficient) <= max(1e-9, direct * 1e-12),
        "best_finite": np.isfinite(best_profile(well, candidates)).all(),
        "duplicate_invariance": abs(float(np.min(profile_sse(well, candidates))) - float(np.min(profile_sse(well, duplicate)))) <= 1e-12,
        "pair_shape": pair.shape == (9, GRID),
        "metric_decomposition": abs(error_metric(residual).sse - (error_metric(residual).datum_sse + error_metric(residual).trend_sse + error_metric(residual).shape_sse)) <= 1e-8,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "sse_delta": abs(direct - sufficient)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_self_test:
        print(json.dumps(synthetic_self_test(), indent=2, sort_keys=True, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj))
        return
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = run(config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
