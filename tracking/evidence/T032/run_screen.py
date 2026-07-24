#!/usr/bin/env python3
"""T032 registered analog-tail retrieval oracle-coverage gate.

The implementation is intentionally an oracle-coverage diagnostic. Registration
and candidate construction use only legal query inputs and outer-training wells.
Hidden query labels are used only to choose the best already-generated neighbor
action and to score coverage. No legal selector or competition prediction is
created here.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEPS = ROOT / "scratch/_t032_deps"
if str(DEPS) not in sys.path:
    sys.path.append(str(DEPS))
from dtaidistance import dtw, dtw_ndim  # noqa: E402

CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = ROOT / "scratch/agents/t032-analog-gate-20260724/main"
OOF_PATH = ROOT / "artifacts/E011/oof_predictions.csv.gz"
COMPACT_PATH = ROOT / "artifacts/E011/compact_stats_v1.npz"
FEATURE_PATH = ROOT / "experiments/E003/results/legal_features.csv"

U_BINS = 64
GR_BINS = 96
ACTION_BINS = 256
ACTION_CAP_FT = 120.0
GEOMETRY_FEATURES = [
    "hidden_fraction",
    "hidden_z_delta",
    "horizontal_hidden_distance",
    "md_hidden_span",
    "x_hidden_delta",
    "y_hidden_delta",
    "hidden_z_slope",
    "hidden_rows",
]


@dataclass(frozen=True)
class Context:
    map_index: int
    fold: int
    key: str
    library: np.ndarray
    query: np.ndarray


@dataclass(frozen=True)
class WellMetric:
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


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
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


def resample(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("resample requires a nonempty vector")
    if values.size == 1:
        return np.full(bins, float(values[0]), dtype=np.float64)
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, bins)
    return np.interp(target, source, values).astype(np.float64)


def robust_normalize(values: np.ndarray, finite_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values) if finite_mask is None else np.asarray(finite_mask, dtype=bool) & np.isfinite(values)
    if not mask.any():
        return np.zeros(values.size, dtype=np.float64), np.zeros(values.size, dtype=np.float64)
    finite_index = np.flatnonzero(mask)
    clean = values[mask]
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    scale = max(1e-6, 1.4826 * mad, float(np.std(clean)) * 0.1)
    normalized = np.empty(values.size, dtype=np.float64)
    normalized[mask] = np.clip((values[mask] - median) / scale, -8.0, 8.0)
    missing = np.flatnonzero(~mask)
    if finite_index.size == 1:
        normalized[missing] = normalized[finite_index[0]]
    elif missing.size:
        normalized[missing] = np.interp(missing, finite_index, normalized[finite_index])
    return normalized, mask.astype(np.float64)


def gr_profile(values: np.ndarray, bins: int) -> np.ndarray:
    normalized, mask = robust_normalize(values)
    return np.column_stack([resample(normalized, bins), np.clip(resample(mask, bins), 0.0, 1.0)])


def visible_u_profile(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    visible = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
    finite = np.isfinite(visible)
    if not finite.any() or finite.all():
        raise ValueError("TVT_input must contain a visible prefix and hidden suffix")
    known = int(np.flatnonzero(~finite)[0])
    if not finite[:known].all() or finite[known:].any():
        raise ValueError("TVT_input mask is not a contiguous prefix")
    z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(float)
    u = visible[:known] + z[:known]
    normalized, _ = robust_normalize(u)
    return resample(normalized, U_BINS), known


def load_fold_assignments(well_ids: np.ndarray) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for index in range(1, 6):
        data = json.loads((ROOT / f"folds/v{index}.json").read_text(encoding="utf-8"))
        assignments = data["assignments"]
        array = np.asarray([int(assignments[str(well_id)]) for well_id in well_ids], dtype=np.int8)
        if set(map(int, np.unique(array))) != set(range(5)):
            raise ValueError(f"fold map v{index} is incomplete")
        output.append(array)
    return output


def contexts_from_folds(folds: list[np.ndarray]) -> list[Context]:
    contexts: list[Context] = []
    for map_index, assignment in enumerate(folds):
        for fold in range(5):
            query = np.flatnonzero(assignment == fold)
            library = np.flatnonzero(assignment != fold)
            if query.size == 0 or library.size == 0 or np.intersect1d(query, library).size:
                raise ValueError("invalid outer context")
            contexts.append(Context(map_index, fold, f"v{map_index + 1}_f{fold}", library, query))
    return contexts


def build_profiles(config: dict[str, Any], output_dir: Path, force: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "profiles_v1.npz"
    manifest_path = output_dir / "profile_manifest.json"
    if cache.exists() and manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    compact = np.load(COMPACT_PATH, allow_pickle=False)
    well_ids = compact["well_ids"].astype(str)
    if len(well_ids) != int(config.get("expected_wells", 773)) and len(well_ids) != 773:
        raise ValueError("unexpected well count")
    if len(set(well_ids.tolist())) != len(well_ids):
        raise ValueError("duplicate well IDs")

    legal = pd.read_csv(FEATURE_PATH).set_index("well_id").loc[well_ids]
    geometry = legal[GEOMETRY_FEATURES].to_numpy(float)
    if not np.isfinite(geometry).all():
        raise ValueError("geometry features are not finite")

    u_profiles = np.empty((len(well_ids), U_BINS), dtype=np.float64)
    h_profiles = np.empty((len(well_ids), GR_BINS, 2), dtype=np.float64)
    t_profiles = np.empty((len(well_ids), GR_BINS, 2), dtype=np.float64)
    known_rows = np.empty(len(well_ids), dtype=np.int32)
    total_rows = np.empty(len(well_ids), dtype=np.int32)

    started = time.time()
    for i, well_id in enumerate(well_ids):
        horizontal = pd.read_csv(
            ROOT / "data/train" / f"{well_id}__horizontal_well.csv",
            usecols=["Z", "GR", "TVT_input"],
        )
        typewell = pd.read_csv(
            ROOT / "data/train" / f"{well_id}__typewell.csv",
            usecols=["GR"],
        )
        u_profiles[i], known_rows[i] = visible_u_profile(horizontal)
        h_profiles[i] = gr_profile(pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float), GR_BINS)
        t_profiles[i] = gr_profile(pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(float), GR_BINS)
        total_rows[i] = len(horizontal)

    radius_u = max(1, int(math.ceil(float(config["registration"]["dtw_radius_fraction"]) * U_BINS)))
    radius_gr = max(1, int(math.ceil(float(config["registration"]["dtw_radius_fraction"]) * GR_BINS)))
    u_distance = np.asarray(
        dtw.distance_matrix_fast(
            [np.ascontiguousarray(row, dtype=np.double) for row in u_profiles],
            window=radius_u,
            parallel=True,
            compact=False,
            only_triu=False,
            use_c=True,
        ),
        dtype=np.float64,
    )
    horizontal_distance = np.asarray(
        dtw_ndim.distance_matrix_fast(
            [np.ascontiguousarray(row, dtype=np.double) for row in h_profiles],
            ndim=2,
            window=radius_gr,
            parallel=True,
            compact=False,
            only_triu=False,
            use_c=True,
        ),
        dtype=np.float64,
    )
    typewell_distance = np.asarray(
        dtw_ndim.distance_matrix_fast(
            [np.ascontiguousarray(row, dtype=np.double) for row in t_profiles],
            ndim=2,
            window=radius_gr,
            parallel=True,
            compact=False,
            only_triu=False,
            use_c=True,
        ),
        dtype=np.float64,
    )
    for name, matrix in [
        ("u", u_distance),
        ("horizontal", horizontal_distance),
        ("typewell", typewell_distance),
    ]:
        if matrix.shape != (len(well_ids), len(well_ids)) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} distance matrix is invalid")
        if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
            raise ValueError(f"{name} distance matrix is asymmetric")
        if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0):
            raise ValueError(f"{name} distance diagonal is nonzero")

    np.savez_compressed(
        cache,
        well_ids=well_ids,
        u_profiles=u_profiles.astype(np.float32),
        horizontal_profiles=h_profiles.astype(np.float32),
        typewell_profiles=t_profiles.astype(np.float32),
        geometry=geometry.astype(np.float64),
        known_rows=known_rows,
        total_rows=total_rows,
        u_distance=u_distance,
        horizontal_distance=horizontal_distance,
        typewell_distance=typewell_distance,
    )
    elapsed = time.time() - started
    manifest = {
        "schema_version": 1,
        "status": "BUILD_ONLY_PASS",
        "well_count": int(len(well_ids)),
        "u_shape": list(u_profiles.shape),
        "horizontal_shape": list(h_profiles.shape),
        "typewell_shape": list(t_profiles.shape),
        "geometry_shape": list(geometry.shape),
        "distance_shape": list(u_distance.shape),
        "dtw_radius_u": radius_u,
        "dtw_radius_gr": radius_gr,
        "cache_sha256": sha256_file(cache),
        "well_ids_sha256": sha256_array(well_ids.astype("S8")),
        "u_distance_sha256": sha256_array(u_distance.astype("<f8")),
        "horizontal_distance_sha256": sha256_array(horizontal_distance.astype("<f8")),
        "typewell_distance_sha256": sha256_array(typewell_distance.astype("<f8")),
        "runtime_seconds": elapsed,
        "input_files": {
            str(COMPACT_PATH.relative_to(ROOT)): sha256_file(COMPACT_PATH),
            str(FEATURE_PATH.relative_to(ROOT)): sha256_file(FEATURE_PATH),
        },
    }
    dump_json(manifest_path, manifest)
    return manifest


def positive_scale(matrix: np.ndarray, library: np.ndarray) -> float:
    block = matrix[np.ix_(library, library)]
    values = block[np.triu_indices(len(library), 1)]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0
    return max(1e-9, float(np.median(values)))


def geometry_distances(geometry: np.ndarray, library: np.ndarray, query: np.ndarray) -> np.ndarray:
    train = geometry[library]
    median = np.median(train, axis=0)
    q25, q75 = np.quantile(train, [0.25, 0.75], axis=0)
    scale = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
    train_z = np.clip((train - median) / scale, -8.0, 8.0)
    query_z = np.clip((geometry[query] - median) / scale, -8.0, 8.0)
    diff = query_z[:, None, :] - train_z[None, :, :]
    return np.sqrt(np.mean(diff * diff, axis=2))


def metric_for(error: np.ndarray) -> WellMetric:
    error = np.asarray(error, dtype=np.float64)
    rows = int(error.size)
    x = np.arange(rows, dtype=np.float64)
    sum_error = float(error.sum())
    sum_error_sq = float(np.dot(error, error))
    sum_x = float(x.sum())
    sum_x_sq = float(np.dot(x, x))
    sum_x_error = float(np.dot(x, error))
    n = float(rows)
    centered_x_sq = sum_x_sq - sum_x * sum_x / n
    centered_x_error = sum_x_error - sum_x * sum_error / n
    slope = centered_x_error / centered_x_sq if centered_x_sq > 0.0 else 0.0
    datum_sse = sum_error * sum_error / n
    trend_sse = slope * slope * centered_x_sq
    shape_sse = sum_error_sq - datum_sse - trend_sse
    tolerance = max(1e-9, sum_error_sq * 1e-12)
    if shape_sse < 0.0 and abs(shape_sse) <= tolerance:
        shape_sse = 0.0
    if shape_sse < 0.0:
        raise ArithmeticError("negative shape SSE")
    return WellMetric(rows, sum_error_sq, datum_sse, trend_sse, shape_sse, math.sqrt(sum_error_sq / n))


def summarize(metrics: Iterable[WellMetric]) -> dict[str, float | int]:
    items = list(metrics)
    rows = sum(item.rows for item in items)
    sse = sum(item.sse for item in items)
    datum = sum(item.datum_sse for item in items)
    trend = sum(item.trend_sse for item in items)
    shape = sum(item.shape_sse for item in items)
    return {
        "rows": int(rows),
        "wells": int(len(items)),
        "sse": float(sse),
        "rmse": float(math.sqrt(sse / rows)),
        "datum_sse": float(datum),
        "trend_sse": float(trend),
        "shape_sse": float(shape),
    }


def config_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in config["grid"]["neighbors"]:
        for gw in config["grid"]["geometry_weights"]:
            for tw in config["grid"]["typewell_weights"]:
                for beta in config["grid"]["action_beta"]:
                    name = f"k{k:02d}_g{gw:.2f}_t{tw:.2f}_b{beta:.2f}".replace(".", "p")
                    rows.append({
                        "name": name,
                        "neighbors": int(k),
                        "geometry_weight": float(gw),
                        "typewell_weight": float(tw),
                        "beta": float(beta),
                    })
    if len(rows) != 81 or len({row["name"] for row in rows}) != 81:
        raise ValueError("configuration grid is not exactly 81 unique points")
    return rows


def source_profile(residual: np.ndarray) -> np.ndarray:
    profile = resample(np.asarray(residual, dtype=np.float64), ACTION_BINS)
    return np.clip(profile, -ACTION_CAP_FT, ACTION_CAP_FT).astype(np.float32)


def action_for(profile: np.ndarray, rows: int) -> np.ndarray:
    return np.interp(
        np.linspace(0.0, 1.0, rows),
        np.linspace(0.0, 1.0, ACTION_BINS),
        np.asarray(profile, dtype=np.float64),
    )


def deterministic_order(distances: np.ndarray, library: np.ndarray, well_ids: np.ndarray) -> np.ndarray:
    ids = well_ids[library]
    local = np.lexsort((ids, np.asarray(distances, dtype=np.float64)))
    return library[local]


def derangement(indices: np.ndarray, seed: int) -> dict[int, int]:
    values = np.asarray(indices, dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        perm = rng.permutation(values)
        if np.all(perm != values):
            return {int(a): int(b) for a, b in zip(values, perm)}
    perm = np.roll(values, 1)
    return {int(a): int(b) for a, b in zip(values, perm)}


def build_cross_reversed(profiles: np.ndarray, window: int) -> np.ndarray:
    n = len(profiles)
    reversed_profiles = [np.ascontiguousarray(row[::-1], dtype=np.double) for row in profiles]
    originals = [np.ascontiguousarray(row, dtype=np.double) for row in profiles]
    combined = reversed_profiles + originals
    matrix = np.asarray(
        dtw_ndim.distance_matrix_fast(
            combined,
            ndim=int(profiles.shape[2]),
            window=window,
            block=((0, n), (n, 2 * n)),
            parallel=True,
            compact=False,
            only_triu=False,
            use_c=True,
        ),
        dtype=np.float64,
    )
    cross = matrix[:n, n:]
    if cross.shape != (n, n) or not np.isfinite(cross).all():
        raise ValueError("reversed cross-distance matrix is invalid")
    return cross


def run(config: dict[str, Any], output_dir: Path, build_only: bool, force_rebuild: bool) -> dict[str, Any]:
    started = time.time()
    manifest = build_profiles(config, output_dir, force_rebuild)
    if build_only:
        return {"status": "BUILD_ONLY_PASS", "profile_manifest": manifest}

    cache = np.load(output_dir / "profiles_v1.npz", allow_pickle=False)
    well_ids = cache["well_ids"].astype(str)
    n_wells = len(well_ids)
    geometry = cache["geometry"].astype(float)
    u_distance = cache["u_distance"].astype(float)
    horizontal_distance = cache["horizontal_distance"].astype(float)
    typewell_distance = cache["typewell_distance"].astype(float)
    horizontal_profiles = cache["horizontal_profiles"].astype(float)
    typewell_profiles = cache["typewell_profiles"].astype(float)
    folds = load_fold_assignments(well_ids)
    contexts = contexts_from_folds(folds)

    compact = np.load(COMPACT_PATH, allow_pickle=False)
    compact_ids = compact["well_ids"].astype(str)
    if not np.array_equal(compact_ids, well_ids):
        raise ValueError("compact well order differs from profile order")
    legacy_spatial = compact["spatial_assignment"].astype(int)
    legacy_typewell = compact["typewell_assignment"].astype(int)
    hidden_rows_expected = compact["rows"].astype(int)

    frame = pd.read_csv(
        OOF_PATH,
        usecols=["well_id", "hidden_index", "target", "spline4_ridge_equal_s075"],
    ).sort_values(["well_id", "hidden_index"])
    by_well = {well_id: group for well_id, group in frame.groupby("well_id", sort=False)}
    truths: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    action_profiles = np.empty((n_wells, ACTION_BINS), dtype=np.float32)
    for i, well_id in enumerate(well_ids):
        group = by_well[str(well_id)]
        hidden_index = group["hidden_index"].to_numpy(int)
        if not np.array_equal(hidden_index, np.arange(len(group))) or len(group) != int(hidden_rows_expected[i]):
            raise ValueError(f"{well_id}: OOF hidden-index contract failed")
        truth = group["target"].to_numpy(float)
        base = group["spline4_ridge_equal_s075"].to_numpy(float)
        if not np.isfinite(truth).all() or not np.isfinite(base).all():
            raise ValueError(f"{well_id}: non-finite OOF rows")
        truths.append(truth)
        bases.append(base)
        errors.append(base - truth)
        action_profiles[i] = source_profile(truth - base)

    baseline_metrics = [metric_for(error) for error in errors]
    baseline_summary = summarize(baseline_metrics)
    configs = config_grid(config)
    config_index = {row["name"]: i for i, row in enumerate(configs)}
    selected = np.full((len(configs), 5, n_wells), -1, dtype=np.int32)
    distance_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    stats_cache: dict[tuple[int, int], tuple[float, float]] = {}

    def candidate_stats(query_index: int, source_index: int) -> tuple[float, float]:
        key = (query_index, source_index)
        cached = stats_cache.get(key)
        if cached is not None:
            return cached
        action = action_for(action_profiles[source_index], len(errors[query_index]))
        cross = float(np.dot(errors[query_index], action))
        norm = float(np.dot(action, action))
        stats_cache[key] = (cross, norm)
        return cross, norm

    weight_pairs = sorted({(row["geometry_weight"], row["typewell_weight"]) for row in configs})
    for context in contexts:
        library = context.library
        query = context.query
        u_scale = positive_scale(u_distance, library)
        h_scale = positive_scale(horizontal_distance, library)
        t_scale = positive_scale(typewell_distance, library)
        geom = geometry_distances(geometry, library, query)
        distance_rows.append({
            "context": context.key,
            "library_wells": len(library),
            "query_wells": len(query),
            "u_scale": u_scale,
            "horizontal_scale": h_scale,
            "typewell_scale": t_scale,
        })
        rankings: dict[tuple[float, float, int], np.ndarray] = {}
        for local_q, q in enumerate(query):
            base_components = u_distance[q, library] / u_scale + horizontal_distance[q, library] / h_scale
            for gw, tw in weight_pairs:
                composite = base_components + tw * typewell_distance[q, library] / t_scale + gw * geom[local_q]
                order = deterministic_order(composite, library, well_ids)
                if q in order:
                    raise AssertionError("same-well retrieval escaped outer library")
                rankings[(gw, tw, int(q))] = order[:9]
        for q in query:
            base_sse = baseline_metrics[int(q)].sse
            for gw, tw in weight_pairs:
                top = rankings[(gw, tw, int(q))]
                cross = np.empty(len(top), dtype=float)
                norm = np.empty(len(top), dtype=float)
                for j, source in enumerate(top):
                    cross[j], norm[j] = candidate_stats(int(q), int(source))
                for row in configs:
                    if row["geometry_weight"] != gw or row["typewell_weight"] != tw:
                        continue
                    k = row["neighbors"]
                    beta = row["beta"]
                    sses = base_sse + 2.0 * beta * cross[:k] + beta * beta * norm[:k]
                    local_best = int(np.argmin(sses))
                    if float(sses[local_best]) < base_sse - 1e-9:
                        source = int(top[local_best])
                        selected[config_index[row["name"]], context.map_index, int(q)] = source
                        improved = True
                        selected_sse = float(sses[local_best])
                    else:
                        source = -1
                        improved = False
                        selected_sse = base_sse
                    selection_rows.append({
                        "context": context.key,
                        "config": row["name"],
                        "query_well": str(well_ids[q]),
                        "selected_source": "" if source < 0 else str(well_ids[source]),
                        "improved": improved,
                        "baseline_sse": base_sse,
                        "selected_sse": selected_sse,
                    })

    config_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    for row in configs:
        ci = config_index[row["name"]]
        beta = row["beta"]
        metrics: list[WellMetric] = []
        positive = 0
        selected_counter: Counter[int] = Counter()
        for q in range(n_wells):
            profile = np.zeros(ACTION_BINS, dtype=np.float64)
            for map_index in range(5):
                source = int(selected[ci, map_index, q])
                if source >= 0:
                    profile += action_profiles[source]
                    selected_counter[source] += 1
            profile /= 5.0
            correction = beta * action_for(profile, len(bases[q]))
            prediction = bases[q] + correction
            metric = metric_for(prediction - truths[q])
            positive += int(metric.sse < baseline_metrics[q].sse - 1e-9)
            metrics.append(metric)
        summary = summarize(metrics)
        shape_reduction = (baseline_summary["shape_sse"] - summary["shape_sse"]) / baseline_summary["shape_sse"]
        gains_spatial: list[float] = []
        gains_typewell: list[float] = []
        for assignment, output in [(legacy_spatial, gains_spatial), (legacy_typewell, gains_typewell)]:
            for group in range(5):
                ids = np.flatnonzero(assignment == group)
                base_group = summarize(baseline_metrics[i] for i in ids)
                candidate_group = summarize(metrics[i] for i in ids)
                output.append(float(base_group["rmse"] - candidate_group["rmse"]))
        total_selected = sum(selected_counter.values())
        entropy = 0.0
        if total_selected:
            for count in selected_counter.values():
                p = count / total_selected
                entropy -= p * math.log(max(p, 1e-300))
        config_rows.append({
            **row,
            "rmse": summary["rmse"],
            "gain_vs_e011": baseline_summary["rmse"] - summary["rmse"],
            "shape_sse_reduction_fraction": shape_reduction,
            "positive_well_fraction": positive / n_wells,
            "minimum_spatial_gain": min(gains_spatial),
            "minimum_typewell_gain": min(gains_typewell),
            "unique_selected_sources": len(selected_counter),
            "selected_source_entropy": entropy,
        })

    best = min(config_rows, key=lambda item: (float(item["rmse"]), str(item["name"])))
    best_name = str(best["name"])
    best_config = configs[config_index[best_name]]
    best_ci = config_index[best_name]
    best_beta = float(best_config["beta"])
    best_predictions: list[np.ndarray] = []
    best_metrics: list[WellMetric] = []
    for q in range(n_wells):
        profile = np.zeros(ACTION_BINS, dtype=np.float64)
        for map_index in range(5):
            source = int(selected[best_ci, map_index, q])
            if source >= 0:
                profile += action_profiles[source]
        profile /= 5.0
        prediction = bases[q] + best_beta * action_for(profile, len(bases[q]))
        best_predictions.append(prediction)
        best_metrics.append(metric_for(prediction - truths[q]))

    group_rows: list[dict[str, Any]] = []
    for system, assignment in [("legacy_spatial", legacy_spatial), ("legacy_typewell", legacy_typewell)]:
        for group in range(5):
            ids = np.flatnonzero(assignment == group)
            base_group = summarize(baseline_metrics[i] for i in ids)
            candidate_group = summarize(best_metrics[i] for i in ids)
            group_rows.append({
                "system": system,
                "group": group,
                "wells": len(ids),
                "baseline_rmse": base_group["rmse"],
                "candidate_rmse": candidate_group["rmse"],
                "gain": base_group["rmse"] - candidate_group["rmse"],
            })
    row_order = np.argsort(hidden_rows_expected, kind="mergesort")
    horizon_group = np.empty(n_wells, dtype=int)
    for rank, idx in enumerate(row_order):
        horizon_group[idx] = min(4, int(rank * 5 / n_wells))
    for group in range(5):
        ids = np.flatnonzero(horizon_group == group)
        base_group = summarize(baseline_metrics[i] for i in ids)
        candidate_group = summarize(best_metrics[i] for i in ids)
        group_rows.append({
            "system": "horizon_quintile",
            "group": group,
            "wells": len(ids),
            "baseline_rmse": base_group["rmse"],
            "candidate_rmse": candidate_group["rmse"],
            "gain": base_group["rmse"] - candidate_group["rmse"],
        })

    map_rows: list[dict[str, Any]] = []
    beta = best_beta
    for map_index in range(5):
        map_metrics: list[WellMetric] = []
        for q in range(n_wells):
            source = int(selected[best_ci, map_index, q])
            correction = np.zeros(len(bases[q]), dtype=float) if source < 0 else beta * action_for(action_profiles[source], len(bases[q]))
            map_metrics.append(metric_for(bases[q] + correction - truths[q]))
        summary = summarize(map_metrics)
        map_rows.append({
            "map": map_index + 1,
            "baseline_rmse": baseline_summary["rmse"],
            "candidate_rmse": summary["rmse"],
            "gain": baseline_summary["rmse"] - summary["rmse"],
        })

    # Destructive controls repeat the entire 81-point oracle screen. This prevents
    # a control from appearing to fail merely because its optimum moved away from
    # the main best configuration. Only the named relation is corrupted; all
    # thresholds, grids, fold exclusions, and outer-library scaling remain frozen.
    radius_gr = int(manifest["dtw_radius_gr"])
    reversed_horizontal = build_cross_reversed(horizontal_profiles, radius_gr)
    reversed_typewell = build_cross_reversed(typewell_profiles, radius_gr)
    control_config_rows: list[dict[str, Any]] = []

    def control_screen(name: str, h_matrix: np.ndarray, t_matrix: np.ndarray, shuffled: bool) -> dict[str, Any]:
        control_selected = np.full((len(configs), 5, n_wells), -1, dtype=np.int32)
        for context in contexts:
            library = context.library
            query = context.query
            u_scale = positive_scale(u_distance, library)
            h_scale = positive_scale(horizontal_distance, library)
            t_scale = positive_scale(typewell_distance, library)
            geom = geometry_distances(geometry, library, query)
            mapping = derangement(library, 32000 + context.map_index * 10 + context.fold) if shuffled else None
            rankings: dict[tuple[float, float, int], np.ndarray] = {}
            for local_q, q in enumerate(query):
                base_components = u_distance[q, library] / u_scale + h_matrix[q, library] / h_scale
                for gw, tw in weight_pairs:
                    composite = base_components + tw * t_matrix[q, library] / t_scale + gw * geom[local_q]
                    rankings[(gw, tw, int(q))] = deterministic_order(composite, library, well_ids)[:9]
            for q in query:
                base_sse = baseline_metrics[int(q)].sse
                for gw, tw in weight_pairs:
                    top = rankings[(gw, tw, int(q))]
                    action_sources = np.asarray(
                        [int(mapping[int(source)]) if mapping is not None else int(source) for source in top],
                        dtype=np.int32,
                    )
                    if np.any(action_sources == int(q)):
                        raise AssertionError("control action mapping escaped outer library")
                    cross = np.empty(len(action_sources), dtype=float)
                    norm = np.empty(len(action_sources), dtype=float)
                    for j, source in enumerate(action_sources):
                        cross[j], norm[j] = candidate_stats(int(q), int(source))
                    for row in configs:
                        if row["geometry_weight"] != gw or row["typewell_weight"] != tw:
                            continue
                        k = int(row["neighbors"])
                        beta_value = float(row["beta"])
                        sses = base_sse + 2.0 * beta_value * cross[:k] + beta_value * beta_value * norm[:k]
                        local_best = int(np.argmin(sses))
                        if float(sses[local_best]) < base_sse - 1e-9:
                            control_selected[config_index[row["name"]], context.map_index, int(q)] = int(action_sources[local_best])

        rows: list[dict[str, Any]] = []
        for row in configs:
            ci = config_index[row["name"]]
            beta_value = float(row["beta"])
            metrics: list[WellMetric] = []
            positive = 0
            for q in range(n_wells):
                profile = np.zeros(ACTION_BINS, dtype=float)
                for map_index in range(5):
                    source = int(control_selected[ci, map_index, q])
                    if source >= 0:
                        profile += action_profiles[source]
                profile /= 5.0
                correction = beta_value * action_for(profile, len(bases[q]))
                metric = metric_for(bases[q] + correction - truths[q])
                positive += int(metric.sse < baseline_metrics[q].sse - 1e-9)
                metrics.append(metric)
            summary_value = summarize(metrics)
            spatial_gains: list[float] = []
            typewell_gains: list[float] = []
            for assignment, output in [(legacy_spatial, spatial_gains), (legacy_typewell, typewell_gains)]:
                for group in range(5):
                    ids = np.flatnonzero(assignment == group)
                    base_group = summarize(baseline_metrics[i] for i in ids)
                    candidate_group = summarize(metrics[i] for i in ids)
                    output.append(float(base_group["rmse"] - candidate_group["rmse"]))
            result = {
                "control": name,
                **row,
                "rmse": summary_value["rmse"],
                "gain_vs_e011": baseline_summary["rmse"] - summary_value["rmse"],
                "shape_sse_reduction_fraction": (
                    baseline_summary["shape_sse"] - summary_value["shape_sse"]
                ) / baseline_summary["shape_sse"],
                "positive_well_fraction": positive / n_wells,
                "minimum_spatial_gain": min(spatial_gains),
                "minimum_typewell_gain": min(typewell_gains),
            }
            result["passes_main_coverage_gates"] = bool(
                float(result["gain_vs_e011"]) >= float(config["go_gate"]["minimum_oracle_rmse_gain"])
                and float(result["shape_sse_reduction_fraction"]) >= float(config["go_gate"]["minimum_shape_sse_reduction_fraction"])
                and float(result["positive_well_fraction"]) >= float(config["go_gate"]["minimum_positive_well_fraction"])
                and float(result["minimum_spatial_gain"]) >= -1e-12
                and float(result["minimum_typewell_gain"]) >= -1e-12
            )
            rows.append(result)
        control_config_rows.extend(rows)
        return min(rows, key=lambda item: (float(item["rmse"]), str(item["name"])))

    controls = [
        control_screen("shuffled_library_well_ids", horizontal_distance, typewell_distance, True),
        control_screen("reversed_horizontal_gr", reversed_horizontal, typewell_distance, False),
        control_screen("reversed_typewell_gr", horizontal_distance, reversed_typewell, False),
    ]

    gates = {
        "minimum_oracle_rmse_gain": float(best["gain_vs_e011"]) >= float(config["go_gate"]["minimum_oracle_rmse_gain"]),
        "minimum_shape_sse_reduction_fraction": float(best["shape_sse_reduction_fraction"]) >= float(config["go_gate"]["minimum_shape_sse_reduction_fraction"]),
        "minimum_positive_well_fraction": float(best["positive_well_fraction"]) >= float(config["go_gate"]["minimum_positive_well_fraction"]),
        "nonnegative_every_legacy_spatial_group": float(best["minimum_spatial_gain"]) >= -1e-12,
        "nonnegative_every_legacy_typewell_group": float(best["minimum_typewell_gain"]) >= -1e-12,
        "all_destructive_controls_fail_main_coverage": all(
            not bool(control["passes_main_coverage_gates"]) for control in controls
        ),
    }
    decision = "ADVANCE_ANALOG_SELECTOR_PREREGISTRATION" if all(gates.values()) else "CLOSE_ANALOG_TAIL_RETRIEVAL"

    edge_checks = {
        "well_count_773": n_wells == 773,
        "contexts_25": len(contexts) == 25,
        "grid_81": len(configs) == 81,
        "all_profile_distances_finite": bool(np.isfinite(u_distance).all() and np.isfinite(horizontal_distance).all() and np.isfinite(typewell_distance).all()),
        "same_well_excluded": all(q not in context.library for context in contexts for q in context.query),
        "well_ids_unique": len(set(well_ids.tolist())) == n_wells,
        "selected_source_bounds": bool(((selected >= -1) & (selected < n_wells)).all()),
        "zero_fallback_exact": bool(np.array_equal(bases[0] + np.zeros_like(bases[0]), bases[0])),
        "all_best_predictions_finite": all(np.isfinite(value).all() for value in best_predictions),
        "legacy_groups_complete": set(map(int, np.unique(legacy_spatial))) == set(range(5)) and set(map(int, np.unique(legacy_typewell))) == set(range(5)),
        "tie_order_deterministic": bool(np.array_equal(
            deterministic_order(np.zeros(len(contexts[0].library)), contexts[0].library, well_ids),
            deterministic_order(np.zeros(len(contexts[0].library)), contexts[0].library, well_ids),
        )),
        "action_bound": float(np.max(np.abs(action_profiles))) <= ACTION_CAP_FT + 1e-12,
        "streaming_only_winner_predictions_materialized": len(best_predictions) == n_wells,
        "destructive_controls_report_complete": all(
            set(control) >= {
                "control", "name", "rmse", "gain_vs_e011", "shape_sse_reduction_fraction",
                "positive_well_fraction", "minimum_spatial_gain", "minimum_typewell_gain",
                "passes_main_coverage_gates",
            }
            for control in controls
        ),
        "control_grid_rows_243": len(control_config_rows) == 3 * len(configs),
    }
    edge_status = "PASS" if all(edge_checks.values()) else "FAIL"

    selected_well_rows: list[dict[str, Any]] = []
    for q, well_id in enumerate(well_ids):
        base_metric = baseline_metrics[q]
        candidate_metric = best_metrics[q]
        selected_well_rows.append({
            "well_id": str(well_id),
            "rows": base_metric.rows,
            "baseline_rmse": base_metric.rmse,
            "candidate_rmse": candidate_metric.rmse,
            "gain": base_metric.rmse - candidate_metric.rmse,
            "baseline_sse": base_metric.sse,
            "candidate_sse": candidate_metric.sse,
            "selected_maps": int(np.sum(selected[best_ci, :, q] >= 0)),
            "legacy_spatial": int(legacy_spatial[q]),
            "legacy_typewell": int(legacy_typewell[q]),
            "horizon_quintile": int(horizon_group[q]),
        })

    implementation = {
        "schema_version": 1,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "dtaidistance": __import__("dtaidistance").__version__,
        "threads": 2,
        "registration": {
            "visible_u_bins": U_BINS,
            "horizontal_gr_bins": GR_BINS,
            "typewell_gr_bins": GR_BINS,
            "gr_channels": ["robust_z", "finite_mask"],
            "geometry_features": GEOMETRY_FEATURES,
            "component_scaling": "outer-library median positive pairwise distance; geometry outer-library median/IQR",
            "composite": "U_DTW + horizontal_GR_DTW + typewell_weight*typewell_GR_DTW + geometry_weight*geometry_distance",
        },
        "candidate_action": {
            "source": "source truth minus exact E011 OOF prediction",
            "nodes": ACTION_BINS,
            "transfer": "linear interpolation on normalized hidden fraction",
            "absolute_cap_ft": ACTION_CAP_FT,
            "placement": "registered beta",
            "fallback": "exact E011 zero action",
        },
        "oracle": "best top-k neighbor action by held-out hidden SSE inside each outer context, then mean of five map actions per well",
        "streaming": "all 81 configurations are scored and discarded sequentially; only the winning configuration predictions are reconstructed",
        "control_gate": "each shuffled/reversed control repeats the full 81-point screen and its own best configuration must fail at least one main coverage gate",
    }

    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "decision": decision,
        "baseline": baseline_summary,
        "best_config": best,
        "gates": gates,
        "controls": controls,
        "edge_status": edge_status,
        "runtime_seconds": time.time() - started,
        "wells": n_wells,
        "contexts": len(contexts),
        "grid_points": len(configs),
    }

    write_csv(output_dir / "config_metrics.csv", config_rows)
    write_csv(output_dir / "distance_contexts.csv", distance_rows)
    write_csv(output_dir / "selection_rows.csv", selection_rows)
    write_csv(output_dir / "selected_well_metrics.csv", selected_well_rows)
    write_csv(output_dir / "group_metrics.csv", group_rows)
    write_csv(output_dir / "map_metrics.csv", map_rows)
    write_csv(output_dir / "control_metrics.csv", controls)
    write_csv(output_dir / "control_config_metrics.csv", control_config_rows)
    dump_json(output_dir / "implementation_receipt.json", implementation)
    dump_json(output_dir / "edge_cases.json", {"schema_version": 1, "status": edge_status, "checks": edge_checks})
    dump_json(output_dir / "summary.json", summary)

    report = [
        "# T032 Result — Registered analog-tail retrieval oracle-coverage gate",
        "",
        f"Decision: **{decision}**",
        "",
        f"Best configuration: `{best_name}`",
        f"- Oracle RMSE: {best['rmse']:.6f}",
        f"- Gain versus E011: {best['gain_vs_e011']:.6f}",
        f"- Shape-SSE reduction: {best['shape_sse_reduction_fraction']:.6%}",
        f"- Positive-well fraction: {best['positive_well_fraction']:.6%}",
        f"- Minimum spatial-group gain: {best['minimum_spatial_gain']:.6f}",
        f"- Minimum typewell-group gain: {best['minimum_typewell_gain']:.6f}",
        "",
        "## Gates",
        "",
    ]
    report.extend(f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in gates.items())
    report.extend([
        "",
        "This is hidden-label oracle coverage only. It does not authorize a selector, package, Kaggle run, or competition submission unless every gate passes.",
    ])
    (output_dir / "RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    artifact_names = [
        "profile_manifest.json", "implementation_receipt.json", "summary.json", "edge_cases.json",
        "config_metrics.csv", "distance_contexts.csv", "selection_rows.csv", "selected_well_metrics.csv",
        "group_metrics.csv", "map_metrics.csv", "control_metrics.csv", "control_config_metrics.csv", "RESULT.md",
    ]
    artifact_manifest = {
        "schema_version": 1,
        "files": [
            {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in artifact_names
        ],
    }
    dump_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return summary


def synthetic_self_test() -> dict[str, Any]:
    a = np.linspace(-2.0, 3.0, ACTION_BINS).astype(np.float32)
    transferred = action_for(a, 407)
    error = np.sin(np.linspace(0.0, 4.0, 407))
    metric = metric_for(error)
    library = np.asarray([3, 1, 2], dtype=int)
    ids = np.asarray(["a", "b", "c", "d"])
    ordered = deterministic_order(np.asarray([1.0, 1.0, 1.0]), library, ids)
    checks = {
        "transfer_shape": transferred.shape == (407,),
        "transfer_finite": bool(np.isfinite(transferred).all()),
        "metric_decomposition": abs(metric.sse - metric.datum_sse - metric.trend_sse - metric.shape_sse) <= 1e-8,
        "tie_order": ordered.tolist() == [1, 2, 3],
        "derangement": all(k != v for k, v in derangement(np.arange(20), 7).items()),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


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
        print(json.dumps(synthetic_self_test(), indent=2, sort_keys=True))
        return
    result = run(config, args.output_dir, args.build_only, args.force_rebuild)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
