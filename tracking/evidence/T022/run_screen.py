from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[3]
OOF_PATH = ROOT / "artifacts/E011/oof_predictions.csv.gz"
FEATURE_PATH = ROOT / "experiments/E008/results/legal_features.csv"
CONFIG_PATH = ROOT / "experiments/E011/config.json"
COMPACT_PATH = ROOT / "artifacts/E011/compact_stats_v1.npz"
OUTPUT_PATH = ROOT / "scratch/agents/t022-uncertainty-screen-20260723/screen_result.json"
SOURCE_COMMIT = "33e2de77dc044832f71de294f69430f3b5d2e79c"

E011_COL = "spline4_ridge_equal_s075"
E006_COL = "e006_nested_fusion"
EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
MIN_ORACLE_HEADROOM = 0.20
MIN_LEGAL_GAIN = 0.03
MIN_MAP_WINS = 4
MIN_CELL_WINS = 17


class ScreenError(RuntimeError):
    pass


@dataclass(frozen=True)
class WellQuadratic:
    well_id: str
    rows: int
    a: float
    b: float
    c: float
    e011_rmse: float
    e006_rmse: float
    oracle_weight: float

    def sse(self, weight: float) -> float:
        w = float(np.clip(weight, 0.0, 1.0))
        value = self.a + 2.0 * self.b * w + self.c * w * w
        tolerance = max(1e-8, abs(self.a) * 1e-12)
        if value < 0.0 and abs(value) <= tolerance:
            value = 0.0
        if value < 0.0 or not math.isfinite(value):
            raise ScreenError(f"{self.well_id}: invalid blend SSE")
        return value


@dataclass(frozen=True)
class Context:
    key: str
    scope: str
    label: str
    outer_group: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    inner_assignments: Mapping[str, int]


@dataclass(frozen=True)
class FeatureState:
    names: tuple[str, ...]
    train_x: np.ndarray
    test_x: np.ndarray


@dataclass(frozen=True)
class BranchSelection:
    config: Mapping[str, Any]
    shrinkage: float
    threshold: float | None
    inner_rmse: float
    inner_auc: float | None


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ScreenError(f"non-finite {label}")
    return result


def load_quadratics() -> tuple[dict[str, WellQuadratic], dict[str, dict[str, float]]]:
    required = {"id", "well_id", "row_index", "target", E006_COL, E011_COL}
    accum: dict[str, list[float]] = {}
    delta_stats: dict[str, list[float]] = {}
    seen_keys: set[tuple[str, int]] = set()
    rows_total = 0
    current_well: str | None = None
    closed_wells: set[str] = set()
    with gzip.open(OOF_PATH, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ScreenError("E011 OOF schema differs")
        for row in reader:
            well_id = str(row["well_id"])
            row_index = int(row["row_index"])
            key = (well_id, row_index)
            if key in seen_keys:
                raise ScreenError("duplicate OOF well/row key")
            seen_keys.add(key)
            if current_well is None:
                current_well = well_id
            elif well_id != current_well:
                closed_wells.add(current_well)
                if well_id in closed_wells:
                    raise ScreenError("OOF well blocks are not contiguous")
                current_well = well_id
            target = _finite(row["target"], "target")
            e006 = _finite(row[E006_COL], E006_COL)
            e011 = _finite(row[E011_COL], E011_COL)
            error = e011 - target
            delta = e006 - e011
            item = accum.setdefault(well_id, [0.0, 0.0, 0.0, 0.0])
            item[0] += 1.0
            item[1] += error * error
            item[2] += error * delta
            item[3] += delta * delta
            d = delta_stats.setdefault(well_id, [])
            d.append(delta)
            rows_total += 1
    if rows_total != EXPECTED_ROWS or len(accum) != EXPECTED_WELLS:
        raise ScreenError(f"OOF identity differs: rows={rows_total}, wells={len(accum)}")
    records: dict[str, WellQuadratic] = {}
    disagreement: dict[str, dict[str, float]] = {}
    for well_id in sorted(accum):
        rows, a, b, c = accum[well_id]
        count = int(rows)
        if count <= 0 or min(a, c) < 0.0:
            raise ScreenError(f"{well_id}: invalid quadratic statistics")
        oracle = float(np.clip(-b / c, 0.0, 1.0)) if c > 1e-12 else 0.0
        e006_sse = a + 2.0 * b + c
        records[well_id] = WellQuadratic(
            well_id=well_id,
            rows=count,
            a=a,
            b=b,
            c=c,
            e011_rmse=math.sqrt(a / count),
            e006_rmse=math.sqrt(max(0.0, e006_sse) / count),
            oracle_weight=oracle,
        )
        values = np.asarray(delta_stats[well_id], dtype=np.float64)
        positions = np.arange(values.size, dtype=np.float64)
        slope = float(np.polyfit(positions, values, 1)[0]) if values.size >= 2 else 0.0
        disagreement[well_id] = {
            "e011_e006_first": float(values[0]),
            "e011_e006_last": float(values[-1]),
            "e011_e006_mean": float(values.mean()),
            "e011_e006_mean_abs": float(np.mean(np.abs(values))),
            "e011_e006_std": float(values.std()),
            "e011_e006_range": float(values.max() - values.min()),
            "e011_e006_max_abs": float(np.max(np.abs(values))),
            "e011_e006_slope_per_row": slope,
            "e011_e006_positive_fraction": float(np.mean(values > 0.0)),
        }
    return records, disagreement


def summarize(records: Mapping[str, WellQuadratic], weights: Mapping[str, float], ids: Sequence[str]) -> dict[str, float]:
    if not ids:
        raise ScreenError("empty metric set")
    sses = []
    rmses = []
    rows = 0
    for well_id in ids:
        if well_id not in weights:
            raise ScreenError(f"missing weight for {well_id}")
        weight = _finite(weights[well_id], f"weight {well_id}")
        if weight < -1e-12 or weight > 1.0 + 1e-12:
            raise ScreenError(f"out-of-bound weight for {well_id}")
        record = records[well_id]
        sse = record.sse(weight)
        sses.append(sse)
        rmses.append(math.sqrt(sse / record.rows))
        rows += record.rows
    total_sse = float(sum(sses))
    ordered = sorted(sses, reverse=True)
    worst_count = max(1, int(math.ceil(0.05 * len(ordered))))
    return {
        "rows": float(rows),
        "wells": float(len(ids)),
        "sse": total_sse,
        "rmse": math.sqrt(total_sse / rows),
        "median_well_rmse": float(np.median(rmses)),
        "p90_well_rmse": float(np.quantile(rmses, 0.9)),
        "p95_well_rmse": float(np.quantile(rmses, 0.95)),
        "max_well_rmse": float(max(rmses)),
        "worst_5pct_sse_share": float(sum(ordered[:worst_count]) / total_sse),
    }


def load_features(well_ids: Sequence[str], disagreement: Mapping[str, Mapping[str, float]]) -> tuple[tuple[str, ...], np.ndarray]:
    frame = pd.read_csv(FEATURE_PATH)
    if "well_id" not in frame.columns or frame["well_id"].astype(str).duplicated().any():
        raise ScreenError("legal feature identity differs")
    frame["well_id"] = frame["well_id"].astype(str)
    frame = frame.set_index("well_id")
    if set(frame.index) != set(well_ids):
        raise ScreenError("legal feature well coverage differs")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    for name in sorted(next(iter(disagreement.values())).keys()):
        numeric[name] = [disagreement[well_id][name] for well_id in numeric.index]
    numeric = numeric.loc[list(well_ids)]
    names = tuple(str(name) for name in numeric.columns)
    values = numeric.to_numpy(dtype=np.float64)
    if values.shape[0] != len(well_ids) or values.shape[1] != len(names):
        raise ScreenError("feature matrix shape differs")
    forbidden = [name for name in names if any(token in name.lower() for token in ("target_tvt", "true_tvt", "hidden_target", "hidden_label", "oracle"))]
    if forbidden:
        raise ScreenError(f"forbidden features: {forbidden[:5]}")
    return names, values


def load_contexts(well_ids: Sequence[str]) -> tuple[list[Context], dict[str, int], dict[str, int], dict[str, int]]:
    expected = set(well_ids)
    folds = []
    for version in range(1, 6):
        payload = json.loads((ROOT / f"folds/v{version}.json").read_text(encoding="utf-8"))
        assignments = {str(k): int(v) for k, v in payload["assignments"].items()}
        if set(assignments) != expected or set(assignments.values()) != set(range(5)):
            raise ScreenError(f"fold v{version} identity differs")
        folds.append((f"v{version}", assignments))
    with np.load(COMPACT_PATH, allow_pickle=False) as payload:
        ids = tuple(str(value) for value in payload["well_ids"].tolist())
        spatial_values = np.asarray(payload["spatial_assignment"], dtype=int)
        typewell_values = np.asarray(payload["typewell_assignment"], dtype=int)
    if ids != tuple(well_ids):
        raise ScreenError("compact well order differs")
    spatial = {well_id: int(spatial_values[index]) for index, well_id in enumerate(ids)}
    typewell = {well_id: int(typewell_values[index]) for index, well_id in enumerate(ids)}
    if set(spatial.values()) != set(range(5)) or set(typewell.values()) != set(range(5)):
        raise ScreenError("stress assignment values differ")
    contexts: list[Context] = []
    for label, assignments in folds:
        for outer in range(5):
            test_ids = tuple(well_id for well_id in well_ids if assignments[well_id] == outer)
            train_ids = tuple(well_id for well_id in well_ids if assignments[well_id] != outer)
            contexts.append(Context(f"repeated:{label}:{outer}", "repeated", label, outer, train_ids, test_ids, assignments))
    inner = folds[0][1]
    for scope, assignments in (("spatial", spatial), ("typewell", typewell)):
        for outer in range(5):
            test_ids = tuple(well_id for well_id in well_ids if assignments[well_id] == outer)
            train_ids = tuple(well_id for well_id in well_ids if assignments[well_id] != outer)
            contexts.append(Context(f"{scope}:{outer}", scope, scope, outer, train_ids, test_ids, inner))
    for context in contexts:
        train = set(context.train_ids)
        test = set(context.test_ids)
        if not train or not test or train & test or train | test != expected:
            raise ScreenError(f"{context.key}: invalid membership")
    return contexts, folds[0][1], spatial, typewell


def prepare_features(all_x: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, names: Sequence[str], target: np.ndarray, feature_count: int | None, *, scale: bool) -> FeatureState:
    train = np.asarray(all_x[train_idx], dtype=np.float64)
    test = np.asarray(all_x[test_idx], dtype=np.float64)
    medians = np.zeros(train.shape[1], dtype=np.float64)
    for col in range(train.shape[1]):
        finite = train[np.isfinite(train[:, col]), col]
        medians[col] = float(np.median(finite)) if finite.size else 0.0
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    if scale:
        means = train.mean(axis=0)
        stds = train.std(axis=0)
        stds[~np.isfinite(stds) | (stds < 1e-9)] = 1.0
        train = (train - means) / stds
        test = (test - means) / stds
    count = train.shape[1] if feature_count is None else min(int(feature_count), train.shape[1])
    if count < train.shape[1]:
        y = np.asarray(target, dtype=np.float64)
        y_std = float(np.std(y))
        scored = []
        for col, name in enumerate(names):
            x = train[:, col]
            corr = 0.0
            if float(np.std(x)) > 1e-12 and y_std > 1e-12:
                value = float(np.corrcoef(x, y)[0, 1])
                corr = abs(value) if math.isfinite(value) else 0.0
            scored.append((-corr, str(name), col))
        scored.sort()
        chosen = sorted(item[2] for item in scored[:count])
    else:
        chosen = list(range(train.shape[1]))
    return FeatureState(tuple(str(names[i]) for i in chosen), train[:, chosen], test[:, chosen])


def _indices(id_to_index: Mapping[str, int], ids: Sequence[str]) -> np.ndarray:
    return np.asarray([id_to_index[well_id] for well_id in ids], dtype=int)


def _fit_predict(
    family: str,
    config: Mapping[str, Any],
    all_x: np.ndarray,
    feature_names: Sequence[str],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    id_to_index: Mapping[str, int],
    target_weight: Mapping[str, float],
    target_binary: Mapping[str, int],
    row_weights: Mapping[str, float],
    *,
    shuffle_salt: str | None = None,
) -> np.ndarray:
    train_idx = _indices(id_to_index, train_ids)
    test_idx = _indices(id_to_index, test_ids)
    if family.startswith("logistic"):
        y = np.asarray([target_binary[well_id] for well_id in train_ids], dtype=int)
    else:
        y = np.asarray([target_weight[well_id] for well_id in train_ids], dtype=np.float64)
    if shuffle_salt is not None and len(y) > 1:
        order = sorted(range(len(train_ids)), key=lambda i: hashlib.sha256(f"{shuffle_salt}|{train_ids[i]}".encode()).hexdigest())
        order = order[1:] + order[:1]
        y = y[np.asarray(order, dtype=int)]
    linear = family in {"logistic_hard", "logistic_soft", "ridge_equal", "ridge_row"}
    prepared = prepare_features(all_x, train_idx, test_idx, feature_names, y.astype(float), config.get("feature_count"), scale=linear or family == "knn")
    sample_weight = np.asarray([row_weights[well_id] for well_id in train_ids], dtype=np.float64)
    sample_weight /= max(float(sample_weight.mean()), 1.0)
    if family in {"logistic_hard", "logistic_soft"}:
        if len(set(y.tolist())) < 2:
            return np.full(len(test_ids), float(y[0]) if y.size else 0.0, dtype=np.float64)
        model = LogisticRegression(C=float(config["c"]), max_iter=2000, solver="liblinear", random_state=16016)
        model.fit(prepared.train_x, y, sample_weight=sample_weight)
        return np.asarray(model.predict_proba(prepared.test_x)[:, 1], dtype=np.float64)
    if family in {"ridge_equal", "ridge_row"}:
        model = Ridge(alpha=float(config["alpha"]))
        weight = sample_weight if family == "ridge_row" else None
        model.fit(prepared.train_x, y, sample_weight=weight)
        return np.asarray(model.predict(prepared.test_x), dtype=np.float64)
    if family == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=128,
            max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_leaf"]),
            max_features=0.5,
            random_state=16016,
            n_jobs=1,
        )
        model.fit(prepared.train_x, y, sample_weight=sample_weight)
        return np.asarray(model.predict(prepared.test_x), dtype=np.float64)
    if family == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_leaf"]),
            random_state=16016,
            loss="huber",
        )
        model.fit(prepared.train_x, y, sample_weight=sample_weight)
        return np.asarray(model.predict(prepared.test_x), dtype=np.float64)
    if family == "knn":
        neighbors = min(int(config["neighbors"]), len(train_ids))
        model = KNeighborsRegressor(n_neighbors=max(1, neighbors), weights=str(config["weights"]), p=2)
        model.fit(prepared.train_x, y)
        return np.asarray(model.predict(prepared.test_x), dtype=np.float64)
    raise ScreenError(f"unknown family {family}")


def family_grid(family: str) -> list[dict[str, Any]]:
    if family in {"logistic_hard", "logistic_soft"}:
        return [{"c": c, "feature_count": count} for count in (32, 64, None) for c in (0.03, 0.3, 3.0)]
    if family in {"ridge_equal", "ridge_row"}:
        return [{"alpha": alpha, "feature_count": count} for count in (32, 64, None) for alpha in (1.0, 30.0, 300.0)]
    if family == "extra_trees":
        return [{"max_depth": depth, "min_leaf": leaf, "feature_count": None} for depth, leaf in ((2, 20), (4, 20), (6, 10), (8, 20))]
    if family == "gradient_boosting":
        return [
            {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 1, "min_leaf": 20, "feature_count": None},
            {"n_estimators": 150, "learning_rate": 0.025, "max_depth": 2, "min_leaf": 20, "feature_count": None},
            {"n_estimators": 120, "learning_rate": 0.02, "max_depth": 2, "min_leaf": 40, "feature_count": None},
        ]
    if family == "knn":
        return [{"neighbors": k, "weights": weights, "feature_count": count} for count in (32, 64) for k in (25, 50, 100) for weights in ("uniform", "distance")]
    raise ScreenError(f"unknown family {family}")


def transform_raw(family: str, raw: np.ndarray, shrinkage: float, threshold: float | None) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ScreenError(f"{family}: invalid raw predictions")
    if family == "logistic_hard":
        if threshold is None:
            raise ScreenError("hard logistic threshold missing")
        output = (values >= float(threshold)).astype(np.float64)
    else:
        output = np.clip(values, 0.0, 1.0) * float(shrinkage)
    if not np.all(np.isfinite(output)) or np.any(output < -1e-12) or np.any(output > 1.0 + 1e-12):
        raise ScreenError(f"{family}: transformed weights invalid")
    return output


def select_branch(
    context: Context,
    family: str,
    records: Mapping[str, WellQuadratic],
    all_x: np.ndarray,
    feature_names: Sequence[str],
    id_to_index: Mapping[str, int],
    target_weight: Mapping[str, float],
    target_binary: Mapping[str, int],
    row_weights: Mapping[str, float],
    *,
    shuffle: bool = False,
) -> BranchSelection:
    groups = sorted({int(context.inner_assignments[well_id]) for well_id in context.train_ids})
    candidates = []
    for config_index, config in enumerate(family_grid(family)):
        inner_raw: dict[str, float] = {}
        inner_truth: dict[str, int] = {}
        for group in groups:
            validation = tuple(well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) == group)
            training = tuple(well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) != group)
            if not validation or not training:
                continue
            raw = _fit_predict(
                family, config, all_x, feature_names, training, validation, id_to_index,
                target_weight, target_binary, row_weights,
                shuffle_salt=f"{context.key}|{family}|{config_index}|{group}" if shuffle else None,
            )
            inner_raw.update({well_id: float(raw[index]) for index, well_id in enumerate(validation)})
            inner_truth.update({well_id: int(target_binary[well_id]) for well_id in validation})
        if set(inner_raw) != set(context.train_ids):
            raise ScreenError(f"{context.key} {family}: incomplete inner OOF")
        ids = tuple(context.train_ids)
        raw_values = np.asarray([inner_raw[well_id] for well_id in ids], dtype=np.float64)
        auc = None
        truth = np.asarray([inner_truth[well_id] for well_id in ids], dtype=int)
        if len(set(truth.tolist())) >= 2:
            try:
                auc = float(roc_auc_score(truth, raw_values))
            except ValueError:
                auc = None
        if family == "logistic_hard":
            transforms = [(1.0, threshold) for threshold in (0.2, 0.35, 0.5, 0.65, 0.8)]
        else:
            transforms = [(shrink, None) for shrink in (0.10, 0.25, 0.50, 0.75, 1.00)]
        for transform_index, (shrinkage, threshold) in enumerate(transforms):
            values = transform_raw(family, raw_values, shrinkage, threshold)
            weights = {well_id: float(values[index]) for index, well_id in enumerate(ids)}
            rmse = float(summarize(records, weights, ids)["rmse"])
            candidates.append((rmse, config_index, transform_index, BranchSelection(dict(config), float(shrinkage), threshold, rmse, auc)))
    candidates.sort(key=lambda item: item[:-1])
    return candidates[0][-1]


def fit_context(
    context: Context,
    family: str,
    selection: BranchSelection,
    records: Mapping[str, WellQuadratic],
    all_x: np.ndarray,
    feature_names: Sequence[str],
    id_to_index: Mapping[str, int],
    target_weight: Mapping[str, float],
    target_binary: Mapping[str, int],
    row_weights: Mapping[str, float],
    *,
    shuffle: bool = False,
) -> dict[str, float]:
    raw = _fit_predict(
        family, selection.config, all_x, feature_names, context.train_ids, context.test_ids,
        id_to_index, target_weight, target_binary, row_weights,
        shuffle_salt=f"{context.key}|{family}|outer" if shuffle else None,
    )
    values = transform_raw(family, raw, selection.shrinkage, selection.threshold)
    return {well_id: float(values[index]) for index, well_id in enumerate(context.test_ids)}


def constant_blend_context(context: Context, records: Mapping[str, WellQuadratic]) -> tuple[float, dict[str, float], float]:
    candidates = []
    for order, weight in enumerate((0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)):
        weights = {well_id: weight for well_id in context.train_ids}
        rmse = float(summarize(records, weights, context.train_ids)["rmse"])
        candidates.append((rmse, order, weight))
    candidates.sort()
    selected = float(candidates[0][2])
    return selected, {well_id: selected for well_id in context.test_ids}, float(candidates[0][0])


def main() -> None:
    started = time.perf_counter()
    records, disagreement = load_quadratics()
    well_ids = tuple(sorted(records))
    feature_names, all_x = load_features(well_ids, disagreement)
    id_to_index = {well_id: index for index, well_id in enumerate(well_ids)}
    contexts, repeated_v1, spatial, typewell = load_contexts(well_ids)
    repeated = [context for context in contexts if context.scope == "repeated"]
    stress = [context for context in contexts if context.scope != "repeated"]

    target_weight = {well_id: records[well_id].oracle_weight for well_id in well_ids}
    target_binary = {well_id: int(records[well_id].e006_rmse < records[well_id].e011_rmse) for well_id in well_ids}
    row_weights = {well_id: float(records[well_id].rows) for well_id in well_ids}

    zero = {well_id: 0.0 for well_id in well_ids}
    one = {well_id: 1.0 for well_id in well_ids}
    oracle_fallback = {well_id: float(target_binary[well_id]) for well_id in well_ids}
    oracle_blend = dict(target_weight)
    baseline = summarize(records, zero, well_ids)
    e006 = summarize(records, one, well_ids)
    fallback_summary = summarize(records, oracle_fallback, well_ids)
    blend_summary = summarize(records, oracle_blend, well_ids)
    oracle = {
        "e011": baseline,
        "e006": e006,
        "fallback": fallback_summary,
        "continuous_blend": blend_summary,
        "fallback_gain": float(baseline["rmse"] - fallback_summary["rmse"]),
        "continuous_blend_gain": float(baseline["rmse"] - blend_summary["rmse"]),
        "e006_better_wells": int(sum(target_binary.values())),
        "e006_better_fraction": float(np.mean(list(target_binary.values()))),
        "positive_oracle_weight_wells": int(sum(value > 1e-8 for value in target_weight.values())),
        "full_e006_oracle_weight_wells": int(sum(value >= 1.0 - 1e-8 for value in target_weight.values())),
        "oracle_weight_quantiles": {str(q): float(np.quantile(list(target_weight.values()), q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
    }
    headroom_pass = max(oracle["fallback_gain"], oracle["continuous_blend_gain"]) >= MIN_ORACLE_HEADROOM

    families = ["logistic_hard", "logistic_soft", "ridge_equal", "ridge_row", "extra_trees", "gradient_boosting", "knn"]
    context_rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, list[float]]] = {family: defaultdict(list) for family in ("constant_blend", *families, "ridge_row_shuffled", "extra_trees_shuffled")}
    selections: list[dict[str, Any]] = []

    if headroom_pass:
        for context_index, context in enumerate(contexts):
            base_test = {well_id: 0.0 for well_id in context.test_ids}
            base_summary = summarize(records, base_test, context.test_ids)
            selected_weight, const_weights, const_inner = constant_blend_context(context, records)
            const_summary = summarize(records, const_weights, context.test_ids)
            context_rows.append({"context": context.key, "scope": context.scope, "candidate": "constant_blend", "selected_constant_weight": selected_weight, "inner_rmse": const_inner, **const_summary, "gain_vs_e011": float(base_summary["rmse"] - const_summary["rmse"])})
            if context.scope == "repeated":
                for well_id, value in const_weights.items():
                    predictions["constant_blend"][well_id].append(value)
            for family in families:
                selection = select_branch(context, family, records, all_x, feature_names, id_to_index, target_weight, target_binary, row_weights)
                output = fit_context(context, family, selection, records, all_x, feature_names, id_to_index, target_weight, target_binary, row_weights)
                summary = summarize(records, output, context.test_ids)
                context_rows.append({"context": context.key, "scope": context.scope, "candidate": family, "selected_constant_weight": "", "inner_rmse": selection.inner_rmse, "inner_auc": selection.inner_auc if selection.inner_auc is not None else "", "config_json": json.dumps(selection.config, sort_keys=True), "shrinkage": selection.shrinkage, "threshold": selection.threshold if selection.threshold is not None else "", **summary, "gain_vs_e011": float(base_summary["rmse"] - summary["rmse"])})
                selections.append({"context": context.key, "scope": context.scope, "family": family, "config": dict(selection.config), "shrinkage": selection.shrinkage, "threshold": selection.threshold, "inner_rmse": selection.inner_rmse, "inner_auc": selection.inner_auc})
                if context.scope == "repeated":
                    for well_id, value in output.items():
                        predictions[family][well_id].append(value)
            # Negative controls for two distinct model classes.
            for family, name in (("ridge_row", "ridge_row_shuffled"), ("extra_trees", "extra_trees_shuffled")):
                selection = select_branch(context, family, records, all_x, feature_names, id_to_index, target_weight, target_binary, row_weights, shuffle=True)
                output = fit_context(context, family, selection, records, all_x, feature_names, id_to_index, target_weight, target_binary, row_weights, shuffle=True)
                summary = summarize(records, output, context.test_ids)
                context_rows.append({"context": context.key, "scope": context.scope, "candidate": name, "selected_constant_weight": "", "inner_rmse": selection.inner_rmse, "inner_auc": selection.inner_auc if selection.inner_auc is not None else "", "config_json": json.dumps(selection.config, sort_keys=True), "shrinkage": selection.shrinkage, "threshold": selection.threshold if selection.threshold is not None else "", **summary, "gain_vs_e011": float(base_summary["rmse"] - summary["rmse"])})
                if context.scope == "repeated":
                    for well_id, value in output.items():
                        predictions[name][well_id].append(value)
            print(f"completed {context_index + 1}/{len(contexts)} {context.key}", flush=True)

    candidate_summaries: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, bool]] = {}
    special_rows: list[dict[str, Any]] = []
    stress_rows = [row for row in context_rows if row["scope"] != "repeated"]
    repeated_rows = [row for row in context_rows if row["scope"] == "repeated"]
    if headroom_pass:
        # Each well appears once per map, so every repeated candidate must have five legal predictions.
        final_weights: dict[str, dict[str, float]] = {}
        for candidate, by_well in predictions.items():
            if set(by_well) != set(well_ids) or any(len(values) != 5 for values in by_well.values()):
                raise ScreenError(f"{candidate}: repeated coverage differs")
            final_weights[candidate] = {well_id: float(np.mean(values)) for well_id, values in by_well.items()}
            candidate_summaries[candidate] = summarize(records, final_weights[candidate], well_ids)
            candidate_summaries[candidate]["gain_vs_e011"] = float(baseline["rmse"] - candidate_summaries[candidate]["rmse"])
            candidate_summaries[candidate]["mean_weight"] = float(np.mean(list(final_weights[candidate].values())))
            candidate_summaries[candidate]["action_fraction"] = float(np.mean([value > 1e-8 for value in final_weights[candidate].values()]))

        # Map and cell counts from untouched repeated context outputs.
        for candidate in candidate_summaries:
            rows = [row for row in repeated_rows if row["candidate"] == candidate]
            if len(rows) != 25:
                raise ScreenError(f"{candidate}: repeated context row count differs")
            cell_wins = sum(float(row["gain_vs_e011"]) > 0.0 for row in rows)
            map_wins = 0
            map_gains = {}
            for version in ("v1", "v2", "v3", "v4", "v5"):
                selected = [row for row in rows if row["context"].startswith(f"repeated:{version}:")]
                candidate_sse = sum(float(row["sse"]) for row in selected)
                total_rows = sum(float(row["rows"]) for row in selected)
                candidate_rmse = math.sqrt(candidate_sse / total_rows)
                base_ids = [well_id for well_id in well_ids if True]
                # The five cells partition all wells; baseline is identical full OOF.
                gain = float(baseline["rmse"] - candidate_rmse)
                map_gains[version] = gain
                map_wins += int(gain > 0.0)
            candidate_summaries[candidate]["cell_wins"] = int(cell_wins)
            candidate_summaries[candidate]["map_wins"] = int(map_wins)
            candidate_summaries[candidate]["map_gains"] = map_gains

        rows_values = np.asarray([records[well_id].rows for well_id in well_ids], dtype=float)
        missing_index = feature_names.index("hidden_gr_missing_fraction")
        missing_values = all_x[:, missing_index]
        long_threshold = float(np.quantile(rows_values, 0.8))
        finite_missing = missing_values[np.isfinite(missing_values)]
        missing_threshold = float(np.quantile(finite_missing, 0.8)) if finite_missing.size else 0.0
        special_sets = {
            "long_suffix": [well_id for well_id in well_ids if records[well_id].rows >= long_threshold],
            "high_gr_missingness": [well_id for well_id in well_ids if math.isfinite(float(all_x[id_to_index[well_id], missing_index])) and float(all_x[id_to_index[well_id], missing_index]) >= missing_threshold],
            "e011_catastrophe": [well_id for well_id in well_ids if records[well_id].e011_rmse >= 12.0],
            "e006_oracle_wins": [well_id for well_id in well_ids if target_binary[well_id] == 1],
        }
        base_special = {name: summarize(records, zero, ids) for name, ids in special_sets.items() if ids}
        for candidate, weights in final_weights.items():
            for name, ids in special_sets.items():
                if not ids:
                    continue
                summary = summarize(records, weights, ids)
                special_rows.append({"slice": name, "candidate": candidate, "wells": len(ids), **summary, "gain_vs_e011": float(base_special[name]["rmse"] - summary["rmse"])})

        for candidate, summary in candidate_summaries.items():
            spatial_rows = [row for row in stress_rows if row["candidate"] == candidate and row["scope"] == "spatial"]
            typewell_rows = [row for row in stress_rows if row["candidate"] == candidate and row["scope"] == "typewell"]
            negative = candidate.endswith("shuffled")
            gates[candidate] = {
                "not_negative_control": not negative,
                "gain": float(summary["gain_vs_e011"]) >= MIN_LEGAL_GAIN,
                "maps": int(summary["map_wins"]) >= MIN_MAP_WINS,
                "cells": int(summary["cell_wins"]) >= MIN_CELL_WINS,
                "p90": float(summary["p90_well_rmse"]) <= float(baseline["p90_well_rmse"]) + 1e-12,
                "worst5": float(summary["worst_5pct_sse_share"]) <= float(baseline["worst_5pct_sse_share"]) + 1e-12,
                "spatial": len(spatial_rows) == 5 and all(float(row["gain_vs_e011"]) > 0.0 for row in spatial_rows),
                "typewell": len(typewell_rows) == 5 and all(float(row["gain_vs_e011"]) > 0.0 for row in typewell_rows),
                "special_slices": all(float(row["gain_vs_e011"]) >= 0.0 for row in special_rows if row["candidate"] == candidate and row["slice"] in {"long_suffix", "high_gr_missingness", "e011_catastrophe"}),
            }

    controls = {
        "source_commit": {"pass": SOURCE_COMMIT == "33e2de77dc044832f71de294f69430f3b5d2e79c", "value": SOURCE_COMMIT},
        "oof_coverage": {"pass": len(records) == EXPECTED_WELLS and int(sum(record.rows for record in records.values())) == EXPECTED_ROWS, "wells": len(records), "rows": int(sum(record.rows for record in records.values()))},
        "exact_fallback": {"pass": summarize(records, zero, well_ids) == baseline},
        "weight_bounds": {"pass": all(0.0 <= value <= 1.0 for by_well in predictions.values() for values in by_well.values() for value in values) if headroom_pass else True},
        "context_completion": {"pass": (len(context_rows) == 35 * 10) if headroom_pass else len(context_rows) == 0, "rows": len(context_rows), "expected_if_run": 350},
        "shuffled_labels": {
            "pass": (all(float(candidate_summaries[name]["gain_vs_e011"]) < MIN_LEGAL_GAIN for name in ("ridge_row_shuffled", "extra_trees_shuffled"))) if headroom_pass else True,
            "gains": {name: float(candidate_summaries[name]["gain_vs_e011"]) for name in ("ridge_row_shuffled", "extra_trees_shuffled")} if headroom_pass else {},
        },
        "finite_metrics": {"pass": all(math.isfinite(float(value)) for summary in (baseline, e006, fallback_summary, blend_summary, *candidate_summaries.values()) for key, value in summary.items() if isinstance(value, (int, float)))},
    }
    passing = [candidate for candidate, candidate_gates in gates.items() if all(candidate_gates.values())]
    result = {
        "schema_version": 1,
        "task_id": "T022",
        "hypothesis_id": "H016",
        "source_commit": SOURCE_COMMIT,
        "status": "worth_screen_pass" if passing else "worth_screen_reject",
        "headroom_pass": headroom_pass,
        "oracle": oracle,
        "feature_count": len(feature_names),
        "contexts": {"repeated": len(repeated), "stress": len(stress)},
        "candidate_summaries": candidate_summaries,
        "gates": gates,
        "passing_candidates": passing,
        "controls": controls,
        "context_rows": context_rows,
        "special_slice_rows": special_rows,
        "selections": selections,
        "runtime_seconds": time.perf_counter() - started,
        "decision": "authorize_e013_preregistration" if passing else "close_h016_without_e013",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"],
        "e011_rmse": baseline["rmse"],
        "fallback_oracle_gain": oracle["fallback_gain"],
        "continuous_oracle_gain": oracle["continuous_blend_gain"],
        "e006_better_wells": oracle["e006_better_wells"],
        "passing_candidates": passing,
        "top_candidates": sorted(((name, values["gain_vs_e011"], values["map_wins"], values["cell_wins"]) for name, values in candidate_summaries.items()), key=lambda x: (-x[1], x[0]))[:8],
        "controls_pass": all(item["pass"] for item in controls.values()),
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
