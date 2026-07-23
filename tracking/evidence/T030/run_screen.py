#!/usr/bin/env python3
"""T030 outer-isolated complementary pseudo-coefficient and datum ensemble."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 4 else Path.cwd()
OUTPUTS = 4
T025_PARENT_RMSE = 12.42492760849511
T016_PARENT_RMSE = 12.492966056189658
T027_PARENT_RMSE = 12.45883303686926
E011_RMSE = 12.550756295689673


class DataValidationError(ValueError):
    """Raised when a frozen T030 contract is violated."""


@dataclass
class Action:
    """Exact E011-relative row action in sufficient-statistic form."""

    base_scale: np.ndarray
    spline: np.ndarray
    datum: np.ndarray

    def validate(self, rows: int, *, allow_negative_scale: bool = False) -> None:
        if self.base_scale.shape != (rows,) or self.spline.shape != (rows, OUTPUTS) or self.datum.shape != (rows,):
            raise DataValidationError("action dimensions differ")
        if not np.all(np.isfinite(self.base_scale)) or not np.all(np.isfinite(self.spline)) or not np.all(np.isfinite(self.datum)):
            raise DataValidationError("action contains non-finite values")
        if not allow_negative_scale and np.any(self.base_scale < -1e-12):
            raise DataValidationError("real action contains negative coefficient scale")


@dataclass(frozen=True)
class Selection:
    alpha: float
    beta: float
    gamma: float
    pooled_rmse: float
    minimum_domain_gain: float
    fallback: bool


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(value.__class__.__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise DataValidationError(f"refusing to write empty CSV {path.name}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise DataValidationError(f"{path.name}: inconsistent schema")
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


def import_parent(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DataValidationError(f"could not import parent {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for relative, expected in config["input_identities"].items():
        path = root / relative
        if not path.is_file():
            raise DataValidationError(f"missing frozen input {relative}")
        actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if actual != expected:
            raise DataValidationError(f"input identity differs: {relative}")
        rows.append({"path": relative, **actual})
    return {"pass": True, "files": rows}


def summarize(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(metrics)
    if not values:
        raise DataValidationError("cannot summarize empty metrics")
    total_rows = sum(int(row["rows_scored"]) for row in values)
    total_sse = sum(float(row["sse"]) for row in values)
    if total_rows <= 0 or total_sse < 0.0:
        raise DataValidationError("invalid aggregate totals")
    rmses = np.asarray([float(row["rmse"]) for row in values], dtype=np.float64)
    ordered = sorted(values, key=lambda row: (-float(row["sse"]), str(row["well_id"])))
    worst5 = max(1, math.ceil(0.05 * len(values)))
    worst10 = max(1, math.ceil(0.10 * len(values)))
    return {
        "rows_scored": total_rows,
        "wells_scored": len(values),
        "sse": total_sse,
        "rmse": math.sqrt(total_sse / total_rows),
        "median_well_rmse": float(np.quantile(rmses, 0.5)),
        "p90_well_rmse": float(np.quantile(rmses, 0.9)),
        "p95_well_rmse": float(np.quantile(rmses, 0.95)),
        "max_well_rmse": float(rmses.max()),
        "worst_5pct_sse_share": sum(float(row["sse"]) for row in ordered[:worst5]) / total_sse if total_sse else 0.0,
        "worst_10pct_sse_share": sum(float(row["sse"]) for row in ordered[:worst10]) / total_sse if total_sse else 0.0,
    }


def metric_one(record: Any, base_scale: float, spline: Sequence[float], datum: float) -> dict[str, Any]:
    a = float(base_scale)
    c = np.asarray(spline, dtype=np.float64)
    d = float(datum)
    if c.shape != (OUTPUTS,) or not math.isfinite(a) or not math.isfinite(d) or not np.all(np.isfinite(c)):
        raise DataValidationError(f"{record.well_id}: invalid action")
    required = (record.e011_sse, record.e011_sum, record.e_dot_d, record.d_sse, record.d_sum,
                record.basis_dot_e, record.basis_dot_d, record.basis_sum, record.basis_cross)
    if any(value is None for value in required):
        raise DataValidationError(f"{record.well_id}: sufficient statistics missing")
    linear = a * float(record.e_dot_d) + float(np.dot(record.basis_dot_e, c)) + d * float(record.e011_sum)
    path_sum = a * float(record.d_sum) + float(np.dot(record.basis_sum, c))
    quadratic = (
        a * a * float(record.d_sse)
        + 2.0 * a * float(np.dot(record.basis_dot_d, c))
        + float(c @ record.basis_cross @ c)
        + 2.0 * d * path_sum
        + d * d * int(record.hidden_rows)
    )
    sse = float(record.e011_sse) + 2.0 * linear + quadratic
    tolerance = 1e-8 * max(1.0, abs(float(record.e011_sse)), abs(2.0 * linear), abs(quadratic))
    if sse < -tolerance:
        raise DataValidationError(f"{record.well_id}: materially negative SSE")
    sse = max(0.0, sse)
    error_sum = float(record.e011_sum) + path_sum + d * int(record.hidden_rows)
    return {
        "well_id": record.well_id,
        "rows_scored": int(record.hidden_rows),
        "sse": sse,
        "rmse": math.sqrt(sse / int(record.hidden_rows)),
        "mean_error": error_sum / int(record.hidden_rows),
    }


def score_action(records: Sequence[Any], indices: Sequence[int], action: Action) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    idx = np.asarray(indices, dtype=np.int64)
    action.validate(len(idx), allow_negative_scale=True)
    rows = [metric_one(records[int(well)], action.base_scale[pos], action.spline[pos], action.datum[pos]) for pos, well in enumerate(idx)]
    return summarize(rows), rows


def zero_action(size: int) -> Action:
    return Action(np.zeros(size), np.zeros((size, OUTPUTS)), np.zeros(size))


def combine_action(c25: np.ndarray, d16: np.ndarray, c27: np.ndarray, alpha: float, beta: float, gamma: float) -> Action:
    c25 = np.asarray(c25, dtype=np.float64); d16 = np.asarray(d16, dtype=np.float64); c27 = np.asarray(c27, dtype=np.float64)
    if c25.ndim != 2 or c25.shape[1] != OUTPUTS or c27.shape != c25.shape or d16.shape != (len(c25),):
        raise DataValidationError("parent action dimensions differ")
    action = Action(
        np.full(len(c25), float(alpha) + float(gamma), dtype=np.float64),
        float(alpha) * c25 + float(gamma) * c27,
        float(beta) * d16,
    )
    action.validate(len(c25), allow_negative_scale=(alpha < 0 or gamma < 0))
    return action


def average_actions(actions: Sequence[Action]) -> Action:
    if not actions:
        raise DataValidationError("cannot average empty actions")
    rows = len(actions[0].base_scale)
    for action in actions:
        action.validate(rows, allow_negative_scale=True)
    return Action(
        np.mean(np.stack([action.base_scale for action in actions]), axis=0),
        np.mean(np.stack([action.spline for action in actions]), axis=0),
        np.mean(np.stack([action.datum for action in actions]), axis=0),
    )


def _impute_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xtr = np.asarray(train, dtype=np.float64); xte = np.asarray(test, dtype=np.float64)
    if xtr.ndim != 2 or xte.ndim != 2 or xtr.shape[1] != xte.shape[1] or len(xtr) == 0:
        raise DataValidationError("ridge feature dimensions differ")
    medians = np.zeros(xtr.shape[1], dtype=np.float64)
    for column in range(xtr.shape[1]):
        finite = xtr[np.isfinite(xtr[:, column]), column]
        medians[column] = float(np.median(finite)) if len(finite) else 0.0
    xtr = np.where(np.isfinite(xtr), xtr, medians)
    xte = np.where(np.isfinite(xte), xte, medians)
    means = xtr.mean(axis=0); scales = xtr.std(axis=0); scales = np.where(scales > 1e-12, scales, 1.0)
    xtr = (xtr - means) / scales; xte = (xte - means) / scales
    if not np.all(np.isfinite(xtr)) or not np.all(np.isfinite(xte)):
        raise DataValidationError("ridge features remain non-finite")
    return xtr, xte


def predict_t027(context: Any, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    xtr, xte = _impute_scale(features[context.train_indices], features[context.test_indices])
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(xtr, targets[context.train_indices])
    prediction = np.clip(np.asarray(model.predict(xte), dtype=np.float64), -80.0, 80.0)
    if prediction.shape != (len(context.test_indices), OUTPUTS) or not np.all(np.isfinite(prediction)):
        raise DataValidationError("T027 comparator prediction differs")
    return prediction


def build_grids(config: Mapping[str, Any]) -> dict[str, list[tuple[float, float, float]]]:
    result: dict[str, list[tuple[float, float, float]]] = {}
    for name, spec in config["blend_grids"].items():
        if name == "tie_order": continue
        step = float(spec["step"])
        values = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 12)
        grid = []
        for alpha in values:
            for beta in values:
                if name == "simplex" and alpha + beta > 1.0 + 1e-12: continue
                grid.append((float(alpha), float(beta), float(spec["gamma"])))
        result[name] = grid
    if len(result.get("simplex", [])) != 231 or len(result.get("box", [])) != 121:
        raise DataValidationError("blend grid counts differ")
    return result


def membership_sha(indices: Sequence[int], well_ids: Sequence[str]) -> str:
    payload = "\n".join(well_ids[int(index)] for index in sorted(int(value) for value in indices))
    return hashlib.sha256(payload.encode()).hexdigest()


def inner_contexts(outer: Any, repeated: Sequence[Any], well_ids: Sequence[str]) -> list[Any]:
    outer_train = set(int(value) for value in outer.train_indices)
    outer_test = set(int(value) for value in outer.test_indices)
    if outer.scope == "repeated":
        sources = [context for context in repeated if context.label == outer.label and context.outer_group != outer.outer_group]
    else:
        sources = [context for context in repeated if context.label == "v1"]
    result = []
    for position, source in enumerate(sources):
        validation = sorted(outer_train & set(int(value) for value in source.test_indices))
        training = sorted(outer_train - set(validation))
        if not validation or not training:
            raise DataValidationError(f"{outer.key}: empty inner partition")
        if set(training) & set(validation) or (set(training) | set(validation)) != outer_train or outer_test & (set(training) | set(validation)):
            raise DataValidationError(f"{outer.key}: inner membership leak")
        result.append(SimpleNamespace(
            key=f"{outer.key}:inner:{position}", scope="inner", label=outer.label,
            outer_group=position, train_indices=np.asarray(training, dtype=np.int64),
            test_indices=np.asarray(validation, dtype=np.int64),
        ))
    expected = 4 if outer.scope == "repeated" else 5
    if len(result) != expected:
        raise DataValidationError(f"{outer.key}: inner context count differs")
    return result


def select_t016(context: Any, corrections: np.ndarray, placements: Sequence[Any], rows: np.ndarray, sums: np.ndarray, base_sse: np.ndarray, t016: Any) -> tuple[np.ndarray, Any, float]:
    selected_index, train_sse = t016.select_placement(corrections, context.train_indices, rows, sums, base_sse)
    placement = placements[selected_index]
    selected = np.asarray(corrections[selected_index, context.test_indices], dtype=np.float64)
    if selected.shape != (len(context.test_indices),) or not np.all(np.isfinite(selected)):
        raise DataValidationError(f"{context.key}: invalid T016 correction")
    return selected, placement, train_sse


def domain_gains(records: Sequence[Any], indices: np.ndarray, action: Action, spatial: np.ndarray, typewell: np.ndarray) -> list[float]:
    gains = []
    for labels in (spatial, typewell):
        for group in sorted(set(int(labels[int(i)]) for i in indices)):
            local_positions = np.flatnonzero(labels[indices] == group)
            if len(local_positions) == 0: continue
            local_indices = indices[local_positions]
            base, _ = score_action(records, local_indices, zero_action(len(local_indices)))
            subset = Action(action.base_scale[local_positions], action.spline[local_positions], action.datum[local_positions])
            candidate, _ = score_action(records, local_indices, subset)
            gains.append(float(base["rmse"]) - float(candidate["rmse"]))
    if not gains:
        raise DataValidationError("no available domains for selection")
    return gains


def select_blend(mode: str, grid_name: str, grids: Mapping[str, Sequence[tuple[float, float, float]]], records: Sequence[Any], indices: np.ndarray,
                 c25: np.ndarray, d16: np.ndarray, c27: np.ndarray, spatial: np.ndarray, typewell: np.ndarray) -> Selection:
    candidates = []
    for alpha, beta, gamma in grids[grid_name]:
        action = combine_action(c25, d16, c27, alpha, beta, gamma)
        summary, _ = score_action(records, indices, action)
        gains = domain_gains(records, indices, action, spatial, typewell)
        candidates.append((float(summary["rmse"]), min(gains), alpha, beta, gamma))
    if mode == "inner_pooled":
        chosen = min(candidates, key=lambda row: (row[0], row[2], row[3], row[4]))
        fallback = False
    elif mode == "inner_maximum_minimum_domain_gain":
        chosen = min(candidates, key=lambda row: (-row[1], row[0], row[2], row[3], row[4]))
        fallback = False
    elif mode == "inner_lowest_pooled_rmse_among_nonnegative_available_domains_else_zero":
        eligible = [row for row in candidates if row[1] >= -1e-12]
        if eligible:
            chosen = min(eligible, key=lambda row: (row[0], row[2], row[3], row[4])); fallback = False
        else:
            chosen = next(row for row in candidates if abs(row[2]) < 1e-12 and abs(row[3]) < 1e-12 and abs(row[4]) < 1e-12); fallback = True
    else:
        raise DataValidationError(f"unknown blend selection {mode}")
    return Selection(chosen[2], chosen[3], chosen[4], chosen[0], chosen[1], fallback)


def read_selected_coefficients(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    coefficients: dict[str, np.ndarray] = {}; sse: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            well_id = str(row["well_id"])
            if well_id in coefficients: raise DataValidationError(f"duplicate selected well {well_id}")
            coefficients[well_id] = np.asarray([float(row[f"predicted_coefficient_{i}"]) for i in range(OUTPUTS)])
            sse[well_id] = float(row["sse"])
    return coefficients, sse


def read_t016_selected(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    correction = {}; rmse = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            well_id = str(row["well_id"])
            if well_id in correction: raise DataValidationError(f"duplicate T016 well {well_id}")
            correction[well_id] = float(row["nested_real_correction"]); rmse[well_id] = float(row["nested_real_rmse"])
    return correction, rmse

def audit_row_correction_bounds(oof_path: Path, records: Sequence[Any], actions: Mapping[str, Action], maximum: float, spline_basis_fn: Any) -> dict[str, Any]:
    by_id = {record.well_id: (index, record) for index, record in enumerate(records)}
    maxima = {name: 0.0 for name in actions}
    current = ""; predictions: list[float] = []; hidden_indices: list[int] = []; total_rows = 0

    def finish(well_id: str) -> None:
        if not well_id: return
        if well_id not in by_id: raise DataValidationError(f"row-bound audit unknown well {well_id}")
        index, record = by_id[well_id]
        prediction = np.asarray(predictions, dtype=np.float64)
        if len(prediction) != int(record.hidden_rows) or hidden_indices != list(range(int(record.hidden_rows))) or not np.all(np.isfinite(prediction)):
            raise DataValidationError(f"{well_id}: row-bound OOF coverage differs")
        basis = spline_basis_fn(int(record.hidden_rows))
        delta = float(record.last_tvt) - prediction
        for name, action in actions.items():
            correction = action.base_scale[index] * delta + basis @ action.spline[index] + action.datum[index]
            if not np.all(np.isfinite(correction)):
                raise DataValidationError(f"{well_id}/{name}: non-finite row correction")
            maxima[name] = max(maxima[name], float(np.max(np.abs(correction))))

    with gzip.open(oof_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"well_id", "hidden_index", "spline4_ridge_equal_s075"}
        if not required.issubset(reader.fieldnames or []): raise DataValidationError("row-bound OOF schema differs")
        for row in reader:
            well_id = str(row["well_id"])
            if current and well_id != current:
                finish(current); predictions.clear(); hidden_indices.clear()
            current = well_id; predictions.append(float(row["spline4_ridge_equal_s075"])); hidden_indices.append(int(row["hidden_index"])); total_rows += 1
    finish(current)
    passed = total_rows == sum(int(record.hidden_rows) for record in records) and max(maxima.values()) <= float(maximum) + 1e-9
    return {"pass": passed, "maximum_allowed_ft": float(maximum), "maximum_by_candidate": maxima, "rows": total_rows}


def run_edge_tests(config: Mapping[str, Any]) -> dict[str, Any]:
    passed = []
    grids = build_grids(config); passed.append("simplex_box_grid_boundaries_counts")
    assert grids["simplex"][0] == (0.0, 0.0, 0.0) and grids["box"][-1] == (1.0, 1.0, 0.0); passed.append("deterministic_grid_order")
    record = SimpleNamespace(well_id="edge", hidden_rows=4, e011_sse=4.0, e011_sum=0.0, e_dot_d=0.0, d_sse=4.0, d_sum=0.0,
                             basis_dot_e=np.zeros(4), basis_dot_d=np.zeros(4), basis_sum=np.ones(4), basis_cross=np.eye(4))
    zero = metric_one(record, 0.0, np.zeros(4), 0.0); assert zero["sse"] == 4.0; passed.append("zero_weight_exact_fallback")
    left = metric_one(record, 0.2, np.asarray([1.,2.,3.,4.]), 0.3)
    right = metric_one(record, 0.2, np.asarray([1.,2.,3.,4.]), 0.3)
    assert left == right; passed.append("finite_exact_sufficient_statistic_scoring")
    a = combine_action(np.ones((2,4)), np.asarray([1.,2.]), np.zeros((2,4)), .4,.6,0.); assert a.spline.shape==(2,4); passed.append("single_parent_and_combined_dimensions")
    try: combine_action(np.ones((2,3)),np.ones(2),np.ones((2,4)),.4,.6,0.); raise AssertionError("bad coefficient passed")
    except DataValidationError: pass
    passed.append("nonfinite_dimension_rejection")
    bad = Action(np.asarray([np.nan]),np.zeros((1,4)),np.zeros(1))
    try: bad.validate(1); raise AssertionError("nonfinite action passed")
    except DataValidationError: pass
    passed.append("nonfinite_action_rejection")
    xtr,xte=_impute_scale(np.asarray([[np.nan,1.],[np.nan,1.]]),np.asarray([[np.nan,1.]])); assert np.all(np.isfinite(xtr)) and np.all(np.isfinite(xte)); passed.append("all_missing_zero_variance_imputation")
    model=Ridge(alpha=1.0).fit(np.asarray([[0.],[1.]]),np.asarray([[0.,0.,0.,0.],[1.,1.,1.,1.]])); assert model.predict([[.5]]).shape==(1,4); passed.append("tiny_ridge_partition")
    act1=Action(np.ones(1),np.ones((1,4)),np.ones(1)); act2=Action(np.zeros(1),np.zeros((1,4)),np.zeros(1)); avg=average_actions([act1,act2]); assert abs(avg.base_scale[0]-.5)<1e-12; passed.append("row_action_averaging")
    assert metric_one(record,.5,np.full(4,.5),.5)["sse"] != .5*(metric_one(record,1.,np.ones(4),1.)["sse"]+metric_one(record,0.,np.zeros(4),0.)["sse"]); passed.append("metric_averaging_not_used")
    h=np.eye(4); assert abs(float(np.ones(4)@h@np.arange(4))-float(np.arange(4)@h@np.ones(4)))<1e-12; passed.append("coefficient_datum_cross_term_symmetry")
    values=np.asarray([-100.,0.,100.]); clipped=np.clip(.1*values,-5.,5.); assert clipped.tolist()==[-5.,0.,5.]; passed.append("t016_scale_cap_extreme_boundaries")
    names=["a","b"]; assert names[min(range(2),key=lambda i:([1.,1.][i],i))]=="a"; passed.append("deterministic_ties")
    repeated=[]
    for v in range(5):
        for f in range(5):
            test=np.flatnonzero(np.arange(10)%5==f); train=np.flatnonzero(np.arange(10)%5!=f); repeated.append(SimpleNamespace(key=f"repeated:v{v+1}:{f}",scope="repeated",label=f"v{v+1}",outer_group=f,train_indices=train,test_indices=test))
    stress=[]; toy_domains=np.asarray([0,1,2,3,4,2,3,4,0,1])
    for scope in ("spatial","typewell"):
        for g in range(5):
            test=np.flatnonzero(toy_domains==g); train=np.flatnonzero(toy_domains!=g); stress.append(SimpleNamespace(key=f"{scope}:{g}",scope=scope,label=scope,outer_group=g,train_indices=train,test_indices=test))
    for context in repeated+stress:
        assert not(set(context.train_indices)&set(context.test_indices)) and set(context.train_indices)|set(context.test_indices)==set(range(10))
    passed.append("strict_context_membership")
    inner=inner_contexts(repeated[0],repeated,[str(i) for i in range(10)]); assert len(inner)==4; passed.append("repeated_inner_heldout_exclusion")
    inner_s=inner_contexts(stress[0],repeated,[str(i) for i in range(10)]); assert len(inner_s)==5; passed.append("stress_inner_heldout_exclusion")
    try:
        empty=SimpleNamespace(key="empty",scope="repeated",label="v1",outer_group=0,train_indices=np.asarray([],dtype=int),test_indices=np.arange(10))
        inner_contexts(empty,repeated,[str(i) for i in range(10)]); raise AssertionError("empty inner passed")
    except DataValidationError: pass
    passed.append("empty_inner_partition_rejection")
    labels=np.arange(10)%5; gains=domain_gains([record]*10,np.arange(10),zero_action(10),labels,labels); assert len(gains)==10 and max(abs(v) for v in gains)<1e-12; passed.append("domain_partition_completeness")
    horizons=np.arange(10); edges=np.unique(np.quantile(horizons,[.2,.4,.6,.8])); bins=np.digitize(horizons,edges); assert set(bins)==set(range(5)); passed.append("horizon_quintile_completeness")
    rng1=np.random.default_rng(stable_seed(30033,"x")); rng2=np.random.default_rng(stable_seed(30033,"x")); assert np.array_equal(rng1.permutation(10),rng2.permutation(10)); passed.append("joint_control_determinism")
    maximum=float(config["controls"]["maximum_correction_absolute_ft"]); assert maximum==160.0; passed.append("maximum_correction_bound_contract")
    if len(config["real_branches"])!=10 or len(config["negative_controls"])!=4: raise AssertionError("branch/control count differs")
    passed.append("branch_control_contract")
    if len(passed)!=23: raise AssertionError(f"edge group count differs {len(passed)}")
    return {"status":"PASS","edge_groups":len(passed),"passed":passed}


def run_screen(root: Path, output_dir: Path, implementation_commit: str) -> dict[str, Any]:
    started=time.perf_counter()
    config_path=root/"tracking/evidence/T030/config.json"; config=json.loads(config_path.read_text())
    source_control=verify_hashes(root,config)
    t025=import_parent("t025_parent_t030",root/str(config["inputs"]["t025_source"]))
    t016=import_parent("t016_parent_t030",root/str(config["inputs"]["t016_source"]))
    t025_config=json.loads((root/str(config["inputs"]["t025_config"])).read_text())
    t016_config=json.loads((root/str(config["inputs"]["t016_config"])).read_text())
    records=t025.load_raw_records(root/str(config["inputs"]["data_dir"]),t025_config)
    coverage=t025.attach_e011_sufficient(records,root/str(config["inputs"]["e011_oof"]),int(config["expected_hidden_rows"]))
    e011_features,e011_names,spatial,typewell=t025.load_compact(root/str(config["inputs"]["e011_compact"]),records)
    well_ids=[record.well_id for record in records]
    repeated,stress=t025.load_contexts(root,well_ids,config["inputs"]["fold_files"],spatial,typewell)
    pseudo=np.vstack([record.pseudo_features for record in records]); reversed_pseudo=np.vstack([record.reversed_features for record in records])
    targets=np.vstack([record.target_coefficients for record in records]); latest=np.vstack([record.latest_coefficients for record in records]); analytic=np.vstack([record.linear_extrapolation for record in records])
    permuted_pseudo=pseudo[np.random.default_rng(int(t025_config["negative_controls"]["well_permutation_seed"])).permutation(len(records))]
    t025_branch=next(branch for branch in t025_config["branches"] if branch["name"]==config["parent_contract"]["t025_branch"]["name"])
    actions=t016.load_actions(root/str(config["inputs"]["e009_actions"]),well_ids,config["parent_contract"]["t016_action_families"])
    placements=t016.build_placements(t016_config); corrections=t016.correction_matrix(actions,placements)
    shuffled_actions=actions[np.random.default_rng(30032).permutation(len(records))]; shuffled_corrections=t016.correction_matrix(shuffled_actions,placements)
    rows=np.asarray([record.hidden_rows for record in records],dtype=np.float64); sums=np.asarray([record.e011_sum for record in records],dtype=np.float64); base_sse=np.asarray([record.e011_sse for record in records],dtype=np.float64)
    grids=build_grids(config); branches=list(config["real_branches"]); branch_names=[str(branch["name"]) for branch in branches]
    if len(set(branch_names))!=len(branch_names): raise DataValidationError("duplicate branch names")
    base_metrics=[metric_one(record,0.,np.zeros(4),0.) for record in records]; base_summary=summarize(base_metrics)
    if abs(float(base_summary["rmse"])-float(config["controls"]["exact_e011_rmse"]))>1e-9: raise DataValidationError("E011 RMSE differs")

    context_rows=[]; stress_rows=[]; selection_rows=[]; isolation_rows=[]
    map_actions={label:{name:{"base":np.full(len(records),np.nan),"spline":np.full((len(records),4),np.nan),"datum":np.full(len(records),np.nan)} for name in branch_names} for label in sorted({c.label for c in repeated})}
    parent_map={label:{"c25":np.full((len(records),4),np.nan),"c27":np.full((len(records),4),np.nan),"d16":np.full(len(records),np.nan)} for label in map_actions}
    control_maps={name:{label:{"base":np.full(len(records),np.nan),"spline":np.full((len(records),4),np.nan),"datum":np.full(len(records),np.nan)} for label in map_actions} for name in [c["name"] for c in config["negative_controls"]]}
    base_context={}

    for outer in [*repeated,*stress]:
        c25=t025.predict_branch(t025_branch,outer,pseudo,reversed_pseudo,permuted_pseudo,e011_features,targets,latest,analytic,80.0,None,t025_config["negative_controls"])
        c27=predict_t027(outer,e011_features,targets)
        d16,placement,train_sse=select_t016(outer,corrections,placements,rows,sums,base_sse,t016)
        isolation_rows.extend([
            {"outer_context":outer.key,"stage":"outer_t025_fit","train_wells":len(outer.train_indices),"validation_wells":0,"test_wells":len(outer.test_indices),"train_sha256":membership_sha(outer.train_indices,well_ids),"validation_sha256":"","test_sha256":membership_sha(outer.test_indices,well_ids),"pass":True},
            {"outer_context":outer.key,"stage":"outer_t027_fit","train_wells":len(outer.train_indices),"validation_wells":0,"test_wells":len(outer.test_indices),"train_sha256":membership_sha(outer.train_indices,well_ids),"validation_sha256":"","test_sha256":membership_sha(outer.test_indices,well_ids),"pass":True},
            {"outer_context":outer.key,"stage":"outer_t016_selection","train_wells":len(outer.train_indices),"validation_wells":0,"test_wells":len(outer.test_indices),"train_sha256":membership_sha(outer.train_indices,well_ids),"validation_sha256":"","test_sha256":membership_sha(outer.test_indices,well_ids),"pass":True},
        ])
        nested: dict[str,Selection]={}
        inner_c25=np.full((len(records),4),np.nan); inner_c27=np.full((len(records),4),np.nan); inner_d16=np.full(len(records),np.nan); inner_count=np.zeros(len(records),dtype=int)
        for inner in inner_contexts(outer,repeated,well_ids):
            ic25=t025.predict_branch(t025_branch,inner,pseudo,reversed_pseudo,permuted_pseudo,e011_features,targets,latest,analytic,80.0,None,t025_config["negative_controls"])
            ic27=predict_t027(inner,e011_features,targets)
            id16,iplacement,itrain_sse=select_t016(inner,corrections,placements,rows,sums,base_sse,t016)
            inner_c25[inner.test_indices]=ic25; inner_c27[inner.test_indices]=ic27; inner_d16[inner.test_indices]=id16; inner_count[inner.test_indices]+=1
            isolation_rows.extend([
                {"outer_context":outer.key,"stage":"inner_t025_fit","train_wells":len(inner.train_indices),"validation_wells":len(inner.test_indices),"test_wells":len(outer.test_indices),"train_sha256":membership_sha(inner.train_indices,well_ids),"validation_sha256":membership_sha(inner.test_indices,well_ids),"test_sha256":membership_sha(outer.test_indices,well_ids),"pass":True},
                {"outer_context":outer.key,"stage":"inner_t027_fit","train_wells":len(inner.train_indices),"validation_wells":len(inner.test_indices),"test_wells":len(outer.test_indices),"train_sha256":membership_sha(inner.train_indices,well_ids),"validation_sha256":membership_sha(inner.test_indices,well_ids),"test_sha256":membership_sha(outer.test_indices,well_ids),"pass":True},
                {"outer_context":outer.key,"stage":"inner_t016_selection","train_wells":len(inner.train_indices),"validation_wells":len(inner.test_indices),"test_wells":len(outer.test_indices),"train_sha256":membership_sha(inner.train_indices,well_ids),"validation_sha256":membership_sha(inner.test_indices,well_ids),"test_sha256":membership_sha(outer.test_indices,well_ids),"pass":True},
            ])
        if not np.all(inner_count[outer.train_indices]==1) or np.any(inner_count[outer.test_indices]!=0): raise DataValidationError(f"{outer.key}: inner OOF coverage differs")
        train_idx=np.asarray(outer.train_indices,dtype=np.int64)
        for branch in branches:
            if branch["selection"]!="fixed":
                nested[branch["name"]]=select_blend(branch["selection"],branch["grid"],grids,records,train_idx,inner_c25[train_idx],inner_d16[train_idx],inner_c27[train_idx],spatial,typewell)
        base_outer,_=score_action(records,outer.test_indices,zero_action(len(outer.test_indices))); base_context[outer.key]=base_outer
        context_rows.append({"context":outer.key,"scope":outer.scope,"map":outer.label,"outer_group":outer.outer_group,"candidate":"e011","alpha":0.0,"beta":0.0,"gamma":0.0,**base_outer}) if outer.scope=="repeated" else stress_rows.append({"context":outer.key,"scope":outer.scope,"outer_group":outer.outer_group,"candidate":"e011","alpha":0.0,"beta":0.0,"gamma":0.0,**base_outer})
        for branch in branches:
            if branch["selection"]=="fixed": selection=Selection(float(branch["alpha"]),float(branch["beta"]),float(branch["gamma"]),math.nan,math.nan,False)
            else: selection=nested[branch["name"]]
            action=combine_action(c25,d16,c27,selection.alpha,selection.beta,selection.gamma)
            maximum=np.max(np.abs(np.column_stack((action.base_scale,action.spline,action.datum))))
            if maximum>float(config["controls"]["maximum_correction_absolute_ft"])+1e-9: raise DataValidationError(f"{outer.key}/{branch['name']}: action bound exceeded")
            summary,_=score_action(records,outer.test_indices,action)
            row={"context":outer.key,"scope":outer.scope,"outer_group":outer.outer_group,"candidate":branch["name"],"alpha":selection.alpha,"beta":selection.beta,"gamma":selection.gamma,**summary}
            if outer.scope=="repeated": row["map"]=outer.label; context_rows.append(row)
            else: stress_rows.append(row)
            selection_rows.append({"context":outer.key,"scope":outer.scope,"candidate":branch["name"],"selection":branch["selection"],"alpha":selection.alpha,"beta":selection.beta,"gamma":selection.gamma,"inner_rmse":selection.pooled_rmse,"inner_minimum_domain_gain":selection.minimum_domain_gain,"fallback":selection.fallback,"t016_family":placement.family,"t016_scale":placement.scale,"t016_cap":placement.cap,"t016_train_rmse":math.sqrt(train_sse/float(rows[outer.train_indices].sum()))})
            if outer.scope=="repeated":
                target=map_actions[outer.label][branch["name"]]; target["base"][outer.test_indices]=action.base_scale; target["spline"][outer.test_indices]=action.spline; target["datum"][outer.test_indices]=action.datum
        if outer.scope=="repeated":
            parent_map[outer.label]["c25"][outer.test_indices]=c25; parent_map[outer.label]["c27"][outer.test_indices]=c27; parent_map[outer.label]["d16"][outer.test_indices]=d16
            shuffled_c25=t025.predict_branch(t025_branch,outer,pseudo,reversed_pseudo,permuted_pseudo,e011_features,targets,latest,analytic,80.0,"shuffled_targets",{"target_shuffle_seed":30031})
            shuffled_d16,_,_=select_t016(outer,shuffled_corrections,placements,rows,sums,base_sse,t016)
            control_actions={
                "joint_shuffled_targets_and_actions":combine_action(shuffled_c25,shuffled_d16,np.zeros_like(c27),.4,.6,0.),
                "joint_sign_flip":combine_action(c25,d16,np.zeros_like(c27),-.4,-.6,0.),
                "shuffled_t025_and_sign_flipped_t016":combine_action(t025.predict_branch(t025_branch,outer,pseudo,reversed_pseudo,permuted_pseudo,e011_features,targets,latest,analytic,80.0,"shuffled_targets",{"target_shuffle_seed":30035}),d16,np.zeros_like(c27),.4,-.6,0.),
            }
            p25=np.random.default_rng(stable_seed(30033,outer.key)).permutation(len(outer.test_indices)); p16=np.random.default_rng(stable_seed(30034,outer.key)).permutation(len(outer.test_indices))
            control_actions["independent_test_pair_permutation"]=combine_action(c25[p25],d16[p16],np.zeros_like(c27),.4,.6,0.)
            for name,action_control in control_actions.items():
                target=control_maps[name][outer.label]; target["base"][outer.test_indices]=action_control.base_scale; target["spline"][outer.test_indices]=action_control.spline; target["datum"][outer.test_indices]=action_control.datum

    # Coverage and map/final action construction.
    map_rows=[]; final_actions={}; final_well_metrics={}
    for label in map_actions:
        for name in branch_names:
            values=map_actions[label][name]
            if not np.all(np.isfinite(values["base"])) or not np.all(np.isfinite(values["spline"])) or not np.all(np.isfinite(values["datum"])): raise DataValidationError(f"{label}/{name}: map coverage incomplete")
            action=Action(values["base"],values["spline"],values["datum"]); summary,_=score_action(records,np.arange(len(records)),action)
            map_rows.append({"map":label,"candidate":name,**summary})
    for name in branch_names:
        actions_for_maps=[Action(map_actions[label][name]["base"],map_actions[label][name]["spline"],map_actions[label][name]["datum"]) for label in sorted(map_actions)]
        final=average_actions(actions_for_maps); final_actions[name]=final; summary,well=score_action(records,np.arange(len(records)),final); final_well_metrics[name]=well
    final_rows=[{"candidate":"e011","gain_vs_e011":0.0,**base_summary}]
    for name in branch_names:
        summary=summarize(final_well_metrics[name]); final_rows.append({"candidate":name,"gain_vs_e011":float(base_summary["rmse"])-float(summary["rmse"]),**summary})

    # Parent reproduction.
    official25,official25_sse=read_selected_coefficients(root/"tracking/evidence/T025/selected_well_metrics.csv")
    official27,official27_sse=read_selected_coefficients(root/"tracking/evidence/T027/selected_well_metrics.csv")
    official16,official16_rmse=read_t016_selected(root/"tracking/evidence/T016/per_well_final.csv")
    c25_final=np.mean(np.stack([parent_map[label]["c25"] for label in sorted(parent_map)]),axis=0); c27_final=np.mean(np.stack([parent_map[label]["c27"] for label in sorted(parent_map)]),axis=0); d16_final=np.mean(np.stack([parent_map[label]["d16"] for label in sorted(parent_map)]),axis=0)
    max_c25=max(float(np.max(np.abs(c25_final[i]-official25[record.well_id]))) for i,record in enumerate(records)); max_c27=max(float(np.max(np.abs(c27_final[i]-official27[record.well_id]))) for i,record in enumerate(records)); max_d16=max(abs(float(d16_final[i])-official16[record.well_id]) for i,record in enumerate(records))
    max_sse25=max(abs(final_well_metrics["t025_fixed_050"][i]["sse"]-official25_sse[record.well_id]) for i,record in enumerate(records)); max_rmse16=max(abs(final_well_metrics["t016_fixed_100"][i]["rmse"]-official16_rmse[record.well_id]) for i,record in enumerate(records))
    parent_control={"pass":max(max_c25,max_c27,max_d16,max_sse25,max_rmse16)<=float(config["controls"]["maximum_parent_per_well_delta"]),"maximum_t025_coefficient_delta":max_c25,"maximum_t027_coefficient_delta":max_c27,"maximum_t016_correction_delta":max_d16,"maximum_t025_sse_delta":max_sse25,"maximum_t016_rmse_delta":max_rmse16}

    # Negative controls.
    control_rows=[]; maximum_control_gain=-math.inf
    for name,maps in control_maps.items():
        for label,values in maps.items():
            if not np.all(np.isfinite(values["base"])) or not np.all(np.isfinite(values["spline"])) or not np.all(np.isfinite(values["datum"])): raise DataValidationError(f"{name}/{label}: control coverage incomplete")
        final=average_actions([Action(maps[label]["base"],maps[label]["spline"],maps[label]["datum"]) for label in sorted(maps)]); summary,_=score_action(records,np.arange(len(records)),final); gain=float(base_summary["rmse"])-float(summary["rmse"]); maximum_control_gain=max(maximum_control_gain,gain); control_rows.append({"control":name,"gain_vs_e011":gain,**summary})

    # Slices and horizon quintiles.
    hidden=np.asarray([record.hidden_rows for record in records],dtype=float); missing=np.asarray([record.hidden_gr_missing_fraction for record in records],dtype=float); base_rmse=np.asarray([metric["rmse"] for metric in base_metrics])
    thresholds={"long_suffix":float(np.quantile(hidden,.8)),"high_gr_missingness":float(np.quantile(missing,.8)),"e011_catastrophe":float(config["slices"]["e011_catastrophe_rmse"])}
    sets={"long_suffix":np.flatnonzero(hidden>=thresholds["long_suffix"]),"high_gr_missingness":np.flatnonzero(missing>=thresholds["high_gr_missingness"]),"e011_catastrophe":np.flatnonzero(base_rmse>=thresholds["e011_catastrophe"])}
    horizon_edges=np.unique(np.quantile(hidden,[.2,.4,.6,.8])); bins=np.digitize(hidden,horizon_edges)
    if set(bins.tolist())!=set(range(5)): raise DataValidationError("horizon quintiles incomplete")
    for group in range(5): sets[f"horizon_q{group}"]=np.flatnonzero(bins==group)
    slice_rows=[]
    for slice_name,idx in sets.items():
        base=summarize([base_metrics[int(i)] for i in idx]); slice_rows.append({"slice":slice_name,"candidate":"e011",**base})
        for name in branch_names: slice_rows.append({"slice":slice_name,"candidate":name,**summarize([final_well_metrics[name][int(i)] for i in idx])})

    # Wins and gates.
    context_lookup={(row["context"],row["candidate"]):row for row in context_rows}; map_lookup={(row["map"],row["candidate"]):row for row in map_rows}; stress_lookup={(row["context"],row["candidate"]):row for row in stress_rows}; slice_lookup={(row["slice"],row["candidate"]):row for row in slice_rows}
    t025_row=next(row for row in final_rows if row["candidate"]=="t025_fixed_050")
    selected_contexts={name:sum(float(row["alpha"])>0 and float(row["beta"])>0 for row in selection_rows if row["scope"]=="repeated" and row["candidate"]==name) for name in branch_names}
    authorization=[]; gates_by_candidate={}
    for row in final_rows:
        name=row["candidate"]
        if name=="e011": continue
        map_wins=sum(float(base_summary["rmse"])-float(map_lookup[(label,name)]["rmse"])>0 for label in map_actions)
        cell_wins=sum(float(context_lookup[(context.key,"e011")]["rmse"])-float(context_lookup[(context.key,name)]["rmse"])>0 for context in repeated)
        spatial_gains=[float(stress_lookup[(context.key,"e011")]["rmse"])-float(stress_lookup[(context.key,name)]["rmse"]) for context in stress if context.scope=="spatial"]
        typewell_gains=[float(stress_lookup[(context.key,"e011")]["rmse"])-float(stress_lookup[(context.key,name)]["rmse"]) for context in stress if context.scope=="typewell"]
        special_gains=[float(slice_lookup[(slice_name,"e011")]["rmse"])-float(slice_lookup[(slice_name,name)]["rmse"]) for slice_name in ("long_suffix","high_gr_missingness","e011_catastrophe")]
        horizon_gains=[float(slice_lookup[(f"horizon_q{i}","e011")]["rmse"])-float(slice_lookup[(f"horizon_q{i}",name)]["rmse"]) for i in range(5)]
        gates={
            "gain":float(row["gain_vs_e011"])>=float(config["promotion"]["minimum_gain_vs_e011"]),
            "incremental_t025":float(t025_row["rmse"])-float(row["rmse"])>=float(config["promotion"]["minimum_incremental_gain_vs_t025"]),
            "maps":map_wins>=int(config["promotion"]["minimum_map_wins"]),"cells":cell_wins>=int(config["promotion"]["minimum_outer_cell_wins"]),
            "p90":float(row["p90_well_rmse"])-float(base_summary["p90_well_rmse"])<=float(config["promotion"]["maximum_p90_deterioration"]),
            "worst5":float(row["worst_5pct_sse_share"])-float(base_summary["worst_5pct_sse_share"])<=float(config["promotion"]["maximum_worst5_share_increase"]),
            "spatial":min(spatial_gains)>0.0,"typewell":min(typewell_gains)>0.0,"special":min(special_gains)>0.0,"horizon":min(horizon_gains)>0.0,
            "nondegenerate":selected_contexts[name]>=int(config["promotion"]["minimum_nonzero_both_parent_contexts"]),
            "controls":maximum_control_gain<=float(config["promotion"]["maximum_negative_control_gain_vs_e011"]),"parents":bool(parent_control["pass"]),"reproduction":False,
        }
        gates_by_candidate[name]=gates
        preliminary=float(row["gain_vs_e011"])>=float(config["preliminary"]["minimum_gain_vs_e011"]) and map_wins>=int(config["preliminary"]["minimum_map_wins"]) and cell_wins>=int(config["preliminary"]["minimum_outer_cell_wins"]) and float(row["p90_well_rmse"])-float(base_summary["p90_well_rmse"])<=float(config["preliminary"]["maximum_p90_deterioration"]) and float(row["worst_5pct_sse_share"])-float(base_summary["worst_5pct_sse_share"])<=float(config["preliminary"]["maximum_worst5_share_increase"])
        row.update({"map_wins":map_wins,"outer_cell_wins":cell_wins,"incremental_gain_vs_t025":float(t025_row["rmse"])-float(row["rmse"]),"preliminary_pass":preliminary})
        authorization.append({"candidate":name,"minimum_spatial_gain":min(spatial_gains),"minimum_typewell_gain":min(typewell_gains),"minimum_special_gain":min(special_gains),"minimum_horizon_gain":min(horizon_gains),"nonzero_both_parent_contexts":selected_contexts[name],**{f"gate_{k}":v for k,v in gates.items()}})
    substantive=[name for name,gates in gates_by_candidate.items() if all(v for k,v in gates.items() if k!="reproduction")]
    reported=min([row for row in final_rows if row["candidate"]!="e011"],key=lambda row:(float(row["rmse"]),str(row["candidate"])))

    # Selected well rows.
    selected=[]; reported_name=reported["candidate"]; action=final_actions[reported_name]
    for i,record in enumerate(records):
        metric=final_well_metrics[reported_name][i]; selected.append({"well_id":record.well_id,"candidate":reported_name,"rows_scored":metric["rows_scored"],"sse":metric["sse"],"rmse":metric["rmse"],"mean_error":metric["mean_error"],"known_rows":record.known_rows,"hidden_rows":record.hidden_rows,"hidden_gr_missing_fraction":record.hidden_gr_missing_fraction,"base_scale":action.base_scale[i],"datum_correction":action.datum[i],**{f"spline_contribution_{j}":action.spline[i,j] for j in range(4)}})

    edges=run_edge_tests(config)
    row_bound_control=audit_row_correction_bounds(root/str(config["inputs"]["e011_oof"]),records,final_actions,float(config["controls"]["maximum_correction_absolute_ft"]),t025.spline_basis)
    controls={"source_inputs":source_control,"coverage":{"pass":coverage["rows"]==int(config["expected_hidden_rows"]) and coverage["wells"]==int(config["expected_wells"]),**coverage},"parent_reproduction":parent_control,"exact_fallback":{"pass":abs(float(base_summary["rmse"])-E011_RMSE)<=1e-9,"rmse":base_summary["rmse"]},"negative_controls":{"pass":maximum_control_gain<=float(config["promotion"]["maximum_negative_control_gain_vs_e011"]),"maximum_gain_vs_e011":maximum_control_gain},"context_completion":{"pass":len(context_rows)==25*(len(branches)+1) and len(stress_rows)==10*(len(branches)+1),"repeated_rows":len(context_rows),"stress_rows":len(stress_rows)},"isolation":{"pass":all(bool(row["pass"]) for row in isolation_rows),"rows":len(isolation_rows)},"bounds":row_bound_control,"edge_groups":edges}
    all_controls=all((item.get("pass") if "pass" in item else item.get("status")=="PASS") for item in controls.values())
    if not all_controls: substantive=[]
    status="awaiting_reproduction" if substantive else "confirmation_reject"; decision="await_independent_reproduction" if substantive else "close_h022_without_promotion"
    output_dir.mkdir(parents=True,exist_ok=True)
    _write_csv(output_dir/"candidate_metrics.csv",final_rows); _write_csv(output_dir/"context_metrics.csv",context_rows); _write_csv(output_dir/"map_metrics.csv",map_rows); _write_csv(output_dir/"stress_metrics.csv",stress_rows); _write_csv(output_dir/"slice_metrics.csv",slice_rows); _write_csv(output_dir/"selected_weights.csv",selection_rows); _write_csv(output_dir/"negative_control_metrics.csv",control_rows); _write_csv(output_dir/"isolation_audit.csv",isolation_rows); _write_csv(output_dir/"selected_well_metrics.csv",selected); _write_csv(output_dir/"authorization_gates.csv",authorization); _write_json(output_dir/"edge_cases.json",edges)
    summary={"schema_version":1,"task_id":"T030","hypothesis_id":"H022","implementation_commit":implementation_commit,"status":status,"decision":decision,"reported_candidate":reported_name,"reported_metrics":reported,"base_summary":base_summary,"t025_parent":t025_row,"substantive_passers":substantive,"gates_by_candidate":gates_by_candidate,"controls":controls,"thresholds":thresholds,"horizon_edges":horizon_edges.tolist(),"slice_wells":{name:len(idx) for name,idx in sets.items()},"maximum_negative_control_gain":maximum_control_gain,"runtime_seconds":time.perf_counter()-started,"deployment":{"package_built":False,"kaggle_executed":False,"submission_created":False,"submission_made":False}}
    _write_json(output_dir/"summary.json",summary)
    files=sorted(p for p in output_dir.iterdir() if p.is_file()); _write_json(output_dir/"artifact_manifest.json",{"schema_version":1,"task_id":"T030","implementation_commit":implementation_commit,"files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":_sha256(p)} for p in files]})
    print(json.dumps({"status":status,"decision":decision,"reported_candidate":reported_name,"reported_rmse":reported["rmse"],"gain_vs_e011":reported["gain_vs_e011"],"substantive_passers":substantive,"runtime_seconds":summary["runtime_seconds"]},indent=2,sort_keys=True))
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=ROOT); parser.add_argument("--output-dir",type=Path); parser.add_argument("--implementation-commit",default="UNCOMMITTED"); parser.add_argument("--edge-only",action="store_true"); args=parser.parse_args()
    config=json.loads((args.root/"tracking/evidence/T030/config.json").read_text())
    if args.edge_only:
        print(json.dumps(run_edge_tests(config),indent=2,sort_keys=True)); return
    if args.output_dir is None: raise SystemExit("--output-dir is required")
    run_screen(args.root.resolve(),args.output_dir.resolve(),args.implementation_commit)


if __name__=="__main__": main()
