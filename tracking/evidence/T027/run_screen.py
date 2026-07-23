#!/usr/bin/env python3
"""T027: target-preserving causal feature-view worth screen."""
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
PLACEMENTS = (0.25, 0.50, 0.75, 1.00)
CONTROL_MODES = (
    "permuted_source_targets",
    "shuffled_task_targets",
    "permuted_view_indicators",
    "permuted_registered_blocks",
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
        values = (
            self.e011_sse, self.e011_sum, self.e_dot_d, self.d_sse, self.d_sum,
            self.basis_dot_e, self.basis_dot_d, self.basis_sum, self.basis_cross,
        )
        if any(value is None for value in values):
            raise DataValidationError(f"{self.well_id}: OOF sufficient statistics incomplete")
        linear = float(self.e_dot_d) + float(np.dot(self.basis_dot_e, c))
        quadratic = float(self.d_sse) + 2.0 * float(np.dot(self.basis_dot_d, c)) + float(c @ self.basis_cross @ c)
        sse = float(self.e011_sse) + 2.0 * w * linear + w * w * quadratic
        tolerance = 1e-8 * max(1.0, abs(float(self.e011_sse)), abs(2.0*w*linear), abs(w*w*quadratic))
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
class ViewPool:
    feature_names: list[str]
    view_names: list[str]
    features_with_indicators: np.ndarray
    features_without_indicators: np.ndarray
    targets: np.ndarray
    original_view_index: int
    registered_indices: np.ndarray
    mask_indices: dict[str, np.ndarray]


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


# ---------- spline target and raw records ----------

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
        raise DataValidationError("spline basis rank deficient")
    return matrix


def solve_coefficients(matrix: np.ndarray, delta: np.ndarray, bound: float) -> tuple[np.ndarray, float]:
    x = np.asarray(matrix, dtype=np.float64); y = np.asarray(delta, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != OUTPUTS or y.shape != (x.shape[0],):
        raise DataValidationError("coefficient solve dimensions differ")
    if x.shape[0] < OUTPUTS or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise DataValidationError("coefficient solve support/finiteness differs")
    if np.linalg.matrix_rank(x) < OUTPUTS:
        raise DataValidationError("coefficient solve rank deficient")
    coefficients = np.clip(np.linalg.lstsq(x, y, rcond=None)[0], -bound, bound)
    residual = x @ coefficients - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    if coefficients.shape != (OUTPUTS,) or not np.all(np.isfinite(coefficients)) or not math.isfinite(rmse):
        raise DataValidationError("coefficient solve emitted non-finite values")
    return coefficients, rmse


def fit_segment(values: Sequence[float], baseline: float, bound: float) -> tuple[np.ndarray, float]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < OUTPUTS:
        raise DataValidationError("segment has insufficient support")
    if not math.isfinite(float(baseline)) or not np.all(np.isfinite(y)):
        raise DataValidationError("segment contains non-finite values")
    return solve_coefficients(spline_basis(len(y)), y - float(baseline), bound)


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
        raise DataValidationError(f"{well_id}: TVT_input requires prefix and suffix")
    known = int(np.flatnonzero(~finite)[0])
    if known <= 0 or finite[known:].any() or not finite[:known].all():
        raise DataValidationError(f"{well_id}: TVT_input not contiguous")
    if np.max(np.abs(arrays["TVT_input"][:known] - arrays["TVT"][:known])) > 1e-8:
        raise DataValidationError(f"{well_id}: visible TVT_input differs from truth")
    return arrays, known


def load_records(data_dir: Path, expected_wells: int, bound: float) -> list[WellRecord]:
    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if len(paths) != expected_wells:
        raise DataValidationError(f"expected {expected_wells} wells, found {len(paths)}")
    records: list[WellRecord] = []; seen: set[str] = set()
    for path in paths:
        well_id = path.name.split("__", 1)[0]
        if well_id in seen: raise DataValidationError(f"duplicate well {well_id}")
        seen.add(well_id)
        arrays, known = validate_horizontal(pd.read_csv(path), well_id)
        hidden = arrays["TVT"][known:]
        coefficients, _ = fit_segment(hidden, arrays["TVT"][known-1], bound)
        hidden_gr = arrays["GR"][known:]
        records.append(WellRecord(
            well_id=well_id,
            known_rows=known,
            hidden_rows=len(hidden),
            last_tvt=float(arrays["TVT"][known-1]),
            hidden_gr_missing_fraction=float(np.mean(~np.isfinite(hidden_gr))) if len(hidden_gr) else 1.0,
            target_coefficients=coefficients,
            raw_hidden_sum=float(hidden.sum()),
            raw_hidden_sum_sq=float(np.dot(hidden, hidden)),
        ))
    return records


# ---------- E011 scoring and compact features ----------

def attach_e011_sufficient(records: Sequence[WellRecord], oof_path: Path, expected_rows: int) -> dict[str, Any]:
    by_id = {record.well_id: record for record in records}
    current = ""; targets: list[float] = []; predictions: list[float] = []; hidden_indices: list[int] = []
    seen_ids: set[str] = set(); seen_wells: set[str] = set(); total_rows = 0
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
        if abs(float(y.sum()) - record.raw_hidden_sum) > sum_tolerance or abs(float(np.dot(y,y)) - record.raw_hidden_sum_sq) > square_tolerance:
            raise DataValidationError(f"{well_id}: OOF target differs from raw truth")
        basis = spline_basis(record.hidden_rows); error = pred-y; delta = record.last_tvt-pred
        record.e011_sse=float(np.dot(error,error)); record.e011_sum=float(error.sum()); record.e_dot_d=float(np.dot(error,delta))
        record.d_sse=float(np.dot(delta,delta)); record.d_sum=float(delta.sum()); record.basis_dot_e=basis.T@error
        record.basis_dot_d=basis.T@delta; record.basis_sum=basis.sum(axis=0); record.basis_cross=basis.T@basis
        seen_wells.add(well_id)
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
            current=well_id; targets.append(float(row["target"])); predictions.append(float(row["spline4_ridge_equal_s075"])); hidden_indices.append(int(row["hidden_index"])); total_rows+=1
    finalize(current)
    if total_rows != expected_rows or seen_wells != set(by_id): raise DataValidationError("OOF coverage differs")
    return {"rows":total_rows,"wells":len(seen_wells),"unique_ids":len(seen_ids)}


def load_compact(path: Path, records: Sequence[WellRecord], expected_features: int) -> tuple[np.ndarray,list[str],np.ndarray,np.ndarray]:
    arrays=np.load(path,allow_pickle=False)
    required={"well_ids","feature_names","features","spatial_assignment","typewell_assignment"}
    if not required.issubset(arrays.files): raise DataValidationError("compact schema differs")
    ids=[str(value) for value in arrays["well_ids"]]
    if ids != [record.well_id for record in records]: raise DataValidationError("compact well order differs")
    names=[str(value) for value in arrays["feature_names"]]; features=np.asarray(arrays["features"],dtype=np.float64)
    spatial=np.asarray(arrays["spatial_assignment"],dtype=np.int64); typewell=np.asarray(arrays["typewell_assignment"],dtype=np.int64)
    if len(names)!=expected_features or len(set(names))!=len(names) or names!=sorted(names) or features.shape!=(len(records),expected_features):
        raise DataValidationError("feature schema/order differs")
    if set(spatial.tolist())!=set(range(5)) or set(typewell.tolist())!=set(range(5)):
        raise DataValidationError("stress groups differ")
    return features,names,spatial,typewell


# ---------- views ----------

def registered_blocks(names: Sequence[str], config: Mapping[str,Any]) -> tuple[dict[str,np.ndarray],np.ndarray]:
    blocks: dict[str,np.ndarray] = {}
    for prefix in config["feature_contract"]["backtest_blocks"]:
        idx=np.asarray([i for i,name in enumerate(names) if name.startswith(str(prefix))],dtype=np.int64)
        if len(idx)!=5: raise DataValidationError(f"{prefix}: expected five features, found {len(idx)}")
        blocks[str(prefix)]=idx
    for suffix in config["feature_contract"]["window_suffixes"]:
        idx=np.asarray([i for i,name in enumerate(names) if name.endswith(str(suffix))],dtype=np.int64)
        if len(idx)!=4: raise DataValidationError(f"{suffix}: expected four features, found {len(idx)}")
        blocks[str(suffix)]=idx
    registered=np.unique(np.concatenate(list(blocks.values()))).astype(np.int64)
    if len(registered)!=27: raise DataValidationError(f"registered feature count differs: {len(registered)}")
    return blocks,registered


def build_view_pool(base: np.ndarray, names: list[str], targets: np.ndarray, config: Mapping[str,Any]) -> ViewPool:
    if targets.shape!=(len(base),OUTPUTS) or not np.all(np.isfinite(targets)):
        raise DataValidationError("target matrix differs")
    blocks,registered=registered_blocks(names,config)
    views=list(config["views"]); view_names=[str(view["name"]) for view in views]
    if len(view_names)!=10 or len(set(view_names))!=10 or view_names[0]!="original_full":
        raise DataValidationError("view catalog differs")
    without=[]; with_indicators=[]; masks={}
    for vi,view in enumerate(views):
        matrix=np.asarray(base,dtype=np.float64).copy(); indices=[]
        for key in [*view["mask_backtests"],*view["mask_windows"]]:
            if str(key) not in blocks: raise DataValidationError(f"unknown view block {key}")
            indices.extend(blocks[str(key)].tolist())
        idx=np.unique(np.asarray(indices,dtype=np.int64)) if indices else np.asarray([],dtype=np.int64)
        matrix[:,idx]=np.nan; masks[view_names[vi]]=idx
        indicators=np.zeros((len(base),len(views)),dtype=np.float64); indicators[:,vi]=1.0
        without.append(matrix); with_indicators.append(np.hstack((matrix,indicators)))
    return ViewPool(names,view_names,np.stack(with_indicators),np.stack(without),targets,0,registered,masks)


def prepare_features(train: np.ndarray,test: np.ndarray,scale: bool) -> tuple[np.ndarray,np.ndarray]:
    xtr=np.asarray(train,dtype=np.float64); xte=np.asarray(test,dtype=np.float64)
    if xtr.ndim!=2 or xte.ndim!=2 or xtr.shape[1]!=xte.shape[1] or len(xtr)==0:
        raise DataValidationError("feature dimensions differ")
    finite_train=np.where(np.isfinite(xtr),xtr,np.nan); med=np.zeros(xtr.shape[1],dtype=np.float64); valid_columns=np.any(np.isfinite(finite_train),axis=0)
    if np.any(valid_columns): med[valid_columns]=np.nanmedian(finite_train[:,valid_columns],axis=0)
    xtr=np.where(np.isfinite(xtr),xtr,med); xte=np.where(np.isfinite(xte),xte,med)
    if scale:
        mean=xtr.mean(axis=0); std=xtr.std(axis=0); std=np.where(std>1e-12,std,1.0)
        xtr=(xtr-mean)/std; xte=(xte-mean)/std
    if not np.all(np.isfinite(xtr)) or not np.all(np.isfinite(xte)):
        raise DataValidationError("prepared features non-finite")
    return xtr,xte


def well_equal_weights(source: np.ndarray,multiplier: np.ndarray|None=None) -> np.ndarray:
    source=np.asarray(source,dtype=np.int64); mult=np.ones(len(source),dtype=np.float64) if multiplier is None else np.asarray(multiplier,dtype=np.float64)
    if source.shape!=mult.shape or len(source)==0: raise DataValidationError("source-weight dimensions differ")
    weights=np.zeros(len(source),dtype=np.float64)
    for well in np.unique(source):
        idx=np.flatnonzero(source==well); total=float(mult[idx].sum())
        if total<=0: raise DataValidationError("source weights nonpositive")
        weights[idx]=mult[idx]/total
    if not np.all(np.isfinite(weights)) or np.any(weights<=0.0): raise DataValidationError("source weights invalid")
    return weights


def training_rows(pool: ViewPool, context: Context, regime: str, config: Mapping[str,Any], indicators: bool=True, original_boost: float=1.0) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,dict[str,Any]]:
    if regime not in config["regimes"]: raise DataValidationError(f"unknown regime {regime}")
    selected_names=[str(value) for value in config["regimes"][regime]]
    if not selected_names or "original_full" not in selected_names: raise DataValidationError(f"{regime}: original view missing")
    view_lookup={name:i for i,name in enumerate(pool.view_names)}
    if any(name not in view_lookup for name in selected_names): raise DataValidationError(f"{regime}: unknown view")
    matrices=pool.features_with_indicators if indicators else pool.features_without_indicators
    x=[]; y=[]; source=[]; view_ids=[]; multipliers=[]
    for name in selected_names:
        vi=view_lookup[name]
        x.append(matrices[vi,context.train_indices]); y.append(pool.targets[context.train_indices]); source.append(context.train_indices.copy()); view_ids.append(np.full(len(context.train_indices),vi,dtype=np.int64))
        multipliers.append(np.full(len(context.train_indices),float(original_boost) if name=="original_full" else 1.0,dtype=np.float64))
    xtr=np.vstack(x); ytr=np.vstack(y); source_arr=np.concatenate(source); view_arr=np.concatenate(view_ids); multiplier=np.concatenate(multipliers)
    if set(source_arr.tolist()) != set(context.train_indices.tolist()) or set(source_arr.tolist()) & set(context.test_indices.tolist()):
        raise DataValidationError(f"{context.key}: held-out view leaked")
    weights=well_equal_weights(source_arr,multiplier)
    totals=np.asarray([weights[source_arr==well].sum() for well in np.unique(source_arr)])
    detail={"tasks":len(xtr),"source_wells":len(np.unique(source_arr)),"views":len(selected_names),"minimum_source_weight":float(totals.min()),"maximum_source_weight":float(totals.max())}
    return xtr,ytr,source_arr,view_arr,weights,detail


# ---------- contexts ----------

def load_contexts(root: Path,well_ids: Sequence[str],fold_files: Sequence[str],spatial: np.ndarray,typewell: np.ndarray) -> tuple[list[Context],list[Context]]:
    id_to_index={well:i for i,well in enumerate(well_ids)}; repeated=[]
    for file_name in fold_files:
        payload=json.loads((root/file_name).read_text()); assignments=payload["assignments"]; version=str(payload["version"])
        if set(assignments)!=set(well_ids) or int(payload["n_folds"])!=5: raise DataValidationError(f"{file_name}: fold contract differs")
        for fold in range(5):
            test=np.asarray([id_to_index[w] for w in well_ids if int(assignments[w])==fold],dtype=np.int64)
            mask=np.ones(len(well_ids),dtype=bool); mask[test]=False; train=np.flatnonzero(mask).astype(np.int64)
            repeated.append(Context(f"repeated:{version}:{fold}","repeated",version,fold,train,test))
    stress=[]
    for scope,labels in (("spatial",spatial),("typewell",typewell)):
        for group in range(5):
            test=np.flatnonzero(labels==group).astype(np.int64); train=np.flatnonzero(labels!=group).astype(np.int64)
            stress.append(Context(f"{scope}:{group}",scope,scope,group,train,test))
    validate_contexts(repeated,stress,len(well_ids)); return repeated,stress


def validate_contexts(repeated: Sequence[Context],stress: Sequence[Context],wells: int) -> None:
    if len(repeated)!=25 or len(stress)!=10: raise DataValidationError("context count differs")
    coverage=np.zeros(wells,dtype=np.int64)
    for context in [*repeated,*stress]:
        train=set(context.train_indices.tolist()); test=set(context.test_indices.tolist())
        if train&test or train|test!=set(range(wells)) or not train or not test: raise DataValidationError(f"{context.key}: membership differs")
        if context.scope=="repeated": coverage[context.test_indices]+=1
    if not np.all(coverage==5): raise DataValidationError("repeated coverage differs")


# ---------- models and controls ----------

def fit_ridge(xtr: np.ndarray,ytr: np.ndarray,xte: np.ndarray,weights: np.ndarray,alpha: float) -> np.ndarray:
    a,b=prepare_features(xtr,xte,True); model=Ridge(alpha=float(alpha),fit_intercept=True); model.fit(a,ytr,sample_weight=weights); return np.asarray(model.predict(b),dtype=np.float64)


def custom_knn_predict(xtr: np.ndarray,ytr: np.ndarray,xte: np.ndarray,sample_weight: np.ndarray,neighbors: int) -> np.ndarray:
    n=min(int(neighbors),len(xtr))
    if n<=0: raise DataValidationError("KNN has no training rows")
    model=NearestNeighbors(n_neighbors=n).fit(xtr); distances,indices=model.kneighbors(xte); output=np.empty((len(xte),OUTPUTS),dtype=np.float64)
    for i in range(len(xte)):
        d=distances[i]; idx=indices[i]; zero=d<=1e-12
        weights=sample_weight[idx]*zero.astype(float) if zero.any() else sample_weight[idx]/np.maximum(d,1e-12)
        output[i]=np.average(ytr[idx],axis=0,weights=weights)
    return output


def permuted_source_targets(targets: np.ndarray,source: np.ndarray,seed: int,context_key: str) -> np.ndarray:
    wells=np.unique(source)
    if len(wells)<2: raise DataValidationError("source-target permutation requires at least two wells")
    shift=1 + stable_seed(seed,context_key,"source_targets") % (len(wells)-1); donors=np.roll(wells,int(shift))
    if np.any(wells==donors): raise DataValidationError("source-target permutation contains self donor")
    mapping={int(w):int(d) for w,d in zip(wells,donors)}; result=np.empty_like(targets)
    target_by_well={int(w):targets[np.flatnonzero(source==w)[0]].copy() for w in wells}
    for i,w in enumerate(source): result[i]=target_by_well[mapping[int(w)]]
    return result


def predict_branch(branch: Mapping[str,Any],context: Context,pool: ViewPool,config: Mapping[str,Any],control_mode: str|None=None,duplicate_original: bool=False) -> tuple[np.ndarray,dict[str,Any]]:
    name=str(branch["name"]); family=str(branch["family"]); indicators=bool(branch.get("indicators",True)); boost=float(branch.get("original_boost",1.0)); regime=str(branch["regime"])
    test_matrix=(pool.features_with_indicators if indicators else pool.features_without_indicators)[pool.original_view_index,context.test_indices]
    if duplicate_original:
        selected=["original_full"]*10; original_regimes=config["regimes"]; temp=dict(original_regimes); temp["__duplicate_original__"]=selected; local_config=dict(config); local_config["regimes"]=temp
        xtr,ytr,source,view_ids,weights,detail=training_rows(pool,context,"__duplicate_original__",local_config,indicators,1.0)
    else:
        xtr,ytr,source,view_ids,weights,detail=training_rows(pool,context,regime,config,indicators,boost)
    if control_mode=="permuted_source_targets": ytr=permuted_source_targets(ytr,source,27027,context.key)
    elif control_mode=="shuffled_task_targets":
        rng=np.random.default_rng(stable_seed(27028,context.key,name)); ytr=ytr[rng.permutation(len(ytr))]
    elif control_mode=="permuted_view_indicators":
        if indicators:
            rng=np.random.default_rng(stable_seed(27029,context.key,name)); start=len(pool.feature_names); xtr[:,start:]=xtr[rng.permutation(len(xtr)),start:]
    elif control_mode=="permuted_registered_blocks":
        rng=np.random.default_rng(stable_seed(27030,context.key,name))
        for vi in np.unique(view_ids):
            rows=np.flatnonzero(view_ids==vi); permutation=rng.permutation(rows)
            xtr[np.ix_(rows,pool.registered_indices)]=xtr[np.ix_(permutation,pool.registered_indices)]
    elif control_mode is not None: raise DataValidationError(f"unknown control {control_mode}")
    if not np.all(np.isfinite(ytr)): raise DataValidationError(f"{name}: non-finite targets")
    if family=="ridge": prediction=fit_ridge(xtr,ytr,test_matrix,weights,float(branch["alpha"]))
    elif family=="per_view_ensemble":
        predictions=[]; lookup={v:i for i,v in enumerate(pool.view_names)}
        for view_name in config["regimes"][regime]:
            vi=lookup[str(view_name)]; matrix=(pool.features_with_indicators if indicators else pool.features_without_indicators)
            local_x=matrix[vi,context.train_indices]; local_y=pool.targets[context.train_indices]; local_w=np.ones(len(local_y),dtype=np.float64)
            predictions.append(fit_ridge(local_x,local_y,test_matrix,local_w,float(branch["alpha"])))
        prediction=np.mean(np.stack(predictions),axis=0); detail["component_models"]=len(predictions)
    elif family=="extra_trees":
        a,b=prepare_features(xtr,test_matrix,False); model=ExtraTreesRegressor(n_estimators=64,min_samples_leaf=8,max_features=0.7,random_state=stable_seed(27027,context.key,name),n_jobs=1); model.fit(a,ytr,sample_weight=weights); prediction=model.predict(b)
    elif family=="hist_gradient":
        a,b=prepare_features(xtr,test_matrix,True); cols=[]
        for output in range(OUTPUTS):
            model=HistGradientBoostingRegressor(max_iter=160,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0,random_state=stable_seed(27027,context.key,name,str(output)))
            model.fit(a,ytr[:,output],sample_weight=weights); cols.append(model.predict(b))
        prediction=np.column_stack(cols)
    elif family=="knn":
        a,b=prepare_features(xtr,test_matrix,True); prediction=custom_knn_predict(a,ytr,b,weights,int(branch.get("neighbors",25)))
    else: raise DataValidationError(f"unknown family {family}")
    bound=float(config["target"]["coefficient_absolute_bound_ft"]); result=np.clip(np.asarray(prediction,dtype=np.float64),-bound,bound)
    if result.shape!=(len(context.test_indices),OUTPUTS) or not np.all(np.isfinite(result)): raise DataValidationError(f"{name}: prediction shape/finiteness differs")
    return result,detail


# ---------- scoring ----------

def metrics_for_indices(records: Sequence[WellRecord],indices: Sequence[int],coefficients: np.ndarray,weight: float) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    array=np.asarray(coefficients,dtype=np.float64)
    if array.shape!=(len(indices),OUTPUTS): raise DataValidationError("coefficient membership differs")
    rows=[records[int(index)].metric(array[pos],weight) for pos,index in enumerate(indices)]
    return summarize(rows),rows


def candidate_name(branch: str,weight: float) -> str: return f"{branch}__w{weight:.2f}"


# ---------- score-blind edge tests ----------

def synthetic_feature_names() -> list[str]:
    names=[]
    for prefix in ("backtest_0p5_","backtest_0p7_","backtest_0p85_"):
        names.extend(prefix+s for s in ("mean","rmse","toe","trend","u_slope_delta"))
    for suffix in ("_w32","_w128","_w512"):
        names.extend(prefix+suffix for prefix in ("visible_tvt_slope","visible_u_scale","visible_u_slope_delta","visible_u_slope"))
    index=0
    while len(names)<142:
        candidate=f"zz_feature_{index:03d}"; index+=1
        if candidate not in names: names.append(candidate)
    return sorted(names)


def run_edge_tests(config: Mapping[str,Any]) -> dict[str,Any]:
    passed=[]; names=synthetic_feature_names(); base=np.arange(3*142,dtype=float).reshape(3,142); targets=np.zeros((3,4))
    blocks,registered=registered_blocks(names,config); assert len(names)==142 and len(registered)==27; passed.append("feature_schema_and_block_counts")
    missing_names=[name for name in names if name != "backtest_0p5_mean"]
    try: registered_blocks(missing_names,config); raise AssertionError("missing registered feature passed")
    except DataValidationError: pass
    bad=names.copy(); bad[-1]=bad[-2]
    try:
        if len(set(bad))!=len(bad): raise DataValidationError("duplicate names")
        raise AssertionError("duplicate names passed")
    except DataValidationError: pass
    passed.append("missing_duplicate_feature_names")
    pool=build_view_pool(base,names,targets,config); assert pool.features_with_indicators.shape==(10,3,152); passed.append("view_catalog_and_indicators")
    bad_config=json.loads(json.dumps(config)); bad_config["views"][1]["mask_windows"]=["_unknown"]
    try: build_view_pool(base,names,targets,bad_config); raise AssertionError("unknown view block passed")
    except DataValidationError: pass
    passed.append("unknown_view_rejection")
    assert np.isnan(pool.features_without_indicators[1,:,blocks["backtest_0p7_"]]).all(); passed.append("registered_mask_application")
    xtr,xte=prepare_features(np.full((3,2),np.nan),np.asarray([[np.nan,np.nan]]),True); assert np.all(np.isfinite(xtr)) and np.all(np.isfinite(xte)); passed.append("all_missing_and_zero_variance")
    context=Context("tiny","repeated","v",0,np.asarray([0,1]),np.asarray([2]))
    x,y,source,view_ids,weights,detail=training_rows(pool,context,"all_views",config,True,1.0); totals=[weights[source==w].sum() for w in np.unique(source)]; assert max(totals)-min(totals)<1e-12; passed.append("source_well_equal_weight")
    _,_,source_b,view_b,weights_b,_=training_rows(pool,context,"all_views",config,True,4.0); assert abs(weights_b[(source_b==0)&(view_b==0)].sum()/weights_b[source_b==0].sum()-4/13)<1e-12; passed.append("original_boost_normalization")
    assert not(set(source.tolist())&set(context.test_indices.tolist())); passed.append("heldout_view_exclusion")
    empty=json.loads(json.dumps(config)); empty["regimes"]["empty"]=[]
    try: training_rows(pool,context,"empty",empty); raise AssertionError("empty regime passed")
    except DataValidationError: pass
    missing=json.loads(json.dumps(config)); missing["regimes"]["empty"]=["backtest_0p5_only"]
    try: training_rows(pool,context,"empty",missing); raise AssertionError("missing original passed")
    except DataValidationError: pass
    passed.append("empty_missing_original_regime")
    pred=fit_ridge(x,y,pool.features_with_indicators[0,context.test_indices],weights,1.0); assert pred.shape==(1,4); passed.append("tiny_ridge_partition")
    a,b=prepare_features(x,pool.features_with_indicators[0,context.test_indices],True); kp=custom_knn_predict(a,y,b,weights,25); assert kp.shape==(1,4); passed.append("knn_neighbor_clipping")
    tree=ExtraTreesRegressor(n_estimators=2,min_samples_leaf=1,random_state=1,n_jobs=1).fit(np.arange(6).reshape(-1,1),np.arange(6),sample_weight=np.ones(6)); assert math.isfinite(float(tree.predict([[1.5]])[0])); passed.append("tree_sample_weight")
    try: fit_segment([1,2,3,math.nan],0.0,80.0); raise AssertionError("nonfinite target passed")
    except DataValidationError: pass
    passed.append("nonfinite_target_rejection")
    clipped,_=solve_coefficients(spline_basis(128),np.full(128,1000.0),5.0); assert np.max(np.abs(clipped))<=5.0; passed.append("coefficient_clipping")
    basis=spline_basis(128); path=basis@clipped; assert np.all(np.isfinite(path)); passed.append("finite_spline_reconstruction")
    record=WellRecord("edgewell",128,128,10.0,0.0,np.zeros(4),0.0,0.0,e011_sse=128.0,e011_sum=0.0,e_dot_d=0.0,d_sse=128.0,d_sum=0.0,basis_dot_e=np.zeros(4),basis_dot_d=np.zeros(4),basis_sum=basis.sum(axis=0),basis_cross=basis.T@basis)
    assert record.metric(np.full(4,80.0),0.0)["sse"]==128.0; passed.append("exact_e011_fallback")
    try: record.metric([0,0,0,np.nan],1.0); raise AssertionError("nonfinite prediction passed")
    except DataValidationError: pass
    passed.append("nonfinite_prediction_rejection")
    assert ["a","b"][min(range(2),key=lambda i:([1.0,1.0][i],i))]=="a"; passed.append("deterministic_ties")
    repeated=[]
    for version in range(5):
        for fold in range(5):
            test=np.flatnonzero(np.arange(10)%5==fold); train=np.flatnonzero(np.arange(10)%5!=fold); repeated.append(Context(f"v{version}:{fold}","repeated",f"v{version}",fold,train,test))
    stress=[]
    for scope in ("spatial","typewell"):
        for group in range(5):
            test=np.flatnonzero(np.arange(10)%5==group); train=np.flatnonzero(np.arange(10)%5!=group); stress.append(Context(f"{scope}:{group}",scope,scope,group,train,test))
    validate_contexts(repeated,stress,10); passed.append("context_membership")
    assert len(config["branches"])==21 and tuple(float(v) for v in config["placements"])==PLACEMENTS; passed.append("branch_placement_contract")
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/"bad.csv.gz"
        with gzip.open(p,"wt",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=["id","well_id","hidden_index","target","spline4_ridge_equal_s075"]); writer.writeheader(); row={"id":"edgewell_128","well_id":"edgewell","hidden_index":0,"target":1.0,"spline4_ridge_equal_s075":1.0}; writer.writerow(row); writer.writerow(row)
        try: attach_e011_sufficient([record],p,2); raise AssertionError("duplicate OOF passed")
        except DataValidationError: pass
    passed.append("duplicate_oof_rejection")
    if len(passed)!=22: raise AssertionError(f"edge count differs: {len(passed)}")
    return {"status":"PASS","edge_groups":len(passed),"passed":passed}


# ---------- full screen ----------
def run_screen(root: Path,output_dir: Path,implementation_commit: str) -> dict[str,Any]:
    started=time.perf_counter(); config=json.loads((root/"tracking/evidence/T027/config.json").read_text())
    if tuple(float(v) for v in config["placements"])!=PLACEMENTS or len(config["branches"])!=21: raise DataValidationError("frozen branch/placement contract differs")
    bound=float(config["target"]["coefficient_absolute_bound_ft"])
    records=load_records(root/str(config["data_dir"]),int(config["expected_wells"]),bound)
    coverage=attach_e011_sufficient(records,root/str(config["e011_oof_path"]),int(config["expected_hidden_rows"]))
    base_features,feature_names,spatial,typewell=load_compact(root/str(config["compact_path"]),records,int(config["feature_contract"]["expected_features"]))
    targets=np.vstack([record.target_coefficients for record in records]); pool=build_view_pool(base_features,feature_names,targets,config)
    repeated,stress=load_contexts(root,[record.well_id for record in records],config["fold_files"],spatial,typewell)
    branches=list(config["branches"]); branch_by_name={str(b["name"]):b for b in branches}
    if len(branch_by_name)!=len(branches): raise DataValidationError("branch names duplicate")
    weights=[float(v) for v in config["placements"]]
    base_metrics=[record.metric(np.zeros(4),0.0) for record in records]; base_summary=summarize(base_metrics)
    oracle_metrics=[record.metric(record.target_coefficients,1.0) for record in records]; oracle_summary=summarize(oracle_metrics)
    all_contexts=[*repeated,*stress]; context_rows=[]; isolation_rows=[]; repeated_base={}; stress_base={}
    map_coefficients={label:{name:np.full((len(records),OUTPUTS),np.nan) for name in branch_by_name} for label in sorted({c.label for c in repeated})}
    context_predictions: dict[tuple[str,str],np.ndarray]={}
    duplicate_predictions={label:np.full((len(records),OUTPUTS),np.nan) for label in map_coefficients}
    for context in all_contexts:
        base,_=metrics_for_indices(records,context.test_indices,np.zeros((len(context.test_indices),4)),0.0)
        (repeated_base if context.scope=="repeated" else stress_base)[context.key]=base
        for branch in branches:
            name=str(branch["name"]); prediction,detail=predict_branch(branch,context,pool,config)
            context_predictions[(context.key,name)]=prediction
            isolation_rows.append({"context":context.key,"scope":context.scope,"branch":name,"tasks":detail["tasks"],"source_wells":detail["source_wells"],"views":detail["views"],"minimum_source_weight":detail["minimum_source_weight"],"maximum_source_weight":detail["maximum_source_weight"],"heldout_source_tasks":0})
            if context.scope=="repeated": map_coefficients[context.label][name][context.test_indices]=prediction
            for weight in weights:
                summary,_=metrics_for_indices(records,context.test_indices,prediction,weight)
                context_rows.append({"context":context.key,"scope":context.scope,"map":context.label,"outer_group":context.outer_group,"branch":name,"weight":weight,"candidate":candidate_name(name,weight),**summary})
        if context.scope=="repeated":
            duplicate,_=predict_branch(branch_by_name["ridge_original_a1"],context,pool,config,duplicate_original=True)
            duplicate_predictions[context.label][context.test_indices]=duplicate
    for label,branch_map in map_coefficients.items():
        for name,values in branch_map.items():
            if not np.all(np.isfinite(values)): raise DataValidationError(f"{label}/{name}: map coverage incomplete")
        if not np.all(np.isfinite(duplicate_predictions[label])): raise DataValidationError(f"{label}: duplicate-original coverage incomplete")
    map_rows=[]; final_coefficients={}; duplicate_max_delta=0.0
    for name in branch_by_name:
        stack=np.stack([map_coefficients[label][name] for label in sorted(map_coefficients)]); final_coefficients[name]=stack.mean(axis=0)
        for label in sorted(map_coefficients):
            for weight in weights:
                summary,_=metrics_for_indices(records,np.arange(len(records)),map_coefficients[label][name],weight)
                map_rows.append({"map":label,"branch":name,"weight":weight,"candidate":candidate_name(name,weight),**summary})
    for label in sorted(map_coefficients): duplicate_max_delta=max(duplicate_max_delta,float(np.max(np.abs(duplicate_predictions[label]-map_coefficients[label]["ridge_original_a1"]))))
    final_rows=[]; final_well_metrics={}
    for name,coefficients in final_coefficients.items():
        for weight in weights:
            candidate=candidate_name(name,weight); summary,well_metrics=metrics_for_indices(records,np.arange(len(records)),coefficients,weight); final_well_metrics[candidate]=well_metrics
            final_rows.append({"candidate":candidate,"branch":name,"weight":weight,"eligible":bool(branch_by_name[name].get("eligible",True)),"gain_vs_e011":float(base_summary["rmse"])-float(summary["rmse"]),"gain_vs_t025":float(config["promotion"]["t025_best_rmse"])-float(summary["rmse"]),**summary})
    context_lookup={(r["context"],r["candidate"]):r for r in context_rows}; map_lookup={(r["map"],r["candidate"]):r for r in map_rows}
    for row in final_rows:
        candidate=str(row["candidate"]); row["map_wins"]=sum(float(base_summary["rmse"])-float(map_lookup[(label,candidate)]["rmse"])>0 for label in sorted(map_coefficients)); row["outer_cell_wins"]=sum(float(repeated_base[c.key]["rmse"])-float(context_lookup[(c.key,candidate)]["rmse"])>0 for c in repeated)
    original_rows=[r for r in final_rows if str(r["branch"]).startswith("ridge_original_")]; best_original=min(original_rows,key=lambda r:(float(r["rmse"]),str(r["candidate"])))
    hidden_rows=np.asarray([record.hidden_rows for record in records],dtype=float); missing=np.asarray([record.hidden_gr_missing_fraction for record in records],dtype=float); base_well_rmse=np.asarray([m["rmse"] for m in base_metrics])
    long_threshold=float(np.quantile(hidden_rows,float(config["stress"]["long_suffix_quantile"]))); missing_threshold=float(np.quantile(missing,float(config["stress"]["high_gr_missingness_quantile"]))); q_edges=np.unique(np.quantile(hidden_rows,np.linspace(0,1,6)[1:-1]))
    slice_sets={"long_suffix":np.flatnonzero(hidden_rows>=long_threshold),"high_gr_missingness":np.flatnonzero(missing>=missing_threshold),"e011_catastrophe":np.flatnonzero(base_well_rmse>=float(config["stress"]["e011_catastrophe_rmse"]))}
    qbins=np.digitize(hidden_rows,q_edges)
    for q in range(5): slice_sets[f"horizon_q{q}"]=np.flatnonzero(qbins==q)
    slice_rows=[]; slice_lookup={}
    for slice_name,indices in slice_sets.items():
        base=summarize([base_metrics[int(i)] for i in indices]); slice_rows.append({"slice":slice_name,"candidate":"e011",**base}); slice_lookup[(slice_name,"e011")]=base
        for row in final_rows:
            candidate=str(row["candidate"]); summary=summarize([final_well_metrics[candidate][int(i)] for i in indices]); slice_rows.append({"slice":slice_name,"candidate":candidate,**summary}); slice_lookup[(slice_name,candidate)]=summary
    control_sums={mode:np.zeros((len(records),OUTPUTS)) for mode in CONTROL_MODES}; control_counts=np.zeros(len(records),dtype=int); core=branch_by_name["ridge_all_views_a1"]
    for context in repeated:
        for mode in CONTROL_MODES:
            prediction,_=predict_branch(core,context,pool,config,control_mode=mode); control_sums[mode][context.test_indices]+=prediction
        control_counts[context.test_indices]+=1
    if not np.all(control_counts==5): raise DataValidationError("control coverage differs")
    control_rows=[]; maximum_control_gain=-math.inf
    for mode in CONTROL_MODES:
        coefficients=control_sums[mode]/control_counts[:,None]
        for weight in weights:
            summary,_=metrics_for_indices(records,np.arange(len(records)),coefficients,weight); gain=float(base_summary["rmse"])-float(summary["rmse"]); maximum_control_gain=max(maximum_control_gain,gain); control_rows.append({"control":mode,"branch":"ridge_all_views_a1","weight":weight,"candidate":f"ridge_all_views_a1__{mode}__w{weight:.2f}","gain_vs_e011":gain,**summary})
    edge=run_edge_tests(config); controls={
        "coverage":{"pass":coverage["rows"]==int(config["expected_hidden_rows"]) and coverage["wells"]==int(config["expected_wells"]),**coverage},
        "feature_schema":{"pass":len(feature_names)==142 and len(pool.registered_indices)==27,"features":len(feature_names),"registered":len(pool.registered_indices)},
        "view_catalog":{"pass":pool.features_with_indicators.shape==(10,773,152),"views":len(pool.view_names),"tasks":10*773},
        "context_completion":{"pass":len(context_rows)==35*21*4,"rows":len(context_rows)},
        "isolation_completion":{"pass":len(isolation_rows)==35*21 and all(int(r["heldout_source_tasks"])==0 for r in isolation_rows),"rows":len(isolation_rows)},
        "source_weight_equality":{"pass":all(abs(float(r["maximum_source_weight"])-float(r["minimum_source_weight"]))<=1e-10 for r in isolation_rows)},
        "duplicate_original":{"pass":duplicate_max_delta<=float(config["promotion"]["duplicate_original_tolerance"]),"maximum_coefficient_delta":duplicate_max_delta},
        "exact_fallback":{"pass":abs(float(base_summary["rmse"])-float(config["promotion"]["e011_rmse"]))<=1e-8,"rmse":base_summary["rmse"]},
        "finite_bounded":{"pass":all(np.all(np.isfinite(v)) and np.max(np.abs(v))<=bound+1e-9 for v in final_coefficients.values())},
        "negative_controls":{"pass":maximum_control_gain<=float(config["promotion"]["maximum_negative_control_gain"]),"maximum_gain_vs_e011":maximum_control_gain},
        "edge_groups":edge,
    }
    all_controls=all((bool(v["pass"]) if "pass" in v else v.get("status")=="PASS") for v in controls.values())
    authorization_rows=[]; gates_by_candidate={}
    for row in final_rows:
        candidate=str(row["candidate"]); spatial_gains=[]; typewell_gains=[]
        for context in stress:
            gain=float(stress_base[context.key]["rmse"])-float(context_lookup[(context.key,candidate)]["rmse"]); (spatial_gains if context.scope=="spatial" else typewell_gains).append(gain)
        special_gains=[float(slice_lookup[(name,"e011")]["rmse"])-float(slice_lookup[(name,candidate)]["rmse"]) for name in ("long_suffix","high_gr_missingness","e011_catastrophe")]
        horizon_gains=[float(slice_lookup[(f"horizon_q{q}","e011")]["rmse"])-float(slice_lookup[(f"horizon_q{q}",candidate)]["rmse"]) for q in range(5)]
        gain_original=float(best_original["rmse"])-float(row["rmse"])
        gates={"eligible":bool(row["eligible"]),"oracle":float(oracle_summary["rmse"])<=float(config["promotion"]["maximum_oracle_rmse"]),"gain":float(row["gain_vs_e011"])>=float(config["promotion"]["minimum_gain_vs_e011"]),"gain_vs_t025":float(row["gain_vs_t025"])>=float(config["promotion"]["minimum_gain_vs_t025_best"]),"gain_vs_original":gain_original>=float(config["promotion"]["minimum_gain_vs_best_original_only"]),"maps":int(row["map_wins"])>=int(config["promotion"]["minimum_map_wins"]),"cells":int(row["outer_cell_wins"])>=int(config["promotion"]["minimum_outer_cell_wins"]),"p90":float(row["p90_well_rmse"])-float(base_summary["p90_well_rmse"])<=float(config["promotion"]["maximum_p90_deterioration"]),"worst5":float(row["worst_5pct_sse_share"])-float(base_summary["worst_5pct_sse_share"])<=float(config["promotion"]["maximum_worst5_share_increase"]),"spatial":min(spatial_gains)>=0.0,"typewell":min(typewell_gains)>=0.0,"special_slices":min(special_gains)>=0.0,"horizon_quintiles":min(horizon_gains)>=0.0,"controls":all_controls,"reproduction":False}
        gates_by_candidate[candidate]=gates; authorization_rows.append({"candidate":candidate,"gain_vs_best_original_only":gain_original,"minimum_spatial_gain":min(spatial_gains),"minimum_typewell_gain":min(typewell_gains),"minimum_special_slice_gain":min(special_gains),"minimum_horizon_quintile_gain":min(horizon_gains),**{f"gate_{k}":v for k,v in gates.items()}})
    substantive_passers=[candidate for candidate,gates in gates_by_candidate.items() if all(v for k,v in gates.items() if k!="reproduction")]
    reported=min(final_rows,key=lambda r:(float(r["rmse"]),str(r["candidate"]))); status="awaiting_reproduction" if substantive_passers else "worth_screen_reject"; decision="await_independent_reproduction" if substantive_passers else "close_h020_without_formal_experiment"
    reported_candidate=str(reported["candidate"]); reported_branch=str(reported["branch"]); reported_weight=float(reported["weight"]); selected_well_rows=[]
    for index,record in enumerate(records):
        metric=final_well_metrics[reported_candidate][index]; selected_well_rows.append({"well_id":record.well_id,"candidate":reported_candidate,"branch":reported_branch,"weight":reported_weight,"rows_scored":metric["rows_scored"],"sse":metric["sse"],"rmse":metric["rmse"],"mean_error":metric["mean_error"],"known_rows":record.known_rows,"hidden_rows":record.hidden_rows,"hidden_gr_missing_fraction":record.hidden_gr_missing_fraction,**{f"predicted_coefficient_{o}":final_coefficients[reported_branch][index,o] for o in range(OUTPUTS)},**{f"target_coefficient_{o}":record.target_coefficients[o] for o in range(OUTPUTS)}})
    output_dir.mkdir(parents=True,exist_ok=True); _write_csv(output_dir/"candidate_metrics.csv",final_rows); _write_csv(output_dir/"context_metrics.csv",context_rows); _write_csv(output_dir/"map_metrics.csv",map_rows); _write_csv(output_dir/"slice_metrics.csv",slice_rows); _write_csv(output_dir/"negative_control_metrics.csv",control_rows); _write_csv(output_dir/"isolation_audit.csv",isolation_rows); _write_csv(output_dir/"authorization_gates.csv",authorization_rows); _write_csv(output_dir/"selected_well_metrics.csv",selected_well_rows); _write_json(output_dir/"edge_cases.json",edge)
    runtime=time.perf_counter()-started; summary={"schema_version":1,"task_id":"T027","hypothesis_id":"H020","implementation_commit":implementation_commit,"status":status,"decision":decision,"formal_experiment_authorized":False,"reported_candidate":reported_candidate,"reported_metrics":reported,"base_summary":base_summary,"oracle_summary":oracle_summary,"best_original_only":best_original,"substantive_passers":substantive_passers,"gates_by_candidate":gates_by_candidate,"controls":controls,"maximum_negative_control_gain":maximum_control_gain,"thresholds":{"long_suffix":long_threshold,"high_gr_missingness":missing_threshold,"e011_catastrophe":config["stress"]["e011_catastrophe_rmse"],"horizon_quintile_edges":q_edges.tolist()},"view_contract":{"features":len(feature_names),"registered_features":len(pool.registered_indices),"views":len(pool.view_names),"tasks":10*len(records)},"runtime_seconds":runtime,"deployment":{"package_built":False,"kaggle_executed":False,"submission_created":False,"submission_made":False}}
    _write_json(output_dir/"summary.json",summary); files=sorted(p for p in output_dir.iterdir() if p.is_file()); _write_json(output_dir/"artifact_manifest.json",{"schema_version":1,"task_id":"T027","implementation_commit":implementation_commit,"files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":_sha256(p)} for p in files]}); return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir"); parser.add_argument("--implementation-commit",default="UNSEALED"); parser.add_argument("--edge-only",action="store_true"); args=parser.parse_args()
    config=json.loads((ROOT/"tracking/evidence/T027/config.json").read_text())
    if args.edge_only:
        print(json.dumps(run_edge_tests(config),indent=2,sort_keys=True)); return
    if not args.output_dir: parser.error("--output-dir is required unless --edge-only")
    summary=run_screen(ROOT,ROOT/args.output_dir,args.implementation_commit); print(json.dumps({"status":summary["status"],"decision":summary["decision"],"reported_candidate":summary["reported_candidate"],"reported_rmse":summary["reported_metrics"]["rmse"],"gain_vs_e011":summary["reported_metrics"]["gain_vs_e011"],"oracle_rmse":summary["oracle_summary"]["rmse"],"runtime_seconds":summary["runtime_seconds"]},indent=2,sort_keys=True))


if __name__=="__main__": main()
