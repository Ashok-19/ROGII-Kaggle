"""E011: memory-safe nonlinear coefficient and control-point learning.

The experiment predicts a small number of whole-suffix correction coefficients
around the frozen E006 path.  All target construction, feature selection,
hyperparameter selection, shrinkage, confidence gating, and fallback decisions
are isolated inside each validation training partition.  Row scoring uses exact
per-well sufficient statistics; a row-by-candidate matrix is never built.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import resource
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .fusion import _group_assignments
from .gr_path import TypewellCurve
from .harness import (
    DataValidationError,
    ErrorAccumulator,
    WellMetric,
    _canonical_json,
    _quantile,
    _sha256,
    _summarize,
    _write_csv,
    _write_json,
    scan_profiles,
)

E011_OOF_FILENAME = "oof_predictions.csv.gz"
E011_RESULT_FILENAMES = (
    "candidate_metrics.csv",
    "coefficient_metrics.csv",
    "map_metrics.csv",
    "outer_cell_metrics.csv",
    "stress_metrics.csv",
    "special_slice_metrics.csv",
    "selected_hyperparameters.csv",
    "selected_feature_frequency.csv",
    "controls.csv",
    "membership_audit.csv",
    "representation_targets.csv",
    "model_predictions.csv",
    "summary.json",
)
LEARNED_REPRESENTATIONS = ("shape3", "spline3", "spline4", "spline5", "spline7")
PRIMARY_REPRESENTATIONS = ("spline4", "spline5", "spline7")
RIDGE_MODELS = ("ridge_equal", "ridge_row_weighted")
BASE_MODELS = ("ridge_equal", "ridge_row_weighted", "extra_trees", "ridge_tree_blend")
COMPARATORS = ("last_known_tvt", "e006_nested_fusion")
NEGATIVE_CONTROLS = ("spline4_shuffled_ridge_s100", "spline4_sign_flipped_ridge_s100")


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
class CandidateSpec:
    name: str
    representation: str
    model: str
    shrinkage: float | None
    confidence_fallback: bool = False
    eligible: bool = True
    negative_control: bool = False


@dataclass(frozen=True)
class PreparedFeatures:
    names: tuple[str, ...]
    train_x: np.ndarray
    test_x: np.ndarray


@dataclass(frozen=True)
class RidgeSelection:
    feature_count: int
    alpha: float
    inner_rmse: float
    inner_predictions: Mapping[str, np.ndarray]
    outer_predictions: Mapping[str, np.ndarray]
    selected_features: tuple[str, ...]


@dataclass(frozen=True)
class FallbackSelection:
    shrinkage: float
    disagreement_quantile: float
    magnitude_quantile: float
    disagreement_threshold: float
    magnitude_threshold: float
    inner_rmse: float
    action_rate: float
    passed: bool


@dataclass
class BasisSufficient:
    representation: str
    rows: int
    sum_x: float
    sum_x_sq: float
    base_sum: float
    base_sse: float
    base_x_sum: float
    basis_sum: np.ndarray
    basis_x_sum: np.ndarray
    basis_base: np.ndarray
    basis_cross: np.ndarray
    target_coefficients: np.ndarray

    def metric(self, well_id: str, coefficients: Sequence[float]) -> WellMetric:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != self.target_coefficients.shape or not np.all(np.isfinite(values)):
            raise DataValidationError(f"{well_id}: invalid E011 coefficient vector for {self.representation}")
        sum_error = self.base_sum + float(self.basis_sum @ values)
        sum_error_sq = self.base_sse + 2.0 * float(self.basis_base @ values) + float(values @ self.basis_cross @ values)
        sum_x_error = self.base_x_sum + float(self.basis_x_sum @ values)
        tolerance = max(1e-9, abs(self.base_sse) * 1e-12)
        if sum_error_sq < 0.0 and abs(sum_error_sq) <= tolerance:
            sum_error_sq = 0.0
        if sum_error_sq < 0.0 or not math.isfinite(sum_error_sq):
            raise DataValidationError(f"{well_id}: invalid E011 sufficient-statistic SSE")
        return ErrorAccumulator(
            rows=self.rows,
            sum_error=sum_error,
            sum_error_sq=sum_error_sq,
            sum_x=self.sum_x,
            sum_x_sq=self.sum_x_sq,
            sum_x_error=sum_x_error,
        ).finalize(well_id)


@dataclass
class CoefficientWell:
    well_id: str
    rows: int
    features: Mapping[str, float | None]
    spatial: tuple[float, float]
    typewell: tuple[float, float, float]
    hidden_gr_missing_fraction: float
    last_metric: WellMetric
    e006_metric: WellMetric
    statistics: Mapping[str, BasisSufficient]


@dataclass
class _StatsBuilder:
    representation: str
    dimensions: int
    rows: int = 0
    sum_x: float = 0.0
    sum_x_sq: float = 0.0
    base_sum: float = 0.0
    base_sse: float = 0.0
    base_x_sum: float = 0.0

    def __post_init__(self) -> None:
        self.basis_sum = np.zeros(self.dimensions, dtype=np.float64)
        self.basis_x_sum = np.zeros(self.dimensions, dtype=np.float64)
        self.basis_base = np.zeros(self.dimensions, dtype=np.float64)
        self.basis_cross = np.zeros((self.dimensions, self.dimensions), dtype=np.float64)

    def add(self, x: float, base_error: float, basis: np.ndarray) -> None:
        if basis.shape != (self.dimensions,) or not np.all(np.isfinite(basis)):
            raise DataValidationError("E011 invalid basis vector")
        self.rows += 1
        self.sum_x += x
        self.sum_x_sq += x * x
        self.base_sum += base_error
        self.base_sse += base_error * base_error
        self.base_x_sum += x * base_error
        self.basis_sum += basis
        self.basis_x_sum += x * basis
        self.basis_base += base_error * basis
        self.basis_cross += np.outer(basis, basis)

    def finalize(self) -> BasisSufficient:
        if self.rows <= 0:
            raise DataValidationError("E011 sufficient statistics contain no rows")
        target = np.linalg.lstsq(self.basis_cross, -self.basis_base, rcond=None)[0]
        if target.shape != (self.dimensions,) or not np.all(np.isfinite(target)):
            raise DataValidationError("E011 coefficient target is invalid")
        return BasisSufficient(
            representation=self.representation,
            rows=self.rows,
            sum_x=self.sum_x,
            sum_x_sq=self.sum_x_sq,
            base_sum=self.base_sum,
            base_sse=self.base_sse,
            base_x_sum=self.base_x_sum,
            basis_sum=self.basis_sum.copy(),
            basis_x_sum=self.basis_x_sum.copy(),
            basis_base=self.basis_base.copy(),
            basis_cross=self.basis_cross.copy(),
            target_coefficients=target,
        )


def _finite(raw: Any, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"E011 invalid numeric {label}") from exc
    if not math.isfinite(value):
        raise DataValidationError(f"E011 non-finite numeric {label}")
    return value


def _representation_spec(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        return config["representations"][name]
    except KeyError as exc:
        raise DataValidationError(f"E011 unknown representation {name}") from exc


def basis_vector(position: float, name: str, config: Mapping[str, Any]) -> np.ndarray:
    s = float(position)
    if not math.isfinite(s) or s < -1e-12 or s > 1.0 + 1e-12:
        raise DataValidationError("E011 normalized suffix position is invalid")
    s = min(1.0, max(0.0, s))
    spec = _representation_spec(config, name)
    dimensions = int(spec["dimensions"])
    if name == "linear1":
        values = np.asarray([s], dtype=np.float64)
    elif name == "quadratic2":
        values = np.asarray([s, s * s], dtype=np.float64)
    elif name == "shape3":
        values = np.asarray([s, 4.0 * s * (1.0 - s), 16.0 * s * (1.0 - s) * (s - 0.5)], dtype=np.float64)
    elif name.startswith("spline"):
        knots = np.asarray(spec["knots"], dtype=np.float64)
        if knots.shape != (dimensions,) or not np.all(np.diff(knots) > 0.0) or abs(float(knots[-1]) - 1.0) > 1e-12:
            raise DataValidationError(f"E011 malformed knot contract for {name}")
        grid = np.concatenate(([0.0], knots))
        values = np.empty(dimensions, dtype=np.float64)
        for index in range(dimensions):
            controls = np.zeros(dimensions + 1, dtype=np.float64)
            controls[index + 1] = 1.0
            values[index] = float(np.interp(s, grid, controls))
    else:
        raise DataValidationError(f"E011 unsupported representation {name}")
    if values.shape != (dimensions,) or not np.all(np.isfinite(values)):
        raise DataValidationError(f"E011 basis width differs for {name}")
    return values


def basis_matrix(rows: int, name: str, config: Mapping[str, Any]) -> np.ndarray:
    count = int(rows)
    if count <= 0:
        raise DataValidationError("E011 basis requires at least one row")
    denominator = max(1, count - 1)
    return np.vstack([basis_vector(index / denominator, name, config) for index in range(count)])


def _peak_correction(coefficients: np.ndarray, name: str, config: Mapping[str, Any]) -> float:
    if name.startswith("spline"):
        return float(np.max(np.abs(coefficients))) if coefficients.size else 0.0
    samples = np.linspace(0.0, 1.0, 257)
    matrix = np.vstack([basis_vector(float(value), name, config) for value in samples])
    return float(np.max(np.abs(matrix @ coefficients)))


def bound_coefficients(coefficients: Sequence[float], name: str, config: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    dimensions = int(_representation_spec(config, name)["dimensions"])
    if values.shape != (dimensions,) or not np.all(np.isfinite(values)):
        raise DataValidationError(f"E011 invalid emitted coefficients for {name}")
    bound_key = "spline_knot_absolute_ft" if name.startswith("spline") else "shape3_coefficient_absolute_ft"
    values = np.clip(values, -float(config["coefficient_bounds"][bound_key]), float(config["coefficient_bounds"][bound_key]))
    peak = _peak_correction(values, name, config)
    maximum = float(config["coefficient_bounds"]["emitted_correction_absolute_ft"])
    if peak > maximum and peak > 0.0:
        values = values * (maximum / peak)
    if _peak_correction(values, name, config) > maximum + 1e-8:
        raise DataValidationError(f"E011 correction bound failed for {name}")
    return values


def candidate_specs(config: Mapping[str, Any]) -> tuple[CandidateSpec, ...]:
    shrinkages = tuple(float(value) for value in config["shrinkages"])
    output: list[CandidateSpec] = []
    for representation in ("shape3", "spline3"):
        for shrinkage in shrinkages:
            output.append(CandidateSpec(f"{representation}_ridge_equal_s{int(round(shrinkage * 100)):03d}", representation, "ridge_equal", shrinkage))
    for representation in PRIMARY_REPRESENTATIONS:
        for model in BASE_MODELS:
            for shrinkage in shrinkages:
                output.append(CandidateSpec(f"{representation}_{model}_s{int(round(shrinkage * 100)):03d}", representation, model, shrinkage))
        output.append(CandidateSpec(f"{representation}_blend_confidence_fallback", representation, "ridge_tree_blend", None, confidence_fallback=True))
    output.append(CandidateSpec(NEGATIVE_CONTROLS[0], "spline4", "ridge_equal", 1.0, eligible=False, negative_control=True))
    output.append(CandidateSpec(NEGATIVE_CONTROLS[1], "spline4", "ridge_equal", 1.0, eligible=False, negative_control=True))
    names = [item.name for item in output]
    if len(names) != len(set(names)):
        raise DataValidationError("E011 candidate names are not unique")
    return tuple(output)


def validate_e011_config(config: Mapping[str, Any]) -> None:
    if str(config.get("experiment_id")) != "E011":
        raise DataValidationError("E011 experiment ID differs")
    if tuple(config.get("learned_representations", ())) != LEARNED_REPRESENTATIONS:
        raise DataValidationError("E011 learned representation order differs")
    if tuple(float(value) for value in config.get("shrinkages", ())) != (0.25, 0.5, 0.75, 1.0):
        raise DataValidationError("E011 shrinkage contract differs")
    if int(config["resource_design"]["maximum_threads"]) != 2:
        raise DataValidationError("E011 maximum thread contract differs")
    if int(config["promotion"]["maximum_rss_mb"]) != 2048:
        raise DataValidationError("E011 memory contract differs")
    if bool(config.get("submission_authorized")):
        raise DataValidationError("E011 submission must remain unauthorized")
    for name in (*LEARNED_REPRESENTATIONS, "linear1", "quadratic2"):
        matrix = basis_matrix(7, name, config)
        if matrix.shape[1] != int(config["representations"][name]["dimensions"]):
            raise DataValidationError(f"E011 representation width differs for {name}")
    for name in ("linear1", "quadratic2"):
        if float(config["representations"][name]["oracle_rmse"]) <= 5.0:
            raise DataValidationError("E011 negative-capacity control unexpectedly below 5")
    for name in LEARNED_REPRESENTATIONS:
        if float(config["representations"][name]["oracle_rmse"]) >= 5.0:
            raise DataValidationError("E011 learned representation lacks sub-5 capacity")
    if len(config["fold_files"]) != 5:
        raise DataValidationError("E011 requires five fold maps")
    if len(candidate_specs(config)) < 50:
        raise DataValidationError("E011 candidate branching is incomplete")


def _verify_parent_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    passed = True
    for name, item in config["parent_artifacts"].items():
        path = root / str(item["path"])
        actual = _sha256(path) if path.is_file() else None
        size = path.stat().st_size if path.is_file() else None
        ok = actual == str(item["sha256"]) and size == int(item["bytes"])
        passed = passed and ok
        rows.append({"name": name, "path": str(item["path"]), "expected_sha256": item["sha256"], "actual_sha256": actual, "expected_bytes": item["bytes"], "actual_bytes": size, "pass": ok})
    return {"pass": passed, "artifacts": rows}


def _read_numeric_features(path: Path) -> tuple[dict[str, dict[str, float | None]], tuple[str, ...], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "well_id" not in reader.fieldnames:
            raise DataValidationError("E011 legal feature table lacks well_id")
        names = tuple(name for name in reader.fieldnames if name != "well_id")
        records: dict[str, dict[str, float | None]] = {}
        for row in reader:
            well_id = str(row["well_id"])
            if not well_id or well_id in records:
                raise DataValidationError("E011 duplicate or empty legal-feature well ID")
            values: dict[str, float | None] = {}
            for name in names:
                raw = row.get(name)
                if raw is None or not raw.strip() or raw.strip().lower() in {"nan", "na", "none", "null"}:
                    values[name] = None
                else:
                    value = _finite(raw, f"feature {name}")
                    values[name] = value
            records[well_id] = values
    return records, names, 1 + len(names)


def _validate_feature_schema(features: Mapping[str, Mapping[str, float | None]], names: Sequence[str], columns: int, config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["legal_feature_contract"]
    if len(features) != int(contract["expected_rows"]) or columns != int(contract["expected_columns"]):
        raise DataValidationError("E011 legal feature shape differs")
    lowered = [name.lower() for name in names]
    forbidden = sorted(name for name, low in zip(names, lowered) if any(token.lower() in low for token in contract["forbidden_name_fragments"]))
    if forbidden:
        raise DataValidationError(f"E011 forbidden legal features: {forbidden[:8]}")
    missing_prefixes = [prefix for prefix in contract["required_prefixes"] if not any(name.startswith(prefix) for name in names)]
    if missing_prefixes:
        raise DataValidationError(f"E011 missing pseudo-cut feature prefixes: {missing_prefixes}")
    return {"pass": True, "wells": len(features), "columns": columns, "feature_count": len(names), "forbidden": forbidden, "missing_prefixes": missing_prefixes}


def _last_metric(well_id: str, rows: Sequence[tuple[float, float, float, float]]) -> WellMetric:
    accumulator = ErrorAccumulator()
    for hidden_index, target, _e006, last in rows:
        accumulator.add(last - target, hidden_index)
    return accumulator.finalize(well_id)


def _read_records(root: Path, train_dir: Path, config: Mapping[str, Any]) -> tuple[dict[str, CoefficientWell], tuple[str, ...], dict[str, Any]]:
    feature_item = config["parent_artifacts"]["e008_legal_features"]
    features, feature_names, feature_columns = _read_numeric_features(root / str(feature_item["path"]))
    feature_control = _validate_feature_schema(features, feature_names, feature_columns, config)
    oof_path = root / str(config["parent_artifacts"]["e010_oof"]["path"])
    required = {"id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt", "e006_nested_fusion"}
    records: dict[str, CoefficientWell] = {}
    current_id: str | None = None
    current_rows: list[tuple[float, float, float, float]] = []
    builders: dict[str, _StatsBuilder] = {}
    previous_row_index: int | None = None
    total_rows = 0
    rolling_ids = hashlib.sha256()
    seen_wells: set[str] = set()

    def finalize_current() -> None:
        nonlocal current_id, current_rows, builders, previous_row_index
        if current_id is None:
            return
        well_id = current_id
        expected = int(_finite(features[well_id].get("hidden_rows"), "hidden_rows"))
        if len(current_rows) != expected or expected <= 0:
            raise DataValidationError(f"{well_id}: E011 hidden row count differs")
        stats = {name: builder.finalize() for name, builder in builders.items()}
        first = stats[LEARNED_REPRESENTATIONS[0]]
        e006_metric = first.metric(well_id, np.zeros_like(first.target_coefficients))
        horizontal = train_dir / f"{well_id}__horizontal_well.csv"
        with horizontal.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first_row = next(reader, None)
            if first_row is None:
                raise DataValidationError(f"{well_id}: empty horizontal file")
            last_row = first_row
            for row in reader:
                last_row = row
        spatial = (0.5 * (_finite(first_row.get("X"), "X") + _finite(last_row.get("X"), "X")), 0.5 * (_finite(first_row.get("Y"), "Y") + _finite(last_row.get("Y"), "Y")))
        curve = TypewellCurve.read(train_dir / f"{well_id}__typewell.csv")
        typewell = (curve.gr_mean, curve.gr_std, curve.maximum_tvt - curve.minimum_tvt)
        records[well_id] = CoefficientWell(
            well_id=well_id,
            rows=expected,
            features=features[well_id],
            spatial=spatial,
            typewell=typewell,
            hidden_gr_missing_fraction=float(features[well_id].get("hidden_gr_missing_fraction") or 0.0),
            last_metric=_last_metric(well_id, current_rows),
            e006_metric=e006_metric,
            statistics=stats,
        )
        current_id = None
        current_rows = []
        builders = {}
        previous_row_index = None

    with gzip.open(oof_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise DataValidationError("E011 E010 OOF schema differs")
        for row in reader:
            well_id = str(row["well_id"])
            if well_id not in features:
                raise DataValidationError(f"E011 OOF well {well_id} is absent from legal features")
            if current_id is None:
                if well_id in seen_wells:
                    raise DataValidationError("E011 parent OOF well blocks are not contiguous")
                current_id = well_id
                seen_wells.add(well_id)
                builders = {name: _StatsBuilder(name, int(config["representations"][name]["dimensions"])) for name in LEARNED_REPRESENTATIONS}
            elif well_id != current_id:
                finalize_current()
                if well_id in seen_wells:
                    raise DataValidationError("E011 parent OOF well blocks repeat")
                current_id = well_id
                seen_wells.add(well_id)
                builders = {name: _StatsBuilder(name, int(config["representations"][name]["dimensions"])) for name in LEARNED_REPRESENTATIONS}
            hidden_index = int(row["hidden_index"])
            row_index = int(row["row_index"])
            if hidden_index != len(current_rows) or (previous_row_index is not None and row_index <= previous_row_index):
                raise DataValidationError(f"{well_id}: E011 parent row order differs")
            previous_row_index = row_index
            target = _finite(row["target"], "target")
            e006 = _finite(row["e006_nested_fusion"], "E006")
            last = _finite(row["last_known_tvt"], "last-known")
            base_error = e006 - target
            expected = int(_finite(features[well_id].get("hidden_rows"), "hidden_rows"))
            position = hidden_index / max(1, expected - 1)
            for name, builder in builders.items():
                builder.add(float(hidden_index), base_error, basis_vector(position, name, config))
            current_rows.append((float(hidden_index), target, e006, last))
            rolling_ids.update(str(row["id"]).encode("utf-8")); rolling_ids.update(b"\n")
            total_rows += 1
    finalize_current()
    if set(records) != set(features) or len(records) != int(config["expected_wells"]) or total_rows != int(config["expected_hidden_rows"]):
        raise DataValidationError("E011 parent well/row identity differs")
    return records, feature_names, {"rows": total_rows, "wells": len(records), "parent_id_sequence_sha256": rolling_ids.hexdigest(), "feature_schema": feature_control}


def _load_folds(root: Path, config: Mapping[str, Any], well_ids: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    expected = set(well_ids)
    for item in config["fold_files"]:
        path = root / str(item["path"])
        if _sha256(path) != str(item["sha256"]) or path.stat().st_size != int(item["bytes"]):
            raise DataValidationError(f"E011 fold hash differs: {path}")
        fold = json.loads(path.read_text(encoding="utf-8"))
        if str(fold.get("data_signature")) != str(config["data_signature"]) or set(fold.get("assignments", {})) != expected:
            raise DataValidationError(f"E011 fold identity differs: {path}")
        output.append(fold)
    versions = [str(item["version"]) for item in output]
    if len(output) != 5 or len(set(versions)) != 5:
        raise DataValidationError("E011 fold versions differ")
    return output


def _build_contexts(records: Mapping[str, CoefficientWell], folds: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[Context], list[dict[str, Any]], dict[str, int], dict[str, int]]:
    ids = tuple(sorted(records))
    contexts: list[Context] = []
    audits: list[dict[str, Any]] = []
    for fold in folds:
        assignments = {well_id: int(value) for well_id, value in fold["assignments"].items()}
        version = str(fold["version"])
        for outer in range(int(fold["n_folds"])):
            test = tuple(well_id for well_id in ids if assignments[well_id] == outer)
            train = tuple(well_id for well_id in ids if assignments[well_id] != outer)
            context = Context(f"repeated:{version}:{outer}", "repeated", version, outer, train, test, assignments)
            contexts.append(context)
            audits.append(_context_audit(context))
    spatial = _group_assignments({well_id: record.spatial for well_id, record in records.items()}, 5)
    typewell = _group_assignments({well_id: record.typewell for well_id, record in records.items()}, 5)
    inner = {well_id: int(folds[0]["assignments"][well_id]) for well_id in ids}
    for scope, assignments in (("spatial", spatial), ("typewell", typewell)):
        for outer in range(5):
            test = tuple(well_id for well_id in ids if assignments[well_id] == outer)
            train = tuple(well_id for well_id in ids if assignments[well_id] != outer)
            context = Context(f"{scope}:{outer}", scope, scope, outer, train, test, inner)
            contexts.append(context)
            audits.append(_context_audit(context))
    if not all(bool(item["pass"]) for item in audits):
        raise DataValidationError("E011 context membership audit failed")
    return contexts, audits, spatial, typewell


def _context_audit(context: Context) -> dict[str, Any]:
    train = set(context.train_ids); test = set(context.test_ids)
    inner_groups = sorted({int(context.inner_assignments[well_id]) for well_id in context.train_ids})
    inner_coverage: set[str] = set()
    inner_detail = []
    for group in inner_groups:
        validation = {well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) == group}
        training = train - validation
        ok = bool(training) and bool(validation) and not (training & validation) and not (validation & test)
        inner_coverage.update(validation)
        inner_detail.append({"group": group, "train_wells": len(training), "validation_wells": len(validation), "pass": ok})
    passed = bool(train) and bool(test) and not (train & test) and inner_coverage == train and all(bool(item["pass"]) for item in inner_detail)
    return {"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "outer_train_wells": len(train), "outer_test_wells": len(test), "train_test_overlap": len(train & test), "inner_groups": len(inner_groups), "inner_coverage": len(inner_coverage), "pass": passed, "inner_detail_json": _canonical_json(inner_detail)}


def _target_matrix(records: Mapping[str, CoefficientWell], ids: Sequence[str], representation: str, *, mode: str = "normal", salt: str = "") -> np.ndarray:
    matrix = np.vstack([records[well_id].statistics[representation].target_coefficients for well_id in ids]).astype(np.float64)
    if mode == "sign_flipped":
        matrix = -matrix
    elif mode == "shuffled":
        order = sorted(range(len(ids)), key=lambda index: (hashlib.sha256(f"E011-shuffle|{salt}|{ids[index]}".encode()).hexdigest(), ids[index]))
        if len(order) > 1:
            order = order[1:] + order[:1]
        matrix = matrix[np.asarray(order, dtype=int)]
    elif mode != "normal":
        raise DataValidationError(f"E011 unknown target mode {mode}")
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.all(np.isfinite(matrix)):
        raise DataValidationError("E011 coefficient target matrix is invalid")
    return matrix


def _raw_feature_matrix(records: Mapping[str, CoefficientWell], ids: Sequence[str], names: Sequence[str]) -> np.ndarray:
    matrix = np.empty((len(ids), len(names)), dtype=np.float64)
    for row, well_id in enumerate(ids):
        for column, name in enumerate(names):
            value = records[well_id].features.get(name)
            matrix[row, column] = np.nan if value is None else float(value)
    return matrix


def _prepare_features(records: Mapping[str, CoefficientWell], train_ids: Sequence[str], test_ids: Sequence[str], names: Sequence[str], train_y: np.ndarray, maximum_features: int) -> PreparedFeatures:
    if not train_ids or not test_ids or train_y.shape[0] != len(train_ids):
        raise DataValidationError("E011 feature preparation membership differs")
    train = _raw_feature_matrix(records, train_ids, names)
    test = _raw_feature_matrix(records, test_ids, names)
    medians = np.zeros(train.shape[1], dtype=np.float64)
    for column in range(train.shape[1]):
        finite = train[np.isfinite(train[:, column]), column]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    train_z = (train - means) / scales
    test_z = (test - means) / scales
    y_mean = train_y.mean(axis=0)
    y_scale = train_y.std(axis=0)
    y_scale[~np.isfinite(y_scale) | (y_scale < 1e-9)] = 1.0
    y_z = (train_y - y_mean) / y_scale
    scores = []
    for index, name in enumerate(names):
        x = train_z[:, index]
        correlations = []
        if float(np.std(x)) > 1e-12:
            for target_index in range(y_z.shape[1]):
                y = y_z[:, target_index]
                if float(np.std(y)) > 1e-12:
                    value = float(np.corrcoef(x, y)[0, 1])
                    correlations.append(abs(value) if math.isfinite(value) else 0.0)
        scores.append((-max(correlations, default=0.0), name, index))
    scores.sort()
    chosen = sorted((item[2] for item in scores[: min(int(maximum_features), len(scores))]), key=lambda index: names[index])
    return PreparedFeatures(tuple(names[index] for index in chosen), train_z[:, chosen], test_z[:, chosen])


def _weights(records: Mapping[str, CoefficientWell], ids: Sequence[str], mode: str) -> np.ndarray | None:
    if mode == "equal_well":
        return None
    if mode == "row_weighted":
        values = np.asarray([records[well_id].rows for well_id in ids], dtype=np.float64)
        return values / max(float(values.mean()), 1.0)
    raise DataValidationError(f"E011 unknown sample weight mode {mode}")


def _fit_prepared_ridge(prepared: PreparedFeatures, train_y: np.ndarray, alpha: float, sample_weight: np.ndarray | None) -> np.ndarray:
    from sklearn.linear_model import Ridge
    y_mean = train_y.mean(axis=0)
    y_scale = train_y.std(axis=0)
    y_scale[~np.isfinite(y_scale) | (y_scale < 1e-9)] = 1.0
    scaled = (train_y - y_mean) / y_scale
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(prepared.train_x, scaled, sample_weight=sample_weight)
    prediction = np.asarray(model.predict(prepared.test_x), dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    values = prediction * y_scale + y_mean
    if values.shape != (prepared.test_x.shape[0], train_y.shape[1]) or not np.all(np.isfinite(values)):
        raise DataValidationError("E011 ridge emitted invalid coefficients")
    return values


def _fit_extra_trees(records: Mapping[str, CoefficientWell], train_ids: Sequence[str], test_ids: Sequence[str], feature_names: Sequence[str], representation: str, config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    from sklearn.ensemble import ExtraTreesRegressor
    train_y = _target_matrix(records, train_ids, representation)
    prepared = _prepare_features(records, train_ids, test_ids, feature_names, train_y, len(feature_names))
    y_mean = train_y.mean(axis=0)
    y_scale = train_y.std(axis=0)
    y_scale[~np.isfinite(y_scale) | (y_scale < 1e-9)] = 1.0
    scaled = (train_y - y_mean) / y_scale
    spec = config["model_families"]["extra_trees"]
    model = ExtraTreesRegressor(
        n_estimators=int(spec["n_estimators"]),
        max_depth=int(spec["max_depth"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        max_features=float(spec["max_features"]),
        bootstrap=bool(spec["bootstrap"]),
        random_state=int(spec["random_state"]),
        n_jobs=int(spec["n_jobs"]),
    )
    model.fit(prepared.train_x, scaled, sample_weight=_weights(records, train_ids, "row_weighted"))
    prediction = np.asarray(model.predict(prepared.test_x), dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    values = prediction * y_scale + y_mean
    if values.shape != (len(test_ids), train_y.shape[1]) or not np.all(np.isfinite(values)):
        raise DataValidationError("E011 ExtraTrees emitted invalid coefficients")
    return {well_id: values[index] for index, well_id in enumerate(test_ids)}, prepared.names


def _metrics_for_coefficients(records: Mapping[str, CoefficientWell], predictions: Mapping[str, np.ndarray], representation: str) -> list[WellMetric]:
    if not predictions:
        raise DataValidationError("E011 coefficient metric set is empty")
    return [records[well_id].statistics[representation].metric(well_id, predictions[well_id]) for well_id in sorted(predictions)]


def _ridge_selection(context: Context, records: Mapping[str, CoefficientWell], feature_names: Sequence[str], representation: str, config: Mapping[str, Any], model_name: str, *, target_mode: str = "normal") -> RidgeSelection:
    if model_name not in RIDGE_MODELS:
        raise DataValidationError("E011 unknown ridge branch")
    spec = config["model_families"][model_name]
    weight_mode = "equal_well" if model_name == "ridge_equal" else "row_weighted"
    grid_predictions: dict[tuple[int, float], dict[str, np.ndarray]] = {(int(features), float(alpha)): {} for features in spec["feature_counts"] for alpha in spec["alphas"]}
    groups = sorted({int(context.inner_assignments[well_id]) for well_id in context.train_ids})
    for group in groups:
        validation = tuple(well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) == group)
        training = tuple(well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) != group)
        if not training or not validation:
            continue
        train_y = _target_matrix(records, training, representation, mode=target_mode, salt=f"{context.key}|{group}|{model_name}")
        for feature_count in spec["feature_counts"]:
            prepared = _prepare_features(records, training, validation, feature_names, train_y, int(feature_count))
            sample_weight = _weights(records, training, weight_mode)
            for alpha in spec["alphas"]:
                values = _fit_prepared_ridge(prepared, train_y, float(alpha), sample_weight)
                grid_predictions[(int(feature_count), float(alpha))].update({well_id: values[index] for index, well_id in enumerate(validation)})
    if any(set(predictions) != set(context.train_ids) for predictions in grid_predictions.values()):
        raise DataValidationError(f"{context.key}: E011 ridge inner OOF coverage failed")
    scored = []
    for key, predictions in grid_predictions.items():
        bounded = {well_id: bound_coefficients(values, representation, config) for well_id, values in predictions.items()}
        rmse = float(_summarize(_metrics_for_coefficients(records, bounded, representation))["rmse"])
        scored.append((rmse, spec["feature_counts"].index(key[0]), spec["alphas"].index(key[1]), key, predictions))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    inner_rmse, _feature_order, _alpha_order, selected_key, inner_predictions = scored[0]
    train_y = _target_matrix(records, context.train_ids, representation, mode=target_mode, salt=f"{context.key}|outer|{model_name}")
    prepared = _prepare_features(records, context.train_ids, context.test_ids, feature_names, train_y, selected_key[0])
    values = _fit_prepared_ridge(prepared, train_y, selected_key[1], _weights(records, context.train_ids, weight_mode))
    outer = {well_id: values[index] for index, well_id in enumerate(context.test_ids)}
    return RidgeSelection(selected_key[0], selected_key[1], inner_rmse, inner_predictions, outer, prepared.names)


def _inner_extra_trees(context: Context, records: Mapping[str, CoefficientWell], feature_names: Sequence[str], representation: str, config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[str, ...]]:
    inner: dict[str, np.ndarray] = {}
    groups = sorted({int(context.inner_assignments[well_id]) for well_id in context.train_ids})
    for group in groups:
        validation = tuple(well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) == group)
        training = tuple(well_id for well_id in context.train_ids if int(context.inner_assignments[well_id]) != group)
        if not training or not validation:
            continue
        prediction, _ = _fit_extra_trees(records, training, validation, feature_names, representation, config)
        inner.update(prediction)
    if set(inner) != set(context.train_ids):
        raise DataValidationError(f"{context.key}: E011 tree inner OOF coverage failed")
    outer, names = _fit_extra_trees(records, context.train_ids, context.test_ids, feature_names, representation, config)
    return inner, outer, names


def _coefficient_magnitude(coefficients: np.ndarray, representation: str, config: Mapping[str, Any]) -> float:
    return _peak_correction(bound_coefficients(coefficients, representation, config), representation, config)


def _fallback_selection(records: Mapping[str, CoefficientWell], representation: str, ridge: Mapping[str, np.ndarray], tree: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> FallbackSelection:
    ids = tuple(sorted(ridge))
    if set(ids) != set(tree):
        raise DataValidationError("E011 fallback component coverage differs")
    disagreements = {well_id: float(np.max(np.abs(np.asarray(ridge[well_id]) - np.asarray(tree[well_id])))) for well_id in ids}
    blends = {well_id: 0.5 * (np.asarray(ridge[well_id]) + np.asarray(tree[well_id])) for well_id in ids}
    magnitudes = {well_id: _coefficient_magnitude(blends[well_id], representation, config) for well_id in ids}
    quantiles = tuple(float(value) for value in config["confidence_fallback"]["quantiles"])
    rows = []
    e006 = _summarize([records[well_id].e006_metric for well_id in ids])
    for shrinkage in config["shrinkages"]:
        for disagreement_quantile in quantiles:
            disagreement_threshold = float(_quantile(list(disagreements.values()), disagreement_quantile))
            for magnitude_quantile in quantiles:
                magnitude_threshold = float(_quantile(list(magnitudes.values()), magnitude_quantile))
                predictions: dict[str, np.ndarray] = {}
                actions = 0
                for well_id in ids:
                    action = disagreements[well_id] <= disagreement_threshold + 1e-12 and magnitudes[well_id] <= magnitude_threshold + 1e-12
                    actions += int(action)
                    raw = float(shrinkage) * blends[well_id] if action else np.zeros_like(blends[well_id])
                    predictions[well_id] = bound_coefficients(raw, representation, config)
                summary = _summarize(_metrics_for_coefficients(records, predictions, representation))
                action_rate = actions / len(ids)
                # Frozen literal gates from PRE_REGISTRATION/config.
                passed = float(e006["rmse"]) - float(summary["rmse"]) >= 0.05 and float(summary["p90_well_rmse"]) - float(e006["p90_well_rmse"]) <= 0.10 and action_rate >= 0.20
                rows.append((not passed, float(summary["rmse"]), config["shrinkages"].index(shrinkage), quantiles.index(disagreement_quantile), quantiles.index(magnitude_quantile), FallbackSelection(float(shrinkage), disagreement_quantile, magnitude_quantile, disagreement_threshold, magnitude_threshold, float(summary["rmse"]), action_rate, passed)))
    passing = [item for item in rows if item[-1].passed]
    if not passing:
        return FallbackSelection(0.0, 0.0, 0.0, -math.inf, -math.inf, float(e006["rmse"]), 0.0, False)
    passing.sort(key=lambda item: item[:-1])
    return passing[0][-1]


def _apply_fallback(selection: FallbackSelection, representation: str, ridge: Mapping[str, np.ndarray], tree: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    output: dict[str, np.ndarray] = {}
    detail: dict[str, dict[str, Any]] = {}
    for well_id in sorted(ridge):
        blend = 0.5 * (np.asarray(ridge[well_id]) + np.asarray(tree[well_id]))
        disagreement = float(np.max(np.abs(np.asarray(ridge[well_id]) - np.asarray(tree[well_id]))))
        magnitude = _coefficient_magnitude(blend, representation, config)
        action = bool(selection.passed and disagreement <= selection.disagreement_threshold + 1e-12 and magnitude <= selection.magnitude_threshold + 1e-12)
        coefficients = bound_coefficients(selection.shrinkage * blend if action else np.zeros_like(blend), representation, config)
        output[well_id] = coefficients
        detail[well_id] = {"action": action, "fallback": not action, "disagreement": disagreement, "magnitude": magnitude}
    return output, detail


def _fixed_candidate_predictions(spec: CandidateSpec, base: Mapping[str, Mapping[str, np.ndarray]], fallback: tuple[Mapping[str, np.ndarray], Mapping[str, Mapping[str, Any]]] | None, config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    if spec.confidence_fallback:
        if fallback is None:
            raise DataValidationError("E011 fallback prediction is unavailable")
        return dict(fallback[0]), {key: dict(value) for key, value in fallback[1].items()}
    if spec.model not in base:
        raise DataValidationError(f"E011 base model {spec.model} is unavailable")
    predictions = {well_id: bound_coefficients(float(spec.shrinkage) * values, spec.representation, config) for well_id, values in base[spec.model].items()}
    return predictions, {well_id: {"action": bool(np.any(np.abs(values) > 0.0)), "fallback": False, "disagreement": None, "magnitude": _coefficient_magnitude(values, spec.representation, config)} for well_id, values in predictions.items()}


def _fit_context_representation(context: Context, records: Mapping[str, CoefficientWell], feature_names: Sequence[str], representation: str, config: Mapping[str, Any], *, need_models: set[str] | None = None, run_negative_controls: bool = False) -> tuple[dict[str, dict[str, np.ndarray]], tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]] | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    needed = set(need_models or ({"ridge_equal"} if representation in {"shape3", "spline3"} else set(BASE_MODELS)))
    if "ridge_tree_blend" in needed:
        needed.update({"ridge_equal", "extra_trees"})
    base: dict[str, dict[str, np.ndarray]] = {}
    inner: dict[str, dict[str, np.ndarray]] = {}
    hyper_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    selected: dict[str, RidgeSelection] = {}
    for model_name in RIDGE_MODELS:
        if model_name not in needed:
            continue
        selection = _ridge_selection(context, records, feature_names, representation, config, model_name)
        selected[model_name] = selection
        base[model_name] = {well_id: np.asarray(value) for well_id, value in selection.outer_predictions.items()}
        inner[model_name] = {well_id: np.asarray(value) for well_id, value in selection.inner_predictions.items()}
        hyper_rows.append({"context": context.key, "scope": context.scope, "representation": representation, "model": model_name, "feature_count": selection.feature_count, "alpha": selection.alpha, "inner_rmse": selection.inner_rmse, "selected_features_json": _canonical_json(selection.selected_features), "fallback_shrinkage": "", "disagreement_quantile": "", "magnitude_quantile": "", "action_rate": ""})
    if "extra_trees" in needed:
        inner_tree, outer_tree, names = _inner_extra_trees(context, records, feature_names, representation, config)
        base["extra_trees"] = outer_tree
        inner["extra_trees"] = inner_tree
        hyper_rows.append({"context": context.key, "scope": context.scope, "representation": representation, "model": "extra_trees", "feature_count": len(names), "alpha": "", "inner_rmse": float(_summarize(_metrics_for_coefficients(records, {well_id: bound_coefficients(values, representation, config) for well_id, values in inner_tree.items()}, representation))["rmse"]), "selected_features_json": _canonical_json(names), "fallback_shrinkage": "", "disagreement_quantile": "", "magnitude_quantile": "", "action_rate": ""})
    if "ridge_tree_blend" in needed:
        base["ridge_tree_blend"] = {well_id: 0.5 * (base["ridge_equal"][well_id] + base["extra_trees"][well_id]) for well_id in context.test_ids}
        inner["ridge_tree_blend"] = {well_id: 0.5 * (inner["ridge_equal"][well_id] + inner["extra_trees"][well_id]) for well_id in context.train_ids}
    fallback = None
    if representation in PRIMARY_REPRESENTATIONS and "ridge_tree_blend" in needed:
        selection = _fallback_selection(records, representation, inner["ridge_equal"], inner["extra_trees"], config)
        fallback = _apply_fallback(selection, representation, base["ridge_equal"], base["extra_trees"], config)
        hyper_rows.append({"context": context.key, "scope": context.scope, "representation": representation, "model": "blend_confidence_fallback", "feature_count": "", "alpha": "", "inner_rmse": selection.inner_rmse, "selected_features_json": "", "fallback_shrinkage": selection.shrinkage, "disagreement_quantile": selection.disagreement_quantile, "magnitude_quantile": selection.magnitude_quantile, "action_rate": selection.action_rate})
    for model_name, predictions in base.items():
        for well_id in sorted(predictions):
            values = np.asarray(predictions[well_id], dtype=np.float64)
            row = {"context": context.key, "scope": context.scope, "representation": representation, "model": model_name, "well_id": well_id}
            for index in range(7):
                row[f"coefficient_{index}"] = float(values[index]) if index < values.size else ""
            model_rows.append(row)
    controls: dict[str, np.ndarray] = {}
    if run_negative_controls and representation == "spline4":
        for mode, name in (("shuffled", NEGATIVE_CONTROLS[0]), ("sign_flipped", NEGATIVE_CONTROLS[1])):
            selection = _ridge_selection(context, records, feature_names, representation, config, "ridge_equal", target_mode=mode)
            controls[name] = np.vstack([selection.outer_predictions[well_id] for well_id in context.test_ids])
            hyper_rows.append({"context": context.key, "scope": context.scope, "representation": representation, "model": name, "feature_count": selection.feature_count, "alpha": selection.alpha, "inner_rmse": selection.inner_rmse, "selected_features_json": _canonical_json(selection.selected_features), "fallback_shrinkage": "", "disagreement_quantile": "", "magnitude_quantile": "", "action_rate": ""})
    return base, fallback, hyper_rows, model_rows, controls


def _candidate_family(spec: CandidateSpec) -> str:
    return spec.model if not spec.confidence_fallback else "confidence_fallback"


def _candidate_order(config: Mapping[str, Any], specs: Sequence[CandidateSpec]) -> dict[str, int]:
    base_order = {name: index for index, name in enumerate(config["candidate_order"])}
    shrink_order = {float(value): index for index, value in enumerate(config["shrinkages"])}
    keys = []
    for spec in specs:
        if spec.negative_control:
            key = (10_000, 0)
        elif spec.confidence_fallback:
            base = f"{spec.representation}_blend_confidence_fallback"
            key = (base_order.get(base, 9_000), 0)
        else:
            base = f"{spec.representation}_{spec.model}"
            key = (base_order.get(base, 9_000), shrink_order.get(float(spec.shrinkage), 99))
        keys.append((key, spec.name))
    keys.sort()
    return {name: index for index, (_key, name) in enumerate(keys)}


def _advance_candidates(specs: Sequence[CandidateSpec], summaries: Mapping[str, Mapping[str, Any]], stage_gates: Mapping[str, Mapping[str, bool]], order: Mapping[str, int], limit: int = 8) -> list[str]:
    passing = [spec for spec in specs if spec.eligible and all(stage_gates[spec.name].values())]
    passing.sort(key=lambda spec: (float(summaries[spec.name]["rmse"]), order[spec.name]))
    if len(passing) <= limit:
        return [spec.name for spec in passing]
    selected: list[CandidateSpec] = []
    for representation in LEARNED_REPRESENTATIONS:
        match = next((spec for spec in passing if spec.representation == representation), None)
        if match and match not in selected and len(selected) < limit:
            selected.append(match)
    for model in ("ridge_equal", "ridge_row_weighted", "extra_trees", "ridge_tree_blend", "confidence_fallback"):
        match = next((spec for spec in passing if _candidate_family(spec) == model), None)
        if match and match not in selected and len(selected) < limit:
            selected.append(match)
    for spec in passing:
        if spec not in selected and len(selected) < limit:
            selected.append(spec)
    return [spec.name for spec in selected]


def _coefficient_metrics(records: Mapping[str, CoefficientWell], final_coefficients: Mapping[str, Mapping[str, np.ndarray]], specs: Mapping[str, CandidateSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, by_well in final_coefficients.items():
        spec = specs[candidate]
        ids = sorted(by_well)
        actual = np.vstack([records[well_id].statistics[spec.representation].target_coefficients for well_id in ids])
        predicted = np.vstack([by_well[well_id] for well_id in ids])
        for index in range(actual.shape[1]):
            y = actual[:, index]; p = predicted[:, index]
            error = p - y
            denominator = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - float(error @ error) / denominator if denominator > 1e-12 else 0.0
            correlation = float(np.corrcoef(y, p)[0, 1]) if float(np.std(y)) > 1e-12 and float(np.std(p)) > 1e-12 else 0.0
            rows.append({"candidate": candidate, "representation": spec.representation, "coefficient_index": index, "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error))), "r2": r2, "pearson": correlation, "actual_std": float(np.std(y)), "predicted_std": float(np.std(p))})
    return rows



def _direct_scoring_control(parent_oof: Path, records: Mapping[str, CoefficientWell], config: Mapping[str, Any]) -> tuple[float, float]:
    """Compare frozen basis/statistics with direct row scoring on deterministic real wells."""
    ids = tuple(sorted(records))
    selected = {ids[0], ids[len(ids) // 2], ids[-1]}
    matrices = {(well_id, representation): basis_matrix(records[well_id].rows, representation, config) for well_id in selected for representation in LEARNED_REPRESENTATIONS}
    coefficients = {
        (well_id, representation): bound_coefficients(0.73 * records[well_id].statistics[representation].target_coefficients, representation, config)
        for well_id in selected for representation in LEARNED_REPRESENTATIONS
    }
    accumulators = {(well_id, representation): ErrorAccumulator() for well_id in selected for representation in LEARNED_REPRESENTATIONS}
    basis_delta = 0.0
    with gzip.open(parent_oof, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            well_id = str(row["well_id"])
            if well_id not in selected:
                continue
            hidden_index = int(row["hidden_index"])
            target = _finite(row["target"], "direct-control target")
            e006 = _finite(row["e006_nested_fusion"], "direct-control E006")
            position = hidden_index / max(1, records[well_id].rows - 1)
            for representation in LEARNED_REPRESENTATIONS:
                direct_basis = basis_vector(position, representation, config)
                matrix_basis = matrices[(well_id, representation)][hidden_index]
                basis_delta = max(basis_delta, float(np.max(np.abs(direct_basis - matrix_basis))))
                error = e006 + float(direct_basis @ coefficients[(well_id, representation)]) - target
                accumulators[(well_id, representation)].add(error, float(hidden_index))
    maximum_relative = 0.0
    for key, accumulator in accumulators.items():
        well_id, representation = key
        direct = accumulator.finalize(well_id)
        sufficient = records[well_id].statistics[representation].metric(well_id, coefficients[key])
        maximum_relative = max(maximum_relative, abs(direct.sse - sufficient.sse) / max(1.0, direct.sse))
        maximum_relative = max(maximum_relative, abs(direct.mean_error - sufficient.mean_error) / max(1.0, abs(direct.mean_error)))
        maximum_relative = max(maximum_relative, abs(direct.trend_per_row - sufficient.trend_per_row) / max(1.0, abs(direct.trend_per_row)))
    return basis_delta, maximum_relative

def _write_gzip_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)


def _safe_relative(path: Path, base: Path) -> str:
    try:
        value = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError as exc:
        raise DataValidationError(f"E011 artifact {path} is outside {base}") from exc
    pure = PurePosixPath(value)
    if pure.is_absolute() or not value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise DataValidationError("E011 unsafe artifact path")
    return value


def finalize_e011_outputs(*, root: Path, output_dir: Path, artifact_dir: Path, config: Mapping[str, Any], code_sha: str) -> dict[str, Any]:
    output_dir = output_dir.resolve(); artifact_dir = artifact_dir.resolve(); root = root.resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise DataValidationError("E011 summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment_id") != "E011" or summary.get("code_sha") != code_sha:
        raise DataValidationError("E011 summary identity differs")
    result_files = [output_dir / name for name in E011_RESULT_FILENAMES]
    if any(not path.is_file() for path in result_files):
        missing = [path.name for path in result_files if not path.is_file()]
        raise DataValidationError(f"E011 result files missing: {missing}")
    oof_path = artifact_dir / E011_OOF_FILENAME
    if not oof_path.is_file():
        raise DataValidationError("E011 OOF artifact is missing")
    with gzip.open(oof_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        count = sum(1 for _ in reader)
    if count != int(config["expected_hidden_rows"]):
        raise DataValidationError("E011 OOF row count differs")
    config_path = root / "experiments/E011/config.json"
    manifest = {
        "schema_version": 1,
        "experiment_id": "E011",
        "code_sha": code_sha,
        "config": {"path": "experiments/E011/config.json", "sha256": _sha256(config_path), "bytes": config_path.stat().st_size},
        "parent_artifacts": [{"path": item["path"], "sha256": _sha256(root / item["path"]), "bytes": (root / item["path"]).stat().st_size} for item in config["parent_artifacts"].values()],
        "fold_files": [{"path": item["path"], "sha256": _sha256(root / item["path"]), "bytes": (root / item["path"]).stat().st_size} for item in config["fold_files"]],
        "files": [{"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size} for path in result_files],
        "external_artifacts": [{"path": _safe_relative(oof_path, artifact_dir), "base": "artifact_dir", "kind": "oof_predictions_gzip", "sha256": _sha256(oof_path), "bytes": oof_path.stat().st_size, "tracked": False}],
    }
    manifest_path = output_dir / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise DataValidationError("E011 manifest round-trip differs")
    return summary


def run_e011(*, root: Path, train_dir: Path, output_dir: Path, artifact_dir: Path, config: Mapping[str, Any], code_sha: str) -> dict[str, Any]:
    started = time.perf_counter()
    validate_e011_config(config)
    root = root.resolve(); train_dir = train_dir.resolve(); output_dir = output_dir.resolve(); artifact_dir = artifact_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True); artifact_dir.mkdir(parents=True, exist_ok=True)
    parent_control = _verify_parent_hashes(root, config)
    if not parent_control["pass"]:
        raise DataValidationError("E011 parent hash audit failed")
    _profiles, data_profile = scan_profiles(train_dir)
    if int(data_profile["well_count"]) != int(config["expected_wells"]) or str(data_profile["data_signature"]) != str(config["data_signature"]):
        raise DataValidationError("E011 competition data identity differs")
    records, feature_names, record_audit = _read_records(root, train_dir, config)
    folds = _load_folds(root, config, sorted(records))
    contexts, membership_rows, spatial_assignments, typewell_assignments = _build_contexts(records, folds, config)
    repeated_contexts = [context for context in contexts if context.scope == "repeated"]
    stress_contexts = [context for context in contexts if context.scope != "repeated"]
    specs_tuple = candidate_specs(config)
    specs = {spec.name: spec for spec in specs_tuple}
    order = _candidate_order(config, specs_tuple)
    well_ids = tuple(sorted(records)); well_index = {well_id: index for index, well_id in enumerate(well_ids)}
    repeated_sums = {spec.name: np.zeros((len(well_ids), int(config["representations"][spec.representation]["dimensions"])), dtype=np.float64) for spec in specs_tuple}
    repeated_counts = {spec.name: np.zeros(len(well_ids), dtype=np.int16) for spec in specs_tuple}
    context_metrics: dict[str, dict[str, dict[str, WellMetric]]] = {}
    outer_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    feature_frequency: Counter[tuple[str, str, str]] = Counter()
    maximum_coefficient = 0.0; maximum_correction = 0.0; exact_fallback_delta = 0.0
    duplicate_model_delta = 0.0

    for context_index, context in enumerate(repeated_contexts):
        by_rep: dict[str, tuple[dict[str, dict[str, np.ndarray]], tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]] | None, dict[str, np.ndarray]]] = {}
        for representation in LEARNED_REPRESENTATIONS:
            base, fallback, rows_h, rows_m, controls = _fit_context_representation(context, records, feature_names, representation, config, run_negative_controls=True)
            by_rep[representation] = (base, fallback, controls)
            hyper_rows.extend(rows_h); model_rows.extend(rows_m)
            for row in rows_h:
                if row["selected_features_json"]:
                    for feature in json.loads(row["selected_features_json"]):
                        feature_frequency[(representation, str(row["model"]), feature)] += 1
            if context_index == 0 and representation == "spline4":
                selection = _ridge_selection(context, records, feature_names, representation, config, "ridge_equal")
                original = base["ridge_equal"]
                for well_id in context.test_ids:
                    duplicate_model_delta = max(duplicate_model_delta, float(np.max(np.abs(original[well_id] - selection.outer_predictions[well_id]))))
        metrics = {name: {} for name in (*COMPARATORS, *(spec.name for spec in specs_tuple))}
        for well_id in context.test_ids:
            metrics["last_known_tvt"][well_id] = records[well_id].last_metric
            metrics["e006_nested_fusion"][well_id] = records[well_id].e006_metric
        for spec in specs_tuple:
            base, fallback, control_predictions = by_rep[spec.representation]
            if spec.negative_control:
                matrix = control_predictions[spec.name]
                predictions = {well_id: bound_coefficients(matrix[index], spec.representation, config) for index, well_id in enumerate(context.test_ids)}
                detail = {well_id: {"fallback": False} for well_id in context.test_ids}
            else:
                predictions, detail = _fixed_candidate_predictions(spec, base, fallback, config)
            for well_id, coefficients in predictions.items():
                index = well_index[well_id]
                repeated_sums[spec.name][index] += coefficients
                repeated_counts[spec.name][index] += 1
                metric = records[well_id].statistics[spec.representation].metric(well_id, coefficients)
                metrics[spec.name][well_id] = metric
                maximum_coefficient = max(maximum_coefficient, float(np.max(np.abs(coefficients))))
                maximum_correction = max(maximum_correction, _peak_correction(coefficients, spec.representation, config))
                if bool(detail[well_id].get("fallback")):
                    exact_fallback_delta = max(exact_fallback_delta, float(np.max(np.abs(coefficients))))
        context_metrics[context.key] = metrics
        for candidate, by_well in metrics.items():
            outer_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "candidate": candidate, **_summarize([by_well[well_id] for well_id in sorted(by_well)])})

    expected_maps = len(folds)
    final_coefficients: dict[str, dict[str, np.ndarray]] = {spec.name: {} for spec in specs_tuple}
    final_metrics: dict[str, dict[str, WellMetric]] = {name: {} for name in (*COMPARATORS, *(spec.name for spec in specs_tuple))}
    for well_id in well_ids:
        final_metrics["last_known_tvt"][well_id] = records[well_id].last_metric
        final_metrics["e006_nested_fusion"][well_id] = records[well_id].e006_metric
    for spec in specs_tuple:
        if np.any(repeated_counts[spec.name] != expected_maps):
            raise DataValidationError(f"E011 repeated coverage differs for {spec.name}")
        averaged = repeated_sums[spec.name] / repeated_counts[spec.name][:, None]
        for index, well_id in enumerate(well_ids):
            coefficients = bound_coefficients(averaged[index], spec.representation, config)
            final_coefficients[spec.name][well_id] = coefficients
            final_metrics[spec.name][well_id] = records[well_id].statistics[spec.representation].metric(well_id, coefficients)
    summaries = {candidate: _summarize([final_metrics[candidate][well_id] for well_id in well_ids]) for candidate in final_metrics}

    map_rows: list[dict[str, Any]] = []
    map_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for fold in folds:
        version = str(fold["version"])
        keys = [f"repeated:{version}:{group}" for group in range(int(fold["n_folds"]))]
        for candidate in final_metrics:
            row = {"map": version, "candidate": candidate, **_summarize([metric for key in keys for metric in context_metrics[key][candidate].values()])}
            map_rows.append(row); map_lookup[(version, candidate)] = row
    outer_lookup = {(row["context"], row["candidate"]): row for row in outer_rows}
    e006_rmse = float(summaries["e006_nested_fusion"]["rmse"])
    last_rmse = float(summaries["last_known_tvt"]["rmse"])
    stage_gates: dict[str, dict[str, bool]] = {}
    map_wins: dict[str, int] = {}; outer_wins: dict[str, int] = {}; retention: dict[str, float] = {}
    for spec in specs_tuple:
        candidate = spec.name
        map_wins[candidate] = sum(float(map_lookup[(str(fold["version"]), "e006_nested_fusion")]["rmse"]) - float(map_lookup[(str(fold["version"]), candidate)]["rmse"]) >= float(config["promotion"]["minimum_map_gain"]) for fold in folds)
        outer_wins[candidate] = sum(float(outer_lookup[(context.key, candidate)]["rmse"]) < float(outer_lookup[(context.key, "e006_nested_fusion")]["rmse"]) for context in repeated_contexts)
        oracle_rmse = float(config["representations"][spec.representation]["oracle_rmse"])
        available = max(1e-12, e006_rmse - oracle_rmse)
        retention[candidate] = (e006_rmse - float(summaries[candidate]["rmse"])) / available
        stage_gates[candidate] = {
            "eligible": spec.eligible,
            "gain": e006_rmse - float(summaries[candidate]["rmse"]) >= float(config["evaluation"]["preliminary_advancement"]["minimum_gain_vs_e006"]),
            "maps": map_wins[candidate] >= int(config["evaluation"]["preliminary_advancement"]["minimum_map_wins"]),
            "outer_cells": outer_wins[candidate] >= int(config["evaluation"]["preliminary_advancement"]["minimum_outer_cell_wins"]),
            "oracle_gain_retention": retention[candidate] >= float(config["evaluation"]["preliminary_advancement"]["minimum_representation_oracle_gain_retention"]),
        }
    advanced = _advance_candidates(specs_tuple, summaries, stage_gates, order, 8)

    stress_rows: list[dict[str, Any]] = []
    stress_lookup: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    needed_by_rep: dict[str, set[str]] = defaultdict(set)
    for candidate in advanced:
        spec = specs[candidate]
        needed_by_rep[spec.representation].add(spec.model)
        if spec.confidence_fallback:
            needed_by_rep[spec.representation].add("ridge_tree_blend")
    for context in stress_contexts:
        by_rep = {}
        for representation, needed in needed_by_rep.items():
            base, fallback, rows_h, rows_m, _controls = _fit_context_representation(context, records, feature_names, representation, config, need_models=needed)
            by_rep[representation] = (base, fallback)
            hyper_rows.extend(rows_h); model_rows.extend(rows_m)
            for row in rows_h:
                if row["selected_features_json"]:
                    for feature in json.loads(row["selected_features_json"]):
                        feature_frequency[(representation, str(row["model"]), feature)] += 1
        metrics = {"e006_nested_fusion": {well_id: records[well_id].e006_metric for well_id in context.test_ids}}
        for candidate in advanced:
            spec = specs[candidate]
            predictions, _detail = _fixed_candidate_predictions(spec, by_rep[spec.representation][0], by_rep[spec.representation][1], config)
            metrics[candidate] = {well_id: records[well_id].statistics[spec.representation].metric(well_id, coefficients) for well_id, coefficients in predictions.items()}
        for candidate, by_well in metrics.items():
            row = {"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "candidate": candidate, **_summarize([by_well[well_id] for well_id in sorted(by_well)])}
            stress_rows.append(row); stress_lookup[(context.scope, context.outer_group, candidate)] = row

    long_threshold = float(_quantile([record.rows for record in records.values()], 0.8))
    missing_threshold = float(_quantile([record.hidden_gr_missing_fraction for record in records.values()], 0.8))
    special_sets = {
        "long_suffix": [well_id for well_id, record in records.items() if record.rows >= long_threshold],
        "high_gr_missingness": [well_id for well_id, record in records.items() if record.hidden_gr_missing_fraction >= missing_threshold],
        "e006_catastrophe": [well_id for well_id, record in records.items() if record.e006_metric.rmse >= 12.0],
    }
    special_rows = []
    for name, ids in special_sets.items():
        if not ids:
            continue
        for candidate in final_metrics:
            special_rows.append({"slice": name, "candidate": candidate, **_summarize([final_metrics[candidate][well_id] for well_id in ids])})
    special_lookup = {(row["slice"], row["candidate"]): row for row in special_rows}

    def special_positive(name: str, candidate: str) -> bool:
        ids = special_sets[name]
        if not ids:
            return True
        return float(special_lookup[(name, "e006_nested_fusion")]["rmse"]) - float(special_lookup[(name, candidate)]["rmse"]) > 0.0

    gates: dict[str, dict[str, bool]] = {}
    mean_map_rmse = {candidate: sum(float(map_lookup[(str(fold["version"]), candidate)]["rmse"]) for fold in folds) / len(folds) for candidate in final_metrics}
    for spec in specs_tuple:
        candidate = spec.name
        spatial_positive = candidate in advanced and all(float(stress_lookup[("spatial", group, "e006_nested_fusion")]["rmse"]) - float(stress_lookup[("spatial", group, candidate)]["rmse"]) > 0.0 for group in range(5))
        typewell_positive = candidate in advanced and all(float(stress_lookup[("typewell", group, "e006_nested_fusion")]["rmse"]) - float(stress_lookup[("typewell", group, candidate)]["rmse"]) > 0.0 for group in range(5))
        gates[candidate] = {
            "advanced": candidate in advanced,
            "gain_vs_e006": e006_rmse - float(summaries[candidate]["rmse"]) >= float(config["promotion"]["minimum_gain_vs_e006"]),
            "gain_vs_last_known": last_rmse - float(summaries[candidate]["rmse"]) >= float(config["promotion"]["minimum_gain_vs_last_known"]),
            "oracle_gain_retention": retention[candidate] >= float(config["promotion"]["minimum_representation_oracle_gain_retention"]),
            "map_wins": map_wins[candidate] >= int(config["promotion"]["minimum_map_wins"]),
            "outer_cells": outer_wins[candidate] >= int(config["promotion"]["minimum_outer_cell_wins"]),
            "p90": float(summaries[candidate]["p90_well_rmse"]) - float(summaries["e006_nested_fusion"]["p90_well_rmse"]) <= float(config["promotion"]["maximum_p90_deterioration_vs_e006"]),
            "worst5": float(summaries[candidate]["worst_5pct_sse_share"]) - float(summaries["e006_nested_fusion"]["worst_5pct_sse_share"]) <= float(config["promotion"]["maximum_worst5_sse_share_increase_vs_e006"]),
            "spatial": spatial_positive if bool(config["promotion"]["require_positive_every_spatial_group"]) else True,
            "typewell": typewell_positive if bool(config["promotion"]["require_positive_every_typewell_group"]) else True,
            "long_suffix": special_positive("long_suffix", candidate) if bool(config["promotion"]["require_positive_long_suffix"]) else True,
            "high_gr_missingness": special_positive("high_gr_missingness", candidate) if bool(config["promotion"]["require_positive_high_gr_missingness"]) else True,
            "e006_catastrophe": special_positive("e006_catastrophe", candidate) if bool(config["promotion"]["require_positive_e006_catastrophic_slice"]) else True,
            "controls": True,
        }

    zero_delta = max(abs(record.e006_metric.sse - record.statistics["spline4"].metric(well_id, np.zeros(4)).sse) for well_id, record in records.items())
    pooled_relative = max(abs(sum(metric.sse for metric in by_well.values()) - float(summaries[candidate]["sse"])) / max(1.0, float(summaries[candidate]["sse"])) for candidate, by_well in final_metrics.items())
    parent_oof = root / str(config["parent_artifacts"]["e010_oof"]["path"])
    basis_direct_delta, sufficient_direct_relative = _direct_scoring_control(parent_oof, records, config)
    shuffled_gain = e006_rmse - float(summaries[NEGATIVE_CONTROLS[0]]["rmse"])
    sign_gain = e006_rmse - float(summaries[NEGATIVE_CONTROLS[1]]["rmse"])
    controls: dict[str, Any] = {
        "parent_hashes": parent_control,
        "data_identity": {"pass": record_audit["wells"] == int(config["expected_wells"]) and record_audit["rows"] == int(config["expected_hidden_rows"]) and data_profile["data_signature"] == config["data_signature"], **record_audit, "data_signature": data_profile["data_signature"]},
        "feature_schema": record_audit["feature_schema"],
        "membership": {"pass": all(bool(row["pass"]) for row in membership_rows), "contexts": len(membership_rows)},
        "target_split_isolation": {"pass": all(not (set(context.train_ids) & set(context.test_ids)) for context in contexts), "contexts": len(contexts)},
        "zero_coefficient_e006": {"pass": zero_delta <= float(config["controls"]["zero_coefficient_equals_e006_max_delta"]), "maximum_sse_delta": zero_delta},
        "basis_reconstruction": {"pass": basis_direct_delta <= float(config["controls"]["basis_reconstruction_max_delta"]), "maximum_delta": basis_direct_delta},
        "sufficient_statistic_direct": {"pass": sufficient_direct_relative <= float(config["controls"]["coefficient_to_path_sufficient_statistic_max_relative_delta"]), "maximum_relative_delta": sufficient_direct_relative},
        "pooled_sse": {"pass": pooled_relative <= float(config["controls"]["pooled_sse_relative_tolerance"]), "maximum_relative_delta": pooled_relative},
        "finite_predictions": {"pass": all(math.isfinite(float(value)) for summary in summaries.values() for value in (summary["rmse"], summary["sse"]))},
        "exact_fallback": {"pass": exact_fallback_delta <= float(config["controls"]["exact_fallback_max_delta"]), "maximum_coefficient_delta": exact_fallback_delta},
        "coefficient_bounds": {"pass": maximum_coefficient <= 80.0 + 1e-8, "maximum_absolute_coefficient": maximum_coefficient},
        "emitted_correction_bounds": {"pass": maximum_correction <= 120.0 + 1e-8, "maximum_absolute_correction": maximum_correction},
        "shuffled_target": {"pass": shuffled_gain <= float(config["controls"]["shuffled_target_maximum_gain_vs_e006"]), "gain_vs_e006": shuffled_gain},
        "sign_flipped_target": {"pass": sign_gain <= float(config["controls"]["sign_flipped_target_maximum_gain_vs_e006"]), "gain_vs_e006": sign_gain},
        "duplicate_model": {"pass": duplicate_model_delta <= float(config["controls"]["duplicate_model_max_delta"]), "maximum_delta": duplicate_model_delta},
        "deterministic_reproduction": {"pass": duplicate_model_delta <= float(config["controls"]["duplicate_model_max_delta"]), "scope": "first repeated context duplicate ridge fit", "maximum_delta": duplicate_model_delta},
        "maximum_threads": {"pass": int(config["resource_design"]["maximum_threads"]) == 2, "maximum_threads": config["resource_design"]["maximum_threads"]},
    }

    # Select the statistically passing candidate before resource controls.  The OOF
    # artifact then remains identical to the final reported candidate even if a
    # runtime or memory control later blocks promotion.
    reported_pre = min((spec.name for spec in specs_tuple if spec.eligible), key=lambda candidate: (float(summaries[candidate]["rmse"]), order[candidate]))
    statistical_passing = [
        spec.name for spec in specs_tuple
        if spec.eligible and all(value for key, value in gates[spec.name].items() if key != "controls")
    ]
    statistical_selected = min(statistical_passing, key=lambda candidate: (mean_map_rmse[candidate], order[candidate])) if statistical_passing else None
    reported_for_oof = statistical_selected or reported_pre
    oof_path = artifact_dir / E011_OOF_FILENAME
    row_count = 0; previous_key: tuple[str, int] | None = None; unique_order = True
    def oof_rows() -> Iterator[dict[str, Any]]:
        nonlocal row_count, previous_key, unique_order
        with gzip.open(parent_oof, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                well_id = str(row["well_id"]); hidden_index = int(row["hidden_index"])
                spec = specs[reported_for_oof]
                coefficients = final_coefficients[reported_for_oof][well_id]
                correction = float(basis_vector(hidden_index / max(1, records[well_id].rows - 1), spec.representation, config) @ coefficients)
                e006 = _finite(row["e006_nested_fusion"], "E006")
                key = (well_id, int(row["row_index"]))
                if previous_key is not None and key == previous_key:
                    unique_order = False
                previous_key = key; row_count += 1
                yield {"id": row["id"], "well_id": well_id, "row_index": row["row_index"], "hidden_index": hidden_index, "target": row["target"], "last_known_tvt": row["last_known_tvt"], "e006_nested_fusion": f"{e006:.8f}", reported_for_oof: f"{e006 + correction:.8f}"}
    _write_gzip_csv(oof_path, ["id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt", "e006_nested_fusion", reported_for_oof], oof_rows())
    controls["oof_identity"] = {"pass": row_count == int(config["expected_hidden_rows"]) and unique_order, "rows": row_count, "ordered_unique_keys": unique_order}

    wall_seconds = time.perf_counter() - started
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    controls["runtime"] = {"pass": wall_seconds <= 60.0 * float(config["promotion"]["maximum_runtime_minutes"]), "wall_seconds": wall_seconds, "budget_minutes": config["promotion"]["maximum_runtime_minutes"]}
    controls["memory"] = {"pass": max_rss_kb <= 1024.0 * float(config["promotion"]["maximum_rss_mb"]), "max_rss_kb": max_rss_kb, "budget_mb": config["promotion"]["maximum_rss_mb"]}
    all_controls = all(bool(item["pass"]) for item in controls.values())
    for spec in specs_tuple:
        gates[spec.name]["controls"] = all_controls
        gates[spec.name]["runtime"] = bool(controls["runtime"]["pass"])
        gates[spec.name]["memory"] = bool(controls["memory"]["pass"])
    passing = [spec.name for spec in specs_tuple if spec.eligible and all(gates[spec.name].values())]
    selected = min(passing, key=lambda candidate: (mean_map_rmse[candidate], order[candidate])) if passing else None
    reported = selected or reported_for_oof
    status = "promoted" if selected else "rejected"

    candidate_rows = [{"candidate": candidate, "eligible": specs[candidate].eligible, "negative_control": specs[candidate].negative_control, "representation": specs[candidate].representation, "model": specs[candidate].model, "shrinkage": specs[candidate].shrinkage if specs[candidate].shrinkage is not None else "", "confidence_fallback": specs[candidate].confidence_fallback, "advanced": candidate in advanced, "selected": candidate == selected, "reported": candidate == reported, "gain_vs_e006": e006_rmse - float(summaries[candidate]["rmse"]), "oracle_gain_retention": retention[candidate], "map_wins": map_wins[candidate], "outer_cell_wins": outer_wins[candidate], "mean_map_rmse": mean_map_rmse[candidate], **summaries[candidate]} for candidate in specs]
    comparator_rows = [{"candidate": name, "eligible": False, "negative_control": False, "representation": "", "model": "", "shrinkage": "", "confidence_fallback": False, "advanced": False, "selected": False, "reported": False, "gain_vs_e006": e006_rmse - float(summaries[name]["rmse"]), "oracle_gain_retention": "", "map_wins": "", "outer_cell_wins": "", "mean_map_rmse": mean_map_rmse[name], **summaries[name]} for name in COMPARATORS]
    target_rows = []
    for well_id in well_ids:
        for representation in LEARNED_REPRESENTATIONS:
            values = records[well_id].statistics[representation].target_coefficients
            metric = records[well_id].statistics[representation].metric(well_id, values)
            row = {"well_id": well_id, "representation": representation, "rows": records[well_id].rows, "oracle_rmse": metric.rmse, "oracle_sse": metric.sse}
            for index in range(7):
                row[f"coefficient_{index}"] = float(values[index]) if index < values.size else ""
            target_rows.append(row)
    coefficient_rows = _coefficient_metrics(records, final_coefficients, specs)
    feature_rows = [{"representation": representation, "model": model, "feature": feature, "outer_fit_count": count} for (representation, model, feature), count in sorted(feature_frequency.items())]
    control_rows = [{"control": name, "pass": bool(item["pass"]), "details": _canonical_json(item)} for name, item in sorted(controls.items())]
    output_map = {
        "candidate_metrics.csv": comparator_rows + candidate_rows,
        "coefficient_metrics.csv": coefficient_rows,
        "map_metrics.csv": map_rows,
        "outer_cell_metrics.csv": outer_rows,
        "stress_metrics.csv": stress_rows,
        "special_slice_metrics.csv": special_rows,
        "selected_hyperparameters.csv": hyper_rows,
        "selected_feature_frequency.csv": feature_rows,
        "controls.csv": control_rows,
        "membership_audit.csv": membership_rows,
        "representation_targets.csv": target_rows,
        "model_predictions.csv": model_rows,
    }
    for filename, rows in output_map.items():
        if not rows:
            raise DataValidationError(f"E011 output {filename} is empty")
        _write_csv(output_dir / filename, list(rows[0]), rows)
    summary = {
        "schema_version": 1,
        "experiment_id": "E011",
        "status": status,
        "code_sha": code_sha,
        "selected_candidate": selected,
        "reported_candidate": reported,
        "eligible_candidates": passing,
        "advanced_candidates": advanced,
        "candidate_count": len(specs_tuple),
        "baseline_metrics": summaries["last_known_tvt"],
        "e006_metrics": summaries["e006_nested_fusion"],
        "candidate_metrics": summaries,
        "representation_oracle_metrics": {name: _summarize([records[well_id].statistics[name].metric(well_id, records[well_id].statistics[name].target_coefficients) for well_id in well_ids]) for name in LEARNED_REPRESENTATIONS},
        "map_wins": map_wins,
        "outer_cell_wins": outer_wins,
        "oracle_gain_retention": retention,
        "preliminary_gates": stage_gates,
        "gates_by_candidate": gates,
        "controls": controls,
        "stress": {"long_suffix_threshold": long_threshold, "high_gr_missingness_threshold": missing_threshold, "special_slice_wells": {name: len(ids) for name, ids in special_sets.items()}, "spatial_assignments": spatial_assignments, "typewell_assignments": typewell_assignments},
        "runtime": {"wall_seconds": wall_seconds, "max_rss_kb": max_rss_kb},
        "deployment": {"statistically_authorized": bool(selected), "local_package_built": False, "local_notebook_parity": False, "private_kaggle_parity": False, "deployment_ready": False, "submission_created": False, "submission_made": False, "reason": "Statistical promotion passed; package and parity remain required." if selected else "No E011 candidate passed every frozen gate; retain E006."},
    }
    _write_json(output_dir / "summary.json", summary)
    finalized = finalize_e011_outputs(root=root, output_dir=output_dir, artifact_dir=artifact_dir, config=config, code_sha=code_sha)
    if finalized != summary:
        raise DataValidationError("E011 finalized summary differs")
    return summary
