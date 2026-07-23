#!/usr/bin/env python3
"""T016: E011-relative placement screen for saved E009 legal datum actions."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.harness import DataValidationError, WellMetric, _summarize  # noqa: E402

BASE_FAMILIES = (
    "ridge_all_a25_f32",
    "ridge_all_a100_f32",
    "ridge_all_a25_f64",
    "ridge_no_e007_a25_f32",
    "ridge_path_evidence_a25_f24",
    "ridge_visible_geometry_a25_f32",
)


@dataclass(frozen=True)
class WellStats:
    well_id: str
    rows: int
    sum_error: float
    base_sse: float
    trend_sse: float
    shape_sse: float

    def metric(self, correction: float) -> WellMetric:
        if not math.isfinite(correction):
            raise DataValidationError(f"{self.well_id}: non-finite correction")
        n = float(self.rows)
        new_sum = self.sum_error + n * correction
        sse = self.base_sse + 2.0 * correction * self.sum_error + n * correction * correction
        tolerance = max(1e-8, self.base_sse * 1e-11)
        if sse < 0.0 and abs(sse) <= tolerance:
            sse = 0.0
        if sse < 0.0:
            raise ArithmeticError(f"{self.well_id}: negative SSE {sse}")
        datum_sse = new_sum * new_sum / n
        shape_sse = self.shape_sse
        reconstructed = datum_sse + self.trend_sse + shape_sse
        if abs(reconstructed - sse) > max(1e-6, sse * 1e-10):
            raise ArithmeticError(f"{self.well_id}: SSE decomposition mismatch")
        return WellMetric(
            well_id=self.well_id,
            rows_scored=self.rows,
            rmse=math.sqrt(sse / n),
            mean_error=new_sum / n,
            sse=sse,
            datum_sse=datum_sse,
            trend_sse=self.trend_sse,
            shape_sse=shape_sse,
            trend_per_row=0.0,
        )


@dataclass(frozen=True)
class Placement:
    family_index: int
    family: str
    scale: float
    cap: float


@dataclass(frozen=True)
class Context:
    key: str
    scope: str
    label: str
    outer_group: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise DataValidationError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise DataValidationError(f"invalid {field}={value!r}") from exc
    if not math.isfinite(result):
        raise DataValidationError(f"non-finite {field}={value!r}")
    return result


def load_actions(path: Path, expected_ids: Sequence[str], family_names: Sequence[str]) -> np.ndarray:
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = (set(BASE_FAMILIES) | {"well_id"}) - fields
        if missing:
            raise DataValidationError(f"action table missing columns {sorted(missing)}")
        for row in reader:
            well_id = str(row["well_id"])
            if well_id in rows:
                raise DataValidationError(f"duplicate action well {well_id}")
            rows[well_id] = {name: _finite(str(row[name]), field=f"{well_id}:{name}") for name in BASE_FAMILIES}
    expected = list(expected_ids)
    if set(rows) != set(expected):
        missing = sorted(set(expected) - set(rows))[:10]
        extra = sorted(set(rows) - set(expected))[:10]
        raise DataValidationError(f"action well mismatch missing={missing} extra={extra}")
    base = np.asarray([[rows[well_id][name] for name in BASE_FAMILIES] for well_id in expected], dtype=float)
    wide32 = base[:, 0]
    wide64 = base[:, 2]
    path24 = base[:, 4]
    derived = np.column_stack(
        [
            np.mean(base, axis=1),
            np.median(base, axis=1),
            np.median(np.column_stack([wide64, path24, wide32]), axis=1),
            0.75 * wide64 + 0.25 * path24,
            0.50 * wide64 + 0.50 * path24,
            0.25 * wide64 + 0.75 * path24,
        ]
    )
    matrix = np.column_stack([base, derived])
    if matrix.shape != (len(expected), len(family_names)):
        raise DataValidationError(f"action matrix shape mismatch {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise DataValidationError("action matrix contains non-finite values")
    return matrix


def load_oof_stats(path: Path, expected_ids: Sequence[str], expected_rows: int = 3_783_989) -> list[WellStats]:
    expected_set = set(expected_ids)
    completed: set[str] = set()
    output: dict[str, WellStats] = {}
    current: str | None = None
    expected_hidden = 0
    previous_row_index = -1
    rows = 0
    n = 0
    sum_error = 0.0
    sse = 0.0
    sum_x = 0.0
    sum_x_sq = 0.0
    sum_x_error = 0.0

    def finish() -> None:
        nonlocal current, n, sum_error, sse, sum_x, sum_x_sq, sum_x_error
        if current is None:
            return
        if n <= 0:
            raise DataValidationError(f"{current}: empty OOF well")
        nf = float(n)
        centered_x_sq = sum_x_sq - sum_x * sum_x / nf
        centered_x_error = sum_x_error - sum_x * sum_error / nf
        slope = centered_x_error / centered_x_sq if centered_x_sq > 0.0 else 0.0
        datum_sse = sum_error * sum_error / nf
        trend_sse = slope * slope * centered_x_sq
        shape_sse = sse - datum_sse - trend_sse
        tolerance = max(1e-8, sse * 1e-10)
        if shape_sse < 0.0 and abs(shape_sse) <= tolerance:
            shape_sse = 0.0
        if shape_sse < 0.0:
            raise ArithmeticError(f"{current}: negative shape SSE {shape_sse}")
        output[current] = WellStats(current, n, sum_error, sse, trend_sse, shape_sse)
        completed.add(current)

    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "well_id", "row_index", "hidden_index", "target", "spline4_ridge_equal_s075"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"E011 OOF missing columns {sorted(missing)}")
        for row in reader:
            well_id = str(row["well_id"])
            if well_id not in expected_set:
                raise DataValidationError(f"unknown OOF well {well_id}")
            if well_id != current:
                finish()
                if well_id in completed:
                    raise DataValidationError(f"noncontiguous duplicate OOF well {well_id}")
                current = well_id
                expected_hidden = 0
                previous_row_index = -1
                n = 0
                sum_error = sse = sum_x = sum_x_sq = sum_x_error = 0.0
            hidden = int(row["hidden_index"])
            row_index = int(row["row_index"])
            if hidden != expected_hidden:
                raise DataValidationError(f"{well_id}: hidden index {hidden} != {expected_hidden}")
            if row_index <= previous_row_index:
                raise DataValidationError(f"{well_id}: non-increasing or duplicate row_index {row_index}")
            if str(row["id"]) != f"{well_id}_{row_index}":
                raise DataValidationError(f"{well_id}: malformed OOF id {row['id']!r}")
            target = _finite(str(row["target"]), field="target")
            prediction = _finite(str(row["spline4_ridge_equal_s075"]), field="e011")
            error = prediction - target
            x = float(hidden)
            n += 1
            rows += 1
            sum_error += error
            sse += error * error
            sum_x += x
            sum_x_sq += x * x
            sum_x_error += x * error
            expected_hidden += 1
            previous_row_index = row_index
    finish()
    if set(output) != expected_set:
        raise DataValidationError("OOF well coverage mismatch")
    if rows != expected_rows:
        raise DataValidationError(f"OOF row count {rows} != {expected_rows}")
    return [output[well_id] for well_id in expected_ids]


def load_contexts(root: Path, ids: Sequence[str], spatial: np.ndarray, typewell: np.ndarray) -> tuple[list[Context], list[Context]]:
    id_to_index = {well_id: i for i, well_id in enumerate(ids)}
    repeated: list[Context] = []
    for version in range(1, 6):
        data = json.loads((root / "folds" / f"v{version}.json").read_text(encoding="utf-8"))
        assignments = data.get("assignments")
        if not isinstance(assignments, dict) or set(assignments) != set(ids):
            raise DataValidationError(f"fold v{version} membership mismatch")
        if int(data.get("n_folds", 0)) != 5:
            raise DataValidationError(f"fold v{version} must have five folds")
        for fold in range(5):
            test = np.asarray([id_to_index[w] for w in ids if int(assignments[w]) == fold], dtype=int)
            train = np.asarray([i for i in range(len(ids)) if i not in set(test.tolist())], dtype=int)
            if len(test) == 0 or len(train) == 0 or len(test) + len(train) != len(ids):
                raise DataValidationError(f"invalid repeated context v{version}:{fold}")
            repeated.append(Context(f"repeated:v{version}:{fold}", "repeated", f"v{version}", fold, train, test))
    stress: list[Context] = []
    for scope, groups in (("spatial", spatial), ("typewell", typewell)):
        values = sorted(set(int(x) for x in groups.tolist()))
        if values != [0, 1, 2, 3, 4]:
            raise DataValidationError(f"{scope} groups are not 0..4")
        for group in values:
            test = np.flatnonzero(groups == group).astype(int)
            train = np.flatnonzero(groups != group).astype(int)
            if len(test) == 0 or len(train) == 0 or len(test) + len(train) != len(ids):
                raise DataValidationError(f"invalid {scope} group {group}")
            stress.append(Context(f"{scope}:{group}", scope, scope, group, train, test))
    return repeated, stress


def build_placements(config: Mapping[str, Any]) -> list[Placement]:
    families = [str(x) for x in config["action_families"]]
    placements: list[Placement] = []
    for family_index, family in enumerate(families):
        for scale in config["placement_grid"]["scales"]:
            for cap in config["placement_grid"]["caps_ft"]:
                placements.append(Placement(family_index, family, float(scale), float(cap)))
    return placements


def correction_matrix(actions: np.ndarray, placements: Sequence[Placement]) -> np.ndarray:
    output = np.empty((len(placements), actions.shape[0]), dtype=float)
    for i, placement in enumerate(placements):
        output[i] = np.clip(placement.scale * actions[:, placement.family_index], -placement.cap, placement.cap)
    if not np.all(np.isfinite(output)):
        raise DataValidationError("placement correction matrix contains non-finite values")
    return output


def select_placement(corrections: np.ndarray, train_indices: np.ndarray, rows: np.ndarray, sum_error: np.ndarray, base_sse: np.ndarray) -> tuple[int, float]:
    if len(train_indices) == 0:
        raise DataValidationError("cannot select placement on empty training partition")
    c = corrections[:, train_indices]
    scores = float(np.sum(base_sse[train_indices])) + np.sum(2.0 * c * sum_error[train_indices] + c * c * rows[train_indices], axis=1)
    if not np.all(np.isfinite(scores)):
        raise DataValidationError("non-finite placement selection scores")
    index = int(np.argmin(scores))
    return index, float(scores[index])


def summarize(stats: Sequence[WellStats], indices: np.ndarray, correction: np.ndarray) -> dict[str, Any]:
    if len(indices) == 0:
        raise DataValidationError("cannot summarize empty well set")
    metrics = [stats[int(i)].metric(float(correction[int(i)])) for i in indices]
    return dict(_summarize(metrics))


def run_nested(
    name: str,
    actions: np.ndarray,
    placements: Sequence[Placement],
    repeated: Sequence[Context],
    stress: Sequence[Context],
    stats: Sequence[WellStats],
    rows: np.ndarray,
    sum_error: np.ndarray,
    base_sse: np.ndarray,
) -> dict[str, Any]:
    corrections = correction_matrix(actions, placements)
    n_wells = len(stats)
    repeated_sum = np.zeros(n_wells, dtype=float)
    repeated_count = np.zeros(n_wells, dtype=int)
    map_corrections = {f"v{version}": np.zeros(n_wells, dtype=float) for version in range(1, 6)}
    context_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected_context_corrections: dict[str, np.ndarray] = {}
    zero = np.zeros(n_wells, dtype=float)
    for context in repeated:
        selected_index, train_sse = select_placement(corrections, context.train_indices, rows, sum_error, base_sse)
        placement = placements[selected_index]
        selected = corrections[selected_index]
        selected_context_corrections[context.key] = selected.copy()
        repeated_sum[context.test_indices] += selected[context.test_indices]
        repeated_count[context.test_indices] += 1
        map_corrections[context.label][context.test_indices] = selected[context.test_indices]
        base = summarize(stats, context.test_indices, zero)
        placed = summarize(stats, context.test_indices, selected)
        for candidate, metric in (("e011", base), (name, placed)):
            context_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "candidate": candidate, **metric})
        selection_rows.append({
            "context": context.key,
            "scope": context.scope,
            "candidate": name,
            "family": placement.family,
            "scale": placement.scale,
            "cap": placement.cap,
            "train_sse": train_sse,
            "train_rmse": math.sqrt(train_sse / float(np.sum(rows[context.train_indices]))),
        })
    if not np.all(repeated_count == 5):
        raise DataValidationError(f"{name}: final repeated coverage is not exactly five per well")
    final_correction = repeated_sum / repeated_count
    final_summary = summarize(stats, np.arange(n_wells, dtype=int), final_correction)
    map_rows: list[dict[str, Any]] = []
    for label, correction in map_corrections.items():
        for candidate, metric in (("e011", summarize(stats, np.arange(n_wells), zero)), (name, summarize(stats, np.arange(n_wells), correction))):
            map_rows.append({"map": label, "candidate": candidate, **metric})
    stress_rows: list[dict[str, Any]] = []
    stress_corrections: dict[str, np.ndarray] = {}
    for context in stress:
        selected_index, train_sse = select_placement(corrections, context.train_indices, rows, sum_error, base_sse)
        placement = placements[selected_index]
        selected = corrections[selected_index]
        stress_corrections[context.key] = selected.copy()
        base = summarize(stats, context.test_indices, zero)
        placed = summarize(stats, context.test_indices, selected)
        for candidate, metric in (("e011", base), (name, placed)):
            stress_rows.append({"context": context.key, "scope": context.scope, "outer_group": context.outer_group, "candidate": candidate, **metric})
        selection_rows.append({
            "context": context.key,
            "scope": context.scope,
            "candidate": name,
            "family": placement.family,
            "scale": placement.scale,
            "cap": placement.cap,
            "train_sse": train_sse,
            "train_rmse": math.sqrt(train_sse / float(np.sum(rows[context.train_indices]))),
        })
    return {
        "name": name,
        "final_correction": final_correction,
        "final_summary": final_summary,
        "map_corrections": map_corrections,
        "context_corrections": selected_context_corrections,
        "stress_corrections": stress_corrections,
        "context_rows": context_rows,
        "map_rows": map_rows,
        "stress_rows": stress_rows,
        "selection_rows": selection_rows,
    }


def sign_control(real: Mapping[str, Any], repeated: Sequence[Context], stress: Sequence[Context], stats: Sequence[WellStats]) -> dict[str, Any]:
    name = "sign_flipped_selected"
    n_wells = len(stats)
    zero = np.zeros(n_wells, dtype=float)
    repeated_sum = np.zeros(n_wells, dtype=float)
    repeated_count = np.zeros(n_wells, dtype=int)
    context_rows: list[dict[str, Any]] = []
    map_corrections = {f"v{version}": np.zeros(n_wells, dtype=float) for version in range(1, 6)}
    for context in repeated:
        correction = -np.asarray(real["context_corrections"][context.key], dtype=float)
        repeated_sum[context.test_indices] += correction[context.test_indices]
        repeated_count[context.test_indices] += 1
        map_corrections[context.label][context.test_indices] = correction[context.test_indices]
        for candidate, metric in (("e011", summarize(stats, context.test_indices, zero)), (name, summarize(stats, context.test_indices, correction))):
            context_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, "candidate": candidate, **metric})
    if not np.all(repeated_count == 5):
        raise DataValidationError("sign control repeated coverage failure")
    final_correction = repeated_sum / repeated_count
    final_summary = summarize(stats, np.arange(n_wells), final_correction)
    map_rows = []
    for label, correction in map_corrections.items():
        for candidate, metric in (("e011", summarize(stats, np.arange(n_wells), zero)), (name, summarize(stats, np.arange(n_wells), correction))):
            map_rows.append({"map": label, "candidate": candidate, **metric})
    stress_rows = []
    for context in stress:
        correction = -np.asarray(real["stress_corrections"][context.key], dtype=float)
        for candidate, metric in (("e011", summarize(stats, context.test_indices, zero)), (name, summarize(stats, context.test_indices, correction))):
            stress_rows.append({"context": context.key, "scope": context.scope, "outer_group": context.outer_group, "candidate": candidate, **metric})
    return {
        "name": name,
        "final_correction": final_correction,
        "final_summary": final_summary,
        "map_corrections": map_corrections,
        "context_rows": context_rows,
        "map_rows": map_rows,
        "stress_rows": stress_rows,
        "selection_rows": [],
    }


def edge_tests(actions: np.ndarray, placements: Sequence[Placement], repeated: Sequence[Context], stress: Sequence[Context], base_summary: Mapping[str, Any]) -> dict[str, Any]:
    passed: list[str] = []
    family_names = []
    for placement in placements:
        if placement.family not in family_names:
            family_names.append(placement.family)

    def action_row(well_id: str, value: str = "1.0") -> dict[str, str]:
        return {"well_id": well_id, **{name: value for name in BASE_FAMILIES}}

    def write_actions(path: Path, rows_to_write: Sequence[Mapping[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["well_id", *BASE_FAMILIES])
            writer.writeheader()
            writer.writerows(rows_to_write)

    def write_oof(path: Path, rows_to_write: Sequence[Mapping[str, Any]]) -> None:
        fields = ["id", "well_id", "row_index", "hidden_index", "target", "spline4_ridge_equal_s075"]
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows_to_write)

    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        action_path = temp / "actions.csv"
        write_actions(action_path, [action_row("a"), action_row("a")])
        try:
            load_actions(action_path, ["a"], family_names)
            raise AssertionError("duplicate action well did not fail")
        except DataValidationError:
            pass
        write_actions(action_path, [action_row("a")])
        try:
            load_actions(action_path, ["a", "b"], family_names)
            raise AssertionError("missing action well did not fail")
        except DataValidationError:
            pass
        passed.append("missing_duplicate_action_wells")

        valid_oof = [
            {"id": "a_1", "well_id": "a", "row_index": 1, "hidden_index": 0, "target": 1.0, "spline4_ridge_equal_s075": 1.5},
            {"id": "a_2", "well_id": "a", "row_index": 2, "hidden_index": 1, "target": 2.0, "spline4_ridge_equal_s075": 2.5},
        ]
        oof_path = temp / "oof.csv.gz"
        write_oof(oof_path, valid_oof)
        loaded = load_oof_stats(oof_path, ["a"], expected_rows=2)
        assert len(loaded) == 1 and loaded[0].rows == 2
        unknown = [dict(valid_oof[0], id="b_1", well_id="b")]
        write_oof(oof_path, unknown)
        try:
            load_oof_stats(oof_path, ["a"], expected_rows=1)
            raise AssertionError("unknown OOF well did not fail")
        except DataValidationError:
            pass
        duplicate = [valid_oof[0], dict(valid_oof[1], id="a_1", row_index=1)]
        write_oof(oof_path, duplicate)
        try:
            load_oof_stats(oof_path, ["a"], expected_rows=2)
            raise AssertionError("duplicate OOF row did not fail")
        except DataValidationError:
            pass
        passed.append("unknown_duplicate_oof_ids")

        write_actions(action_path, [action_row("a", "nan")])
        try:
            load_actions(action_path, ["a"], family_names)
            raise AssertionError("non-finite action did not fail")
        except DataValidationError:
            pass
        nonfinite_oof = [dict(valid_oof[0], target="nan")]
        write_oof(oof_path, nonfinite_oof)
        try:
            load_oof_stats(oof_path, ["a"], expected_rows=1)
            raise AssertionError("non-finite OOF target did not fail")
        except DataValidationError:
            pass
        passed.append("nonfinite_inputs")

    zero_actions = np.zeros_like(actions)
    zero_corrections = correction_matrix(zero_actions, placements)
    assert np.max(np.abs(zero_corrections)) == 0.0
    constant_actions = np.full_like(actions, 2.0)
    constant_corrections = correction_matrix(constant_actions, placements)
    assert np.all(np.isfinite(constant_corrections))
    passed.append("constant_zero_actions")

    synthetic_rows = np.asarray([1.0])
    synthetic_sums = np.asarray([0.0])
    synthetic_sses = np.asarray([1.0])
    one = np.zeros((1, 1), dtype=float)
    selected, score = select_placement(one, np.asarray([0]), synthetic_rows, synthetic_sums, synthetic_sses)
    assert selected == 0 and score == 1.0
    try:
        select_placement(one, np.asarray([], dtype=int), synthetic_rows, synthetic_sums, synthetic_sses)
        raise AssertionError("empty selection did not fail")
    except DataValidationError:
        pass
    passed.append("one_well_empty_partitions")

    values = np.asarray([[1000.0] * actions.shape[1]])
    test_placements = [Placement(0, "x", 1.0, 5.0), Placement(0, "x", -1.0, 10.0), Placement(0, "x", 0.0, 20.0)]
    clipped = correction_matrix(values, test_placements)
    assert clipped[0, 0] == 5.0 and clipped[1, 0] == -10.0 and clipped[2, 0] == 0.0
    passed.append("scale_cap_boundaries")

    tie = np.zeros((3, 2), dtype=float)
    selected, _ = select_placement(tie, np.asarray([0, 1]), np.ones(2), np.zeros(2), np.ones(2))
    assert selected == 0
    passed.append("deterministic_ties")

    assert any(p.scale < 0.0 for p in placements) and any(p.scale == 0.0 for p in placements)
    passed.append("negative_zero_scales")

    assert np.max(np.abs(clipped)) <= 10.0 and np.max(np.abs(correction_matrix(values, [Placement(0, "x", 1.0, 30.0)]))) == 30.0
    passed.append("large_action_clipping")

    assert len(repeated) == 25 and len(stress) == 10
    for context in [*repeated, *stress]:
        assert len(set(context.train_indices.tolist()) & set(context.test_indices.tolist())) == 0
        assert len(context.train_indices) + len(context.test_indices) == actions.shape[0]
    passed.append("context_membership")

    counts = np.zeros(actions.shape[0], dtype=int)
    for context in repeated:
        counts[context.test_indices] += 1
    assert np.all(counts == 5)
    passed.append("five_map_averaging")

    assert abs(float(base_summary["rmse"]) - 12.550756295689673) < 1e-8
    passed.append("exact_e011_fallback")
    assert len(passed) == 12
    return {"status": "PASS", "edge_groups": len(passed), "passed": passed}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "tracking" / "evidence" / "T016" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    compact = np.load(ROOT / config["inputs"]["e011_compact"], allow_pickle=False)
    ids = [str(x) for x in compact["well_ids"].tolist()]
    if len(ids) != 773 or len(set(ids)) != 773:
        raise DataValidationError("compact well identity failure")
    family_names = [str(x) for x in config["action_families"]]
    actions = load_actions(ROOT / config["inputs"]["e009_well_actions"], ids, family_names)
    stats = load_oof_stats(ROOT / config["inputs"]["e011_oof"], ids)
    rows = np.asarray([item.rows for item in stats], dtype=float)
    sum_error = np.asarray([item.sum_error for item in stats], dtype=float)
    base_sse = np.asarray([item.base_sse for item in stats], dtype=float)
    spatial = np.asarray(compact["spatial_assignment"], dtype=int)
    typewell = np.asarray(compact["typewell_assignment"], dtype=int)
    repeated, stress = load_contexts(ROOT, ids, spatial, typewell)
    placements = build_placements(config)
    if len(placements) != 480:
        raise DataValidationError(f"expected 480 placements, got {len(placements)}")
    all_indices = np.arange(len(ids), dtype=int)
    zero = np.zeros(len(ids), dtype=float)
    base_summary = summarize(stats, all_indices, zero)

    real = run_nested("nested_real", actions, placements, repeated, stress, stats, rows, sum_error, base_sse)
    rng = np.random.default_rng(16016)
    shuffled_actions = actions[rng.permutation(len(ids))]
    shuffled = run_nested("shuffled_wells", shuffled_actions, placements, repeated, stress, stats, rows, sum_error, base_sse)
    rng = np.random.default_rng(26016)
    permuted_actions = np.empty_like(actions)
    for i in range(len(ids)):
        permuted_actions[i] = actions[i, rng.permutation(actions.shape[1])]
    permuted = run_nested("permuted_families", permuted_actions, placements, repeated, stress, stats, rows, sum_error, base_sse)
    signed = sign_control(real, repeated, stress, stats)
    runs = [real, shuffled, permuted, signed]

    grid = correction_matrix(actions, placements)
    grid_rows = []
    for i, placement in enumerate(placements):
        summary = summarize(stats, all_indices, grid[i])
        grid_rows.append({"placement_index": i, "family": placement.family, "scale": placement.scale, "cap": placement.cap, "gain_vs_e011": float(base_summary["rmse"]) - float(summary["rmse"]), **summary})

    context_rows = []
    map_rows = []
    stress_rows = []
    selection_rows = []
    for run in runs:
        context_rows.extend(run["context_rows"])
        map_rows.extend(run["map_rows"])
        stress_rows.extend(run["stress_rows"])
        selection_rows.extend(run["selection_rows"])

    feature_names = [str(x) for x in compact["feature_names"].tolist()]
    hidden_rows_index = feature_names.index("hidden_rows")
    missing_index = feature_names.index("hidden_gr_missing_fraction")
    features = np.asarray(compact["features"], dtype=float)
    long_threshold = float(np.quantile(features[:, hidden_rows_index], 0.8))
    missing_threshold = float(np.quantile(features[:, missing_index], 0.8))
    base_well_rmse = np.asarray([math.sqrt(item.base_sse / item.rows) for item in stats], dtype=float)
    special_sets = {
        "long_suffix": np.flatnonzero(features[:, hidden_rows_index] >= long_threshold),
        "high_gr_missingness": np.flatnonzero(features[:, missing_index] >= missing_threshold),
        "e011_catastrophe": np.flatnonzero(base_well_rmse >= 12.0),
    }
    special_rows = []
    for slice_name, indices in special_sets.items():
        for candidate, correction in [("e011", zero)] + [(run["name"], np.asarray(run["final_correction"])) for run in runs]:
            special_rows.append({"slice": slice_name, "candidate": candidate, **summarize(stats, indices, correction)})

    final_summaries = {"e011": base_summary, **{run["name"]: run["final_summary"] for run in runs}}
    context_lookup = {(row["context"], row["candidate"]): row for row in context_rows}
    map_lookup = {(row["map"], row["candidate"]): row for row in map_rows}
    stress_lookup = {(row["context"], row["candidate"]): row for row in stress_rows}
    special_lookup = {(row["slice"], row["candidate"]): row for row in special_rows}
    real_name = "nested_real"
    gain = float(base_summary["rmse"]) - float(final_summaries[real_name]["rmse"])
    map_wins = sum(float(map_lookup[(f"v{i}", "e011")]["rmse"]) > float(map_lookup[(f"v{i}", real_name)]["rmse"]) for i in range(1, 6))
    cell_wins = sum(float(context_lookup[(context.key, "e011")]["rmse"]) > float(context_lookup[(context.key, real_name)]["rmse"]) for context in repeated)
    spatial_gains = {str(i): float(stress_lookup[(f"spatial:{i}", "e011")]["rmse"]) - float(stress_lookup[(f"spatial:{i}", real_name)]["rmse"]) for i in range(5)}
    typewell_gains = {str(i): float(stress_lookup[(f"typewell:{i}", "e011")]["rmse"]) - float(stress_lookup[(f"typewell:{i}", real_name)]["rmse"]) for i in range(5)}
    special_gains = {name: float(special_lookup[(name, "e011")]["rmse"]) - float(special_lookup[(name, real_name)]["rmse"]) for name in special_sets}
    control_gains = {run["name"]: float(base_summary["rmse"]) - float(run["final_summary"]["rmse"]) for run in [shuffled, permuted, signed]}
    gate_config = config["advancement_gate_for_full_stability_refit"]
    gates = {
        "gain": gain >= float(gate_config["minimum_gain_vs_e011"]),
        "maps": map_wins >= int(gate_config["minimum_map_wins"]),
        "cells": cell_wins >= int(gate_config["minimum_outer_cell_wins"]),
        "p90": float(final_summaries[real_name]["p90_well_rmse"]) - float(base_summary["p90_well_rmse"]) <= float(gate_config["maximum_p90_deterioration"]),
        "worst5": float(final_summaries[real_name]["worst_5pct_sse_share"]) - float(base_summary["worst_5pct_sse_share"]) <= float(gate_config["maximum_worst5_sse_share_increase"]),
        "spatial": all(value > 0.0 for value in spatial_gains.values()),
        "typewell": all(value > 0.0 for value in typewell_gains.values()),
        "special_slices": all(value > 0.0 for value in special_gains.values()),
        "negative_controls": all(value <= float(gate_config["maximum_negative_control_gain"]) for value in control_gains.values()),
    }
    edge = edge_tests(actions, placements, repeated, stress, base_summary)
    gates["edge_groups"] = edge["status"] == "PASS"
    gates["reproduction"] = False

    correction_bound = max(float(np.max(np.abs(np.asarray(run["final_correction"])))) for run in runs)
    controls = {
        "source_inputs": {"pass": all((ROOT / path).exists() for path in [config["inputs"]["e009_well_actions"], config["inputs"]["e011_oof"], config["inputs"]["e011_compact"]])},
        "coverage": {"pass": len(ids) == 773 and int(np.sum(rows)) == 3_783_989, "wells": len(ids), "rows": int(np.sum(rows))},
        "action_table": {"pass": actions.shape == (773, 12) and np.all(np.isfinite(actions)), "shape": list(actions.shape)},
        "context_completion": {"pass": len(repeated) == 25 and len(stress) == 10 and len([r for r in context_rows if r["candidate"] != "e011"]) == 100 and len([r for r in stress_rows if r["candidate"] != "e011"]) == 40},
        "exact_fallback": {"pass": abs(float(base_summary["rmse"]) - 12.550756295689673) < 1e-8, "rmse": float(base_summary["rmse"])},
        "finite_bounded": {"pass": all(math.isfinite(float(value)) for summary in final_summaries.values() for value in [summary["rmse"], summary["sse"], summary["p90_well_rmse"]]) and correction_bound <= 30.0 + 1e-12, "maximum_final_correction": correction_bound},
        "pooled_sse": {"pass": abs(float(base_summary["sse"]) - float(np.sum(base_sse))) <= max(1e-5, float(base_summary["sse"]) * 1e-12)},
        "negative_controls": {"pass": gates["negative_controls"], "gains": control_gains},
        "edge_groups": {"pass": edge["status"] == "PASS", "count": edge["edge_groups"]},
    }
    all_controls = all(bool(item["pass"]) for item in controls.values())
    gates["controls"] = all_controls
    preliminary_pass = all(value for key, value in gates.items() if key != "reproduction")
    authorized = False
    status = "awaiting_reproduction" if preliminary_pass else "worth_screen_reject"
    decision = "await_independent_reproduction" if preliminary_pass else "close_h012_without_refit"

    per_well_rows = []
    for i, well_id in enumerate(ids):
        row = {"well_id": well_id, "rows": int(rows[i]), "e011_rmse": base_well_rmse[i]}
        for run in runs:
            correction = float(run["final_correction"][i])
            row[f"{run['name']}_correction"] = correction
            row[f"{run['name']}_rmse"] = stats[i].metric(correction).rmse
        per_well_rows.append(row)

    runtime = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "task_id": "T016",
        "hypothesis_id": "H012",
        "source_commit": str(config["source_commit"]),
        "implementation_commit": str(args.implementation_commit),
        "status": status,
        "decision": decision,
        "authorized_full_refit": authorized,
        "base_summary": base_summary,
        "final_summaries": final_summaries,
        "gain_vs_e011": gain,
        "map_wins": map_wins,
        "outer_cell_wins": cell_wins,
        "spatial_gains": spatial_gains,
        "typewell_gains": typewell_gains,
        "special_slice_gains": special_gains,
        "negative_control_gains": control_gains,
        "gates": gates,
        "controls": controls,
        "edge_cases": edge,
        "thresholds": {"long_suffix": long_threshold, "high_gr_missingness": missing_threshold, "e011_catastrophe": 12.0},
        "special_slice_wells": {name: int(len(indices)) for name, indices in special_sets.items()},
        "placements": len(placements),
        "contexts": {"repeated": len(repeated), "stress": len(stress)},
        "runtime_seconds": runtime,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "edge_cases.json", edge)
    _write_csv(output_dir / "candidate_grid_metrics.csv", grid_rows)
    _write_csv(output_dir / "context_metrics.csv", context_rows)
    _write_csv(output_dir / "map_metrics.csv", map_rows)
    _write_csv(output_dir / "stress_metrics.csv", stress_rows)
    _write_csv(output_dir / "selected_placements.csv", selection_rows)
    _write_csv(output_dir / "special_slice_metrics.csv", special_rows)
    _write_csv(output_dir / "per_well_final.csv", per_well_rows)
    manifest_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest_rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_json(output_dir / "artifact_manifest.json", {"schema_version": 1, "task_id": "T016", "files": manifest_rows})
    print(json.dumps({"status": status, "decision": decision, "gain_vs_e011": gain, "map_wins": map_wins, "outer_cell_wins": cell_wins, "authorized_full_refit": authorized, "runtime_seconds": runtime}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
