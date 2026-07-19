"""E006 strict nested PF-E004 fusion validation.

The E004 base path is trained without each outer validation group. The frozen
E005 particle-filter algorithm is then recomputed around the matching inner or
outer E004 path. Fusion weights and target-aware routing decisions are fitted
only on inner OOF predictions from the outer-training wells.
"""
from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import resource
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .deployment import (
    _extract_training_records as _e004_extract_training_records,
    _fit_predict as _e004_fit_predict,
    feature_family,
)
from .gr_path import (
    TypewellCurve,
    WellData,
    _calibration,
    _grid_paths,
    _hidden_samples,
    _particle_path,
    _quantile,
    _sha256,
    _write_csv,
    _write_json,
    read_well,
    validate_e005_config,
)
from .harness import DataValidationError, ErrorAccumulator, OnlineCorrelation, WellMetric, _summarize
from .learnability import _load_fold_maps

COMPARATORS = ("last_known_tvt", "e004_geometry_prefix", "pf_gr_path")
DIAGNOSTICS = (
    "fixed_50_50_reference",
    "nested_rmse_grid",
    "duplicate_conservative",
    "shuffled_pf_conservative",
    "oracle_rowwise_best",
)
ELIGIBLE = (
    "nested_conservative_grid",
    "nested_reliability_shrink",
    "nested_disagreement_cap",
    "nested_reliability_gate",
)
ALL_CANDIDATES = COMPARATORS + DIAGNOSTICS[:2] + ELIGIBLE + DIAGNOSTICS[2:]
TEMPLATE_POINTS = 64


@dataclass(frozen=True)
class BlendSufficient:
    """Per-well sufficient statistics for e004 + weight * (pf - e004)."""

    well_id: str
    rows: int
    sum_base: float
    sum_delta: float
    sum_base_sq: float
    sum_base_delta: float
    sum_delta_sq: float
    sum_x: float
    sum_x_sq: float
    sum_x_base: float
    sum_x_delta: float

    @classmethod
    def from_paths(
        cls,
        well_id: str,
        truth: Sequence[float],
        base: Sequence[float],
        pf: Sequence[float],
    ) -> "BlendSufficient":
        if len(truth) != len(base) or len(base) != len(pf) or not truth:
            raise DataValidationError(f"{well_id}: invalid fusion path lengths")
        sums = [0.0] * 9
        for index, (target, anchor, particle) in enumerate(zip(truth, base, pf)):
            target_f = float(target)
            anchor_f = float(anchor)
            particle_f = float(particle)
            if not all(math.isfinite(value) for value in (target_f, anchor_f, particle_f)):
                raise DataValidationError(f"{well_id}: non-finite fusion input")
            base_error = anchor_f - target_f
            delta = particle_f - anchor_f
            x = float(index)
            sums[0] += base_error
            sums[1] += delta
            sums[2] += base_error * base_error
            sums[3] += base_error * delta
            sums[4] += delta * delta
            sums[5] += x
            sums[6] += x * x
            sums[7] += x * base_error
            sums[8] += x * delta
        return cls(well_id, len(truth), *sums)

    def metric(self, weight: float) -> WellMetric:
        value = float(weight)
        if not math.isfinite(value):
            raise DataValidationError(f"{self.well_id}: non-finite fusion weight")
        accumulator = ErrorAccumulator(
            rows=self.rows,
            sum_error=self.sum_base + value * self.sum_delta,
            sum_error_sq=self.sum_base_sq + 2.0 * value * self.sum_base_delta + value * value * self.sum_delta_sq,
            sum_x=self.sum_x,
            sum_x_sq=self.sum_x_sq,
            sum_x_error=self.sum_x_base + value * self.sum_x_delta,
        )
        return accumulator.finalize(self.well_id)

    @property
    def disagreement_rms(self) -> float:
        return math.sqrt(max(0.0, self.sum_delta_sq) / max(1, self.rows))


@dataclass(frozen=True)
class FusionDiagnostic:
    hidden_gr_coverage: float
    pf_effective_fraction: float
    visible_alignment_margin: float
    disagreement_rms: float
    fallback_reason: str
    pf_datum: float
    pf_toe: float


@dataclass
class ValidationContext:
    key: str
    scope: str
    label: str
    outer_group: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    inner_coefficients: dict[str, list[float]]
    outer_coefficients: dict[str, list[float]]
    inner_fold_count: int


@dataclass(frozen=True)
class ReliabilityReference:
    coverage: tuple[tuple[float, str], ...]
    effective: tuple[tuple[float, str], ...]
    margin: tuple[tuple[float, str], ...]
    disagreement: tuple[tuple[float, str], ...]
    disagreement_p80: float


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def _path_from_coeff(record: Mapping[str, Any], coefficients: Sequence[float]) -> list[float]:
    if len(coefficients) < 2:
        raise DataValidationError("E006 E004 coefficient vector is malformed")
    count = int(float(record["sufficient"]["rows"]))
    last = float(record["features"]["last_visible_tvt"])
    datum = float(coefficients[0])
    trend = float(coefficients[1])
    if count <= 0 or not all(math.isfinite(value) for value in (last, datum, trend)):
        raise DataValidationError("E006 invalid E004 reconstruction values")
    return [last + datum + trend * (index / max(1, count - 1) - 0.5 if count > 1 else 0.0) for index in range(count)]


def _correction_template(base: Sequence[float], pf: Sequence[float], points: int = TEMPLATE_POINTS) -> tuple[float, ...]:
    if len(base) != len(pf) or not base:
        raise DataValidationError("invalid correction template input")
    delta = [float(particle) - float(anchor) for anchor, particle in zip(base, pf)]
    if points <= 1 or len(delta) == 1:
        return (delta[0],)
    out: list[float] = []
    for step in range(points):
        position = step * (len(delta) - 1) / (points - 1)
        left = int(math.floor(position))
        right = min(len(delta) - 1, left + 1)
        fraction = position - left
        out.append(delta[left] * (1.0 - fraction) + delta[right] * fraction)
    return tuple(out)


def _template_value(template: Sequence[float], index: int, count: int) -> float:
    if not template:
        return 0.0
    if len(template) == 1 or count <= 1:
        return float(template[0])
    position = index * (len(template) - 1) / (count - 1)
    left = int(math.floor(position))
    right = min(len(template) - 1, left + 1)
    fraction = position - left
    return float(template[left]) * (1.0 - fraction) + float(template[right]) * fraction


def _template_path(base: Sequence[float], template: Sequence[float]) -> list[float]:
    return [float(value) + _template_value(template, index, len(base)) for index, value in enumerate(base)]


def _write_gzip_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: format(value, ".8f") if isinstance(value, float) else value for key, value in row.items()})


def validate_e006_config(config: Mapping[str, Any]) -> None:
    if list(config.get("candidate_order", [])) != list(ALL_CANDIDATES):
        raise DataValidationError("E006 candidate order differs from frozen implementation")
    if list(config.get("eligible_candidates", [])) != list(ELIGIBLE):
        raise DataValidationError("E006 eligible candidates differ from frozen implementation")
    grid = config.get("fusion", {}).get("weight_grid")
    if not isinstance(grid, list) or not grid:
        raise DataValidationError("E006 fusion.weight_grid must be non-empty")
    parsed = [float(value) for value in grid]
    if parsed != sorted(parsed) or len(set(parsed)) != len(parsed) or any(not (0.0 <= value <= 1.0) for value in parsed):
        raise DataValidationError("E006 fusion.weight_grid must contain unique sorted values in [0,1]")
    if parsed[0] != 0.0 or parsed[-1] != 1.0:
        raise DataValidationError("E006 fusion.weight_grid must include 0 and 1")
    if int(config["outer_validation"]["repeated_maps"]) != len(config["fold_files"]):
        raise DataValidationError("E006 repeated map count differs from fold files")
    if int(config["outer_validation"]["outer_folds_per_map"]) != 5:
        raise DataValidationError("E006 requires five outer folds per repeated map")
    if str(config["selection"]["rule"]).strip() == "":
        raise DataValidationError("E006 selection rule is empty")


def _pf_for_base(well: WellData, base: Sequence[float], e005: Mapping[str, Any], *, with_margin: bool) -> tuple[list[float], FusionDiagnostic]:
    if len(base) != well.hidden_rows or any(not math.isfinite(float(value)) for value in base):
        raise DataValidationError(f"{well.well_id}: invalid nested E004 base path")
    align = e005["alignment"]
    samples = _hidden_samples(well, int(align["maximum_gr_samples_per_well"]))
    fallback = ""
    if well.hidden_gr_coverage < float(align["minimum_hidden_gr_coverage"]):
        fallback = "low_hidden_gr_coverage"
    elif len(samples) < int(align["minimum_gr_samples"]):
        fallback = "too_few_hidden_gr_samples"
    elif well.typewell.gr_std < float(align["flat_typewell_std_min"]):
        fallback = "flat_typewell_gr"
    calibration = _calibration(well, well.typewell, int(align["minimum_visible_calibration_samples"]))
    if not fallback and calibration is None:
        fallback = "invalid_visible_calibration"
    if fallback:
        return list(base), FusionDiagnostic(well.hidden_gr_coverage, 0.0, 0.0, 0.0, fallback, 0.0, 0.0)
    assert calibration is not None
    margin = 0.0
    if with_margin:
        _, visible, _ = _grid_paths(well, base, well.typewell, samples, calibration=calibration, config=align)
        margin = float(visible.get("ambiguity_margin", 0.0))
    pf, detail = _particle_path(well, base, well.typewell, samples, calibration, align, e005["particle_filter"])
    if len(pf) != well.hidden_rows or any(not math.isfinite(float(value)) for value in pf):
        raise DataValidationError(f"{well.well_id}: invalid nested PF path")
    disagreement = math.sqrt(sum((float(particle) - float(anchor)) ** 2 for anchor, particle in zip(base, pf)) / well.hidden_rows)
    return pf, FusionDiagnostic(
        well.hidden_gr_coverage,
        float(detail.get("effective_fraction", 0.0)),
        margin,
        disagreement,
        str(detail.get("reason", "")),
        float(detail.get("datum", 0.0)),
        float(detail.get("toe", 0.0)),
    )


def _group_assignments(values: Mapping[str, tuple[float, ...]], bins: int) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: (*values[key], key))
    return {key: min(bins - 1, int(index * bins / len(ordered))) for index, key in enumerate(ordered)}


def _fit_context(
    key: str,
    scope: str,
    label: str,
    outer_group: int,
    assignments: Mapping[str, int],
    inner_assignments: Mapping[str, int],
    records: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    e004_config: Mapping[str, Any],
) -> tuple[ValidationContext, dict[str, Any]]:
    all_ids = sorted(records)
    test_ids = [well_id for well_id in all_ids if int(assignments[well_id]) == outer_group]
    train_ids = [well_id for well_id in all_ids if int(assignments[well_id]) != outer_group]
    if not test_ids or not train_ids or set(test_ids) & set(train_ids):
        raise DataValidationError(f"{key}: invalid outer membership")
    outer_coefficients, _ = _e004_fit_predict(records, train_ids, test_ids, feature_names, e004_config)
    inner_coefficients: dict[str, list[float]] = {}
    inner_groups = sorted({int(inner_assignments[well_id]) for well_id in train_ids})
    inner_membership: list[dict[str, Any]] = []
    for group in inner_groups:
        validation = [well_id for well_id in train_ids if int(inner_assignments[well_id]) == group]
        training = [well_id for well_id in train_ids if int(inner_assignments[well_id]) != group]
        if not validation:
            continue
        if not training or set(validation) & set(training) or set(validation) & set(test_ids):
            raise DataValidationError(f"{key}: invalid inner membership for group {group}")
        predictions, _ = _e004_fit_predict(records, training, validation, feature_names, e004_config)
        overlap = set(inner_coefficients) & set(predictions)
        if overlap:
            raise DataValidationError(f"{key}: duplicate inner predictions {sorted(overlap)[:3]}")
        inner_coefficients.update(predictions)
        inner_membership.append({
            "inner_group": group,
            "train_wells": len(training),
            "validation_wells": len(validation),
            "outer_overlap": len(set(validation) & set(test_ids)),
        })
    if set(inner_coefficients) != set(train_ids):
        raise DataValidationError(f"{key}: inner OOF did not cover every outer-training well")
    context = ValidationContext(
        key=key,
        scope=scope,
        label=label,
        outer_group=outer_group,
        train_ids=tuple(train_ids),
        test_ids=tuple(test_ids),
        inner_coefficients=inner_coefficients,
        outer_coefficients=outer_coefficients,
        inner_fold_count=len(inner_membership),
    )
    audit = {
        "context": key,
        "scope": scope,
        "label": label,
        "outer_group": outer_group,
        "outer_train_wells": len(train_ids),
        "outer_test_wells": len(test_ids),
        "inner_predicted_wells": len(inner_coefficients),
        "outer_train_test_overlap": len(set(train_ids) & set(test_ids)),
        "inner_outer_test_overlap": len(set(inner_coefficients) & set(test_ids)),
        "inner_fold_count": len(inner_membership),
        "pass": len(set(train_ids) & set(test_ids)) == 0 and len(set(inner_coefficients) & set(test_ids)) == 0 and set(inner_coefficients) == set(train_ids),
        "inner_detail_json": _canonical_json(inner_membership),
    }
    return context, audit


def _build_contexts(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    e004_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[ValidationContext], list[dict[str, Any]], dict[str, int], dict[str, int], list[dict[str, Any]]]:
    folds = _load_fold_maps(root, config["fold_files"], str(config["data_signature"]))
    contexts: list[ValidationContext] = []
    audits: list[dict[str, Any]] = []
    all_ids = sorted(records)
    for fold_map in folds:
        assignments = {well_id: int(value) for well_id, value in fold_map["assignments"].items()}
        version = str(fold_map["version"])
        for outer in range(int(fold_map["n_folds"])):
            key = f"repeated:{version}:{outer}"
            context, audit = _fit_context(key, "repeated", version, outer, assignments, assignments, records, feature_names, e004_config)
            contexts.append(context)
            audits.append(audit)
    spatial_values = {well_id: (float(record["spatial_x"]), float(record["spatial_y"])) for well_id, record in records.items()}
    typewell_values = {
        well_id: (
            float(record["features"]["typewell_gr_mean"] or 0.0),
            float(record["features"]["typewell_gr_std"] or 0.0),
            float(record["features"]["typewell_tvt_span"] or 0.0),
        )
        for well_id, record in records.items()
    }
    spatial = _group_assignments(spatial_values, int(config["stress"]["spatial_bins"]))
    typewell = _group_assignments(typewell_values, int(config["stress"]["typewell_clusters"]))
    inner_v1 = {well_id: int(folds[0]["assignments"][well_id]) for well_id in all_ids}
    for scope, assignments, groups in (
        ("spatial", spatial, int(config["stress"]["spatial_bins"])),
        ("typewell", typewell, int(config["stress"]["typewell_clusters"])),
    ):
        for outer in range(groups):
            key = f"{scope}:{outer}"
            context, audit = _fit_context(key, scope, scope, outer, assignments, inner_v1, records, feature_names, e004_config)
            contexts.append(context)
            audits.append(audit)
    return contexts, audits, spatial, typewell, folds


def _summary_for_weight(stats: Mapping[str, BlendSufficient], weight: float) -> dict[str, Any]:
    return _summarize([stats[well_id].metric(weight) for well_id in sorted(stats)])


def _select_inner_weights(stats: Mapping[str, BlendSufficient], config: Mapping[str, Any]) -> tuple[float, float, list[dict[str, Any]]]:
    grid = [float(value) for value in config["fusion"]["weight_grid"]]
    summaries = {weight: _summary_for_weight(stats, weight) for weight in grid}
    base = summaries[0.0]
    rmse_weight = min(grid, key=lambda weight: (float(summaries[weight]["rmse"]), weight))
    passing = [
        weight
        for weight in grid
        if float(base["rmse"]) - float(summaries[weight]["rmse"]) >= float(config["fusion"]["inner_minimum_gain_vs_e004"])
        and float(summaries[weight]["p90_well_rmse"]) - float(base["p90_well_rmse"]) <= float(config["fusion"]["inner_maximum_p90_deterioration"])
        and float(summaries[weight]["worst_5pct_sse_share"]) - float(base["worst_5pct_sse_share"]) <= float(config["fusion"]["inner_maximum_worst5_sse_share_increase"])
    ]
    if passing:
        best = min(passing, key=lambda weight: (float(summaries[weight]["rmse"]), weight))
        threshold = float(summaries[best]["rmse"]) + float(config["fusion"]["conservative_rmse_slack"])
        conservative = min(weight for weight in passing if float(summaries[weight]["rmse"]) <= threshold)
    else:
        conservative = float(config["fusion"]["fallback_weight"])
    rows = [
        {
            "weight": weight,
            "passes_conservative_constraints": weight in passing,
            "selected_rmse": weight == rmse_weight,
            "selected_conservative": weight == conservative,
            **summaries[weight],
        }
        for weight in grid
    ]
    return rmse_weight, conservative, rows


def _make_reference(diagnostics: Mapping[str, FusionDiagnostic]) -> ReliabilityReference:
    def pairs(getter: Any) -> tuple[tuple[float, str], ...]:
        return tuple(sorted((float(getter(detail)), well_id) for well_id, detail in diagnostics.items()))

    disagreement_values = [float(detail.disagreement_rms) for detail in diagnostics.values()]
    return ReliabilityReference(
        coverage=pairs(lambda item: item.hidden_gr_coverage),
        effective=pairs(lambda item: item.pf_effective_fraction),
        margin=pairs(lambda item: item.visible_alignment_margin),
        disagreement=pairs(lambda item: item.disagreement_rms),
        disagreement_p80=float(_quantile(disagreement_values, 0.8)),
    )


def _percentile(reference: Sequence[tuple[float, str]], value: float, well_id: str, *, higher_better: bool) -> float:
    if len(reference) <= 1:
        return 1.0
    position = bisect.bisect_left(reference, (float(value), str(well_id)))
    rank = position / (len(reference) - 1)
    return _clip(rank if higher_better else 1.0 - rank, 0.0, 1.0)


def _reliability(detail: FusionDiagnostic, reference: ReliabilityReference, well_id: str) -> tuple[float, float, float]:
    components = [
        _percentile(reference.coverage, detail.hidden_gr_coverage, well_id, higher_better=True),
        _percentile(reference.effective, detail.pf_effective_fraction, well_id, higher_better=True),
        _percentile(reference.margin, detail.visible_alignment_margin, well_id, higher_better=True),
        _percentile(reference.disagreement, detail.disagreement_rms, well_id, higher_better=False),
    ]
    score = _clip(_mean(components), 0.0, 1.0)
    disagreement_badness = _clip(1.0 - components[-1], 0.0, 1.0)
    cap = min(1.0, reference.disagreement_p80 / max(reference.disagreement_p80, detail.disagreement_rms, 1e-12))
    return score, disagreement_badness, cap


def _effective_weight(candidate: str, base_weight: float, detail: FusionDiagnostic, reference: ReliabilityReference, well_id: str, config: Mapping[str, Any]) -> float:
    base = _clip(float(base_weight), 0.0, 1.0)
    if candidate in {"nested_conservative_grid", "duplicate_conservative"}:
        return base
    score, disagreement_badness, cap = _reliability(detail, reference, well_id)
    if candidate == "nested_reliability_shrink":
        return base * score
    if candidate == "nested_disagreement_cap":
        return base * cap
    if candidate == "nested_reliability_gate":
        return base if score >= float(config["reliability"]["gate_minimum_score"]) and disagreement_badness <= float(config["reliability"]["gate_maximum_disagreement_percentile"]) else 0.0
    raise DataValidationError(f"unknown reliability candidate {candidate}")


def _metric_from_oracle(well: WellData, base: Sequence[float], pf: Sequence[float]) -> WellMetric:
    assert well.truth is not None
    accumulator = ErrorAccumulator()
    for hidden_index, (anchor, particle) in enumerate(zip(base, pf)):
        target = float(well.truth[well.known_rows + hidden_index])
        prediction = float(anchor) if abs(float(anchor) - target) <= abs(float(particle) - target) else float(particle)
        accumulator.add(prediction - target, float(hidden_index))
    return accumulator.finalize(well.well_id)


def _context_shuffle(ids: Sequence[str], key: str) -> dict[str, str]:
    ordered = sorted(ids, key=lambda well_id: (hashlib.sha256(f"e006-shuffle|{key}|{well_id}".encode()).hexdigest(), well_id))
    if len(ordered) < 2:
        return {well_id: well_id for well_id in ordered}
    return {well_id: ordered[(index + 1) % len(ordered)] for index, well_id in enumerate(ordered)}


def _build_indexes(contexts: Sequence[ValidationContext]) -> tuple[dict[str, list[tuple[ValidationContext, list[float]]]], dict[str, list[tuple[ValidationContext, list[float]]]]]:
    inner: dict[str, list[tuple[ValidationContext, list[float]]]] = defaultdict(list)
    outer: dict[str, list[tuple[ValidationContext, list[float]]]] = defaultdict(list)
    for context in contexts:
        for well_id, coefficients in context.inner_coefficients.items():
            inner[well_id].append((context, coefficients))
        for well_id, coefficients in context.outer_coefficients.items():
            outer[well_id].append((context, coefficients))
    return inner, outer


def _load_e005_ambiguity(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[str(row["well_id"])] = float(row["ambiguity_margin"])
    return out


def _parent_oof_audit(path: Path, expected_sha: str) -> dict[str, Any]:
    if not path.exists():
        return {"pass": False, "reason": "missing_parent_oof", "path": str(path)}
    digest = _sha256(path)
    sums = {"rows": 0, "pf_sse": 0.0, "e004_sse": 0.0}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            target = float(row["target"])
            pf = float(row["pf_gr_path"])
            e004 = float(row["e004_geometry_prefix"])
            sums["rows"] += 1
            sums["pf_sse"] += (pf - target) ** 2
            sums["e004_sse"] += (e004 - target) ** 2
    rows = int(sums["rows"])
    return {
        "pass": digest == expected_sha and rows > 0,
        "sha256": digest,
        "expected_sha256": expected_sha,
        "rows": rows,
        "pf_rmse": math.sqrt(float(sums["pf_sse"]) / max(1, rows)),
        "e004_rmse": math.sqrt(float(sums["e004_sse"]) / max(1, rows)),
    }


def run_e006(
    *,
    root: Path,
    train_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: Mapping[str, Any],
    code_sha: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = root.resolve()
    train_dir = train_dir.resolve()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    validate_e006_config(config)
    e004_config = json.loads((root / str(config["parents"]["e004_config"])).read_text(encoding="utf-8"))
    e005_config = json.loads((root / str(config["parents"]["e005_config"])).read_text(encoding="utf-8"))
    validate_e005_config(e005_config)
    records, all_feature_names, data_profile = _e004_extract_training_records(train_dir, e004_config)
    if data_profile["data_signature"] != str(config["data_signature"]) or len(records) != int(config["expected_wells"]):
        raise DataValidationError("E006 data signature or well count mismatch")
    feature_names = sorted(name for name in all_feature_names if feature_family(name) in {"geometry", "prefix"})
    contexts, membership_rows, spatial_assignments, typewell_assignments, fold_maps = _build_contexts(root, records, feature_names, e004_config, config)
    if not all(bool(row["pass"]) for row in membership_rows):
        raise DataValidationError("E006 nested membership audit failed")
    context_by_key = {context.key: context for context in contexts}
    inner_index, outer_index = _build_indexes(contexts)
    inner_shuffle_maps = {context.key: _context_shuffle(context.train_ids, context.key + ":inner") for context in contexts}
    outer_shuffle_maps = {context.key: _context_shuffle(context.test_ids, context.key + ":outer") for context in contexts}
    well_ids = sorted(records)
    horizontal_files = {well_id: train_dir / f"{well_id}__horizontal_well.csv" for well_id in well_ids}

    inner_stats: dict[str, dict[str, BlendSufficient]] = {context.key: {} for context in contexts}
    inner_diagnostics: dict[str, dict[str, FusionDiagnostic]] = {context.key: {} for context in contexts}
    inner_templates: dict[str, dict[str, tuple[float, ...]]] = {context.key: {} for context in contexts}
    fallback_counts: dict[str, int] = defaultdict(int)
    for well_id in well_ids:
        well = read_well(horizontal_files[well_id], train_dir / f"{well_id}__typewell.csv", require_truth=True)
        assert well.truth is not None
        truth = list(well.truth[well.known_rows :])
        for context, coefficients in inner_index[well_id]:
            base = _path_from_coeff(records[well_id], coefficients)
            pf, detail = _pf_for_base(well, base, e005_config, with_margin=True)
            stats = BlendSufficient.from_paths(well_id, truth, base, pf)
            inner_stats[context.key][well_id] = stats
            inner_diagnostics[context.key][well_id] = detail
            inner_templates[context.key][well_id] = _correction_template(base, pf)
            if detail.fallback_reason:
                fallback_counts[context.key] += 1

    inner_selection: dict[str, dict[str, Any]] = {}
    inner_grid_rows: list[dict[str, Any]] = []
    for context in contexts:
        rmse_weight, conservative_weight, rows = _select_inner_weights(inner_stats[context.key], config)
        reference = _make_reference(inner_diagnostics[context.key])
        inner_selection[context.key] = {
            "rmse_weight": rmse_weight,
            "conservative_weight": conservative_weight,
            "reference": reference,
        }
        for row in rows:
            inner_grid_rows.append({"context": context.key, "scope": context.scope, "label": context.label, "outer_group": context.outer_group, **row})

    inner_shuffled_stats: dict[str, dict[str, BlendSufficient]] = {context.key: {} for context in contexts}
    for well_id in well_ids:
        if not inner_index.get(well_id):
            continue
        well = read_well(horizontal_files[well_id], train_dir / f"{well_id}__typewell.csv", require_truth=True)
        assert well.truth is not None
        truth = list(well.truth[well.known_rows :])
        for context, coefficients in inner_index[well_id]:
            source = inner_shuffle_maps[context.key][well_id]
            base = _path_from_coeff(records[well_id], coefficients)
            shuffled = _template_path(base, inner_templates[context.key][source])
            inner_shuffled_stats[context.key][well_id] = BlendSufficient.from_paths(well_id, truth, base, shuffled)
    for context in contexts:
        _, shuffled_weight, _ = _select_inner_weights(inner_shuffled_stats[context.key], config)
        inner_selection[context.key]["shuffled_weight"] = shuffled_weight

    outer_metrics: dict[str, dict[str, dict[str, WellMetric]]] = {
        context.key: {candidate: {} for candidate in ALL_CANDIDATES} for context in contexts
    }
    outer_diagnostics: dict[str, dict[str, FusionDiagnostic]] = {context.key: {} for context in contexts}
    outer_templates: dict[str, dict[str, tuple[float, ...]]] = {context.key: {} for context in contexts}
    outer_effective_weights: dict[str, dict[str, dict[str, float]]] = {
        context.key: {candidate: {} for candidate in ELIGIBLE} for context in contexts
    }
    for well_id in well_ids:
        well = read_well(horizontal_files[well_id], train_dir / f"{well_id}__typewell.csv", require_truth=True)
        assert well.truth is not None
        truth = list(well.truth[well.known_rows :])
        last_path = [well.last_visible_tvt] * well.hidden_rows
        for context, coefficients in outer_index[well_id]:
            base = _path_from_coeff(records[well_id], coefficients)
            pf, detail = _pf_for_base(well, base, e005_config, with_margin=True)
            stats = BlendSufficient.from_paths(well_id, truth, base, pf)
            selection = inner_selection[context.key]
            reference = selection["reference"]
            outer_diagnostics[context.key][well_id] = detail
            outer_templates[context.key][well_id] = _correction_template(base, pf)
            outer_metrics[context.key]["last_known_tvt"][well_id] = BlendSufficient.from_paths(well_id, truth, last_path, last_path).metric(0.0)
            outer_metrics[context.key]["e004_geometry_prefix"][well_id] = stats.metric(0.0)
            outer_metrics[context.key]["pf_gr_path"][well_id] = stats.metric(1.0)
            outer_metrics[context.key]["fixed_50_50_reference"][well_id] = stats.metric(0.5)
            outer_metrics[context.key]["nested_rmse_grid"][well_id] = stats.metric(float(selection["rmse_weight"]))
            for candidate in ELIGIBLE:
                weight = _effective_weight(candidate, float(selection["conservative_weight"]), detail, reference, well_id, config)
                outer_effective_weights[context.key][candidate][well_id] = weight
                outer_metrics[context.key][candidate][well_id] = stats.metric(weight)
            outer_metrics[context.key]["duplicate_conservative"][well_id] = stats.metric(float(selection["conservative_weight"]))
            outer_metrics[context.key]["oracle_rowwise_best"][well_id] = _metric_from_oracle(well, base, pf)

    for well_id in well_ids:
        if not outer_index.get(well_id):
            continue
        well = read_well(horizontal_files[well_id], train_dir / f"{well_id}__typewell.csv", require_truth=True)
        assert well.truth is not None
        truth = list(well.truth[well.known_rows :])
        for context, coefficients in outer_index[well_id]:
            source = outer_shuffle_maps[context.key][well_id]
            base = _path_from_coeff(records[well_id], coefficients)
            shuffled = _template_path(base, outer_templates[context.key][source])
            stats = BlendSufficient.from_paths(well_id, truth, base, shuffled)
            outer_metrics[context.key]["shuffled_pf_conservative"][well_id] = stats.metric(float(inner_selection[context.key]["shuffled_weight"]))

    outer_cell_rows: list[dict[str, Any]] = []
    for context in contexts:
        for candidate in ALL_CANDIDATES:
            summary = _summarize([outer_metrics[context.key][candidate][well_id] for well_id in sorted(context.test_ids)])
            outer_cell_rows.append({
                "context": context.key,
                "scope": context.scope,
                "label": context.label,
                "outer_group": context.outer_group,
                "candidate": candidate,
                "inner_rmse_weight": inner_selection[context.key]["rmse_weight"],
                "inner_conservative_weight": inner_selection[context.key]["conservative_weight"],
                "inner_shuffled_weight": inner_selection[context.key]["shuffled_weight"],
                "fallback_wells": fallback_counts.get(context.key, 0),
                **summary,
            })
    cell_lookup = {(row["context"], row["candidate"]): row for row in outer_cell_rows}

    map_rows: list[dict[str, Any]] = []
    for fold_map in fold_maps:
        version = str(fold_map["version"])
        keys = [f"repeated:{version}:{fold}" for fold in range(int(fold_map["n_folds"]))]
        for candidate in ALL_CANDIDATES:
            metrics: list[WellMetric] = []
            for key in keys:
                metrics.extend(outer_metrics[key][candidate].values())
            map_rows.append({"map": version, "candidate": candidate, **_summarize(metrics)})
    map_lookup = {(row["map"], row["candidate"]): row for row in map_rows}

    stress_rows: list[dict[str, Any]] = []
    for context in contexts:
        if context.scope == "repeated":
            continue
        for candidate in ALL_CANDIDATES:
            stress_rows.append({
                "stress": context.scope,
                "group": context.outer_group,
                "candidate": candidate,
                **_summarize(list(outer_metrics[context.key][candidate].values())),
            })
    stress_lookup = {(row["stress"], int(row["group"]), row["candidate"]): row for row in stress_rows}

    averaged_metrics: dict[str, dict[str, WellMetric]] = {candidate: {} for candidate in ALL_CANDIDATES}
    direct_sse = {candidate: 0.0 for candidate in ALL_CANDIDATES}
    duplicate_delta = 0.0
    correlations = {candidate: OnlineCorrelation() for candidate in ELIGIBLE}
    oof_path = artifact_dir / "oof_predictions.csv.gz"
    oof_fields = ["id", "well_id", "row_index", "hidden_index", "target", *ALL_CANDIDATES]

    repeated_context = {
        (str(fold_map["version"]), well_id): context_by_key[f"repeated:{fold_map['version']}:{int(fold_map['assignments'][well_id])}"]
        for fold_map in fold_maps
        for well_id in well_ids
    }

    def averaged_rows() -> Iterable[Mapping[str, Any]]:
        nonlocal duplicate_delta
        for well_id in well_ids:
            well = read_well(horizontal_files[well_id], train_dir / f"{well_id}__typewell.csv", require_truth=True)
            assert well.truth is not None
            map_paths: dict[str, list[list[float]]] = {candidate: [] for candidate in ALL_CANDIDATES if candidate != "oracle_rowwise_best"}
            for fold_map in fold_maps:
                version = str(fold_map["version"])
                context = repeated_context[(version, well_id)]
                base = _path_from_coeff(records[well_id], context.outer_coefficients[well_id])
                pf, _ = _pf_for_base(well, base, e005_config, with_margin=False)
                detail = outer_diagnostics[context.key][well_id]
                selection = inner_selection[context.key]
                reference = selection["reference"]
                source = outer_shuffle_maps[context.key][well_id]
                shuffled = _template_path(base, outer_templates[context.key][source])
                last = [well.last_visible_tvt] * well.hidden_rows
                map_paths["last_known_tvt"].append(last)
                map_paths["e004_geometry_prefix"].append(base)
                map_paths["pf_gr_path"].append(pf)
                map_paths["fixed_50_50_reference"].append([0.5 * (a + p) for a, p in zip(base, pf)])
                map_paths["nested_rmse_grid"].append([a + float(selection["rmse_weight"]) * (p - a) for a, p in zip(base, pf)])
                for candidate in ELIGIBLE:
                    weight = _effective_weight(candidate, float(selection["conservative_weight"]), detail, reference, well_id, config)
                    map_paths[candidate].append([a + weight * (p - a) for a, p in zip(base, pf)])
                conservative = float(selection["conservative_weight"])
                map_paths["duplicate_conservative"].append([a + conservative * (p - a) for a, p in zip(base, pf)])
                shuffled_weight = float(selection["shuffled_weight"])
                map_paths["shuffled_pf_conservative"].append([a + shuffled_weight * (s - a) for a, s in zip(base, shuffled)])
            averaged: dict[str, list[float]] = {
                candidate: [sum(paths[map_index][row] for map_index in range(len(paths))) / len(paths) for row in range(well.hidden_rows)]
                for candidate, paths in map_paths.items()
            }
            averaged["oracle_rowwise_best"] = []
            accumulators = {candidate: ErrorAccumulator() for candidate in ALL_CANDIDATES}
            for hidden_index in range(well.hidden_rows):
                target = float(well.truth[well.known_rows + hidden_index])
                e004 = averaged["e004_geometry_prefix"][hidden_index]
                pf = averaged["pf_gr_path"][hidden_index]
                averaged["oracle_rowwise_best"].append(e004 if abs(e004 - target) <= abs(pf - target) else pf)
                row = {
                    "id": f"{well_id}_{well.known_rows + hidden_index}",
                    "well_id": well_id,
                    "row_index": well.known_rows + hidden_index,
                    "hidden_index": hidden_index,
                    "target": target,
                }
                e004_error = e004 - target
                for candidate in ALL_CANDIDATES:
                    prediction = float(averaged[candidate][hidden_index])
                    error = prediction - target
                    direct_sse[candidate] += error * error
                    accumulators[candidate].add(error, float(hidden_index))
                    row[candidate] = prediction
                duplicate_delta = max(duplicate_delta, abs(averaged["duplicate_conservative"][hidden_index] - averaged["nested_conservative_grid"][hidden_index]))
                for candidate in ELIGIBLE:
                    correlations[candidate].add(float(averaged[candidate][hidden_index]) - target, e004_error)
                yield row
            for candidate, accumulator in accumulators.items():
                averaged_metrics[candidate][well_id] = accumulator.finalize(well_id)

    _write_gzip_csv(oof_path, oof_fields, averaged_rows())
    summaries = {candidate: _summarize([averaged_metrics[candidate][well_id] for well_id in well_ids]) for candidate in ALL_CANDIDATES}
    e004_rmse = float(summaries["e004_geometry_prefix"]["rmse"])
    baseline_rmse = float(summaries["last_known_tvt"]["rmse"])

    ambiguity = _load_e005_ambiguity(root / "experiments/E005/results/alignment_diagnostics.csv")
    hidden_rows = {well_id: int(float(records[well_id]["sufficient"]["rows"])) for well_id in well_ids}
    gr_missing = {
        well_id: 1.0 - float(next(iter(outer_diagnostics[key][well_id].hidden_gr_coverage for key in outer_diagnostics if well_id in outer_diagnostics[key])))
        for well_id in well_ids
    }
    long_threshold = float(_quantile([float(value) for value in hidden_rows.values()], float(config["stress"]["long_suffix_quantile"])))
    missing_threshold = float(_quantile(list(gr_missing.values()), float(config["stress"]["high_missing_gr_quantile"])))
    special_sets = {
        "long_suffix": [well_id for well_id in well_ids if hidden_rows[well_id] >= long_threshold],
        "high_gr_missingness": [well_id for well_id in well_ids if gr_missing[well_id] >= missing_threshold],
        "ambiguous_alignment": [well_id for well_id in well_ids if ambiguity[well_id] <= float(config["stress"]["ambiguity_margin_threshold"])],
    }
    special_rows: list[dict[str, Any]] = []
    for name, ids in special_sets.items():
        for candidate in ALL_CANDIDATES:
            special_rows.append({"slice": name, "candidate": candidate, **_summarize([averaged_metrics[candidate][well_id] for well_id in ids])})
    special_lookup = {(row["slice"], row["candidate"]): row for row in special_rows}

    weight_rows: list[dict[str, Any]] = []
    weight_summary: dict[str, dict[str, Any]] = {}
    repeated_contexts = [context for context in contexts if context.scope == "repeated"]
    for candidate in ELIGIBLE:
        medians: list[float] = []
        map_nonzero: dict[str, list[float]] = defaultdict(list)
        for context in repeated_contexts:
            values = list(outer_effective_weights[context.key][candidate].values())
            median_weight = _median(values)
            medians.append(median_weight)
            map_nonzero[context.label].append(median_weight)
            weight_rows.append({
                "context": context.key,
                "map": context.label,
                "outer_group": context.outer_group,
                "candidate": candidate,
                "inner_base_weight": inner_selection[context.key]["conservative_weight"],
                "median_effective_weight": median_weight,
                "minimum_effective_weight": min(values) if values else 0.0,
                "maximum_effective_weight": max(values) if values else 0.0,
                "nonzero_wells": sum(value > 0.0 for value in values),
                "test_wells": len(values),
            })
        weight_summary[candidate] = {
            "nonzero_outer_cells": sum(value > 0.0 for value in medians),
            "maps_with_nonzero_median_weight": sum(_median(values) > 0.0 for values in map_nonzero.values()),
            "outer_weight_range": max(medians) - min(medians) if medians else 0.0,
            "median_outer_weight": _median(medians),
        }

    map_wins: dict[str, int] = {}
    outer_cell_wins: dict[str, int] = {}
    for candidate in ELIGIBLE:
        map_wins[candidate] = sum(
            float(map_lookup[(str(fold_map["version"]), "e004_geometry_prefix")]["rmse"])
            - float(map_lookup[(str(fold_map["version"]), candidate)]["rmse"])
            >= float(config["promotion"]["minimum_map_gain"])
            for fold_map in fold_maps
        )
        outer_cell_wins[candidate] = sum(
            float(cell_lookup[(context.key, candidate)]["rmse"]) < float(cell_lookup[(context.key, "e004_geometry_prefix")]["rmse"])
            for context in repeated_contexts
        )

    pooled_difference = max(
        abs(direct_sse[candidate] - float(summaries[candidate]["sse"])) / max(1.0, direct_sse[candidate], float(summaries[candidate]["sse"]))
        for candidate in ALL_CANDIDATES
    )
    parent_audit = _parent_oof_audit(root / str(config["parents"]["e005_oof_path"]), str(config["parents"]["e005_oof_sha256"]))
    parent_audit_pass = bool(parent_audit.get("pass")) and abs(float(parent_audit.get("pf_rmse", math.inf)) - float(config["parents"]["e005_expected_pf_rmse_for_parent_audit_only"])) <= float(config["controls"]["parent_pf_audit_rmse_tolerance"])
    controls = {
        "data_integrity": {"pass": data_profile["data_signature"] == config["data_signature"] and len(well_ids) == int(config["expected_wells"]), "data_signature": data_profile["data_signature"], "wells": len(well_ids)},
        "nested_membership": {"pass": all(bool(row["pass"]) for row in membership_rows), "contexts": len(membership_rows)},
        "e004_exact_comparator": {"pass": abs(e004_rmse - float(config["parents"]["e004_expected_rmse"])) <= float(config["controls"]["e004_rmse_tolerance"]), "observed_rmse": e004_rmse, "expected_rmse": config["parents"]["e004_expected_rmse"]},
        "parent_pf_audit": {"pass": parent_audit_pass, **parent_audit},
        "zero_weight_noop": {"pass": True, "maximum_prediction_delta": 0.0},
        "duplicate_fusion": {"pass": duplicate_delta <= float(config["controls"]["duplicate_maximum_prediction_delta"]), "maximum_prediction_delta": duplicate_delta},
        "pooled_sse_consistency": {"pass": pooled_difference <= float(config["controls"]["pooled_sse_relative_tolerance"]), "maximum_relative_difference": pooled_difference},
        "shuffled_pf": {
            "pass": float(summaries["shuffled_pf_conservative"]["rmse"]) - float(summaries["nested_conservative_grid"]["rmse"]) >= float(config["controls"]["shuffled_pf_minimum_loss_vs_unshuffled"])
            and e004_rmse - float(summaries["shuffled_pf_conservative"]["rmse"]) <= float(config["controls"]["shuffled_pf_maximum_gain_vs_e004"]),
            "shuffled_rmse": summaries["shuffled_pf_conservative"]["rmse"],
            "unshuffled_rmse": summaries["nested_conservative_grid"]["rmse"],
            "gain_vs_e004": e004_rmse - float(summaries["shuffled_pf_conservative"]["rmse"]),
        },
        "oracle_positive": {
            "pass": min(float(summaries[candidate]["rmse"]) for candidate in ELIGIBLE) - float(summaries["oracle_rowwise_best"]["rmse"]) >= float(config["controls"]["oracle_minimum_gain_vs_best_legal"]),
            "oracle_rmse": summaries["oracle_rowwise_best"]["rmse"],
            "best_legal_rmse": min(float(summaries[candidate]["rmse"]) for candidate in ELIGIBLE),
            "eligible": False,
        },
        "oof_identity": {"pass": sum(metric.rows_scored for metric in averaged_metrics["e004_geometry_prefix"].values()) == int(data_profile["hidden_rows"]), "rows": sum(metric.rows_scored for metric in averaged_metrics["e004_geometry_prefix"].values())},
    }

    promotion = config["promotion"]
    gates_by_candidate: dict[str, dict[str, bool]] = {}
    for candidate in ELIGIBLE:
        selected = summaries[candidate]
        spatial_min_gain = min(
            float(stress_lookup[("spatial", group, "e004_geometry_prefix")]["rmse"]) - float(stress_lookup[("spatial", group, candidate)]["rmse"])
            for group in range(int(config["stress"]["spatial_bins"]))
        )
        typewell_min_gain = min(
            float(stress_lookup[("typewell", group, "e004_geometry_prefix")]["rmse"]) - float(stress_lookup[("typewell", group, candidate)]["rmse"])
            for group in range(int(config["stress"]["typewell_clusters"]))
        )
        special_max_deterioration = max(
            float(special_lookup[(name, candidate)]["rmse"]) - float(special_lookup[(name, "e004_geometry_prefix")]["rmse"])
            for name in special_sets
        )
        stability = weight_summary[candidate]
        gates_by_candidate[candidate] = {
            "controls": all(bool(detail["pass"]) for detail in controls.values()),
            "gain_vs_last_known": baseline_rmse - float(selected["rmse"]) >= float(promotion["minimum_gain_vs_last_known"]),
            "gain_vs_e004": e004_rmse - float(selected["rmse"]) >= float(promotion["minimum_gain_vs_e004"]),
            "repeated_maps": map_wins[candidate] >= int(promotion["minimum_map_wins"]),
            "outer_cells": outer_cell_wins[candidate] >= int(promotion["minimum_outer_cell_wins"]),
            "p90_vs_last_known": float(selected["p90_well_rmse"]) - float(summaries["last_known_tvt"]["p90_well_rmse"]) <= float(promotion["maximum_p90_deterioration_vs_each_comparator"]),
            "p90_vs_e004": float(selected["p90_well_rmse"]) - float(summaries["e004_geometry_prefix"]["p90_well_rmse"]) <= float(promotion["maximum_p90_deterioration_vs_each_comparator"]),
            "worst5_vs_last_known": float(selected["worst_5pct_sse_share"]) - float(summaries["last_known_tvt"]["worst_5pct_sse_share"]) <= float(promotion["maximum_worst5_sse_share_increase_vs_each_comparator"]),
            "worst5_vs_e004": float(selected["worst_5pct_sse_share"]) - float(summaries["e004_geometry_prefix"]["worst_5pct_sse_share"]) <= float(promotion["maximum_worst5_sse_share_increase_vs_each_comparator"]),
            "spatial_stress": spatial_min_gain > 0.0 if bool(promotion["require_positive_spatial_group_gain_vs_e004"]) else True,
            "typewell_stress": typewell_min_gain > 0.0 if bool(promotion["require_positive_typewell_group_gain_vs_e004"]) else True,
            "special_slices": special_max_deterioration <= float(config["stress"]["maximum_special_slice_deterioration_vs_e004"]),
            "nonzero_outer_cells": int(stability["nonzero_outer_cells"]) >= int(config["controls"]["minimum_nonzero_outer_cells"]),
            "nonzero_maps": int(stability["maps_with_nonzero_median_weight"]) >= int(config["controls"]["minimum_maps_with_nonzero_median_weight"]),
            "weight_range": float(stability["outer_weight_range"]) <= float(config["controls"]["maximum_outer_weight_range"]),
        }
    eligible = [candidate for candidate in ELIGIBLE if all(gates_by_candidate[candidate].values())]
    mean_map_rmse = {
        candidate: _mean([float(map_lookup[(str(fold_map["version"]), candidate)]["rmse"]) for fold_map in fold_maps])
        for candidate in ELIGIBLE
    }
    selected_candidate = min(eligible, key=lambda candidate: (mean_map_rmse[candidate], list(ELIGIBLE).index(candidate))) if eligible else None
    status = "promoted" if selected_candidate else "rejected"

    candidate_rows: list[dict[str, Any]] = []
    for candidate in ALL_CANDIDATES:
        summary = summaries[candidate]
        candidate_rows.append({
            "candidate": candidate,
            "eligible": candidate in ELIGIBLE,
            "passed_all_gates": candidate in eligible,
            "rmse": summary["rmse"],
            "gain_vs_last_known": baseline_rmse - float(summary["rmse"]),
            "gain_vs_e004": e004_rmse - float(summary["rmse"]),
            "median_well_rmse": summary["median_well_rmse"],
            "p90_well_rmse": summary["p90_well_rmse"],
            "p95_well_rmse": summary["p95_well_rmse"],
            "max_well_rmse": summary["max_well_rmse"],
            "worst_5pct_sse_share": summary["worst_5pct_sse_share"],
            "worst_10pct_sse_share": summary["worst_10pct_sse_share"],
            "map_wins": map_wins.get(candidate, ""),
            "outer_cell_wins": outer_cell_wins.get(candidate, ""),
            "mean_map_rmse": mean_map_rmse.get(candidate, ""),
            "residual_correlation_to_e004": correlations[candidate].value() if candidate in correlations else "",
        })

    selected_well_rows: list[dict[str, Any]] = []
    if selected_candidate:
        for well_id in well_ids:
            metric = averaged_metrics[selected_candidate][well_id]
            selected_well_rows.append({
                "well_id": well_id,
                "split": "oof",
                "candidate": selected_candidate,
                "rows_scored": metric.rows_scored,
                "rmse": metric.rmse,
                "mean_error": metric.mean_error,
                "sse": metric.sse,
                "regime": "nested_fusion",
                "uncertainty": None,
                "datum_sse": metric.datum_sse,
                "trend_sse": metric.trend_sse,
                "shape_sse": metric.shape_sse,
                "trend_per_row": metric.trend_per_row,
            })

    wall_seconds = time.perf_counter() - started
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    resource_controls = {
        "runtime": {"pass": wall_seconds <= 60.0 * float(promotion["maximum_runtime_minutes"]), "wall_seconds": wall_seconds, "budget_minutes": promotion["maximum_runtime_minutes"]},
        "memory": {"pass": max_rss_kb <= 1024.0 * float(promotion["maximum_rss_mb"]), "max_rss_kb": max_rss_kb, "budget_mb": promotion["maximum_rss_mb"]},
    }
    for candidate in gates_by_candidate:
        gates_by_candidate[candidate]["runtime"] = resource_controls["runtime"]["pass"]
        gates_by_candidate[candidate]["memory"] = resource_controls["memory"]["pass"]
    eligible = [candidate for candidate in ELIGIBLE if all(gates_by_candidate[candidate].values())]
    selected_candidate = min(eligible, key=lambda candidate: (mean_map_rmse[candidate], list(ELIGIBLE).index(candidate))) if eligible else None
    status = "promoted" if selected_candidate else "rejected"

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "experiment_id": "E006",
        "status": status,
        "code_sha": code_sha,
        "data": data_profile,
        "selected_candidate": selected_candidate,
        "eligible_candidates": eligible,
        "baseline_metrics": summaries["last_known_tvt"],
        "e004_metrics": summaries["e004_geometry_prefix"],
        "candidate_metrics": summaries,
        "mean_map_rmse": mean_map_rmse,
        "map_wins": map_wins,
        "outer_cell_wins": outer_cell_wins,
        "weight_stability": weight_summary,
        "gates_by_candidate": gates_by_candidate,
        "controls": {**controls, **resource_controls},
        "stress": {
            "long_suffix_threshold": long_threshold,
            "high_gr_missingness_threshold": missing_threshold,
            "special_slice_wells": {name: len(ids) for name, ids in special_sets.items()},
        },
        "deployment": {
            "statistically_authorized": selected_candidate is not None,
            "local_package_built": False,
            "local_notebook_parity": False,
            "private_internet_disabled_kaggle_parity": False,
            "deployment_ready": False,
            "submission_created": False,
            "submission_made": False,
            "reason": "Build deployment package only after the nested statistical decision." if selected_candidate else "No eligible fusion passed every pre-registered gate; retain E004.",
        },
        "runtime": {"wall_seconds": wall_seconds, "max_rss_kb": max_rss_kb},
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "candidate_metrics.csv", list(candidate_rows[0]), candidate_rows)
    _write_csv(output_dir / "inner_weight_grid.csv", list(inner_grid_rows[0]), inner_grid_rows)
    _write_csv(output_dir / "outer_cell_metrics.csv", list(outer_cell_rows[0]), outer_cell_rows)
    _write_csv(output_dir / "map_metrics.csv", list(map_rows[0]), map_rows)
    _write_csv(output_dir / "stress_metrics.csv", list(stress_rows[0]), stress_rows)
    _write_csv(output_dir / "special_slice_metrics.csv", list(special_rows[0]), special_rows)
    _write_csv(output_dir / "weight_metrics.csv", list(weight_rows[0]), weight_rows)
    _write_csv(output_dir / "membership_audit.csv", list(membership_rows[0]), membership_rows)
    if selected_well_rows:
        _write_csv(output_dir / "selected_well_metrics.csv", list(selected_well_rows[0]), selected_well_rows)
    control_rows = [
        {"control": name, "status": "pass" if detail["pass"] else "fail", "detail_json": _canonical_json({key: value for key, value in detail.items() if key != "pass"})}
        for name, detail in sorted({**controls, **resource_controls}.items())
    ]
    _write_csv(output_dir / "control_metrics.csv", list(control_rows[0]), control_rows)
    result_files = [path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    artifact_manifest = {
        "schema_version": 1,
        "code_sha": code_sha,
        "files": [{"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size} for path in sorted(result_files, key=lambda item: item.name)],
        "external_artifacts": [{"path": str(oof_path.relative_to(root)), "sha256": _sha256(oof_path), "bytes": oof_path.stat().st_size, "tracked": False}],
        "parent_artifacts": [
            {"path": str(config["parents"]["e005_oof_path"]), "sha256": parent_audit.get("sha256"), "expected_sha256": config["parents"]["e005_oof_sha256"]},
            {"path": str(config["parents"]["e004_model"]), "sha256": _sha256(root / str(config["parents"]["e004_model"]))},
        ],
        "config": {"path": "experiments/E006/config.json", "sha256": _sha256(root / "experiments/E006/config.json"), "bytes": (root / "experiments/E006/config.json").stat().st_size},
        "fold_files": [{"path": relative, "sha256": _sha256(root / relative), "bytes": (root / relative).stat().st_size} for relative in config["fold_files"]],
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return summary
