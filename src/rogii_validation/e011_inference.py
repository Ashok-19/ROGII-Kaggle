"""Deterministic E011 spline-coefficient inference.

The deployed rule is the statistically verified ``spline4_ridge_equal_s075``
branch. It reconstructs the exact E008 legal feature contract from one supplied
well, predicts four bounded spline-knot corrections, and applies them around the
frozen E006 path. Hidden TVT, formation surfaces, absolute spatial coordinates,
leaderboard information, and external artifacts are never read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .e004_inference import (
    DeploymentDataError,
    SUBMISSION_DECIMAL_PLACES,
    extract_well_features,
    predict_coefficients as predict_e004_coefficients,
    read_horizontal,
    reconstruct_hidden,
)
from .e006_inference import E006TypewellCurve, particle_path, validate_e006_model
from .residual_action import _extract_horizontal_features, _path_summary
from .self_correlation import SelfCorrDiagnostic, read_horizontal_selfcorr, selfcorr_path

E011_MODEL_NAME = "spline4_ridge_equal_s075"
E011_REPRESENTATION = "spline4"
E011_OUTPUTS = 4


def _finite_number(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DeploymentDataError(f"E011 model contains invalid {name}") from exc
    if not math.isfinite(parsed):
        raise DeploymentDataError(f"E011 model contains non-finite {name}")
    return parsed


def validate_e011_model(model: Mapping[str, Any]) -> None:
    if not isinstance(model, Mapping) or int(model.get("schema_version", 0)) != 1:
        raise DeploymentDataError("unsupported E011 model schema")
    if model.get("experiment_id") != "E011" or model.get("model_name") != E011_MODEL_NAME:
        raise DeploymentDataError("unsupported E011 deployment candidate")
    if model.get("representation") != E011_REPRESENTATION:
        raise DeploymentDataError("unsupported E011 representation")
    if bool(model.get("internet_required")) or bool(model.get("external_artifacts_required")):
        raise DeploymentDataError("E011 deployment model must be offline and self-contained")
    if bool(model.get("surfaces_required")):
        raise DeploymentDataError("E011 deployment model must not require formation surfaces")
    if int(model.get("training_wells", 0)) != 773:
        raise DeploymentDataError("E011 deployment model training-well identity differs")
    selected = list(model.get("selected_features", []))
    arrays = [
        list(model.get("feature_medians", [])),
        list(model.get("feature_means", [])),
        list(model.get("feature_scales", [])),
        list(model.get("ridge_coefficients", [])),
    ]
    if len(selected) != 142 or any(len(values) != len(selected) for values in arrays):
        raise DeploymentDataError("E011 feature arrays have inconsistent lengths")
    if len(set(selected)) != len(selected) or selected != sorted(selected):
        raise DeploymentDataError("E011 selected feature order is not unique and canonical")
    coefficients = arrays[-1]
    if any(not isinstance(row, (list, tuple)) or len(row) != E011_OUTPUTS for row in coefficients):
        raise DeploymentDataError("E011 ridge coefficient matrix has invalid shape")
    intercept = list(model.get("ridge_scaled_intercept", []))
    target_means = list(model.get("target_means", []))
    target_scales = list(model.get("target_scales", []))
    if not (len(intercept) == len(target_means) == len(target_scales) == E011_OUTPUTS):
        raise DeploymentDataError("E011 target arrays have invalid shape")
    for name, values in (
        ("feature median", arrays[0]),
        ("feature mean", arrays[1]),
        ("feature scale", arrays[2]),
        ("ridge intercept", intercept),
        ("target mean", target_means),
        ("target scale", target_scales),
    ):
        parsed = [_finite_number(value, name) for value in values]
        if name == "feature scale" and any(abs(value) <= 1e-12 for value in parsed):
            raise DeploymentDataError("E011 feature scale is zero")
        if name == "target scale" and any(abs(value) <= 1e-12 for value in parsed):
            raise DeploymentDataError("E011 target scale is zero")
    for row in coefficients:
        for value in row:
            _finite_number(value, "ridge coefficient")
    if abs(_finite_number(model.get("ridge_alpha"), "ridge alpha") - 1.0) > 1e-12:
        raise DeploymentDataError("E011 ridge alpha differs from the verified rule")
    if abs(_finite_number(model.get("shrinkage"), "shrinkage") - 0.75) > 1e-12:
        raise DeploymentDataError("E011 shrinkage differs from the verified rule")
    if abs(_finite_number(model.get("coefficient_absolute_bound_ft"), "coefficient bound") - 80.0) > 1e-12:
        raise DeploymentDataError("E011 coefficient bound differs from the verified rule")
    knots = [_finite_number(value, "spline knot") for value in model.get("spline_knots", [])]
    if knots != [0.25, 0.5, 0.75, 1.0]:
        raise DeploymentDataError("E011 spline knots differ from the verified rule")
    e006_model = model.get("e006_model")
    if not isinstance(e006_model, Mapping):
        raise DeploymentDataError("E011 model is missing the embedded E006 model")
    validate_e006_model(e006_model)
    e007_config = model.get("e007_config")
    e008_config = model.get("e008_config")
    if not isinstance(e007_config, Mapping) or not isinstance(e007_config.get("self_correlation"), Mapping):
        raise DeploymentDataError("E011 model is missing E007 self-correlation settings")
    if not isinstance(e008_config, Mapping) or not isinstance(e008_config.get("feature_contract"), Mapping):
        raise DeploymentDataError("E011 model is missing E008 feature settings")


def load_e011_model(path: Path) -> dict[str, Any]:
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentDataError(f"{path}: could not load E011 model") from exc
    validate_e011_model(model)
    return model


def _diagnostic_mapping(detail: SelfCorrDiagnostic) -> dict[str, float | str]:
    return {
        "visible_gr_coverage": float(detail.visible_gr_coverage),
        "hidden_gr_coverage": float(detail.hidden_gr_coverage),
        "template_states": float(detail.template_states),
        "query_anchors": float(detail.query_anchors),
        "median_best_fingerprint_distance": float(detail.median_best_distance),
        "median_fingerprint_margin": float(detail.median_margin),
        "visible_pseudo_holdout_gain": float(detail.pseudo_gain),
        "pseudo_base_rmse": float(detail.pseudo_base_rmse),
        "pseudo_fixed_rmse": float(detail.pseudo_fixed_rmse),
        "fallback_reason": str(detail.fallback_reason),
        "maximum_absolute_correction": float(detail.maximum_absolute_correction),
    }


def legal_feature_vector(
    *,
    horizontal_path: Path,
    e004_path: Sequence[float],
    e006_path: Sequence[float],
    selfcorr_raw_path: Sequence[float],
    selfcorr_detail: SelfCorrDiagnostic,
    e008_config: Mapping[str, Any],
) -> dict[str, float | None]:
    if not e006_path or not (len(e004_path) == len(e006_path) == len(selfcorr_raw_path)):
        raise DeploymentDataError("E011 parent paths have inconsistent lengths")
    if any(not math.isfinite(float(value)) for path in (e004_path, e006_path, selfcorr_raw_path) for value in path):
        raise DeploymentDataError("E011 parent path contains non-finite values")
    well_id = horizontal_path.name.split("__", 1)[0]
    path_features: dict[str, float | None] = {}
    last_visible = read_horizontal_selfcorr(horizontal_path).last_visible_tvt
    last_path = [float(last_visible)] * len(e006_path)
    try:
        _path_summary(path_features, "e006_path", e006_path)
        _path_summary(path_features, "e006_minus_last", [a - b for a, b in zip(e006_path, last_path)])
        _path_summary(path_features, "e006_minus_e004", [a - b for a, b in zip(e006_path, e004_path)])
        _path_summary(path_features, "e004_minus_last", [a - b for a, b in zip(e004_path, last_path)])
        _path_summary(path_features, "selfcorr_raw_minus_e006", [a - b for a, b in zip(selfcorr_raw_path, e006_path)])
        features, _spatial, hidden_rows = _extract_horizontal_features(
            train_dir=horizontal_path.parent,
            well_id=well_id,
            path_features=path_features,
            diagnostic=_diagnostic_mapping(selfcorr_detail),
            config=e008_config,
        )
    except Exception as exc:
        if isinstance(exc, DeploymentDataError):
            raise
        raise DeploymentDataError(f"{well_id}: could not reconstruct the E008 legal feature vector") from exc
    if hidden_rows != len(e006_path):
        raise DeploymentDataError(f"{well_id}: feature/path hidden-row counts differ")
    return features


def predict_spline_coefficients(model: Mapping[str, Any], features: Mapping[str, float | None]) -> tuple[float, float, float, float]:
    validate_e011_model(model)
    selected = list(model["selected_features"])
    medians = list(model["feature_medians"])
    means = list(model["feature_means"])
    scales = list(model["feature_scales"])
    coefficients = list(model["ridge_coefficients"])
    intercept = [float(value) for value in model["ridge_scaled_intercept"]]
    scaled_targets = list(intercept)
    for feature_index, name in enumerate(selected):
        raw = features.get(name)
        value = float(raw) if raw is not None and math.isfinite(float(raw)) else float(medians[feature_index])
        standardized = (value - float(means[feature_index])) / float(scales[feature_index])
        if not math.isfinite(standardized):
            raise DeploymentDataError(f"E011 standardized feature is non-finite: {name}")
        for output in range(E011_OUTPUTS):
            scaled_targets[output] += standardized * float(coefficients[feature_index][output])
    raw_targets = [
        float(model["target_means"][index]) + float(model["target_scales"][index]) * scaled_targets[index]
        for index in range(E011_OUTPUTS)
    ]
    shrinkage = float(model["shrinkage"])
    bound = float(model["coefficient_absolute_bound_ft"])
    bounded = [min(bound, max(-bound, shrinkage * value)) for value in raw_targets]
    if any(not math.isfinite(value) for value in bounded):
        raise DeploymentDataError("E011 emitted non-finite spline coefficient")
    return tuple(bounded)  # type: ignore[return-value]


def spline4_correction(hidden_rows: int, coefficients: Sequence[float]) -> list[float]:
    if hidden_rows <= 0 or len(coefficients) != E011_OUTPUTS:
        raise DeploymentDataError("E011 spline correction has invalid dimensions")
    values = [float(value) for value in coefficients]
    if any(not math.isfinite(value) for value in values):
        raise DeploymentDataError("E011 spline coefficient is non-finite")
    knots = (0.25, 0.5, 0.75, 1.0)
    output: list[float] = []
    for index in range(hidden_rows):
        position = index / max(1, hidden_rows - 1)
        if position <= knots[0]:
            correction = values[0] * (position / knots[0])
        else:
            correction = values[-1]
            previous_position = knots[0]
            previous_value = values[0]
            for knot_index in range(1, len(knots)):
                right_position = knots[knot_index]
                right_value = values[knot_index]
                if position <= right_position:
                    fraction = (position - previous_position) / (right_position - previous_position)
                    correction = previous_value * (1.0 - fraction) + right_value * fraction
                    break
                previous_position = right_position
                previous_value = right_value
        if not math.isfinite(correction):
            raise DeploymentDataError("E011 reconstructed non-finite spline correction")
        output.append(correction)
    return output


def predict_e011_well(model: Mapping[str, Any], horizontal_path: Path, typewell_path: Path) -> tuple[list[float], dict[str, Any]]:
    validate_e011_model(model)
    well_id = horizontal_path.name.split("__", 1)[0]
    if not typewell_path.exists():
        raise DeploymentDataError(f"missing typewell for {well_id}")
    e006_model = model["e006_model"]
    e004_model = e006_model["e004_model"]
    extracted = extract_well_features(
        horizontal_path,
        typewell_path,
        visible_slope_windows=e004_model["visible_slope_windows"],
        visible_backtest_fractions=e004_model["visible_backtest_fractions"],
        require_truth=False,
    )
    datum, trend = predict_e004_coefficients(e004_model, extracted["features"])
    metadata = extracted["metadata"]
    e004_path = reconstruct_hidden(metadata["last_visible_tvt"], metadata["hidden_rows"], datum, trend)
    horizontal = read_horizontal(horizontal_path, require_truth=False)
    curve = E006TypewellCurve.read(typewell_path)
    pf_path, e006_detail = particle_path(
        horizontal,
        curve,
        e004_path,
        e006_model["alignment"],
        e006_model["particle_filter"],
    )
    weight = float(e006_model["fusion_weight"])
    e006_path = [float(base) + weight * (float(pf) - float(base)) for base, pf in zip(e004_path, pf_path)]
    selfcorr_well = read_horizontal_selfcorr(horizontal_path)
    selfcorr_raw, selfcorr_detail = selfcorr_path(
        selfcorr_well,
        e006_path,
        model["e007_config"]["self_correlation"],
    )
    features = legal_feature_vector(
        horizontal_path=horizontal_path,
        e004_path=e004_path,
        e006_path=e006_path,
        selfcorr_raw_path=selfcorr_raw,
        selfcorr_detail=selfcorr_detail,
        e008_config=model["e008_config"],
    )
    coefficients = predict_spline_coefficients(model, features)
    correction = spline4_correction(len(e006_path), coefficients)
    prediction = [float(base) + float(delta) for base, delta in zip(e006_path, correction)]
    if len(prediction) != int(metadata["hidden_rows"]) or any(not math.isfinite(value) for value in prediction):
        raise DeploymentDataError(f"{well_id}: invalid E011 prediction path")
    return prediction, {
        "well_id": well_id,
        "known_rows": int(metadata["known_rows"]),
        "hidden_rows": int(metadata["hidden_rows"]),
        "coefficients": list(coefficients),
        "maximum_absolute_correction": max(abs(value) for value in correction),
        "e006_fallback_reason": str(e006_detail.get("fallback_reason", "")),
        "selfcorr_fallback_reason": str(selfcorr_detail.fallback_reason),
        "feature_missing_count": sum(features.get(name) is None for name in model["selected_features"]),
    }


def build_e011_prediction_map(model: Mapping[str, Any], test_dir: Path) -> tuple[dict[str, float], list[dict[str, Any]]]:
    validate_e011_model(model)
    horizontal_files = sorted(test_dir.glob("*__horizontal_well.csv"))
    if not horizontal_files:
        raise DeploymentDataError(f"{test_dir}: no horizontal well files")
    predictions: dict[str, float] = {}
    well_rows: list[dict[str, Any]] = []
    for horizontal_path in horizontal_files:
        well_id = horizontal_path.name.split("__", 1)[0]
        typewell_path = test_dir / f"{well_id}__typewell.csv"
        values, detail = predict_e011_well(model, horizontal_path, typewell_path)
        for offset, value in enumerate(values, start=int(detail["known_rows"])):
            key = f"{well_id}_{offset}"
            if key in predictions:
                raise DeploymentDataError(f"duplicate prediction id {key}")
            predictions[key] = value
        well_rows.append(detail)
    return predictions, well_rows


def _sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise DeploymentDataError(f"could not read input file {path}") from exc
    with handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def profile_e011_test_input(test_dir: Path, sample_submission: Path) -> dict[str, Any]:
    """Validate and hash one competition-style E011 test input without reading hidden TVT."""
    horizontal_files = sorted(test_dir.glob("*__horizontal_well.csv"))
    typewell_files = sorted(test_dir.glob("*__typewell.csv"))
    if not horizontal_files:
        raise DeploymentDataError(f"{test_dir}: no horizontal well files")
    horizontal_ids = [path.name.split("__", 1)[0] for path in horizontal_files]
    typewell_ids = [path.name.split("__", 1)[0] for path in typewell_files]
    if len(horizontal_ids) != len(set(horizontal_ids)) or len(typewell_ids) != len(set(typewell_ids)):
        raise DeploymentDataError("duplicate well files in E011 test input")
    if horizontal_ids != typewell_ids:
        missing_typewell = sorted(set(horizontal_ids) - set(typewell_ids))[:5]
        extra_typewell = sorted(set(typewell_ids) - set(horizontal_ids))[:5]
        raise DeploymentDataError(
            f"horizontal/typewell well mismatch; missing_typewell={missing_typewell}, extra_typewell={extra_typewell}"
        )

    expected_ids: list[str] = []
    well_rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for horizontal_path in horizontal_files:
        well_id = horizontal_path.name.split("__", 1)[0]
        typewell_path = test_dir / f"{well_id}__typewell.csv"
        well = read_horizontal_selfcorr(horizontal_path)
        typewell = E006TypewellCurve.read(typewell_path)
        expected_ids.extend(f"{well_id}_{index}" for index in range(well.known_rows, len(well.md)))
        well_rows.append(
            {
                "well_id": well_id,
                "rows": len(well.md),
                "known_rows": well.known_rows,
                "hidden_rows": well.hidden_rows,
                "visible_gr_coverage": well.visible_gr_coverage,
                "hidden_gr_coverage": well.hidden_gr_coverage,
            }
        )
        horizontal_legal = json.dumps(
            {
                "MD": well.md,
                "X": well.x,
                "Y": well.y,
                "Z": well.z,
                "GR": well.gr,
                "TVT_input": well.tvt_input,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        typewell_legal = json.dumps(
            {"TVT": typewell.tvt, "GR": typewell.gr},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        files.extend(
            [
                {
                    "path": f"test/{horizontal_path.name}",
                    "bytes": horizontal_path.stat().st_size,
                    "legal_sha256": hashlib.sha256(horizontal_legal).hexdigest(),
                    "legal_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
                },
                {
                    "path": f"test/{typewell_path.name}",
                    "bytes": typewell_path.stat().st_size,
                    "legal_sha256": hashlib.sha256(typewell_legal).hexdigest(),
                    "legal_columns": ["TVT", "GR"],
                },
            ]
        )

    sample_ids: list[str] = []
    seen: set[str] = set()
    try:
        handle = sample_submission.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise DeploymentDataError(f"could not read sample submission {sample_submission}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != ["id", "tvt"]:
            raise DeploymentDataError(f"{sample_submission}: expected columns id,tvt")
        for row in reader:
            key = str(row.get("id") or "")
            if not key or key in seen:
                raise DeploymentDataError(f"{sample_submission}: empty or duplicate id {key!r}")
            seen.add(key)
            sample_ids.append(key)
    if set(sample_ids) != set(expected_ids):
        missing = sorted(set(sample_ids) - set(expected_ids))[:5]
        extra = sorted(set(expected_ids) - set(sample_ids))[:5]
        raise DeploymentDataError(f"sample/test ID mismatch; unknown_sample_ids={missing}, absent_sample_ids={extra}")
    sample_entry = {
        "path": "sample_submission.csv",
        "bytes": sample_submission.stat().st_size,
        "legal_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "legal_columns": ["id"],
    }
    files.append(sample_entry)
    files.sort(key=lambda item: str(item["path"]))
    signature_files = [
        {
            "path": item["path"],
            "legal_sha256": item["legal_sha256"],
            "legal_columns": item["legal_columns"],
        }
        for item in files
    ]
    signature_payload = json.dumps(signature_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "wells": len(well_rows),
        "rows": sum(int(row["rows"]) for row in well_rows),
        "known_rows": sum(int(row["known_rows"]) for row in well_rows),
        "hidden_rows": sum(int(row["hidden_rows"]) for row in well_rows),
        "sample_rows": len(sample_ids),
        "well_ids_sha256": hashlib.sha256("\n".join(horizontal_ids).encode("utf-8")).hexdigest(),
        "sample_id_order_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "input_files_signature": hashlib.sha256(signature_payload).hexdigest(),
        "sample_submission": sample_entry,
        "files": files,
        "well_profiles": well_rows,
    }


def write_e011_submission(model: Mapping[str, Any], test_dir: Path, sample_submission: Path, output_path: Path) -> dict[str, Any]:
    prediction_map, well_rows = build_e011_prediction_map(model, test_dir)
    sample_ids: list[str] = []
    seen: set[str] = set()
    try:
        handle = sample_submission.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise DeploymentDataError(f"could not read sample submission {sample_submission}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != ["id", "tvt"]:
            raise DeploymentDataError(f"{sample_submission}: expected columns id,tvt")
        for row in reader:
            key = str(row.get("id") or "")
            if not key or key in seen:
                raise DeploymentDataError(f"{sample_submission}: empty or duplicate id {key!r}")
            seen.add(key)
            sample_ids.append(key)
    if set(sample_ids) != set(prediction_map):
        missing = sorted(set(sample_ids) - set(prediction_map))[:5]
        extra = sorted(set(prediction_map) - set(sample_ids))[:5]
        raise DeploymentDataError(f"sample/prediction ID mismatch; missing={missing}, extra={extra}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "tvt"], lineterminator="\n")
        writer.writeheader()
        for key in sample_ids:
            writer.writerow({"id": key, "tvt": format(prediction_map[key], f".{SUBMISSION_DECIMAL_PLACES}f")})
    return {
        "rows": len(sample_ids),
        "wells": len(well_rows),
        "minimum_prediction": min(prediction_map.values()),
        "maximum_prediction": max(prediction_map.values()),
        "maximum_absolute_spline_correction": max(float(row["maximum_absolute_correction"]) for row in well_rows),
        "well_predictions": well_rows,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "bytes": output_path.stat().st_size,
    }
