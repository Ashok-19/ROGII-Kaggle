import csv
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.fusion import (
    ALL_CANDIDATES,
    BlendSufficient,
    FusionDiagnostic,
    _context_shuffle,
    _correction_template,
    _effective_weight,
    _fit_context,
    _pf_for_base,
    _make_reference,
    _eligible_from_gates,
    _mark_candidate_gate_rows,
    _parent_pf_control,
    _select_inner_weights,
    _template_path,
    validate_e006_config,
)
from rogii_validation.e004_inference import read_horizontal
from rogii_validation.e006_inference import (
    E006TypewellCurve,
    build_e006_prediction_map,
    load_e006_model,
    particle_path,
    validate_e006_model,
    write_e006_submission,
)
from rogii_validation.e004_inference import DeploymentDataError
from rogii_validation.gr_path import TypewellCurve, WellData, _calibration, _hidden_samples, _particle_path, read_well
from rogii_validation.harness import DataValidationError, ErrorAccumulator


class E006FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "experiments/E006/config.json").read_text(encoding="utf-8"))

    def test_frozen_config_matches_candidate_order(self) -> None:
        validate_e006_config(self.config)
        self.assertEqual(self.config["candidate_order"], list(ALL_CANDIDATES))
        broken = json.loads(json.dumps(self.config))
        broken["fusion"]["weight_grid"] = [0.0, 0.5, 0.5, 1.0]
        with self.assertRaises(DataValidationError):
            validate_e006_config(broken)
        for mutate in (
            lambda item: item["reliability"].update(gate_minimum_score=math.nan),
            lambda item: item["fusion"].update(fallback_weight=2.0),
            lambda item: item["selection"].update(no_post_score_threshold_changes=False),
            lambda item: item["deployment"].update(internet=True),
            lambda item: item.update(fold_files=["folds/v1.json"] * 5),
        ):
            malformed = json.loads(json.dumps(self.config))
            mutate(malformed)
            with self.assertRaises(DataValidationError):
                validate_e006_config(malformed)

    def test_blend_sufficient_matches_direct_accumulator(self) -> None:
        truth = [10.0, 11.0, 12.0, 13.0]
        base = [9.0, 10.5, 12.5, 14.0]
        pf = [11.0, 11.5, 11.5, 12.0]
        sufficient = BlendSufficient.from_paths("aaaaaaaa", truth, base, pf)
        for weight in (0.0, 0.125, 0.5, 1.0):
            direct = ErrorAccumulator()
            for index, (target, anchor, particle) in enumerate(zip(truth, base, pf)):
                prediction = anchor + weight * (particle - anchor)
                direct.add(prediction - target, float(index))
            expected = direct.finalize("aaaaaaaa")
            actual = sufficient.metric(weight)
            self.assertAlmostEqual(actual.sse, expected.sse, places=12)
            self.assertAlmostEqual(actual.rmse, expected.rmse, places=12)
            self.assertAlmostEqual(actual.mean_error, expected.mean_error, places=12)
            self.assertAlmostEqual(actual.trend_sse, expected.trend_sse, places=12)

    def test_template_resampling_preserves_affine_correction(self) -> None:
        base = [100.0 + index for index in range(17)]
        pf = [value + 2.0 - 5.0 * index / 16.0 for index, value in enumerate(base)]
        template = _correction_template(base, pf)
        reconstructed = _template_path(base, template)
        self.assertEqual(len(template), 64)
        self.assertEqual(len(reconstructed), len(pf))
        self.assertLess(max(abs(left - right) for left, right in zip(reconstructed, pf)), 1e-12)

    def test_conservative_inner_selection_falls_back_or_uses_smallest_near_best_weight(self) -> None:
        improving = {}
        for index in range(40):
            truth = [0.0] * 20
            base = [1.0 + 0.01 * index] * 20
            pf = [-0.2] * 20
            improving[f"{index:08x}"] = BlendSufficient.from_paths(f"{index:08x}", truth, base, pf)
        rmse_weight, conservative, rows = _select_inner_weights(improving, self.config)
        self.assertGreater(rmse_weight, 0.0)
        self.assertGreater(conservative, 0.0)
        passing = [row for row in rows if row["passes_conservative_constraints"]]
        self.assertTrue(passing)
        best = min(float(row["rmse"]) for row in passing)
        self.assertLessEqual(next(float(row["rmse"]) for row in rows if row["weight"] == conservative), best + self.config["fusion"]["conservative_rmse_slack"] + 1e-12)

        harmful = {
            "aaaaaaaa": BlendSufficient.from_paths("aaaaaaaa", [0.0] * 20, [0.1] * 20, [3.0] * 20),
            "bbbbbbbb": BlendSufficient.from_paths("bbbbbbbb", [0.0] * 20, [0.2] * 20, [4.0] * 20),
        }
        _, fallback, _ = _select_inner_weights(harmful, self.config)
        self.assertEqual(fallback, 0.0)

    def test_reliability_candidates_are_finite_bounded_and_deterministic(self) -> None:
        diagnostics = {
            "aaaaaaaa": FusionDiagnostic(0.2, 0.05, 0.001, 20.0, "", 0.0, 0.0),
            "bbbbbbbb": FusionDiagnostic(0.5, 0.2, 0.01, 8.0, "", 0.0, 0.0),
            "cccccccc": FusionDiagnostic(0.9, 0.5, 0.05, 1.0, "", 0.0, 0.0),
        }
        reference = _make_reference(diagnostics)
        for candidate in self.config["eligible_candidates"]:
            values = [
                _effective_weight(candidate, 0.75, detail, reference, well_id, self.config)
                for well_id, detail in diagnostics.items()
            ]
            repeated = [
                _effective_weight(candidate, 0.75, detail, reference, well_id, self.config)
                for well_id, detail in diagnostics.items()
            ]
            self.assertEqual(values, repeated)
            self.assertTrue(all(math.isfinite(value) and 0.0 <= value <= 0.75 for value in values))
        strong = _effective_weight("nested_reliability_shrink", 0.75, diagnostics["cccccccc"], reference, "cccccccc", self.config)
        weak = _effective_weight("nested_reliability_shrink", 0.75, diagnostics["aaaaaaaa"], reference, "aaaaaaaa", self.config)
        self.assertGreater(strong, weak)

    def test_reliability_equal_ties_extremes_and_nonfinite_values(self) -> None:
        tied = {
            well_id: FusionDiagnostic(0.5, 0.5, 0.1, 10.0, "", 0.0, 0.0)
            for well_id in ("aaaaaaaa", "bbbbbbbb", "cccccccc")
        }
        reference = _make_reference(tied)
        values = [
            _effective_weight("nested_reliability_shrink", 0.75, detail, reference, well_id, self.config)
            for well_id, detail in tied.items()
        ]
        self.assertTrue(all(math.isfinite(value) and 0.0 <= value <= 0.75 for value in values))
        extreme = FusionDiagnostic(1.0, 1.0, 1.0, 1e9, "", 0.0, 0.0)
        capped = _effective_weight("nested_disagreement_cap", 0.75, extreme, reference, "dddddddd", self.config)
        self.assertGreaterEqual(capped, 0.0)
        self.assertLess(capped, 0.75)
        with self.assertRaises(DataValidationError):
            _make_reference({})
        with self.assertRaises(DataValidationError):
            _make_reference({"aaaaaaaa": FusionDiagnostic(math.nan, 0.5, 0.1, 1.0, "", 0.0, 0.0)})
        with self.assertRaises(DataValidationError):
            _effective_weight("nested_conservative_grid", math.inf, tied["aaaaaaaa"], reference, "aaaaaaaa", self.config)

    def test_context_shuffle_is_deterministic_and_has_no_fixed_points(self) -> None:
        ids = [f"{index:08x}" for index in range(25)]
        first = _context_shuffle(ids, "test")
        second = _context_shuffle(list(reversed(ids)), "test")
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(ids))
        self.assertEqual(set(first.values()), set(ids))
        self.assertTrue(all(source != recipient for recipient, source in first.items()))

    def test_pf_all_missing_gr_abstains_exactly_and_rejects_nonfinite_base(self) -> None:
        curve = TypewellCurve(
            tvt=(0.0, 1.0, 2.0, 3.0),
            gr=(10.0, 11.0, 12.0, 13.0),
            minimum_tvt=0.0,
            maximum_tvt=3.0,
            gr_mean=11.5,
            gr_std=math.sqrt(1.25),
            digest="d" * 64,
        )
        well = WellData(
            well_id="aaaaaaaa",
            md=(0.0, 1.0, 2.0, 3.0),
            x=(0.0, 1.0, 2.0, 3.0),
            y=(0.0, 0.0, 0.0, 0.0),
            z=(0.0, 0.0, 0.0, 0.0),
            gr=(10.0, 11.0, 12.0, None),
            tvt_input=(0.0, 1.0, 2.0, None),
            truth=(0.0, 1.0, 2.0, 2.5),
            known_rows=3,
            typewell=curve,
        )
        e005 = json.loads((ROOT / "experiments/E005/config.json").read_text(encoding="utf-8"))
        base = [2.25]
        pf, detail = _pf_for_base(well, base, e005, with_margin=True)
        self.assertEqual(pf, base)
        self.assertEqual(detail.fallback_reason, "low_hidden_gr_coverage")
        with self.assertRaises(DataValidationError):
            _pf_for_base(well, [math.inf], e005, with_margin=False)
        with self.assertRaises(DataValidationError):
            _pf_for_base(well, [2.0, 3.0], e005, with_margin=False)

    def test_fit_context_outer_fold_is_absent_from_all_inner_predictions(self) -> None:
        ids = [f"{index:08x}" for index in range(10)]
        records = {well_id: {} for well_id in ids}
        assignments = {well_id: index % 5 for index, well_id in enumerate(ids)}

        def fake_fit(_records, train_ids, test_ids, _features, _config):
            self.assertFalse(set(train_ids) & set(test_ids))
            return {well_id: [0.0, 0.0, 0.0, 0.0] for well_id in test_ids}, {}

        with patch("rogii_validation.fusion._e004_fit_predict", side_effect=fake_fit):
            context, audit = _fit_context(
                "synthetic:v1:0",
                "repeated",
                "v1",
                0,
                assignments,
                assignments,
                records,
                [],
                {},
            )
        self.assertTrue(audit["pass"])
        self.assertFalse(set(context.test_ids) & set(context.train_ids))
        self.assertFalse(set(context.test_ids) & set(context.inner_coefficients))
        self.assertEqual(set(context.inner_coefficients), set(context.train_ids))
        self.assertEqual(context.inner_fold_count, 4)

    def test_template_resampling_handles_single_and_different_length_paths(self) -> None:
        single = _correction_template([5.0], [7.0])
        self.assertEqual(single, (2.0,))
        recipient = _template_path([10.0, 11.0, 12.0, 13.0, 14.0], (0.0, 3.0))
        self.assertEqual(recipient, [10.0, 11.75, 13.5, 15.25, 17.0])

    def test_nonfinite_fusion_inputs_are_rejected(self) -> None:
        with self.assertRaises(DataValidationError):
            BlendSufficient.from_paths("aaaaaaaa", [0.0], [math.nan], [0.0])
        with self.assertRaises(DataValidationError):
            BlendSufficient.from_paths("aaaaaaaa", [], [], [])
        valid = BlendSufficient.from_paths("aaaaaaaa", [0.0], [1.0], [2.0])
        with self.assertRaises(DataValidationError):
            valid.metric(math.inf)

    def test_parent_pf_control_requires_both_hash_audit_and_rmse_tolerance(self) -> None:
        base = {"pass": True, "pf_rmse": 15.0, "sha256": "a" * 64}
        self.assertTrue(_parent_pf_control(base, 15.0, 1e-9)["pass"])
        self.assertFalse(_parent_pf_control(base, 14.0, 1e-9)["pass"])
        self.assertFalse(_parent_pf_control({**base, "pass": False}, 15.0, 1e-9)["pass"])

    def test_runtime_or_memory_failure_clears_final_candidate_pass_flags(self) -> None:
        gates = {candidate: {"statistical": True, "runtime": True, "memory": True} for candidate in self.config["eligible_candidates"]}
        gates["nested_conservative_grid"]["runtime"] = False
        eligible = _eligible_from_gates(gates)
        self.assertNotIn("nested_conservative_grid", eligible)
        rows = [{"candidate": candidate, "passed_all_gates": True} for candidate in self.config["candidate_order"]]
        _mark_candidate_gate_rows(rows, eligible)
        lookup = {row["candidate"]: row["passed_all_gates"] for row in rows}
        self.assertFalse(lookup["nested_conservative_grid"])
        self.assertFalse(lookup["e004_geometry_prefix"])

    def test_deployment_model_validation_rejects_malformed_models(self) -> None:
        model = json.loads((ROOT / "experiments/E006/results/model.json").read_text(encoding="utf-8"))
        validate_e006_model(model)
        for mutate in (
            lambda item: item.update(schema_version=2),
            lambda item: item.update(fusion_weight=math.nan),
            lambda item: item["alignment"].update(datum_offsets_ft=[]),
            lambda item: item["particle_filter"].update(temperature=0.0),
            lambda item: item.update(e004_model={}),
            lambda item: item.update(model_name="unavailable_candidate"),
            lambda item: item.update(internet_required=True),
            lambda item: item["e004_model"].update(feature_scales=[]),
        ):
            broken = json.loads(json.dumps(model))
            mutate(broken)
            with self.assertRaises(DeploymentDataError):
                validate_e006_model(broken)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(DeploymentDataError):
                load_e006_model(path)

    def test_deployment_all_missing_gr_falls_back_exactly_to_e004(self) -> None:
        e005 = json.loads((ROOT / "experiments/E005/config.json").read_text(encoding="utf-8"))
        curve = E006TypewellCurve((0.0, 1.0, 2.0, 3.0), (10.0, 11.0, 12.0, 13.0), 0.0, 3.0, 11.5, math.sqrt(1.25))
        horizontal = {
            "known_rows": 3,
            "row_count": 5,
            "columns": {
                "GR": [10.0, 11.0, 12.0, None, None],
                "TVT_input": [0.0, 1.0, 2.0, None, None],
            },
        }
        base = [2.25, 2.5]
        predicted, detail = particle_path(horizontal, curve, base, e005["alignment"], e005["particle_filter"])
        self.assertEqual(predicted, base)
        self.assertEqual(detail["fallback_reason"], "low_hidden_gr_coverage")

    def test_deployment_particle_path_rejects_bad_base_contract(self) -> None:
        e005 = json.loads((ROOT / "experiments/E005/config.json").read_text(encoding="utf-8"))
        horizontal_path = ROOT / "data/train/000d7d20__horizontal_well.csv"
        typewell_path = ROOT / "data/train/000d7d20__typewell.csv"
        horizontal = read_horizontal(horizontal_path, require_truth=False)
        curve = E006TypewellCurve.read(typewell_path)
        hidden_rows = int(horizontal["row_count"]) - int(horizontal["known_rows"])
        with self.assertRaises(DeploymentDataError):
            particle_path(horizontal, curve, [1.0] * (hidden_rows - 1), e005["alignment"], e005["particle_filter"])
        bad = [1.0] * hidden_rows
        bad[-1] = math.inf
        with self.assertRaises(DeploymentDataError):
            particle_path(horizontal, curve, bad, e005["alignment"], e005["particle_filter"])

    def test_typewell_reader_averages_duplicates_and_rejects_degenerate_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.csv"
            duplicate.write_text("TVT,GR\n1,10\n1,14\n2,20\n", encoding="utf-8")
            curve = E006TypewellCurve.read(duplicate)
            self.assertEqual(curve.tvt, (1.0, 2.0))
            self.assertEqual(curve.gr, (12.0, 20.0))
            missing = root / "missing.csv"
            missing.write_text("TVT,Other\n1,10\n2,20\n", encoding="utf-8")
            with self.assertRaises(DeploymentDataError):
                E006TypewellCurve.read(missing)
            degenerate = root / "degenerate.csv"
            degenerate.write_text("TVT,GR\n1,10\n1,12\n", encoding="utf-8")
            with self.assertRaises(DeploymentDataError):
                E006TypewellCurve.read(degenerate)

    def test_submission_contract_rejects_bad_ids_and_serializes_deterministically(self) -> None:
        predictions = {"aaaaaaaa_3": 1.23456789, "aaaaaaaa_4": -2.0}
        wells = [{"well_id": "aaaaaaaa"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "submission.csv"
            with patch("rogii_validation.e006_inference.build_e006_prediction_map", return_value=(predictions, wells)):
                valid = root / "valid.csv"
                valid.write_text("id,tvt\naaaaaaaa_3,0\naaaaaaaa_4,0\n", encoding="utf-8")
                first = write_e006_submission({}, root, valid, output)
                fixed = output.read_bytes()
                second = write_e006_submission({}, root, valid, output)
                self.assertEqual(fixed, output.read_bytes())
                self.assertEqual(first["sha256"], second["sha256"])
                self.assertIn(b"1.23457", fixed)
                cases = {
                    "duplicate": "id,tvt\naaaaaaaa_3,0\naaaaaaaa_3,0\n",
                    "missing": "id,tvt\naaaaaaaa_3,0\n",
                    "extra": "id,tvt\naaaaaaaa_3,0\naaaaaaaa_4,0\naaaaaaaa_5,0\n",
                    "empty": "id,tvt\n,0\naaaaaaaa_4,0\n",
                    "wrong_header": "tvt,id\n0,aaaaaaaa_3\n0,aaaaaaaa_4\n",
                }
                for name, text in cases.items():
                    sample = root / f"{name}.csv"
                    sample.write_text(text, encoding="utf-8")
                    with self.assertRaises(DeploymentDataError, msg=name):
                        write_e006_submission({}, root, sample, output)

    def test_prediction_map_rejects_empty_test_directory_and_malformed_model(self) -> None:
        model = json.loads((ROOT / "experiments/E006/results/model.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory)
            with self.assertRaises(DeploymentDataError):
                build_e006_prediction_map(model, empty)
        broken = json.loads(json.dumps(model))
        broken["fusion_weight"] = 2.0
        with self.assertRaises(DeploymentDataError):
            build_e006_prediction_map(broken, ROOT / "data/test")

    def test_deployment_particle_path_matches_frozen_training_implementation(self) -> None:
        well_id = "000d7d20"
        base = []
        with gzip.open(ROOT / "artifacts/E005/oof_predictions.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["well_id"] == well_id:
                    base.append(float(row["e004_geometry_prefix"]))
                elif base:
                    break
        e005 = json.loads((ROOT / "experiments/E005/config.json").read_text(encoding="utf-8"))
        horizontal_path = ROOT / f"data/train/{well_id}__horizontal_well.csv"
        typewell_path = ROOT / f"data/train/{well_id}__typewell.csv"
        well = read_well(horizontal_path, typewell_path, require_truth=True)
        calibration = _calibration(well, well.typewell, int(e005["alignment"]["minimum_visible_calibration_samples"]))
        self.assertIsNotNone(calibration)
        samples = _hidden_samples(well, int(e005["alignment"]["maximum_gr_samples_per_well"]))
        expected, expected_detail = _particle_path(
            well,
            base,
            well.typewell,
            samples,
            calibration,
            e005["alignment"],
            e005["particle_filter"],
        )
        deployed, deployed_detail = particle_path(
            read_horizontal(horizontal_path, require_truth=False),
            E006TypewellCurve.read(typewell_path),
            base,
            e005["alignment"],
            e005["particle_filter"],
        )
        self.assertEqual(len(deployed), len(expected))
        self.assertLess(max(abs(left - right) for left, right in zip(deployed, expected)), 1e-12)
        self.assertAlmostEqual(deployed_detail["effective_fraction"], expected_detail["effective_fraction"], places=12)
        self.assertAlmostEqual(deployed_detail["pf_datum"], expected_detail["datum"], places=12)
        self.assertAlmostEqual(deployed_detail["pf_toe"], expected_detail["toe"], places=12)


if __name__ == "__main__":
    unittest.main()
