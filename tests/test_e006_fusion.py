import csv
import gzip
import json
import math
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
    _select_inner_weights,
    _template_path,
    validate_e006_config,
)
from rogii_validation.e004_inference import read_horizontal
from rogii_validation.e006_inference import E006TypewellCurve, particle_path
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
