from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

import numpy as np

from rogii_validation.e010_paths import (
    basis_matrix,
    candidate_count,
    coefficient_grid,
    dtw_path,
    grid_oracle,
    grid_values,
    hmm_path,
    reconstruct_path,
    valid_coefficients,
)
from rogii_validation.gr_path import TypewellCurve, WellData
from rogii_validation.harness import DataValidationError
from rogii_validation.nonlinear_selector import validate_e010_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments/E010/config.json").read_text(encoding="utf-8"))


def synthetic_well(*, hidden_rows: int = 60, missing_hidden: bool = False, flat_reference: bool = False) -> tuple[WellData, np.ndarray]:
    known = 40
    total = known + hidden_rows
    md = np.arange(total, dtype=float)
    tvt_full = 120.0 + 0.35 * md + 2.5 * np.sin(md / 17.0)
    ref_tvt = np.linspace(80.0, 240.0, 801)
    ref_gr = np.full_like(ref_tvt, 50.0) if flat_reference else 50.0 + 18.0 * np.sin(ref_tvt / 5.7) + 7.0 * np.sin(ref_tvt / 13.1)
    curve = TypewellCurve(
        tvt=tuple(float(value) for value in ref_tvt),
        gr=tuple(float(value) for value in ref_gr),
        minimum_tvt=float(ref_tvt[0]),
        maximum_tvt=float(ref_tvt[-1]),
        gr_mean=float(ref_gr.mean()),
        gr_std=float(ref_gr.std()),
        digest="synthetic",
    )
    gr = np.interp(tvt_full, ref_tvt, ref_gr)
    if missing_hidden:
        gr[known:] = np.nan
    well = WellData(
        well_id="synthetic",
        md=tuple(float(value) for value in md),
        x=tuple(float(value) for value in md),
        y=tuple(0.0 for _ in md),
        z=tuple(0.0 for _ in md),
        gr=tuple(None if not math.isfinite(value) else float(value) for value in gr),
        tvt_input=tuple(float(tvt_full[index]) if index < known else None for index in range(total)),
        truth=tuple(float(value) for value in tvt_full),
        known_rows=known,
        typewell=curve,
    )
    return well, tvt_full[known:].copy()


class E010BasisEdgeTests(unittest.TestCase):
    def test_config_identity_and_frozen_branch_counts(self):
        validate_e010_config(CONFIG)
        self.assertEqual(len(CONFIG["candidate_bank"]["families"]), 11)
        self.assertEqual(len(CONFIG["development_screen"]["hmm_variants"]), 4)
        self.assertEqual(len(CONFIG["development_screen"]["dtw_variants"]), 4)
        self.assertEqual(len(CONFIG["selector"]["branches"]), 6)

    def test_changed_selector_order_is_rejected(self):
        bad = copy.deepcopy(CONFIG)
        bad["selector"]["branches"] = list(reversed(bad["selector"]["branches"]))
        with self.assertRaises(DataValidationError):
            validate_e010_config(bad)

    def test_grid_endpoints_and_bad_step(self):
        np.testing.assert_allclose(
            grid_values({"minimum": -5, "maximum": 5, "step": 2.5}),
            [-5, -2.5, 0, 2.5, 5],
        )
        with self.assertRaises(DataValidationError):
            grid_values({"minimum": 0, "maximum": 1, "step": 0})

    def test_one_row_basis_is_finite_and_centered(self):
        for family in CONFIG["candidate_bank"]["families"]:
            matrix = basis_matrix(1, family)
            self.assertTrue(np.all(np.isfinite(matrix)))
            self.assertEqual(matrix.shape[0], 1)
            self.assertEqual(float(matrix[0, 1]), 0.0)
            if matrix.shape[1] == 3:
                self.assertAlmostEqual(float(matrix[:, 2].mean()), 0.0)

    def test_nonlinear_bases_are_mean_centered(self):
        for family in CONFIG["candidate_bank"]["families"]:
            matrix = basis_matrix(101, family)
            if matrix.shape[1] == 3:
                self.assertAlmostEqual(float(matrix[:, 2].mean()), 0.0, places=12)

    def test_candidate_count_survives_bound_filter(self):
        total, counts = candidate_count(CONFIG)
        self.assertEqual(total, 130252)
        self.assertGreaterEqual(total, CONFIG["controls"]["minimum_candidate_count"])
        self.assertTrue(all(value > 0 for value in counts.values()))

    def test_every_sampled_valid_path_respects_global_bound(self):
        maximum = CONFIG["candidate_bank"]["maximum_absolute_correction_ft"]
        for family in CONFIG["candidate_bank"]["families"]:
            coefficients = valid_coefficients(family, CONFIG)
            indices = np.linspace(0, len(coefficients) - 1, min(25, len(coefficients))).astype(int)
            matrix = basis_matrix(1001, family)
            self.assertLessEqual(float(np.max(np.abs(matrix @ coefficients[indices].T))), maximum + 1e-9)

    def test_zero_coefficients_reconstruct_anchor_exactly(self):
        anchor = np.linspace(10.0, 20.0, 17)
        for family in CONFIG["candidate_bank"]["families"]:
            zero = np.zeros(basis_matrix(len(anchor), family).shape[1])
            np.testing.assert_array_equal(reconstruct_path(anchor, family, zero, 160.0), anchor)

    def test_nonfinite_coefficients_are_rejected(self):
        family = CONFIG["candidate_bank"]["families"][1]
        with self.assertRaises(DataValidationError):
            reconstruct_path([1.0, 2.0], family, [0.0, 0.0, math.nan], 160.0)

    def test_grid_oracle_recovers_known_quadratic_candidate(self):
        family = CONFIG["candidate_bank"]["families"][1]
        anchor = np.linspace(100.0, 105.0, 120)
        chosen = np.asarray([10.0, -20.0, 30.0])
        truth = anchor + basis_matrix(len(anchor), family) @ chosen
        result = grid_oracle(anchor, truth, family, valid_coefficients(family, CONFIG))
        self.assertLess(result.rmse, 1e-6)
        np.testing.assert_allclose(result.coefficients, chosen)

    def test_equal_loss_tie_is_deterministic(self):
        family = CONFIG["candidate_bank"]["families"][0]
        coefficients = np.asarray([[0.0, -10.0], [0.0, 10.0]], dtype=float)
        first = grid_oracle([1.0], [1.0], family, coefficients)
        second = grid_oracle([1.0], [1.0], family, coefficients)
        self.assertEqual(first.candidate_index, 0)
        self.assertEqual(first, second)

    def test_grid_shape_matches_basis_width(self):
        for family in CONFIG["candidate_bank"]["families"]:
            grid = coefficient_grid(family, CONFIG)
            self.assertEqual(grid.shape[1], basis_matrix(5, family).shape[1])


class E010StatePathEdgeTests(unittest.TestCase):
    @staticmethod
    def hmm_variant():
        variant = copy.deepcopy(CONFIG["development_screen"]["hmm_variants"][1])
        variant.update({"grid_step_ft": 1.0, "position_span_ft": 20.0, "rate_states": 5, "maximum_observations": 48, "emission_weight": 0.25})
        return variant

    @staticmethod
    def dtw_variant():
        variant = copy.deepcopy(CONFIG["development_screen"]["dtw_variants"][1])
        variant.update({"offset_min_ft": -20.0, "offset_max_ft": 20.0, "offset_step_ft": 1.0, "maximum_observations": 60})
        return variant

    def test_hmm_is_finite_deterministic_bounded_and_useful(self):
        well, truth = synthetic_well()
        base = truth + 8.0
        first = hmm_path(well, base, self.hmm_variant(), 160.0)
        second = hmm_path(well, base, self.hmm_variant(), 160.0)
        self.assertEqual(first.fallback_reason, "")
        self.assertTrue(np.all(np.isfinite(first.path)))
        self.assertLessEqual(float(np.max(np.abs(first.path - base))), 160.0)
        np.testing.assert_allclose(first.path, second.path, rtol=0, atol=1e-12)
        self.assertLess(np.sqrt(np.mean((first.path - truth) ** 2)), np.sqrt(np.mean((base - truth) ** 2)))

    def test_hmm_missing_hidden_gr_returns_exact_fallback(self):
        well, truth = synthetic_well(missing_hidden=True)
        base = truth + 4.0
        result = hmm_path(well, base, self.hmm_variant(), 160.0)
        self.assertTrue(result.fallback_reason)
        np.testing.assert_array_equal(result.path, base)

    def test_hmm_flat_reference_returns_exact_fallback(self):
        well, truth = synthetic_well(flat_reference=True)
        base = truth + 4.0
        result = hmm_path(well, base, self.hmm_variant(), 160.0)
        self.assertEqual(result.fallback_reason, "invalid_visible_calibration")
        np.testing.assert_array_equal(result.path, base)

    def test_hmm_degenerate_rate_grid_is_rejected(self):
        well, truth = synthetic_well()
        variant = self.hmm_variant()
        variant["rate_states"] = 1
        with self.assertRaises(DataValidationError):
            hmm_path(well, truth, variant, 160.0)

    def test_dtw_is_finite_deterministic_bounded_and_useful(self):
        well, truth = synthetic_well()
        base = truth + 8.0
        first = dtw_path(well, base, self.dtw_variant(), 160.0)
        second = dtw_path(well, base, self.dtw_variant(), 160.0)
        self.assertEqual(first.fallback_reason, "")
        self.assertTrue(np.all(np.isfinite(first.path)))
        self.assertLessEqual(float(np.max(np.abs(first.path - base))), 160.0)
        np.testing.assert_array_equal(first.path, second.path)
        self.assertLess(np.sqrt(np.mean((first.path - truth) ** 2)), np.sqrt(np.mean((base - truth) ** 2)))

    def test_dtw_missing_hidden_gr_returns_exact_fallback(self):
        well, truth = synthetic_well(missing_hidden=True)
        base = truth + 4.0
        result = dtw_path(well, base, self.dtw_variant(), 160.0)
        self.assertEqual(result.fallback_reason, "all_hidden_gr_missing")
        np.testing.assert_array_equal(result.path, base)

    def test_dtw_bad_state_step_is_rejected(self):
        well, truth = synthetic_well()
        variant = self.dtw_variant()
        variant["offset_step_ft"] = 0.0
        with self.assertRaises(DataValidationError):
            dtw_path(well, truth, variant, 160.0)


if __name__ == "__main__":
    unittest.main()
