from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.coefficient_learning import Context
from rogii_validation.harness import DataValidationError
from rogii_validation.regime_specialists import (
    BranchSpec,
    _feature_arrays,
    _regime_mean_delta,
    branch_specs,
    fit_branch,
    quantile_regimes,
    validate_e012_config,
)

ROOT = Path(__file__).resolve().parents[1]


class RegimeSpecialistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "experiments/E012/config.json").read_text(encoding="utf-8"))
        cls.e011 = json.loads((ROOT / "experiments/E011/config.json").read_text(encoding="utf-8"))

    def test_config_and_branch_contract(self) -> None:
        validate_e012_config(self.config)
        specs = branch_specs(self.config)
        self.assertEqual(len(specs), 9)
        self.assertEqual(sum(spec.negative_control for spec in specs), 1)
        bad = dict(self.config)
        bad["minimum_regime_support"] = 10
        with self.assertRaises(DataValidationError):
            validate_e012_config(bad)
        bad = dict(self.config)
        bad["submission_authorized"] = True
        with self.assertRaises(DataValidationError):
            validate_e012_config(bad)

    def test_quantile_regimes_tied_values_are_deterministic(self) -> None:
        train = np.ones(9)
        test = np.ones(3)
        first = quantile_regimes(train, test)
        second = quantile_regimes(train, test)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(set(first[0].tolist())), 1)
        with self.assertRaises(DataValidationError):
            quantile_regimes(np.asarray([1.0, np.nan]), np.asarray([1.0]))

    def test_regime_mean_support_and_exact_fallback(self) -> None:
        residual = np.arange(24, dtype=float).reshape(6, 4)
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        test_labels = np.asarray([0, 1, 2])
        delta, fallback, minimum, regimes = _regime_mean_delta(residual, labels, test_labels, 3)
        np.testing.assert_allclose(delta[0], residual[:3].mean(axis=0))
        np.testing.assert_allclose(delta[1], residual[3:].mean(axis=0))
        np.testing.assert_allclose(delta[2], 0.0)
        np.testing.assert_array_equal(fallback, [False, False, True])
        self.assertEqual(minimum, 3)
        self.assertEqual(regimes, 2)
        delta, fallback, _, _ = _regime_mean_delta(residual, labels, test_labels, 4)
        np.testing.assert_allclose(delta, 0.0)
        self.assertTrue(np.all(fallback))
        with self.assertRaises(DataValidationError):
            _regime_mean_delta(np.asarray([[np.nan] * 4]), np.asarray([0]), np.asarray([0]), 1)

    def test_feature_arrays_impute_all_missing_and_reject_bad_membership(self) -> None:
        records = {
            "a": SimpleNamespace(features={"x": None, "y": 1.0}),
            "b": SimpleNamespace(features={"x": None, "y": 2.0}),
            "c": SimpleNamespace(features={"x": None, "y": None}),
        }
        train, test, train_z, test_z = _feature_arrays(records, ("a", "b"), ("c",), ("x", "y"))
        self.assertTrue(np.all(np.isfinite(train)))
        self.assertTrue(np.all(np.isfinite(test)))
        self.assertTrue(np.all(np.isfinite(train_z)))
        self.assertTrue(np.all(np.isfinite(test_z)))
        with self.assertRaises(DataValidationError):
            _feature_arrays(records, ("a", "a"), ("c",), ("x", "y"))
        with self.assertRaises(DataValidationError):
            _feature_arrays(records, ("a", "b"), ("b",), ("x", "y"))
        with self.assertRaises(DataValidationError):
            _feature_arrays(records, (), ("c",), ("x", "y"))

    def _fit_fixture(self, family: str, *, train_count: int = 60, residual_mode: str = "varying"):
        train_ids = tuple(f"t{i:03d}" for i in range(train_count))
        test_ids = ("u000", "u001", "u002")
        context = Context("repeated:v1:0", "repeated", "v1", 0, train_ids, test_ids, {})
        names = ("hidden_rows", "backtest_0p85_rmse")
        train_raw = np.column_stack((np.arange(train_count), np.linspace(0.0, 1.0, train_count)))
        test_raw = np.asarray([[3.0, 0.1], [30.0, 0.5], [59.0, 0.9]])
        means = train_raw.mean(axis=0)
        scales = train_raw.std(axis=0)
        train_z = (train_raw - means) / scales
        test_z = (test_raw - means) / scales
        if residual_mode == "constant":
            residual = np.ones((train_count, 4))
        else:
            residual = np.column_stack(
                (
                    np.linspace(-2.0, 2.0, train_count),
                    np.sin(np.linspace(0.0, 2.0, train_count)),
                    np.cos(np.linspace(0.0, 2.0, train_count)),
                    np.linspace(1.0, -1.0, train_count),
                )
            )
        base = np.zeros((len(test_ids), 4))
        cfg = dict(self.config)
        cfg["e011_config"] = self.e011
        spec = BranchSpec("candidate", family, 0.25, True)
        return fit_branch(
            spec,
            context=context,
            records={},
            feature_names=names,
            train_raw=train_raw,
            test_raw=test_raw,
            train_z=train_z,
            test_z=test_z,
            train_residual=residual,
            test_base=base,
            config=cfg,
        )

    def test_insufficient_support_and_knn_return_exact_parent(self) -> None:
        branch, detail = self._fit_fixture("length3_mean", train_count=30)
        self.assertEqual(detail["fallback_wells"], 3)
        for values in branch.values():
            np.testing.assert_allclose(values, 0.0)
        branch, detail = self._fit_fixture("knn", train_count=20)
        self.assertEqual(detail["fallback_wells"], 3)
        for values in branch.values():
            np.testing.assert_allclose(values, 0.0)

    def test_collapsed_latent_regimes_fallback_exactly(self) -> None:
        branch, detail = self._fit_fixture("latent3", residual_mode="constant")
        self.assertEqual(detail["fallback_wells"], 3)
        self.assertLess(detail["regimes"], 3)
        for values in branch.values():
            np.testing.assert_allclose(values, 0.0)

    def test_deterministic_legal_routing_and_bounds(self) -> None:
        first, detail_first = self._fit_fixture("kmeans2_mean")
        second, detail_second = self._fit_fixture("kmeans2_mean")
        self.assertEqual(detail_first, detail_second)
        for key in first:
            np.testing.assert_allclose(first[key], second[key], atol=0.0, rtol=0.0)
            self.assertLessEqual(float(np.max(np.abs(first[key]))), 80.0 + 1e-12)
        ridge_first, _ = self._fit_fixture("global_ridge")
        ridge_second, _ = self._fit_fixture("global_ridge")
        for key in ridge_first:
            np.testing.assert_allclose(ridge_first[key], ridge_second[key], atol=0.0, rtol=0.0)

    def test_invalid_family_and_nonfinite_residual_fail(self) -> None:
        with self.assertRaises(DataValidationError):
            self._fit_fixture("unknown")
        train_ids = tuple(f"t{i:03d}" for i in range(60))
        context = Context("x", "repeated", "v1", 0, train_ids, ("u",), {})
        cfg = dict(self.config)
        cfg["e011_config"] = self.e011
        with self.assertRaises(DataValidationError):
            fit_branch(
                BranchSpec("bad", "global_ridge", 0.25, True),
                context=context,
                records={},
                feature_names=("hidden_rows", "backtest_0p85_rmse"),
                train_raw=np.ones((60, 2)),
                test_raw=np.ones((1, 2)),
                train_z=np.ones((60, 2)),
                test_z=np.ones((1, 2)),
                train_residual=np.full((60, 4), np.nan),
                test_base=np.zeros((1, 4)),
                config=cfg,
            )


if __name__ == "__main__":
    unittest.main()
