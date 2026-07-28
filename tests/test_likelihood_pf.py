from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rogii_validation.likelihood_pf import LikelihoodPFConfig, predict_likelihood_pf


class LikelihoodPFTests(unittest.TestCase):
    @staticmethod
    def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        n_visible = 20
        n_hidden = 15
        n = n_visible + n_hidden
        md = np.arange(1000.0, 1000.0 + n)
        z = -8000.0 + 0.2 * np.arange(n)
        true_tvt = 10000.0 + 0.05 * np.arange(n)
        tvt_input = true_tvt.copy()
        tvt_input[n_visible:] = np.nan
        gr = 80.0 + 12.0 * np.sin((true_tvt - true_tvt.min()) / 1.5)
        horizontal = pd.DataFrame({"MD": md, "Z": z, "GR": gr, "TVT_input": tvt_input})
        tw_tvt = np.linspace(9980.0, 10030.0, 501)
        typewell = pd.DataFrame({"TVT": tw_tvt, "GR": 80.0 + 12.0 * np.sin((tw_tvt - true_tvt.min()) / 1.5)})
        return horizontal, typewell

    @staticmethod
    def config() -> LikelihoodPFConfig:
        return LikelihoodPFConfig(n_particles=32, seed_ids=(0, 1, 2, 3))

    def test_deterministic_and_visible_prefix_exact(self) -> None:
        horizontal, typewell = self.frames()
        first = predict_likelihood_pf(horizontal, typewell, self.config())
        second = predict_likelihood_pf(horizontal, typewell, self.config())
        visible = horizontal["TVT_input"].notna().to_numpy()
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first[visible], horizontal.loc[visible, "TVT_input"].to_numpy(float)))
        self.assertTrue(np.isfinite(first).all())

    def test_seed_order_invariance(self) -> None:
        horizontal, typewell = self.frames()
        config = self.config()
        forward = predict_likelihood_pf(horizontal, typewell, config)
        reverse = predict_likelihood_pf(horizontal, typewell, config, seed_ids=reversed(config.seed_ids))
        self.assertLessEqual(float(np.max(np.abs(forward - reverse))), 1e-9)

    def test_missing_hidden_gr_is_finite(self) -> None:
        horizontal, typewell = self.frames()
        hidden = horizontal["TVT_input"].isna()
        horizontal.loc[hidden, "GR"] = np.nan
        prediction = predict_likelihood_pf(horizontal, typewell, self.config())
        self.assertTrue(np.isfinite(prediction).all())

    def test_no_emission_is_finite(self) -> None:
        horizontal, typewell = self.frames()
        prediction = predict_likelihood_pf(horizontal, typewell, self.config(), use_emission=False)
        self.assertTrue(np.isfinite(prediction).all())

    def test_rejects_noncontiguous_visible_prefix(self) -> None:
        horizontal, typewell = self.frames()
        horizontal.loc[len(horizontal) - 1, "TVT_input"] = 10001.0
        with self.assertRaisesRegex(ValueError, "contiguous visible prefix"):
            predict_likelihood_pf(horizontal, typewell, self.config())


if __name__ == "__main__":
    unittest.main()
