# E012 Pre-registration — Regime-specialized coefficient experts

Frozen at 2026-07-23 before any nested E012 score was computed.

## Question

Can a legally routed, sufficiently supported specialist improve the verified E011 `spline4_ridge_equal_s075` coefficient path while unsupported wells return the exact E011 coefficients?

## Evidence before implementation

Two score-separated worth screens were completed before writing the formal experiment. The first tested 109 hard-regime combinations across suffix length, GR missingness, pseudo-holdout quality, self-correlation, automatically selected legal features, legal-feature K-means, supervised tree leaves, regime means, and regime-local ridge experts. The second tested 117 global residual, ExtraTrees, KNN, latent-residual routing, and partial-pooling combinations. None produced positive pooled gain over reconstructed E011. The best deviations were small and negative, so broad specialist development is not justified.

E012 is a bounded nested confirmation covering one representative from each distinct mechanism rather than another open search.

## Frozen branches

All branches predict a bounded correction to the split-local E011 coefficient vector and use fixed shrinkage 0.25 unless marked as a negative control:

1. `length3_mean_s025` — three split-local hidden-row quantile regimes, regime-mean residual.
2. `pseudo3_mean_s025` — three split-local `backtest_0p85_rmse` regimes, regime-mean residual.
3. `kmeans2_mean_s025` — two legal-feature K-means regimes, regime-mean residual.
4. `latent3_s025` — three residual-coefficient clusters on training wells, routed by a legal-feature ExtraTrees classifier.
5. `global_ridge_a10_s025` — global residual ridge control.
6. `extratrees_l20_d2_s025` — shallow nonlinear residual model.
7. `knn25_distance_s025` — local similarity residual model.
8. `partial_pseudo_a10_s025` — global residual ridge plus pseudo-regime mean remainder.
9. `shuffled_length_mean_s100` — deterministic shuffled-routing negative control.

The minimum training support is 50 wells. Unsupported regimes emit zero residual correction, which is exact E011 fallback.

## Validation

For every one of the 25 immutable repeated map/cell contexts, E011 ridge feature/alpha selection and train residuals are reconstructed inside the context. No held-out target enters model fitting, regime construction, routing, support checks, shrinkage, or fallback. Every branch is evaluated on all repeated contexts. Every non-control branch is also evaluated on all five spatial and five typewell holdouts.

Final OOF coefficients are the average of the five repeated-map predictions, matching E011 placement. Report pooled and well-tail metrics, map and cell wins, every spatial/typewell gain, long-suffix, high-GR-missingness, and E011-catastrophe slices.

## Promotion gates

A branch must satisfy all of the following:

- pooled gain versus E011 at least 0.05 RMSE;
- at least 4/5 map wins and 17/25 outer-cell wins;
- no p90 deterioration;
- worst-5% SSE-share increase at most 0.002;
- positive gain in every spatial and typewell holdout;
- positive gain on long suffix, high GR missingness, and E011-catastrophe slices;
- minimum split-local support at least 50 for every active regime;
- exact E011 fallback for unsupported regimes;
- deterministic rerun and finite/bounded coefficients;
- shuffled-routing gain no greater than 0.01 RMSE.

If no branch passes, reject H015 and retain E011 unchanged. No Kaggle notebook or submission is authorized for a rejected experiment.

## Edge cases

Tests cover constant/tied regime features, all-missing features after train-only imputation, one surviving regime, insufficient regime support, empty train/test partitions, non-finite residuals, collapsed latent clusters, deterministic K-means/classifier routing, coefficient bounds, exact fallback, unknown branch families, and duplicated well IDs.
