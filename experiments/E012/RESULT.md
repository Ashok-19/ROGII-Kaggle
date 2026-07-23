# E012 Result — Regime-Specialized Coefficient Experts

Status: **rejected; retain E011; no deployment package or Kaggle submission**  
Hypothesis: `H015`  
Task: `T021`  
Frozen source: `200fa40c8bdee8cebbd5a6514950e20ec0a1f81e`

## Decision

Reject regime-specialized coefficient experts as an addition to E011. Keep `spline4_ridge_equal_s075` unchanged as the deployment primary. E006 remains the secondary fallback and E004 the exact fallback.

The rejection is not based on a single failed model. Two preimplementation worth screens completed **226 branches** across hard legal regimes, regime-local means and ridges, unsupervised feature clusters, supervised tree leaves, global residual models, ExtraTrees, KNN, latent residual clusters with legal routing, and partial pooling. None improved pooled OOF RMSE in those screens. The formal nested confirmation then completed nine representative branches over all 25 repeated contexts and all ten spatial/typewell holdouts.

## Formal nested result

The strongest branch was `knn25_distance_s025`:

| Metric | E011 | KNN specialist | Change / gain |
|---|---:|---:|---:|
| Pooled RMSE | 12.550756295700 | 12.537058045100 | **0.013698250593 gain** |
| p90 well RMSE | 17.831024 | 17.801191 | 0.029833 improvement |
| Worst-5% SSE share | 0.315322 | 0.314468 | 0.000854 improvement |
| Repeated map wins | — | 5/5 | passes map direction only |
| Outer-cell wins | — | 15/25 | fails 17/25 gate |

The gain is only **0.013698 RMSE**, below the frozen 0.05 minimum. It also fails complete transfer:

- spatial group 0 gain: -0.018120
- spatial group 1 gain: -0.010678
- typewell group 0 gain: -0.015565

The same branch improves all three registered slices:

- long suffix: 0.027302
- high GR missingness: 0.035205
- E011-catastrophe wells: 0.049475

This is useful diagnostic evidence that local legal-feature similarity contains a very small residual signal, but it is not a stable system improvement and does not justify additional KNN or regime-threshold tuning.

Every other eligible formal branch is worse than E011 in pooled RMSE. The latent three-regime branch has minimum split-local support 69, above the frozen support floor, yet still degrades pooled and transfer metrics. Adequate support alone therefore does not rescue specialization.

## Controls and edge cases

All registered controls pass:

- exact E011 reconstruction: 12.550756295690 RMSE;
- all 25 repeated and 10 stress contexts completed;
- all nine branches completed with full OOF coverage over 773 wells;
- finite and bounded coefficients;
- exact zero-delta E011 fallback for unsupported regimes;
- deterministic duplicate run with maximum coefficient delta 0;
- shuffled routing loses 0.006243 RMSE versus E011;
- no submission was authorized or made.

Focused tests cover tied and constant regimes, all-missing feature imputation, one surviving regime, insufficient support, empty or overlapping membership, non-finite residuals, collapsed latent clusters, deterministic K-means and ridge behavior, coefficient bounds, exact fallback, unknown families, and duplicate IDs.

## Independent reproduction

The official run completed in 123.508 seconds. An independent run from the same frozen source completed in 116.444 seconds. `candidate_metrics.csv`, `context_metrics.csv`, `stress_metrics.csv`, `support_metrics.csv`, `special_slice_metrics.csv`, and `controls.csv` are byte-identical; the normalized summaries are equal after removing measured runtime.

## Conclusion

H015 is closed as rejected. Do not continue broad regime definitions, KNN hyperparameter tuning, latent-cluster routing, or partial-pooling variants around E011. Preserve the tiny KNN slice signal only as a diagnostic lead. The next modeling experiment should require a new mechanism—such as branch uncertainty or physical continuity—not another partition of the same 142-feature coefficient model.
