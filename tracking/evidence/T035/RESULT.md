# T035 Result — Actual rank-8 PCA coordinate identifiability

Decision: **STOP_CLOSE_COMPACT_MANIFOLD_IDENTIFICATION**

Best legal branch: `raw_pca_ridge`
- TVT RMSE: 12.267293806947
- Gain versus E011: 0.283462488742
- Rank-8 oracle gain retention: 2.728375%
- Coordinate R2: 0.034133
- Median coordinate Pearson: 0.478248
- Positive wells: 61.707633%

Privileged geology diagnostic:
- TVT RMSE: 10.897200140737
- Coordinate R2: 0.251894

## Gates
- FAIL — maximum_best_legal_tvt_rmse
- FAIL — minimum_oracle_gain_retention
- FAIL — minimum_multivariate_coordinate_r2
- FAIL — minimum_median_coordinate_pearson
- FAIL — minimum_positive_well_fraction
- PASS — positive_every_legacy_group
- PASS — negative_controls_below_gain_cap

T035 is the final bounded manifold-identifiability gate. No deployment, Kaggle run, or submission is automatic.
