# T034 Result — Legal residual-action identifiability and privileged-geology mediation gate

Decision: **STOP_CLOSE_CURRENT_LEGAL_ACTION_IDENTIFICATION**

Best legal branch: `summary_ridge`
- TVT RMSE: 13.007269287161
- Gain versus E011: -0.456512991472
- Oracle-gain retention: -4.599342%
- Multivariate coefficient R2: 0.219148
- Median coefficient Pearson: 0.475349
- Positive wells: 45.148771%

Best legal-neighbor enrichment: `raw_binned`, k=5
- Action-distance reduction: 6.032264%

Privileged geology diagnostic:
- TVT RMSE: 11.316958496961
- Multivariate coefficient R2: 0.467597

## Gates
- FAIL — maximum_best_legal_tvt_rmse
- FAIL — minimum_oracle_gain_retention
- FAIL — minimum_multivariate_coefficient_r2
- FAIL — minimum_median_coefficient_pearson
- FAIL — minimum_nearest_neighbor_distance_reduction_fraction
- FAIL — positive_every_legacy_group
- PASS — negative_controls_below_gain_cap

T034 is an identifiability diagnostic. The privileged branch is nondeployable, and no package, Kaggle run, or submission is authorized automatically.
