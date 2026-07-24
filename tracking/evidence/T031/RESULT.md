# T031 Result — Raw-sequence latent formation-displacement identifiability gate

Decision: **HARD_STOP_CLOSE_COMPACT_RAW_SEQUENCE_STATE_RECOVERY**

Selected legal raw-sequence branch: `raw_binned_ridge`
- Pearson: 0.261732
- Spearman: 0.181220
- MAE: 15.125630 ft
- RMSE: 20.306342 ft
- Legacy group Pearson floor: 0.101133
- New target-free group Pearson floor: 0.152300

## Comparators

- Summary ridge Pearson/MAE: 0.431034 / 13.685349 ft
- Coordinate-only Pearson/MAE: 0.219849 / 14.639195 ft
- Permuted-target Pearson: -0.036954

## Gate outcomes

- FAIL — maximum_mae_ft
- PASS — maximum_permuted_abs_pearson
- FAIL — minimum_control_gain_destruction_fraction
- FAIL — minimum_every_legacy_group_pearson
- FAIL — minimum_every_new_group_pearson
- FAIL — minimum_mae_advantage_over_summary_ridge_ft
- FAIL — minimum_pearson
- FAIL — minimum_pearson_advantage_over_coordinate_control
- FAIL — minimum_pearson_advantage_over_summary_ridge
- FAIL — minimum_spearman

The best raw sequence branch is materially worse than the frozen 142-summary ridge and falls below the preregistered hard-stop Pearson threshold. Compact raw-sequence latent-state recovery is closed; no T033 joint TVT model is authorized.

This task is an auxiliary-state gate only. It does not emit a competition prediction.
