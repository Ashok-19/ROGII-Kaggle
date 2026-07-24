# T032 Result — Registered analog-tail retrieval oracle-coverage gate

Decision: **CLOSE_ANALOG_TAIL_RETRIEVAL**

Best configuration: `k09_g0p50_t0p50_b0p75`
- Oracle RMSE: 8.674114
- Gain versus E011: 3.876642
- Shape-SSE reduction: 14.231437%
- Positive-well fraction: 85.899094%
- Minimum spatial-group gain: 3.630408
- Minimum typewell-group gain: 3.469701

## Gates

- PASS — minimum_oracle_rmse_gain
- FAIL — minimum_shape_sse_reduction_fraction
- PASS — minimum_positive_well_fraction
- PASS — nonnegative_every_legacy_spatial_group
- PASS — nonnegative_every_legacy_typewell_group
- FAIL — all_destructive_controls_fail_main_coverage

This is hidden-label oracle coverage only. It does not authorize a selector, package, Kaggle run, or competition submission unless every gate passes.
