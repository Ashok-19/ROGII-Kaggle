# T028 result — rejected before implementation

## Decision

Close H021/T028 without implementing a domain-robust coefficient model. E011 remains deployment primary; E006 and E004 remain fallbacks.

## Measured evidence

- T027 original-only candidate: `12.4588330369` RMSE at placement 0.50.
- Globally optimal placement: `0.5454051778`, RMSE `12.4581891219`.
- Extra gain from global re-placement: only `0.0006439149` RMSE.
- Common positive placement interval across all ten domains: `[0.0, 0.0]`; spatial group 3 and typewell group 4 have positive first derivative at zero, so no nonzero common placement improves every domain.
- Illegal spatial-domain oracle placement adds `0.0397890310` RMSE beyond the current candidate.
- Illegal typewell-domain oracle placement adds `0.0494021213` RMSE beyond the current candidate.
- The frozen domain-headroom floor was 0.05 RMSE; the weaker gain is `0.0397890310`, so the decisive gate fails.
- Per-well oracle placement reaches `11.9324309941` and shows `0.5264020428` RMSE opportunity, but this requires hidden-label well routing and cannot authorize a legal objective.
- Seven of ten domain-optimal weights differ materially from the global optimum, confirming conflict rather than a missing scalar placement.
- Coefficient residual RMSE is `14.1695782492`, but the pooled residual mean norm is only `0.1802086287`; domain mean norms range from `1.1232153112` to `3.5045372737`.

## Gate result

`{"additional_per_well_headroom": true, "domain_conflict_headroom": false, "material_domain_weight_heterogeneity": true, "simple_global_placement_not_sufficient": true}`

`worth_screen_authorized = false`

## Completion

The corrected audit covers all 773 wells, 3,783,989 hidden rows, 142 legal features, five spatial groups, five typewell groups, global placement, per-domain placement, per-well placement, and coefficient-residual diagnostics. Its JSON output reproduced byte-for-byte. No model, package, Kaggle notebook, competition submission, or inference route was created.
