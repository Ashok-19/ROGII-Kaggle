# E005 Pre-registration

## Question

Can a clean, test-available typewell-horizontal GR path candidate improve both last-known TVT and the exact E004 geometry-prefix deployment comparator without worsening tail risk or failing distribution-shift controls?

## Immutable evidence boundary

Predictors may read only the complete horizontal-well `MD`, `X`, `Y`, `Z`, and `GR` columns, the contiguous visible prefix of `TVT_input`, and the paired typewell `TVT` and `GR` columns. Hidden training `TVT` is evaluator-only. Formation surfaces, typewell `Geology`, leaderboard feedback, external labels, copied artifacts, and opaque mounts are forbidden.

The E001 whole-well folds and data signature are immutable. E004 must be independently regenerated from its frozen geometry-prefix feature family and reproduce RMSE `15.491306398267565` within `1e-9`. E003's surface-assisted RMSE is diagnostic only and is not a deployment comparator.

## Frozen candidate bank

1. `last_known_tvt`: exact baseline.
2. `e004_geometry_prefix`: exact cross-fitted E004 comparator.
3. `align_affine`: bounded affine datum/toe correction chosen by normalized GR mismatch.
4. `align_visible_path`: the same frozen grid after a visible-prefix-only GR calibration.
5. `pf_gr_path`: deterministic sequential particle weighting over the frozen affine grid.
6. `trellis_gr_path`: deterministic Viterbi offset path with bounded transitions.
7. `no_gr_geometry_prefix`: exact E004 fallback.
8. `no_gr_typewell_affine`: damped visible-prefix extrapolation clipped to the typewell TVT range.
9. `axis_confusion`: wrong-axis negative control.
10. `duplicate_align`: exact duplicate of `align_visible_path`.
11. `shuffled_typewell_gr`: deterministic cross-well typewell-GR shuffle control.
12. `oracle_target`: evaluator-only leakage sentinel.

Only candidates 3-6 are promotion-eligible. Diagnostic fixed blends with E004 at weights 0.25, 0.50, and 0.75 are reported but cannot be selected or deployed in E005. No candidate, threshold, selector, correction, or blend may be added after official target scoring.

## Frozen fitting and abstention

At most 256 deterministic GR samples per well are used for alignment. A GR candidate abstains to E004 when hidden coverage is below 10%, fewer than 24 usable hidden samples exist, the typewell is too short or effectively flat, overlap is absent, calibration is invalid, or a finite bounded path cannot be produced. Corrections are clipped to 60 ft and typewell support plus a 20-ft margin. Particle and trellis settings are exactly those in `config.json`.

## Official evaluation

Every candidate emits the same hidden-row IDs under each immutable E001 map. Scoring is pooled row RMSE with well-level datum/trend/shape decomposition, p90 and p95 well RMSE, maximum well RMSE, worst-5% and worst-10% SSE shares, repeated-map cells, long-suffix/high-missing/ambiguous tails, spatial blocks, typewell clusters, residual correlations, and fixed-blend diagnostics.

The shuffled typewell-GR control must be at least 0.10 RMSE worse than its unshuffled counterpart. The duplicate candidate must be identical within `1e-12`; oracle RMSE must be at most `1e-12`; pooled direct and well-aggregated SSE must agree within `1e-12` relative error. Runtime must not exceed 45 minutes and peak RSS must not exceed 1536 MB.

## Promotion and deployment

An eligible candidate is promoted only if all gates in `config.json` pass. In particular it must improve at least 0.35 RMSE versus last-known and 0.10 versus E004, win at least four of five maps, remain within the frozen p90 and worst-5% tail limits against both comparators, and preserve positive gains in both spatial and typewell-cluster stress. The pre-registered diversity leg requires residual correlation to E004 no greater than 0.995.

Private internet-disabled Kaggle parity is run only after statistical promotion and local deployment checks justify packaging. Deployment readiness does not itself authorize a leaderboard submission. No submission is made unless a separate durable decision explicitly authorizes it.

## Reproducibility

The frozen design is committed before implementation scoring. The official run records the committed code SHA. A second independent output root must reproduce every deterministic artifact byte-for-byte. Results, verification, hashes, records, Learning Lab content, tests, tracker state, and validation evidence are committed using explicit intended paths only. Unrelated pre-existing files remain untouched.
