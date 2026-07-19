# E007 Pre-registration

## Question

Can same-well horizontal GR self-correlation recover local structural direction that is complementary to E006, without typewell matching, and add repeatable value after strictly cross-fitted placement?

## Evidence boundary

The self-correlation leg may use only horizontal `MD`, `X`, `Y`, `Z`, `GR`, the contiguous visible `TVT_input` prefix, and the frozen E006 averaged OOF path as an already cross-fitted anchor. Typewell TVT/GR may be used by the frozen E006 parent and to define evaluator-only typewell stress groups, but it is forbidden from every E007 self-correlation fingerprint, slope target, path, reliability diagnostic, and route.

Actual hidden-suffix `TVT` is evaluator-only. It cannot define fingerprints, slopes, candidate strengths, thresholds, reliability, or fallback rules. Training-only formation surfaces, typewell `Geology`, visible test authoring examples, leaderboard feedback, external labels, copied predictions, and opaque artifacts are forbidden.

## Visible-only design audit

Before this registration, candidate mechanics were compared only by moving the visibility boundary inside the supplied prefix at fractions 0.65, 0.75, and 0.85. The final visible segment was treated as pseudo-hidden; the real hidden suffix was never scored.

Absolute TVT-residual transfer was falsified and is permanently excluded because every tested strength materially worsened the visible-only pseudo-holdout. Local structural-slope transfer was positive at every pseudo-boundary. The frozen design uses five nearest fingerprints and records 0.10 as the fixed visible-only reference strength. Hidden scoring may not add candidates or change these mechanics.

## Frozen self-correlation path

1. Define the structural coordinate `U = TVT + Z` on the visible prefix.
2. Normalize all finite GR from the supplied horizontal well.
3. At each row, form a seven-component fingerprint: local normalized GR means at radii 0, 6, 18, and 48 rows, plus right-minus-left means at radii 6, 18, and 48.
4. Use visible template states every eight rows, excluding the 12-row slope boundary.
5. Estimate each template state's local `dU/drow` over a centered 12-row radius.
6. Query hidden GR every 16 rows. Select the five smallest finite fingerprint distances using deterministic row-index tie breaks, weight them by `exp(-min(20, distance))`, clip the matched slope to ±0.12 ft/row, and linearly interpolate between query anchors.
7. Integrate matched slopes from the exact last visible `U`, then convert back with `TVT = U - Z`.
8. Bound the correction around frozen E006 with `40*tanh((raw_selfcorr - E006)/40)`.

If visible GR coverage is below 0.35, hidden GR coverage is below 0.20, fewer than 40 template states or four query anchors are available, or any path diagnostic is invalid, the self-correlation leg abstains exactly to E006.

## Frozen candidate bank

Comparators:

- `last_known_tvt`.
- `e004_geometry_prefix` from the audited E006 parent artifact.
- `e006_nested_fusion`.

Diagnostics and controls:

- `selfcorr_raw_bounded`, full bounded correction.
- `crossfit_rmse_grid`, unconstrained outer-training RMSE weight, diagnostic only.
- `duplicate_conservative`, exact duplicate control.
- `shuffled_selfcorr_conservative`, deterministic whole-well correction-template shuffle.
- `oracle_rowwise_best`, evaluator-only leakage sentinel.

Promotion-eligible candidates:

1. `visible_fixed_0p10`: the 0.10 strength selected only from visible-prefix pseudo-holdouts.
2. `crossfit_conservative`: fit a weight from `[0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1]` on the outer-training wells. A weight must gain at least 0.01 RMSE over E006, deteriorate p90 by no more than 0.20, and increase worst-5% SSE share by no more than 0.005. Choose the smallest weight within 0.01 RMSE of the best passing weight; otherwise use zero.
3. `crossfit_reliability_shrink`: multiply the conservative weight by the mean favorable outer-training percentile of visible pseudo-holdout gain, hidden GR coverage, reversed median best fingerprint distance, and fingerprint margin.
4. `crossfit_positive_gate`: apply the conservative weight only when pseudo-holdout gain is positive, hidden GR coverage is at least 0.35, and reliability is at least 0.5; otherwise use E006 exactly.

For each of the five immutable fold maps and every outer fold, weight/reliability fitting uses only the other four folds. Frozen E006 OOF and self-correlation paths are target-independent for the held-out well. The same leave-group-out placement is repeated for five spatial and five typewell evaluator groups. Five map-specific predictions are averaged for the final OOF path.

## Controls

- Parent E006 OOF SHA-256 must equal `9b9f26cd8ad82e32dbc0f9ba1466cb38424cd3bbf7001c1079a9867fb5a324b4` and reproduce E006 RMSE `14.933140787236884` within `1e-9`.
- Parent E004 and last-known columns must reproduce their registered RMSE values within `1e-9`.
- Zero weight must reproduce E006 within `1e-12`.
- Duplicate placement must be identical within `1e-12`.
- Direct and well-aggregated SSE must agree within `1e-12` relative error.
- The visible-only design audit must retain positive pooled gain at all three frozen pseudo-boundaries.
- The shuffled control circularly permutes whole-well correction templates in SHA-256 order, resamples to recipient length, and must lose at least 0.03 RMSE versus the corresponding unshuffled conservative candidate while improving E006 by no more than 0.01.
- The rowwise oracle is permanently ineligible and must beat the best legal candidate by at least 0.05.
- At least 15 of 25 outer cells and four of five maps must use nonzero median self-correlation weight; outer-cell median-weight range may not exceed 0.75.
- Raw bounded self-correlation residual correlation to E006 must not exceed 0.98, demonstrating nontrivial independent error structure.

## Promotion

An eligible candidate is promoted only when all frozen gates pass:

- at least 0.45 RMSE gain versus last-known and 0.03 versus E006;
- at least four of five maps improve E006 by at least 0.01;
- at least 17 of 25 outer cells improve E006;
- p90 deterioration is at most 0.25 versus E006 and 0.50 versus last-known;
- worst-5% SSE-share increase is at most 0.01 versus both comparators;
- every leave-spatial-group-out and leave-typewell-group-out result improves E006;
- long-suffix, high-GR-missingness, poor-pseudo-gain, and low-match-margin deterioration is at most 0.10;
- all parent, identity, no-op, duplicate, shuffle, oracle, diversity, stability, runtime, and memory controls pass.

Among candidates passing every gate, select lowest mean repeated-map RMSE, breaking ties by frozen candidate order. If none pass, reject E007 and retain E006 as primary deployment-ready surface-free candidate.

## Deployment

Only a promoted candidate may be packaged. Deployment must embed the audited E006 model and pure-standard-library self-correlation runtime, remain offline and self-contained, preserve exact sample IDs/order, emit finite values, fall back exactly to E006, stay below 100 MB, and pass local plus private internet-disabled Kaggle raw-byte parity. Promotion does not authorize a leaderboard submission.

## Reproducibility

The design and thresholds are committed before hidden-suffix scoring. Results must include the visible-only design audit, parent artifact audit, per-candidate and per-well metrics, repeated maps and outer cells, spatial/typewell stress, special slices, weights, reliability diagnostics, controls, runtime/RSS, deterministic OOF predictions, artifact hashes, and an independent output-root reproduction.
