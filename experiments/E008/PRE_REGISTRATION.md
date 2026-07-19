# E008 Pre-registration

## Question

Can a compact model trained only inside each validation training split learn a bounded signed residual datum/trend correction around E006 from surface-free test-available evidence, while avoiding the fixed E007 p90 and subgroup failures?

## Evidence boundary

The predictor may use only horizontal-well `MD`, `X`, `Y`, `Z`, `GR`, the contiguous visible `TVT_input` prefix, the frozen E006 prediction path, and target-independent E007 horizontal self-correlation paths and diagnostics. Hidden-zone geometry and GR are legal because they are supplied in test. Direct typewell features, typewell identity, absolute spatial coordinates, formation surfaces, typewell `Geology`, hidden TVT, leaderboard feedback, copied predictions, external labels, and opaque artifacts are forbidden predictor inputs.

Actual hidden TVT is used only to create residual targets for wells inside a model's training partition and to score untouched held-out wells. Feature screening, imputation, scaling, ridge fitting, conservative action-scale selection, and every target-derived statistic must be recomputed inside the applicable training split.

## Frozen target and action

For each training well and each hidden row, define residual `r = TVT - E006`. Let `x = hidden_index/(hidden_rows-1) - 0.5`, or zero for a one-row suffix. The two targets are:

- residual datum: the mean of `r`;
- residual trend: the least-squares coefficient of `r` on centered `x`, representing end-to-end trend change.

The emitted correction is `scale * (datum + trend*x)`. Predicted datum is soft-capped as `30*tanh(raw/30)` and trend as `60*tanh(raw/60)`. The final path is E006 plus this bounded correction. Scale zero is an exact E006 fallback.

The ±30/±60 caps are frozen from the previously recorded E003 target distribution: approximately 95% of baseline datum targets lie within about 21 ft and 95% of trend targets within about 34 ft. E008 does not inspect held-out E008 outcomes to set these values.

## Frozen feature contract

The surface-free table contains deterministic summaries from these families:

1. row counts, visible/hidden fractions, and MD spans;
2. relative X/Y/Z movement, hidden horizontal distance, and hidden azimuth sine/cosine, excluding absolute X/Y location;
3. visible TVT, visible U=`TVT+Z`, visible Z, and hidden Z summaries;
4. visible, hidden, and whole horizontal GR summaries, coverage, missing fractions, and longest missing run;
5. visible U/TVT robust slopes over 32, 128, and 512 rows and visible-prefix pseudo-backtests at 0.5, 0.7, and 0.85;
6. E006 hidden-path summaries and its corrections relative to E004 and last-known;
7. E007 raw bounded correction summaries plus visible/hidden GR coverage, template/query counts, fingerprint distance, margin, visible pseudo-gain, fallback state, and maximum correction.

Every feature must be available at inference. Columns containing target, oracle, residual labels, hidden TVT, surface names, direct typewell values, or absolute spatial coordinates are forbidden. Missing values are imputed by the training-split median. Features and targets are standardized from training data only. Within each fit, retain at most 32 features ranked by maximum absolute training correlation across standardized datum and trend targets, breaking ties by feature name.

## Frozen model and candidate bank

The model is the deterministic multi-output ridge solver previously audited in E003, with `alpha=25` and no third-party runtime dependency.

Comparators and diagnostics:

- `last_known_tvt`, `e004_geometry_prefix`, and `e006_nested_fusion`;
- fold-training mean residual datum and datum+trend actions;
- `ridge_no_e007_ablation`, trained under the same split after removing all E007 feature columns;
- `duplicate_feature_ridge`, duplicating one standardized feature as two `1/sqrt(2)` copies to preserve the ridge penalty;
- `shuffled_target_ridge`, trained on a deterministic whole-well derangement of datum/trend targets;
- evaluator-only oracle residual datum and datum+trend actions.

Promotion-eligible candidates:

1. `ridge_residual_datum`: predict and apply only soft-capped residual datum at scale 1.
2. `ridge_residual_datum_trend`: predict both soft-capped residual datum and trend at scale 1.
3. `ridge_residual_conservative`: use the same datum+trend model with scale selected from `[0, 0.25, 0.5, 0.75, 1]` by inner OOF evidence on the outer-training wells. A scale must gain at least 0.01 RMSE versus E006, deteriorate p90 by no more than 0.10, and increase worst-5% SSE share by no more than 0.005. Choose the smallest scale within 0.01 RMSE of the best passing scale; otherwise choose zero.

## Cross-fitting and stress evaluation

For every immutable fold map and outer fold, fit only on the other four folds and score the untouched fold. The conservative candidate generates inner OOF predictions on the outer-training wells using the remaining groups of that same fold map before selecting scale. Five map-specific OOF paths are averaged for final repeated-map reporting.

Repeat the final fit under five leave-spatial-group-out and five leave-typewell-group-out evaluator contexts. Direct typewell data may define evaluator groups only and never enters the feature table. Conservative scale in stress contexts is selected from inner OOF predictions formed by frozen `v1` assignments restricted to the context's training wells.

## Controls

All controls are mandatory:

- E006 and E007 parent artifacts must match frozen SHA-256 values. E006, E004, and last-known parent RMSEs must reproduce within `1e-9`.
- Feature leakage scan must reject hidden target, oracle, correction-label, surface, direct typewell, and absolute spatial columns.
- Zero action must reproduce E006 within `1e-12`.
- Duplicate-feature predictions must match the corresponding ridge predictions within `1e-8`.
- Direct and well-aggregated SSE must agree within `1e-12` relative error.
- Deterministically shuffled targets may improve E006 by at most 0.03 RMSE and may have absolute datum or trend correlation above neither 0.15.
- The rowwise residual oracle is permanently ineligible and must beat the best legal candidate by at least 0.05 RMSE.
- E007 evidence is considered used only if at least one E007 feature is selected in at least three repeated maps. The full model may trail the no-E007 ablation by no more than 0.01 RMSE.
- Emitted correction magnitude must not exceed 60 ft and every prediction must be finite.
- Conservative scale must be nonzero in at least 15 of 25 outer cells and have nonzero median scale in at least four maps; outer-cell median-scale range may not exceed 0.75.
- Runtime must not exceed 30 minutes and RSS must not exceed 1,536 MB.
- An independent clean-output reproduction and artifact-hash audit are required.

## Promotion

An eligible candidate is promoted only if every frozen gate passes:

- gain of at least 0.05 RMSE versus E006 and 0.50 versus last-known;
- at least four of five maps improve E006 by at least 0.01;
- at least 17 of 25 repeated outer cells improve E006;
- p90 deterioration of at most 0.10 versus E006 and 0.50 versus last-known;
- worst-5% SSE-share increase of at most 0.005 versus E006 and 0.01 versus last-known;
- positive gain in every spatial and every typewell evaluator group;
- deterioration of at most 0.05 on long suffix, high GR missingness, poor E007 pseudo-gain, and low fingerprint-margin slices;
- all parent, leakage, no-op, duplicate, shuffle, oracle, E007-use, correction-bound, stability, runtime, memory, identity, and reproduction controls pass.

Among candidates passing all gates, select the lowest mean repeated-map RMSE, breaking ties by frozen candidate order. If none pass, reject E008 and retain E006.

## Deployment

Only a statistically promoted candidate may be packaged. Deployment must embed the audited E006 model, E007 horizontal self-correlation runtime, the full-training E008 feature transform and ridge coefficients, and the frozen action scale. It must remain offline and standard-library-only, preserve exact sample IDs/order, emit finite predictions, fall back exactly to E006 when E007 or E008 evidence is invalid, remain below 100 MB, and pass local plus private internet-disabled Kaggle raw-byte parity. Promotion does not authorize a leaderboard submission.

## Reproducibility

The design, feature contract, targets, caps, regularization, candidates, scale grid, controls, and gates are committed before E008 hidden-suffix scoring. Results must include legal features, residual targets, selected-feature frequency, model coefficients, per-candidate and per-well metrics, repeated maps and outer cells, spatial/typewell stress, special slices, scale selection, target diagnostics, controls, runtime/RSS, deterministic OOF predictions, hashes, and an independent output-root reproduction.
