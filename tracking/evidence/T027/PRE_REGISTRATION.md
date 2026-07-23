# T027 Pre-registration — Target-preserving causal feature-view worth screen

## Hypothesis

Deterministic causal subviews of the frozen 142-feature E011 legal vector can regularize prediction of the same original competition-boundary last-known+spline4 target. Repeating source wells is admissible only when every repeated view keeps exactly the same source target, each source well has controlled total training weight, and held-out wells are excluded from every fit and transform.

## Evidence boundary

This document is frozen before any T027 candidate coefficient, coefficient error, RMSE, map result, outer-cell result, stress result, horizon result, special-slice result, or negative-control result is computed.

T026 supplies the causal reason for this test: mask augmentation improved a weak original-only comparator by 0.284538 RMSE, but its view-local targets differed from the real original target by 14.278922 coefficient RMSE. T027 changes the target relationship, not the architecture size.

## Legal inputs and target

Inputs:

- the exact 142-feature E011 compact legal vector in canonical feature order;
- exact E011 OOF predictions and sufficient statistics for placement scoring;
- raw training TVT only to reconstruct the fixed original-boundary last-known+spline4 target;
- immutable repeated, spatial, and typewell assignments.

Target:

- four bounded piecewise-linear control coefficients fitted to each well's real hidden suffix relative to its real last visible TVT;
- knots 0.25, 0.50, 0.75, and 1.00;
- correction fixed to zero at suffix start;
- coefficient bounds ±80 ft;
- every auxiliary view from a source well receives this exact same target vector.

Forbidden:

- view-local or synthetic-boundary target coefficients;
- hidden target/residual values as features;
- formation surfaces, Geology, absolute X/Y, well ID, evaluator group, leaderboard values, overlap identity, public constants, unknown-license artifacts, or external model outputs;
- any held-out-well view in fitting, imputation, scaling, weighting, model selection, or control construction.

## Frozen 142-feature parity

All views retain the exact 142 E011 feature columns and canonical order. A view changes only registered causal prefix-evidence cells by replacing them with missing values; split-local median imputation handles those cells. Nonregistered feature values are unchanged.

Registered feature blocks:

- `backtest_0p5_*`: five features;
- `backtest_0p7_*`: five features;
- `backtest_0p85_*`: five features;
- window-32 prefix features: names ending `_w32`;
- window-128 prefix features: names ending `_w128`;
- window-512 prefix features: names ending `_w512`.

View catalog:

1. `original_full`: no masked cells.
2. `backtest_0p5_only`: mask 0.70 and 0.85 backtest blocks.
3. `backtest_0p7_only`: mask 0.50 and 0.85 blocks.
4. `backtest_0p85_only`: mask 0.50 and 0.70 blocks.
5. `backtest_0p5_0p7`: mask the 0.85 block.
6. `slope_w32_only`: mask window-128 and window-512 features.
7. `slope_w128_only`: mask window-32 and window-512 features.
8. `slope_w512_only`: mask window-32 and window-128 features.
9. `local_prefix`: keep the 0.85 backtest block and window-32/window-128 features; mask 0.50/0.70 and window-512.
10. `long_prefix`: keep 0.50/0.70 and window-512; mask 0.85 and window-32/window-128.

View indicators are ten appended one-hot columns unless the branch explicitly says `no_indicator`. Test predictions always use `original_full` and its indicator.

## Training regimes

- `original_only`: one original view per source well.
- `backtest_single`: original plus the three single-backtest views.
- `backtest_cumulative`: original, `backtest_0p5_only`, and `backtest_0p5_0p7`.
- `slope_single`: original plus the three single-window views.
- `local_long`: original plus `local_prefix` and `long_prefix`.
- `all_views`: all ten views.
- `all_views_original_boost4`: original view receives weight four before per-source normalization.
- `all_views_no_indicator`: all views without indicator columns.
- `per_view_ensemble`: one ridge model per view type; all ten predictions on the original held-out view are averaged.

Except the explicit original boost, every source well has total training weight one. Duplicate views never increase a source well's total weight.

## Complete branch set

1. `ridge_original_a1`
2. `ridge_original_a10`
3. `ridge_original_a100`
4. `ridge_backtest_single_a1`
5. `ridge_backtest_single_a10`
6. `ridge_backtest_cumulative_a1`
7. `ridge_backtest_cumulative_a10`
8. `ridge_slope_single_a1`
9. `ridge_slope_single_a10`
10. `ridge_local_long_a1`
11. `ridge_local_long_a10`
12. `ridge_all_views_a1`
13. `ridge_all_views_a10`
14. `ridge_all_views_a100`
15. `ridge_all_views_no_indicator_a1`
16. `ridge_all_views_original_boost4_a1`
17. `ridge_all_views_original_boost4_a10`
18. `ridge_per_view_ensemble_a1`
19. `extra_trees_all_views`
20. `hist_gradient_all_views`
21. `knn25_all_views`

Fixed settings:

- ridge alpha exactly as named;
- ExtraTrees: 64 estimators, minimum leaf 8, maximum features 0.7, deterministic context seed, one thread;
- HistGradientBoosting: 160 iterations, learning rate 0.05, maximum leaf nodes 15, minimum leaf 20, L2 1.0;
- KNN: 25 distance-weighted neighbors after split-local median imputation and standardization;
- multioutput targets are fitted directly and clipped to ±80 ft.

No hyperparameter or branch selection occurs inside the screen. Every branch finishes every context.

## Placement and validation

Each branch predicts a direct last-known+spline4 path. It is blended against exact E011 OOF at fixed weights 0.25, 0.50, 0.75, and 1.00. Weight zero is the exact E011 fallback.

Validation:

- all 773 wells and 3,783,989 ordered hidden rows;
- all five immutable maps and 25 repeated outer cells;
- all five spatial and all five typewell holdouts for every branch;
- long suffix, high GR missingness, E011-catastrophe, and five original hidden-horizon quintiles;
- pooled row RMSE primary, plus per-well p90 and worst-5% SSE concentration;
- view-isolation and source-weight audits for every branch/context;
- independent clean-output reproduction.

## Authorization gates

A later formal experiment is authorized only if one eligible fixed candidate passes every gate:

- hidden-label last-known+spline4 oracle RMSE <= 5.0;
- gain versus exact E011 >= 0.15 RMSE;
- RMSE improvement versus T025 best >= 0.03;
- gain versus the best original-only candidate at the same placement grid >= 0.05;
- at least 4/5 map wins and 17/25 repeated-cell wins;
- p90 deterioration versus E011 <= 0.25;
- worst-5% SSE-share increase <= 0.02;
- nonnegative gain in every spatial group, every typewell group, all three special slices, and every horizon quintile;
- every corrupted control <= 0.03 gain;
- duplicate-original equality, all isolation/weight/schema/edge controls, and independent reproduction pass.

## Negative controls

Apply the core `ridge_all_views_a1` branch under:

- deterministic permutation of source-well target blocks;
- deterministic shuffle of task targets inside each outer training split;
- deterministic permutation of view-indicator rows while feature masks remain unchanged;
- deterministic within-view permutation of the registered masked-feature blocks across outer-training wells;
- exact duplicate-original views with per-source normalization, which must reproduce `ridge_original_a1` within 1e-10 coefficient and metric tolerance;
- exact zero-weight E011 fallback.

## Edge cases

The implementation must test at least:

- exact 142-feature order and all registered block counts;
- missing or duplicate feature names;
- unknown view names and duplicate view rows;
- all-missing registered blocks and zero-variance indicators;
- one-well and tiny training partitions;
- source-well total-weight equality and original-boost normalization;
- held-out source exclusion from all task rows and transforms;
- empty view regime and missing original view rejection;
- non-finite target, E011 prediction, feature, or model output handling;
- KNN neighbor clipping and tree sample-weight support;
- coefficient clipping and finite spline reconstruction;
- exact E011 fallback identity;
- deterministic seeds and tie ordering;
- complete repeated/spatial/typewell membership;
- duplicate/unknown/out-of-order OOF IDs;
- all 21 branches, four placements, 35 contexts, controls, and slice groups present.

## Escalation boundary

T027 is a bounded preimplementation worth screen. It does not authorize a neural network, formal experiment, package, Kaggle execution, or competition submission. Failure closes this exact feature-view mask catalog, training regimes, models, and placements. Success authorizes only a separately preregistered formal experiment.
