# T026 Pre-registration — Randomized long-horizon mask-task coefficient meta-learning worth screen

## Hypothesis

Many causal end-of-well mask tasks generated only from outer-training wells can teach a mask-conditioned four-control spline coefficient predictor that transfers better than T025’s four fixed within-prefix summaries. The expanded task distribution must improve true hidden-suffix prediction, not merely exploit repeated source-well identity or short-horizon smoothness.

## Evidence boundary

This document is frozen before any T026 task target coefficient, feature-target correlation, candidate coefficient, RMSE, map result, cell result, stress result, special-slice result, or negative-control result is computed.

Only non-target support metadata was inspected to freeze the task pool. Original known prefixes span 851–2,392 rows. Original hidden suffixes span 407–10,052 rows with median 4,840. The frozen randomized pool contains 12,216 tasks, 8–16 per well, with median horizon 4,878 rows.

## Outer-well isolation

For every repeated, spatial, or typewell context:

- task rows may originate only from context training wells;
- all tasks from every held-out well are forbidden from fitting, imputation, scaling, weighting, bin-edge construction, and model selection;
- no well identifier, row ID, target-derived group, hidden coefficient, hidden RMSE, or leaderboard value may be used as a feature;
- hyperparameters, placements, task-generation constants, and gates are fixed here and are not selected from context results.

Precomputing deterministic task features and targets is permitted only if context filtering is exact and verified before fitting.

## Randomized end-of-well task pool

Each task masks from a legal cut to the actual end of a training well. Its target is the frozen four-control piecewise-linear spline fit to `TVT[cut:]` relative to `TVT[cut-1]`, with knots 0.25, 0.50, 0.75, and 1.00 and coefficient bounds ±80 ft.

Base prefix grid:

```text
900, 1180, 1346, 1423, 1504, 1565, 1650, 1703,
1745, 1802, 1860, 1945, 2053, 2214, 2350
```

For each source well and base cut, deterministic SHA-256-derived jitter in [-64, +64] rows is generated with seed 26023. The original competition boundary is always included. A task is kept only when:

- prefix rows >= 851;
- end-of-well horizon rows >= 407;
- MD is strictly increasing and required legal columns are valid;
- its target and all causal history segments have sufficient finite support.

Duplicate cuts within a source well are removed.

## Legal task features

### Raw mask-conditioned features

Reconstruct the E004-style legal feature family at the task cut using only:

- `TVT` before the cut, representing visible `TVT_input`;
- full non-target `MD`, `X`, `Y`, `Z`, and `GR`, which are available at inference;
- the paired typewell `TVT` and `GR` summary;
- visible-prefix TVT/U/GR summaries and robust slopes;
- hidden geometry/GR summaries and missingness;
- visible pseudo-backtests at 0.50, 0.70, and 0.85;
- known/horizon rows and fractions.

No TVT at or after the task cut is used in any feature.

### Horizon-matched causal coefficient history

Let `H` be the task horizon and `P` the prefix length. Define `M = min(H, P - 1)`. Fit four historical spline targets on segments ending immediately before the task cut with lengths 0.40M, 0.60M, 0.80M, and 1.00M. These historical segments are entirely visible at task time. Record their bounded coefficients, fit RMSEs, support ratios, first differences, and per-output log-horizon extrapolations to H.

The minimum frozen history support is 128 rows. Score-blind metadata proves the smallest possible 0.40M support exceeds this boundary.

## Task weighting

The following fixed training regimes are tested:

- `original_only`: original competition-boundary task from each training well;
- `masks_only`: randomized masks excluding the original task;
- `all_well_equal`: all tasks, with each source well receiving total weight one;
- `all_joint_balanced`: well-equal weights multiplied by inverse frequency in a 5×5 prefix/horizon bin grid whose edges are computed from original tasks in the current outer training partition only;
- `all_original_boost4`: original task weight multiplied by four before per-well normalization;
- `long_only`: horizon at least 3,830 rows, the frozen original hidden-length 20th percentile;
- `short_only`: horizon below 3,830 rows; registered as an ineligible horizon ablation.

## Complete branch set

Every branch and every context must finish:

1. `history_latest_identity`
2. `history_log_extrapolation`
3. `ridge_original_combined_a10`
4. `ridge_masks_combined_a10`
5. `ridge_all_raw_a10_well_equal`
6. `ridge_all_history_a10_well_equal`
7. `ridge_all_combined_a1_well_equal`
8. `ridge_all_combined_a10_well_equal`
9. `ridge_all_combined_a100_well_equal`
10. `ridge_all_combined_a10_joint_balanced`
11. `ridge_all_combined_a10_original_boost4`
12. `ridge_delta_history_a10_well_equal`
13. `ridge_long_combined_a10_well_equal`
14. `ridge_short_combined_a10_well_equal` — ineligible ablation
15. `ridge_horizon_expert_a10`
16. `extra_trees_combined_well_equal`
17. `hist_gradient_combined_well_equal`
18. `knn25_combined`
19. `ridge_horizon_only_a10` — ineligible comparator

Fixed model settings:

- ridge alphas exactly 1, 10, or 100 as named;
- ExtraTrees: 64 estimators, minimum leaf 8, maximum features 0.7, deterministic context seed;
- HistGradientBoosting: maximum 160 iterations, learning rate 0.05, maximum leaf nodes 15, minimum leaf 20, L2 regularization 1.0;
- KNN: 25 distance-weighted neighbors after split-local median imputation and standardization;
- horizon expert: five train-original horizon bins, same bin plus adjacent bins, minimum 128 task rows, otherwise exact global ridge fallback.

Each branch is placed against exact E011 OOF with fixed weights 0.25, 0.50, 0.75, and 1.00. Weight zero is the exact E011 fallback.

## Validation

- all 773 wells and 3,783,989 ordered hidden rows;
- all five immutable maps and 25 repeated outer cells;
- all five spatial and five typewell holdouts for every branch, not only preliminary passers;
- long suffix, high GR missingness, and E011-catastrophe slices;
- pooled row RMSE primary, plus per-well p90 and worst-5% SSE concentration;
- per-horizon-bin gains and original-task versus randomized-task coefficient diagnostics;
- independent clean-output reproduction.

## Worth authorization gates

A later formal experiment is authorized only if one eligible fixed candidate passes every gate:

- hidden-label last-known+spline4 oracle RMSE <= 5.0;
- gain versus exact E011 >= 0.15 RMSE;
- RMSE improvement versus T025 best aggregate candidate >= 0.03;
- at least 4/5 map wins and 17/25 repeated-cell wins;
- p90 deterioration versus E011 <= 0.25;
- worst-5% SSE-share increase <= 0.02;
- nonnegative gain in every spatial group and every typewell group;
- nonnegative gain in all three special slices;
- nonnegative gain in every frozen original hidden-horizon quintile;
- at least 0.05 RMSE gain over the best original-only branch at the same placement grid;
- all controls, edge groups, isolation checks, and independent reproduction pass.

## Negative controls

Apply the core `ridge_all_combined_a10_well_equal` branch under:

- deterministic permutation of complete source-well target blocks;
- shuffled task targets inside each outer training split;
- reversed order of the four historical horizons while retaining values;
- deterministic permutation of task horizon labels/features;
- exact zero-weight E011 fallback.

No corrupted control may gain more than 0.03 RMSE versus E011.

## Edge cases

The implementation must test at least:

- exact 851-row prefix and 407-row horizon boundaries;
- one-row violations on both boundaries;
- deterministic jitter limits and duplicate-cut removal;
- contiguous visibility and strict MD ordering;
- missing/duplicate source wells and OOF IDs;
- outer-held-out task exclusion from every fit and transform;
- constant TVT, constant/non-finite GR, non-finite TVT/geometry/typewell values;
- history support exactly 128 and 127 rows;
- rank-deficient spline solves;
- zero-variance raw/history features;
- empty and tiny task bins;
- all tasks from one source well receiving equal total weight;
- joint-bin inverse-frequency normalization;
- tree sample-weight support;
- KNN neighbor clipping;
- coefficient clipping and finite path reconstruction;
- exact E011 fallback identity;
- deterministic seeds/ties;
- complete repeated/spatial/typewell membership;
- complete ordered OOF coverage;
- all registered branches, placements, controls, and contexts present.

## Escalation boundary

T026 is a bounded preimplementation worth screen. It does not authorize a neural network, formal experiment, package, Kaggle run, or competition submission. A complete pass authorizes only a new preregistered formal experiment. A failure closes H019 as tested and prohibits retuning this exact mask grid, feature family, weighting grid, model settings, or placement grid.
