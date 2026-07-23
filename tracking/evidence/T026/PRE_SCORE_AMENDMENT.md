# T026 pre-score amendment — control alignment and scoring boundary

Frozen before any T026 target coefficient, feature-target correlation, branch coefficient, RMSE, map, cell, stress, slice, horizon-quintile, or control score is computed.

## Source-well target-block permutation

For each outer context, source wells are deterministically permuted with seed 26031. Each receiving well keeps its own task features and task count. The donor well contributes its complete ordered target trajectory after sorting both blocks by `(prefix_rows, horizon_rows, task_id)`. When donor and receiver block sizes differ, each of the four target dimensions is linearly interpolated on normalized block rank from the complete donor block to the receiver block length. This preserves donor-well target-trajectory shape while destroying feature-to-source-well correspondence. A donor may not equal its receiver when the context contains more than one training well.

## Shuffled-task target control

All task target rows inside the outer training split are permuted with deterministic seed 26033 derived from the context key. Features and sample weights remain fixed.

## Reversed-history control

For every task, reverse the four horizon-ordered history coefficient, fit-RMSE, and support-ratio blocks, then recompute first differences and log-horizon extrapolations from the reversed order. Raw mask-conditioned features and targets remain unchanged.

## Permuted-horizon-feature control

Within each outer training split, deterministically permute the complete horizon-conditioned feature subvector as one block with seed 26039 derived from the context key. The block contains explicit prefix/horizon counts and fractions, all four history support ratios, all history fit RMSEs, first differences, and log-horizon extrapolations. Remaining raw geometry/GR/typewell features and task targets remain fixed.

## Baseline scoring

`artifacts/E011/compact_stats_v1.npz` contains E006/last-known sufficient statistics, not exact E011 candidate statistics. T026 therefore reconstructs exact E011 per-well sufficient statistics from `artifacts/E011/oof_predictions.csv.gz`, using the same audited stream and formulas as T025. The compact artifact is used only for frozen E011 legal features and spatial/typewell assignments.

## Comparator interpretation

The `minimum_gain_vs_best_original_only` gate compares each eligible candidate with the lowest-RMSE fixed placement of `ridge_original_combined_a10` over the same four-placement grid. Analytic branches and ineligible ablations are not original-only comparators.
