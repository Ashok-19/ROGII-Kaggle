# T025 Pre-registration — Multi-cut spline coefficient dynamics worth screen

## Hypothesis

The sequence of four-control spline targets fitted on multiple observed pseudo-tails inside a well's known prefix contains transportable information about the well's final hidden-tail nonlinear coefficients. Cross-well models over this coefficient trajectory can produce a path that adds material, stable value to exact E011.

## Evidence boundary

This document is frozen before any final hidden coefficient correlation, branch RMSE, candidate gain, map result, stress result, or control result is computed.

Legal feature construction for a held-out well may use only rows at or before the original `TVT_input` prediction start. Each pseudo-tail lies entirely inside that known prefix. Actual hidden TVT, hidden spline targets, spatial/typewell evaluator groups, leaderboard feedback, public constants, formation surfaces, and external artifacts are forbidden as inference features.

## Pseudo-cut target

For cut fractions 0.40, 0.55, 0.70, and 0.85 of the original known prefix:

1. use TVT at the cut as a last-known baseline;
2. treat the remaining observed prefix rows as a pseudo-hidden segment;
3. fit the frozen E011 four-control piecewise-linear spline with knots 0.25, 0.50, 0.75, and 1.00 by least squares;
4. record four bounded coefficients and the pseudo-fit residual diagnostics.

The final training target is the same four-control fit over the actual hidden suffix relative to the real last-known TVT. It is used only inside the applicable training split and for scoring after prediction.

## Complete branch set

All branches must finish; none may be abandoned after one fold:

- `latest_cut_identity`
- `linear_cut_extrapolation`
- `ridge_pseudo_a1`
- `ridge_pseudo_a10`
- `ridge_pseudo_a100`
- `ridge_pseudo_e011_a1`
- `ridge_pseudo_e011_a10`
- `ridge_pseudo_e011_a100`
- `extra_trees_pseudo`
- `hist_gradient_pseudo`
- `knn25_pseudo`
- `ridge_e011_only_a10`
- `ridge_e011_only_a100`

Each branch is placed against exact E011 with fixed candidate weights 0.25, 0.50, 0.75, and 1.00. Weight 0 is the exact fallback comparator.

## Validation

- 773 wells and 3,783,989 hidden rows.
- Five immutable whole-well maps and all 25 repeated outer cells.
- Five spatial and five typewell holdouts for every branch that passes the preliminary repeated gates.
- Long suffix, high GR missingness, and E011-catastrophe slices.
- Pooled row RMSE is primary; p90 and worst-5% SSE share are mandatory.

## Worth authorization gates

A later formal experiment is authorized only if one fixed candidate passes every gate:

- final last-known+spline4 hidden-label oracle RMSE no worse than 5.0;
- at least 0.15 RMSE gain versus E011;
- at least 4/5 map wins and 17/25 outer-cell wins;
- p90 deterioration no greater than 0.25;
- worst-5% SSE-share increase no greater than 0.02;
- nonnegative gain in every spatial group and every typewell group;
- nonnegative gain in all three special slices;
- at least 0.05 RMSE incremental gain over the best E011-only comparator with the same target and placement grid;
- all scientific controls, edge groups, and independent reproduction pass.

## Negative controls

- deterministic well permutation of the complete pseudo-cut trajectory;
- shuffled final coefficient targets inside each training split;
- reversed cut order while retaining the same coefficient values;
- exact zero-weight E011 fallback.

No negative control may gain more than 0.03 RMSE versus E011.

## Edge cases

The implementation must test contiguous-prefix enforcement, missing/duplicate wells and IDs, one- and two-row segment rejection, exact 128-row support boundary, constant TVT, non-finite TVT/coefficients/predictions, degenerate spline matrices, zero-variance features, tiny train partitions, KNN neighbor clipping, coefficient and path clipping, deterministic ties, complete fold membership, exact fallback identity, and complete ordered OOF coverage.

## Escalation boundary

T025 is a preimplementation worth screen. A pass authorizes only a new preregistered formal experiment. A failure closes H018 as tested. No Kaggle run, package, notebook execution, or competition submission is authorized by this screen.
