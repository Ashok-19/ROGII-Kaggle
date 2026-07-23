# T030 preregistration — outer-isolated complementary pseudo-coefficient and datum ensemble

## Decision boundary

T029 demonstrated only a full-label covariance opportunity. It did not authorize a blend. T030 is the formal confirmation. No T030 score may be read until this document, `config.json`, D036, and the active tracker records are committed.

## Scientific question

Can the T025 pseudo-coefficient action and the nearly uncorrelated T016 datum action preserve their complementary gain when every parent action and blend choice is produced without access to the untouched outer wells?

## Frozen legal inputs

- T025 legal four-cut pseudo features and the frozen 142-feature E011 vector.
- Original-boundary four-control spline target on training wells only.
- The 12 saved cross-fitted E009 datum-action columns used by T016.
- E011 OOF sufficient statistics for scoring and training-well-only selection.
- Frozen fold maps and evaluator-only spatial/typewell assignments.

Forbidden at inference or fitting: hidden labels from held-out wells, absolute spatial coordinates as model inputs, typewell identity, subgroup routing, leaderboard values, visible-test overlap identity, formation surfaces, Geology, unknown-license artifacts, or any newly fit feature/model family.

## Outer isolation

T030 has 25 repeated whole-well cells and ten leave-domain-out stress contexts.

For every outer context:

1. Fit `ridge_pseudo_e011_a1` only on outer-training wells and predict outer-test coefficients.
2. Select the T016 action family/scale/cap only from outer-training wells, then apply the selected legal saved action to outer-test wells.
3. For nested blend branches, create inner OOF parent actions entirely within the outer-training set:
   - repeated outer cells use the other four folds from the same map version;
   - stress outer cells use v1 folds restricted to outer-training wells.
4. Select blend weights only from those inner OOF actions.
5. Audit every train/validation/test membership and reject any overlap.

## Row-action definition

For a well and one context:

`correction = alpha * (last_known_tvt - E011 + B @ c_T025) + beta * d_T016 + gamma * (last_known_tvt - E011 + B @ c_T027)`

where `B` is the fixed four-control spline basis. Final repeated OOF scoring averages each well’s five row-correction actions. It never averages RMSE values or independently averages coefficients and weights.

## Frozen real branches

1. T025 fixed 0.50.
2. T016 fixed 1.00.
3. T029 every-domain diagnostic 0.40/0.60.
4. T029 minimax diagnostic 0.25/0.45.
5. T029 exact pair-box diagnostic 0.4623841380462943/0.896042906654928.
6. T029 pooled three-leg comparator 0.40 T025 / 0.55 T016 / 0.05 T027.
7. Inner pooled selection on a 0.05 simplex grid.
8. Inner pooled selection on a 0.10 box grid.
9. Inner maximum-minimum available-domain selection on the simplex grid.
10. Inner lowest-pooled-RMSE domain-safe simplex selection, with exact-zero fallback if no point is nonnegative in every available spatial/typewell group.

The T027 leg is never eligible for nested selection and appears only in branch 6 to confirm T029’s zero-weight conclusion.

## Frozen controls

Four joint controls corrupt both parents so a surviving real leg cannot falsely fail the control gate:

- shuffled T025 targets plus shuffled T016 well rows;
- sign-flipped complete T025 and T016 directions;
- independently permuted outer-test T025 and T016 actions;
- shuffled T025 targets plus sign-flipped T016.

The maximum corrupted-control gain versus E011 is 0.03 RMSE.

## Frozen reporting

- pooled RMSE, median/p90/p95/max well RMSE;
- worst 5% and 10% SSE shares;
- five map wins and 25 outer-cell wins;
- all five spatial and five typewell gains;
- long suffix, high hidden-GR missingness, and E011-catastrophe slices;
- five hidden-horizon quintiles;
- selected weights and T016 placements by context;
- parent reproduction, isolation, edge, fallback, bound, and negative-control receipts.

## Promotion gate

A candidate is promotable only if all conditions hold:

- gain versus E011 at least 0.15 RMSE;
- incremental gain versus the reproduced T025 parent at least 0.03 RMSE;
- wins all five maps and at least 17/25 outer cells;
- no p90 deterioration versus E011;
- worst-5% SSE-share increase no more than 0.005;
- positive gain in every spatial group, every typewell group, all three special slices, and all five horizon quintiles;
- both T025 and T016 are active in at least 17/25 repeated contexts for any selected-weight branch;
- corrupted controls gain no more than 0.03;
- exact parent reproduction, exact E011 fallback, all edge groups, complete isolation audit, and independent exact reproduction pass.

Failure closes H022/T030. Passing authorizes only a deployment-candidate review; it does not authorize packaging, Kaggle execution, or submission.
