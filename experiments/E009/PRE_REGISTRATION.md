# E009 Pre-registration — Residual-model consensus abstention

## Question

Can disagreement and sign consensus across independently trained, surface-free datum models identify when bounded residual datum action around E006 is transferable, while returning exactly to E006 when consensus is weak?

## Evidence boundary

E009 uses the frozen E008 legal feature table, E008 residual-datum targets, and E008 OOF rows only after verifying their registered SHA-256 hashes. The feature table contains only test-available horizontal geometry/GR, visible-prefix evidence, E006 path summaries, and target-independent E007 summaries. Formation surfaces, direct typewell predictors, absolute spatial coordinates, hidden TVT outside the applicable training partition, leaderboard feedback, copied predictions, and external artifacts are forbidden. Spatial and typewell data may define evaluator groups only.

Observed E008 subgroup outcomes, action scales, and post-score thresholds are not inputs to any E009 rule. Every model fit, feature screen, imputation statistic, standardization statistic, and rule selection is recomputed inside the applicable training split.

## Frozen model bank

All models predict only residual datum around E006 and use the same 30 ft soft cap. The bank deliberately branches across regularization, feature count, and legal feature families:

1. all legal features, ridge alpha 25, top 32;
2. all legal features, ridge alpha 100, top 32;
3. all legal features, ridge alpha 25, top 64;
4. no E007/self-correlation features, alpha 25, top 32;
5. path/evidence features only, alpha 25, top 24;
6. visible-prefix/relative-geometry features only, alpha 25, top 32.

Feature screening is training-only absolute target correlation with deterministic name tie-breaking. These branches are evaluated together; no branch is abandoned after one poor result.

## Frozen consensus evidence and rules

For each held-out well, compute the six model datum predictions. Inference-time uncertainty may use only: median prediction, sign-agreement fraction, median absolute deviation, relative MAD = MAD/max(abs(median), 1 ft), and absolute action magnitude. It may not use well identity, spatial/typewell group, hidden outcomes, E008 subgroup labels, or a target-derived risk model.

Three fixed rules are scored:

- `consensus_majority`: median action, sign fraction at least 2/3, relative MAD at most 1.0, and |median| at least 1 ft.
- `consensus_strict`: median action, sign fraction at least 0.8, relative MAD at most 0.75, and |median| at least 1 ft.
- `consensus_tight_absolute`: median action, sign fraction at least 0.8, absolute MAD at most 3 ft, and |median| at least 1 ft.

If any rule fails, its correction is exactly zero and its prediction is byte-equivalent to E006 before serialization. Unabstained mean and median model actions are diagnostics, not promotion candidates.

`nested_consensus` chooses among the three fixed rules or exact E006 using inner OOF predictions on each outer-training partition. A rule must gain at least 0.01 RMSE, keep p90 deterioration within 0.10, keep worst-5% SSE-share increase within 0.005, and act on at least 10% of training wells. Among passing rules within 0.01 RMSE of the best, frozen candidate order wins. No action scale is tuned.

## Cross-fitting and evaluation

Use immutable whole-well maps v1–v5. Each outer fit trains all six models on four folds and predicts the untouched fold. The nested rule uses only inner OOF predictions from the outer-training wells. Repeat final placement under five leave-spatial-group-out and five leave-typewell-group-out contexts; typewell information remains evaluator-only.

Report pooled row RMSE, median/p90/p95/max well RMSE, worst-5%/10% SSE share, map and outer-cell wins, action/abstention rates, target correlation and material-sign accuracy, model residual correlations, rule overlap, spatial/typewell groups, long suffix, high GR missingness, low consensus margin, and high dispersion.

## Mandatory controls and edge cases

- Parent hashes, well sets, row IDs/order, and E006/last-known RMSE reproduce.
- Feature leakage scan rejects hidden target, oracle, surfaces, absolute spatial fields, and direct typewell fields.
- Zero action reproduces E006 within 1e-12.
- Duplicating one model in a duplicate-invariant consensus formulation changes action by at most 1e-10.
- Independently deranging model predictions across wells and sign-flipping alternating model outputs may improve E006 by at most 0.03 RMSE.
- Oracle residual datum beats the best legal candidate by at least 0.05 RMSE and remains ineligible.
- Direct row SSE and sufficient-statistic SSE agree within 1e-12 relative error.
- All corrections are finite and bounded by 30 ft.
- Exact fallback is tested for zero, sub-threshold magnitude, split signs, excessive relative dispersion, excessive absolute dispersion, missing/non-finite model output, and one-row suffixes.
- Nested action fraction must remain between 10% and 90%; otherwise it has collapsed to near-no-op or near-direct action.
- Runtime is at most 30 minutes and RSS at most 1,536 MB.
- Independent clean-output reproduction and artifact-hash audit are required.

## Promotion

A candidate is promoted only if every gate passes: at least 0.05 RMSE gain versus E006 and 0.50 versus last-known; at least four of five maps improve E006 by 0.01; at least 17 of 25 outer cells improve; p90 deterioration at most 0.10 versus E006 and 0.50 versus last-known; worst-5% SSE-share increase at most 0.005 versus E006 and 0.01 versus last-known; positive gain in every spatial and every typewell evaluator group; no registered special slice deteriorates by more than 0.05; all controls, runtime, memory, identity, and reproduction checks pass.

If no candidate passes, reject E009 and retain E006. Only a statistically promoted candidate may be packaged and run for local/private offline Kaggle byte parity. Promotion never authorizes a leaderboard submission.
