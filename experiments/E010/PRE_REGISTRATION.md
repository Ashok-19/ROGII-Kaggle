# E010 Pre-registration — Nonlinear candidate coverage and selector regret

Frozen at 2026-07-19T19:25:22Z, before any new E010 hidden-label score was computed.

## Question

Does a fixed legal bank contain sufficiently accurate nonlinear whole-well paths, and can legal multi-cut visible-prefix evidence select them without catastrophic regret?

This experiment separates **generation coverage** from **legal selection**. Hidden truth may identify the best fixed whole trajectory only as a retrospective diagnostic. Oracle family identity and coefficients never enter validation-well fitting or inference.

## Evidence before implementation

- Published full-data diagnostics report 9.0354 RMSE for an oracle constant, 6.6972 for an oracle line, and 3.1106 for a robust smooth nonlinear curve. Datum-only or line-only work cannot be the full gold route.
- E006 remains the only promoted deployable legal family at 14.9331 RMSE.
- E009's cross-fitted wide-64 ridge is a useful but unstable 14.6353 diagnostic anchor.
- The exact-HMM public source was audited at SHA-256 `2321997c...31dc`; its five-well score is preliminary. Four independently implemented variants must earn full expansion.
- E005 PF/trellis evidence is complementary but weak. Four constrained multiscale DTW variants are screened so one failed alignment is not treated as final.

## Frozen branching rule

The core lattice always runs. HMM and DTW each receive four deterministic variants on 48 target-independently selected, regime-balanced wells. A branch advances to all 773 wells only if its best declared variant:

1. improves E006 by at least 0.10 RMSE **or** improves the core-bank oracle by at least 0.05;
2. emits finite paths on every screen well; and
3. projects below 45 minutes for all wells.

A failed screen closes that branch with complete evidence; it does not terminate E010.

## Fixed candidate bank

Raw anchors are last-known, E004, E005 visible alignment, E005 PF, E005 trellis, E006, and the cross-fitted E009 wide-64 action around E006.

The grid contains affine, quadratic, mean-centered hinge, and smooth jump/fault corrections. Datum values are `[-60,60]` by 5 ft; linear and shape values are `[-120,120]` by 10 ft. This defines more than 100,000 globally fixed whole paths. Quadratic sufficient statistics accelerate exact grid scoring; they do not create truth-dependent paths. Rowwise switching is forbidden.

## Legal selectors

Every selector is refit inside each outer training set. Median imputation, scaling, correlation ranking, and feature truncation are also training-only. Each family supplies training-well targets consisting of its grid-oracle coefficients and `log1p(per-well RMSE)`. Validation-well truth is absent from fitting and family choice.

The six frozen branches are two ridge formulations, ExtraTrees, RandomForest, a fixed ridge/ExtraTrees average, and the same average with exact E006 fallback unless predicted gain is at least 0.25 ft. The legal feature table contains visible-prefix pseudo-cuts at 0.50, 0.70, and 0.85.

## Controls and edge cases

The run is invalid unless parent hashes, ID identity, no-op, duplicate candidates, basis reconstruction, pooled SSE, oracle positive, shuffled/sign-flipped negatives, duplicate selector, exact E006 fallback, split membership, finite bounds, runtime, memory, and clean reproduction pass.

Tests cover one-row through long suffixes, singular bases, clipping, equal-loss ties, short prefixes, missing GR, flat/duplicate/short typewells, repeated motifs, invalid MD/prefix contracts, HMM normalization and rate degeneracy, DTW monotonic bounds and flat costs, malformed IDs, and resource-gate invalidation.

## Promotion and rejection

Promotion requires bank oracle below 5.0; legal gain at least 0.15 versus E006 and 1.0 versus last-known; at least 4/5 map wins and 17/25 cell wins; p90 and worst-tail limits; positive every spatial and typewell group; special-slice limit; all controls; independent reproduction; final placement; packaging; and parity.

If the bank oracle is above 5, reject as a generation failure even if a selector improves. If the oracle is below 5 but selectors fail, reject as a selector failure and retain E006. No Kaggle submission is authorized.
