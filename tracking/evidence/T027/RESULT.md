# T027 Result — Target-preserving causal feature-view worth screen

Status: **completed; rejected; independently reproduced; no formal experiment; no neural model; no Kaggle execution; no submission**

Implementation commit: `f6bce2caa5e886723153ba9231e1c3cf1945f765`

## Decision

Reject H020 as tested. Do not retune the frozen ten-view mask catalog, view indicators, source-well weighting regimes, model settings, or placement grid. Exact E011 remains deployment primary.

## Complete result

T027 completed all 21 branches, four placements, 25 repeated contexts, ten spatial/typewell contexts, four corrupted-control families, duplicate-original equivalence, eight slice groups, and 22 edge groups. It covered all 773 wells and 3,783,989 hidden rows.

The strongest overall candidate is the ineligible original-view comparator `ridge_original_a1__w0.50`:

- RMSE: **12.4588330369**
- gain versus E011: **+0.0919232588**
- gain versus T025 best: **-0.0339054284**
- map wins: **4/5**
- repeated outer-cell wins: **16/25**
- p90: **17.6326521418** versus E011 **17.8310239265**
- worst-5% SSE share: **0.3060814125** versus E011 **0.3153217243**

The strongest eligible augmented-view branch is `ridge_backtest_cumulative_a1__w0.25`:

- RMSE: **12.5918611025**
- gain versus E011: **-0.0411048068**
- map wins: **0/5**
- repeated outer-cell wins: **5/25**

Every augmented-view candidate is worse than E011. No formal experiment is authorized.

## Transfer

The positive original-only comparator improves all five horizon quintiles and all three global special slices, but fails spatial group 0 by **-0.2840904830** and typewell groups 0, 2, and 4, with typewell group 4 at **-0.3528384927**.

The best eligible view branch regresses four of five spatial groups, four of five typewell groups, every horizon quintile, long suffix, high-GR-missingness, and the E011-catastrophe slice. Its worst spatial and typewell gains are **-0.1723562245** and **-0.1777428523**.

## Interpretation

T027 isolates the augmentation question cleanly. Predicting the fixed original-boundary target from the unmodified 142-feature vector has modest aggregate value. Duplicating that target across masked causal subviews does not regularize the model; every registered view pool degrades exact E011. The failure is therefore no longer target mismatch. It is a feature-view covariate-shift and effective-information-loss problem.

The original-only comparator's positive aggregate result is not deployable because it misses the frozen 0.15 gain, 17-cell, T025, and every-domain-transfer gates. It may be used only as evidence for a future domain-robust training objective, not as a new deployment candidate.

## Controls and edge cases

All four corrupted controls lose; the strongest corrupted-control gain is **-0.1703438041**. Duplicate-original training reproduces the original-only coefficients within **1.256e-11**. All 735 isolation audits, source-well total-weight checks, exact fallback, feature schema, and bounded-finite controls pass.

All **22 edge groups** pass, including registered feature blocks, view catalog, missing and duplicate schema, all-missing imputation, source-weight normalization, original boost, held-out exclusion, tiny partitions, KNN/tree support, target and prediction finiteness, coefficient bounds, exact fallback, contexts, branch coverage, and duplicate OOF IDs.

## Reproduction

A clean independent run reproduces every substantive CSV and edge file byte-for-byte. The summaries are exactly equal after excluding measured runtime.

- official runtime: **2070.793 seconds**
- reproduction runtime: **2045.207 seconds**

## Durable boundary

Do not reopen target-preserving feature masking by adding view types, retuning masks, changing indicators, or using a larger model. A future path must change the training objective rather than the input duplication mechanism. The evidence-supported next question is whether one original view per well can be trained with domain-robust, group-balanced objectives that improve spatial/typewell worst-group transfer without subgroup routing at inference.
