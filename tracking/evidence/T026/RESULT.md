# T026 Result — Randomized long-horizon mask-task coefficient meta-learning worth screen

Status: **completed; rejected; independently reproduced; no formal experiment; no neural model; no Kaggle execution; no submission**

Implementation commit: `71ac3e456e71e9a2a6149b2cf85538ef09c9d6bd`

## Decision

Reject H019 as tested. Do not retune the frozen 12,216-task end-of-well mask pool, target definition, feature family, weighting regimes, model settings, or placement grid. Exact E011 remains the deployment primary.

## Complete result

T026 completed all 19 registered branches, all four placements, all 25 repeated contexts, all ten spatial/typewell contexts, all four corrupted-control families, all registered slices and horizon quintiles, and 25 edge groups. It covered all 773 wells and 3,783,989 ordered hidden rows.

The strongest of 76 fixed candidates is `ridge_all_combined_a1_well_equal__w0.25`:

- RMSE: **12.6256511675**
- gain versus E011: **-0.0748948718**
- gain versus T025 best: **-0.2007235590**
- map wins: **0/5**
- repeated outer-cell wins: **8/25**
- p90: **18.2634625025** versus E011 **17.8310239265**
- worst-5% SSE share: **0.3254277327** versus E011 **0.3153217243**

Every registered branch family is worse than E011 at its strongest placement. Therefore the result is not a near-pass and no formal experiment is authorized.

## Verified capacity and transfer

The last-known plus four-control hidden-label spline oracle is reconfirmed at **4.1137431295 RMSE**. The representation therefore retains sub-5 capacity, but the tested mask-task learner cannot recover it legally.

The reported candidate's spatial gains are:

```text
spatial:0    +0.0087811986
spatial:1    -0.0813586446
spatial:2    -0.2605218203
spatial:3    +0.1790819469
spatial:4    -0.2561035751
```

Its typewell gains are:

```text
typewell:0   -0.0713948892
typewell:1   -0.2013589994
typewell:2   +0.1004884593
typewell:3   -0.2993239855
typewell:4   +0.1022186122
```

It improves only one of five frozen hidden-horizon quintiles. It also regresses the E011-catastrophe slice by 0.1548561565 RMSE and the long-suffix slice by 0.0754623169 RMSE.

## Task-distribution diagnostic

The 11,443 randomized task targets remain correlated with each source well's original-boundary coefficients: **0.5681, 0.7354, 0.7708, and 0.8225** across the four outputs. However, their pooled coefficient RMSE from the source well's original target is **14.2789224568 ft**.

Mask augmentation improves the weak original-only comparator from RMSE 12.9101892120 to 12.6256511675, a within-family improvement of 0.2845380445. It still fails to match E011. The evidence therefore supports a target-mismatch diagnosis: more views help within the weak family, but end-of-well mask coefficients are not interchangeable with the real original-boundary target.

## Controls, isolation, and edge cases

All four corrupted-control families lose. The strongest corrupted control gain is **-0.2374447517**, safely below the +0.03 cap.

All 665 branch-context isolation audits pass. No held-out-well task enters fitting, imputation, scaling, weighting, bin construction, or model selection. The exact E011 fallback, ordered OOF coverage, finite bounded coefficients, and 12,216-task pool checks all pass.

All **25 edge groups** pass, covering support boundaries, deterministic jitter, duplicate cuts, contiguous visibility, strict MD, malformed/non-finite inputs, missing GR, rank-deficient splines, zero-variance features, task weighting, empty bins, tree weights, KNN clipping, coefficient bounds, held-out task exclusion, duplicate OOF IDs, deterministic ties, and exact fallback identity.

## Reproduction and execution history

A clean independent run reproduces every substantive CSV, edge file, and task-diagnostic file byte-for-byte. The normalized summaries are exactly equal.

- official runtime: **2141.155 seconds**
- reproduction runtime: **2425.660 seconds**

Before the official run, process inspection found an older implementation worker and the definitive worker writing to the same scratch directory. Both were terminated and that directory was deleted. The official run then started alone in `main_clean`. No output from the contaminated directory was inspected, retained, or promoted.

## Durable interpretation

T026 falsifies the proposition that end-of-well mask-task coefficients can be pooled as direct substitutes for original-boundary coefficient targets, even with horizon matching, per-well weighting, joint balancing, simple nonlinear models, and exact outer-well isolation.

A future path must change the target relationship rather than add more masks or a larger architecture. The next admissible worth question is whether causal prefix views can predict the **same original-boundary target** while retaining full E011 feature parity. That is a new target-preserving view-learning hypothesis and must be separately preregistered before implementation.
