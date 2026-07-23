# T023 Result — Residual sequence-state worth screen

## Decision

**Reject H017 as tested and do not authorize a GRU, TCN, HMM/PF extension, new path bank, deployment notebook, Kaggle run, or competition submission.**

E011 residuals are extremely smooth and admit large hidden-label spline-oracle headroom, but the preregistered legal sequence proxies do not isolate a stable signed state mechanism. Four branches pass the repeated-context preliminary screen, yet none passes the frozen authorization contract. The strongest transferable branch fails one spatial holdout, and a reversed-profile negative control retains material gain above its cap.

E011 remains the deployment primary, E006 the secondary fallback, and E004 the exact fallback.

## Frozen boundary and completion

- Preregistration commit: `121146dcf5d95bd8d50440cb8967ea5c3c83b79a`
- Sealed implementation commit: `f3a350411735b664d1bc48ef94d9d089110e38d5`
- Implementation SHA-256: `9c0b4b53053a5e038e13d5ca3ed56cf42a1bd2c1719be28baacc02e2587b51d5`
- Wells: 773
- Hidden rows: 3,783,989
- Registered branches: 12, including 3 negative controls
- Repeated contexts: 25
- Preliminary passers: 4
- Stress contexts completed for every passer: 10 each
- Candidate/context metric rows: 325/325
- Stress metric rows: 50/50
- Nested selection rows: 340
- Edge groups: 12/12 passed
- Main runtime: 612.815 seconds
- Independent clean-cache runtime: 597.683 seconds
- No Kaggle execution or competition submission occurred.

## Sequence capacity diagnostics

Residual persistence is real and very strong:

| Diagnostic | Mean | p10 | Median | p90 |
|---|---:|---:|---:|---:|
| ACF lag 1 | 0.999980 | 0.999966 | 0.999993 | 0.999998 |
| ACF lag 16 | 0.998158 | 0.996501 | 0.998620 | 0.999458 |
| ACF lag 64 | 0.975232 | 0.950889 | 0.980196 | 0.992354 |
| ACF lag 256 | 0.742948 | 0.522649 | 0.783951 | 0.912731 |
| Low-frequency energy fraction | 0.982777 | 0.958423 | 0.989769 | 0.998029 |

E011 residual SSE decomposes into 57.44% per-well datum, 15.66% linear trend, and 26.91% remaining shape.

The hidden-label control-point oracle confirms capacity, not legal predictability:

| Controls | Oracle RMSE | Gain versus E011 |
|---:|---:|---:|
| 4 | 4.0810227602 | 8.4697335355 |
| 8 | 2.3313504848 | 10.2194058109 |
| 16 | 1.1867850335 | 11.3639712622 |
| 32 | 0.6380823934 | 11.9126739023 |
| 64 | 0.3976425826 | 12.1531137131 |

The 16-control oracle gains 11.3639712622 RMSE and the 32-control oracle adds another 0.5487026401. Both frozen capacity gates pass strongly.

## Legal branch screen

The best raw pooled branch is `profile_ridge_b32` at 12.1736417960 RMSE, a 0.3771144997 gain with 5/5 maps and 23/25 cells. It is not a preliminary passer because p90 rises from 17.8310239265 to 18.1980019588 and worst-5% SSE share rises from 0.3153217243 to 0.3232278402.

The strongest branch that passes every preliminary repeated-context gate is `block_ridge_b32`:

- RMSE: 12.2651012209
- gain versus E011: 0.2856550748
- map wins: 5/5
- outer-cell wins: 24/25
- p90: 17.6548522347 versus E011 17.8310239265
- worst-5% SSE share: 0.3197026620 versus E011 0.3153217243
- most repeated contexts select ridge alpha 1.0, shrinkage 0.5, and a 10-ft correction cap.

Its stress gains are:

```json
{
  "spatial:0": 0.32112273903248756,
  "spatial:1": 0.3828131768773648,
  "spatial:2": 0.43610639864277445,
  "spatial:3": 0.1492865924462592,
  "spatial:4": -0.1711370494914135,
  "typewell:0": 0.47347492921551293,
  "typewell:1": 0.30753157127838904,
  "typewell:2": 0.00029693164843536124,
  "typewell:3": 0.09594768340243576,
  "typewell:4": 0.07712665538650043
}
```

It improves all five typewell holdouts, but spatial group 4 regresses by 0.1711370495 RMSE. Therefore the every-spatial authorization gate fails.

Its special-slice gains remain positive:

```json
{
  "e011_catastrophe": 0.48704972208836494,
  "high_gr_missingness": 0.37301274971046183,
  "long_suffix": 0.31981603141961834
}
```

The other three preliminary passers also fail spatial and/or typewell transfer. Both KNN branches additionally fail the zero-p90-deterioration authorization gate.

## Negative controls and interpretation

The shuffled-well profile and shuffled-GR/typewell controls lose their gain as required:

- `profile_ridge_b16_shuffled_wells`: 0.0058793738 RMSE gain
- `typewell_block_ridge_b16_shuffled_gr`: -0.0129732118 RMSE gain

The reversed-profile control does not fail:

- `block_ridge_b16_reversed_targets`: 12.4178084493 RMSE
- gain versus E011: 0.1329478464
- maps/cells: 5/5 and 24/25
- frozen maximum negative-control gain: 0.03

Because the reversed signed residual profile still gains 0.1329478464, some apparent benefit is attributable to generic smooth bounded profile movement, shrinkage, or shared profile magnitude rather than correctly recovered signed sequence state. This control failure alone blocks authorization, even before the spatial failures.

The correct conclusion is not that sequence structure is absent. It is that the tested legal profile, block, KNN, HGB, typewell-only, and disagreement-only proxies do not prove a transportable signed mechanism beyond E011.

## Controls and edge cases

The following correctness controls pass:

- source boundary and required inputs;
- 773-well / 3,783,989-row ordered coverage;
- exact zero-correction E011 fallback;
- finite bounded corrections and metrics;
- 325/325 repeated candidate/context rows;
- 50/50 required stress rows;
- deterministic selections and complete nested coverage;
- oracle capacity gate;
- all 12 edge groups.

The negative-control suite fails only because the reversed-profile branch exceeds the frozen 0.03 cap. That is a scientific falsifier, not an execution defect.

The twelve edge groups cover one/two-row suffixes, all-missing GR, constant GR, typewell extrapolation, duplicate typewell TVT values, rejection of non-monotone/duplicate MD, rejection of non-finite base paths, zero-variance features, constant target profiles, tiny KNN partitions, exact boundary/cap enforcement, and full real-data ID/order/block coverage.

## Independent reproduction

A second run rebuilt the raw-data sequence cache and refit every branch, inner selection, repeated context, stress context, control, and edge case.

- Main cache SHA-256: `f7486e680ac3967bb507ebd6b9165dbe449535d2c8406a606bf4ff090c1f4a7a`
- Reproduction cache SHA-256: `f7486e680ac3967bb507ebd6b9165dbe449535d2c8406a606bf4ff090c1f4a7a`
- Cache size: 44,769,818 bytes
- Caches are byte-identical.
- Results are equal field-for-field after excluding measured `runtime_seconds`.
- Candidate summaries, selections, stress rows, controls, edge cases, status, and decision are exactly equal.

## Closure

Close H017 under D025. Do not retune the same residual-profile targets, change only the model family, or launch a neural/state/path implementation from this result. Any future sequence proposal must introduce materially different causal evidence or a different legal state target and must again defeat reversed/shuffled controls plus every spatial/typewell holdout before escalation.
