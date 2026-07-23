# T022 Result — E011 uncertainty and fallback worth screen

## Decision

**Reject H016 and do not preregister E013.**

The screen verifies large hidden-label oracle headroom between E011 and E006, but none of the legal cross-fitted uncertainty placements recovers the frozen minimum gain or transfer stability. E011 remains unchanged as deployment primary.

## Frozen boundary

- Source decision/preregistration commit: `33e2de77dc044832f71de294f69430f3b5d2e79c`
- Wells: 773
- Hidden rows: 3,783,989
- Repeated contexts: 25
- Spatial/typewell stress contexts: 10
- Candidate/context rows completed: 350
- Full screen runtime: 888.676 seconds
- No Kaggle notebook execution or competition submission occurred.

The first invocation failed before reading data because the scratch script resolved the repository root one directory too high. Only that path expression was corrected before the complete run.

## Oracle headroom

| Path | RMSE | Gain versus E011 |
|---|---:|---:|
| E011 | 12.5507562957 | — |
| E006 | 14.9331407872 | -2.3823844915 |
| Per-well E011/E006 oracle fallback | 11.7920303042 | 0.7587259915 |
| Per-well continuous blend oracle | 11.7172383371 | 0.8335179586 |

E006 wins 281/773 wells by per-well RMSE. A positive continuous oracle weight exists for 382 wells and reaches full E006 weight for 190. Thus the D022 0.20-RMSE headroom gate passes strongly; the failure is legal identification, not absence of opportunity.

## Legal screen

Completed branches:

- featureless constant blend;
- hard logistic exact fallback;
- soft logistic blend;
- equal-well ridge weight;
- row-weighted ridge weight;
- ExtraTrees weight;
- gradient-boosted weight;
- KNN weight;
- shuffled row-weighted ridge control;
- shuffled ExtraTrees control.

All hyperparameters, thresholds and shrinkages were selected by inner OOF within each untouched outer context.

The strongest legal branch is `logistic_hard`:

- RMSE: 12.5438879467
- gain versus E011: 0.0068683490
- map wins: 2/5
- outer-cell wins: 5/25
- final action fraction: 0.028461
- mean fallback weight: 0.006468
- p90: 17.8310239265 versus E011 17.8310239265
- worst-5% SSE share: 0.3151125580 versus E011 0.3153217243

Its inner fallback-class AUC is effectively random: median about 0.510, with observed context range 0.464–0.574. Most inner selections choose threshold 0.8 and emit exact E011.

Stress gains for `logistic_hard`:

```json
{
  "spatial:0": 0.0,
  "spatial:1": 0.0,
  "spatial:2": 0.0,
  "spatial:3": 0.0,
  "spatial:4": 0.0,
  "typewell:0": -0.1794802746444084,
  "typewell:1": 0.0,
  "typewell:2": 0.0,
  "typewell:3": 0.0,
  "typewell:4": 0.07410369811153394
}
```

Special-slice gains:

```json
{
  "e006_oracle_wins": 0.042129457182795704,
  "e011_catastrophe": 0.01414697530885789,
  "high_gr_missingness": -0.02847527854276599,
  "long_suffix": 0.015607038551282315
}
```

It regresses typewell group 0 by about 0.1795 RMSE and the high-GR-missingness slice by about 0.0285 RMSE. It fails the 0.03 gain, 4/5 map, 17/25 cell, spatial, typewell and special-slice gates.

Every continuous branch is worse than E011. Inner selection chooses zero for the featureless constant blend, confirming that unconditional movement toward E006 is not useful.

## Controls and edge cases

All controls pass:

- exact source boundary;
- 773-well / 3,783,989-row OOF coverage;
- exact E011 fallback identity;
- 350/350 candidate-context rows;
- finite metrics and weights in [0, 1];
- shuffled ridge gain -0.0249218756;
- shuffled ExtraTrees gain -0.0242128552.

A deterministic reproduction refit all ten candidates over all 35 contexts from the recorded selections. Context RMSE/SSE, final mean weights/RMSE and oracle metrics match at exact zero delta.

Ten focused edge groups pass: identical E011/E006 paths, numerical negative-SSE tolerance, material negative-SSE rejection, missing and non-finite weights, all-missing/constant columns, constant fallback labels, tiny KNN partitions, non-finite model output rejection, weight clipping, and full real-data membership/coverage.

## Interpretation

E011/E006 oracle fallback is valuable only with hidden knowledge. The current 151 legal features—including direct E011/E006 disagreement statistics—cannot identify that knowledge under repeated whole-well validation. This reproduces the project’s durable distinction between risk/headroom and safe signed action.

Do not retune thresholds, add more classifiers, or reopen hard/soft fallback around this feature space. A future path must generate materially new state or sequence evidence rather than route between E011 and E006.
