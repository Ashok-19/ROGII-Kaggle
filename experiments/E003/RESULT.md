# E003 Result — Datum, Trend, and Risk Learnability

Status: **promoted as an OOF modeling substrate; not deployment-ready**  
Run: `R20260719-0535-e003-learnability`  
Code SHA: `fed2541ca1f6a3ebd1879885575efad4713c14f5`

## Outcome

E003 confirms that legal whole-well evidence can predict both signed low-order TVT correction and baseline-well risk under frozen whole-well validation. The pre-registered ridge datum-plus-trend action improves pooled RMSE from **15.9098528707** to **10.9279740918**, a gain of **4.9818787789**. It also beats the best training-fold-mean action by **4.9164503558**.

The deterministic shallow forest independently identifies expensive wells with **0.541500 Spearman**, **0.812465 AUC** for the worst 20%, and **55.35% of baseline SSE captured** by its predicted top 20% wells.

Both capabilities pass all registered controls and contiguous spatial-X stress. Risk and signed action remain separate outputs: the risk score is promoted as evidence, but it does not authorize routing or alter the correction by itself.

This is not yet a Kaggle submission. E003 validates cross-fitted coefficient prediction on training wells; a full-training inference package, test-distribution audit, exact row reconstruction, and Kaggle offline parity still need to be built.

## Signed action result

The selected action predicts two coefficients per well from legal inputs:

- an additive datum shift applied to the hidden suffix;
- a centered linear heel-to-toe trend correction.

The hidden path is reconstructed as the final visible TVT plus the predicted datum and centered trend. The predictor never receives hidden TVT at inference.

| Candidate | Pooled RMSE | Gain versus baseline |
|---|---:|---:|
| Last-known TVT | 15.9098528707 | 0.0000 |
| Fold-mean datum | 15.8444244476 | 0.0654 |
| Ridge datum | 12.1067852607 | 3.8031 |
| **Ridge datum + trend** | **10.9279740918** | **4.9819** |
| Forest datum | 12.8671730031 | 3.0427 |
| Forest datum + trend | 12.0492724017 | 3.8606 |

The selected ridge action improves every repeated map:

| Fold map | RMSE | Gain |
|---|---:|---:|
| v1 | 10.9589775408 | 4.9509 |
| v2 | 10.9649821022 | 4.9449 |
| v3 | 10.9338776456 | 4.9760 |
| v4 | 11.0454221368 | 4.8644 |
| v5 | 10.9850996200 | 4.9248 |

It also improves the tails:

- Median well RMSE: **8.0718**, versus 10.6651.
- p90 well RMSE: **15.9217**, versus 22.9725.
- p95 well RMSE: **19.4681**, versus 29.0108.
- Maximum well RMSE: **33.7270**, versus 70.6394.
- Worst-5% SSE share: **32.96%**, versus 38.99%.
- Worst-10% SSE share: **47.31%**, versus 52.48%.

Under five contiguous spatial-X blocks, ridge datum-plus-trend scores **11.2541337067 RMSE**, retaining a **4.6557191640** gain. This is worse than the repeated random-group maps but remains strongly directionally consistent.

## What is learnable

Averaged cross-fitted ridge target diagnostics are:

| Target | Pearson | Spearman | MAE | Material sign accuracy |
|---|---:|---:|---:|---:|
| Datum correction | 0.790869 | 0.759876 | 5.7367 ft | 88.39% on 465 wells |
| Trend correction | 0.866766 | 0.818820 | 8.0824 ft end-to-end | 89.47% on 570 wells |
| Heel-to-toe U-slope delta | 0.998393 | 0.997645 | 0.001035 ft/MD | 97.41% |

The nearly deterministic U-slope-delta signal is physically consistent with E002: supplied hidden Z and formation-surface movement explain how U should move, while the visible U trend supplies the heel anchor. The useful action is not naive U extrapolation; it is a learned correction to the low-order hidden TVT path.

The 146 legal features include supplied whole-well geometry, formation surfaces, GR availability and summaries, typewell summaries, visible TVT/U statistics, and visible-prefix pseudo-holdouts. Feature selection occurs inside each training fold. The most consistently selected groups are:

- hidden movement and slope of the six supplied formation surfaces;
- surface movement relative to Z;
- hidden Z delta and slope;
- visible U slope and stability across windows;
- visible-prefix pseudo-holdout U-slope changes;
- hidden trajectory direction and GR summaries.

Selection frequency is evidence of predictive use, not a causal claim.

## Risk result

The selected risk model is the deterministic shallow forest over the same legal features.

- Repeated-map averaged Spearman: **0.541500**.
- Worst-20% AUC: **0.812465**.
- Predicted/oracle top-20% overlap: **63.87%**.
- Predicted top-20% SSE capture: **55.35%**.
- Oracle top-20% SSE capture: 66.89%.
- Risk gates pass on all five maps.
- Spatial-block Spearman: **0.522376**.
- Spatial-block AUC: **0.807391**.

Risk is therefore useful for diagnostics, uncertainty features, review prioritization, and later ensemble design. It remains explicitly ineligible for signed routing unless a separately validated routing rule improves a fixed action or blend.

## Oracle headroom

Hidden-label oracle diagnostics remain ineligible for modeling:

- Baseline RMSE: 15.9098528707.
- Oracle datum-only RMSE: **9.0354099881**.
- Oracle datum-plus-linear-trend RMSE: **6.6971947701**.

The promoted ridge action reaches 10.9280. It captures a substantial part of the low-order headroom but still leaves approximately 4.23 RMSE above the datum-plus-trend shape floor.

## Controls

All nine official controls passed:

- Frozen data signature and 773-well integrity.
- Five unique frozen fold maps with exact well coverage.
- Oracle algebra reproducing the E001 datum and datum-plus-trend floors.
- Exact last-known no-op baseline.
- Synthetic positive correlation of 0.998839.
- Regularization-preserving duplicate-feature prediction delta of 1.56e-13.
- Shuffled datum correlation of 0.0406 and only 0.0159 gain over the fold-mean action.
- Hidden-label leakage sentinel marked ineligible.
- Runtime below the 20-minute budget.

Additional verification passed:

- all 13 generated files are byte-identical across independent full runs;
- an independent raw-row scorer reproduces RMSE within 8.1e-13;
- the legal feature table contains 146 columns and no hidden-target, oracle, or correction-label fields;
- TVT-named legal columns refer only to the visible prefix or the supplied typewell.

## Reproducibility

- Official full run: 173.14 seconds, maximum RSS 60,608 KB.
- Independent full run: 175.65 seconds, maximum RSS 60,396 KB.
- Total committed result size: 2,310,084 bytes.
- Exact hashes are stored in `experiments/E003/results/artifact_manifest.json`.
- Verification details are stored in `experiments/E003/verification.json`.

## Decision

Promote the E003 legal feature representation, ridge datum-plus-trend action, and forest risk score into the candidate bank. Do not submit E003 directly and do not treat the OOF score as public-leaderboard evidence.

The immediate next experiment should package this action as a clean full-training candidate interface, add explicit feature-family ablations and test-distribution checks, then compare it with independent physics/GR/PF/trellis candidates under the same OOF IDs. Heavy candidate training and full inference testing should run through Kaggle MCP after exact code and configuration are committed.
