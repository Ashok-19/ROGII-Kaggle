# E006 Result — Nested PF–E004 Fusion and Disagreement Evidence

Status: **completed; promoted; deployment-ready; no leaderboard submission**  
Run: `R20260719-1126-e006-nested-fusion`  
Final frozen design commit: `3735efbaec45e4e17679a9e354cade516b8d1425`  
Frozen scoring commit: `a78ebc5783e67a37a5b801af58ad2bd6376c3f00`  
Final audit commit: `7bd1633310a13d3c19af556eb1b6e131b18c2bf3`

## Decision

Promote `nested_conservative_grid` as the primary deployment-ready surface-free candidate. It scores **14.9331407872 RMSE**, improving last-known TVT by **0.9767120835** and E004 by **0.5581656110**. It wins all **5/5** repeated maps and all **25/25** repeated outer cells, improves every nested spatial and typewell group, and passes every frozen tail, stability, identity, negative-control, runtime, memory, and deployment gate.

E004 remains the exact no-GR/PF-abstention fallback. A valid 14,151-row submission artifact was created for parity testing, but **no leaderboard submission was made**.

## Strict evidence boundary

The observed E005 50/50 diagnostic was motivation only. E006 did not reuse E005 PF rows in nested scoring. For every inner and outer cell, E004 was refitted without the validation wells and the frozen PF algorithm was recomputed around that matching cell-specific E004 path. The outer fold was absent from every inner E004 fit, weight-selection target, reliability reference, and routing calculation.

All 35 membership contexts pass: 25 repeated outer cells, five spatial groups, and five typewell groups. Outer/train overlap and inner/outer-test overlap are zero in every context.

## Candidate ladder

| Candidate | Eligible | RMSE | Gain vs E004 | p90 well RMSE | Worst-5% SSE share |
|---|---:|---:|---:|---:|---:|
| `nested_rmse_grid` | no | 14.9076768681 | 0.5836295302 | 21.7810725596 | 0.3642330796 |
| `fixed_50_50_reference` | no | 14.9210952483 | 0.5702111499 | 21.7683043767 | 0.3678521213 |
| **`nested_conservative_grid`** | **yes** | **14.9331407872** | **0.5581656110** | **21.7855500884** | **0.3685061112** |
| `nested_disagreement_cap` | yes | 15.0024482708 | 0.4888581274 | 21.8280460837 | 0.3730663702 |
| nested PF path | diagnostic | 15.0196801656 | 0.4716262327 | 21.9324631069 | 0.3395476768 |
| `nested_reliability_shrink` | yes | 15.1893372157 | 0.3019691826 | 22.4115357700 | 0.3810863063 |
| E004 geometry-prefix | comparator | 15.4913063983 | 0.0000000000 | 22.4630210994 | 0.3830949846 |
| last-known TVT | comparator | 15.9098528707 | -0.4185464725 | 22.9725365671 | 0.3899072385 |

The lower-scoring fixed 50/50 and unconstrained inner-RMSE candidates remain permanently ineligible. Selection follows the frozen rule among candidates passing every gate: lowest mean repeated-map RMSE, then frozen candidate order.

## Stability and stress

- Registered map wins: **5/5**; per-map gains versus E004 are approximately `0.6208`, `0.4519`, `0.4890`, `0.4843`, and `0.3948` RMSE.
- Repeated outer-cell wins: **25/25**.
- Minimum nested spatial-group gain versus E004: **0.1996631444**.
- Minimum nested typewell-group gain versus E004: **0.2456515254**.
- Long-suffix improvement versus E004: **0.7373409176**.
- High-GR-missingness improvement versus E004: **0.6994400160**.
- Ambiguous-alignment improvement versus E004: **0.6288851777**.
- Outer-cell base weights are confined to `0.375`, `0.5`, and `0.625`; range **0.25**. Every map median is **0.5**.
- Selected residual correlation to E004: **0.9787366461**.

The selected candidate reduces p90 from E004's `22.4630210994` to `21.7855500884`, maximum well RMSE from `68.8470020296` to `64.8699823667`, and worst-5% SSE share from `0.3830949846` to `0.3685061112`.

## Controls

All registered controls pass:

- Frozen data signature and all 773 wells match.
- Exactly 3,783,989 unique hidden-row IDs are emitted.
- Nested E004 reproduces `15.491306398267565` within `1e-9`.
- Parent E005 OOF hash is exactly `5115f7ce...e13ae5`, and PF RMSE reproduces within `1e-9`.
- Zero weight is an exact E004 no-op; duplicate fusion differs by `0.0`.
- Direct and well-aggregated SSE agree within `1.1861e-13` relative error.
- Deterministically shuffled PF loses all gain and returns E004 RMSE.
- The rowwise oracle remains ineligible and beats the best legal candidate by more than the frozen margin.
- All missing GR and PF abstention fall back exactly to E004.
- No forbidden surface columns, network calls, external model artifacts, or third-party runtime packages occur in deployment code.

## Complete audit and edge cases

The audit identified and corrected latent reporting/deployment-contract defects without changing any scored prediction:

1. Parent-PF RMSE tolerance could be hidden by a later hash-only `pass` field overwrite.
2. Candidate `passed_all_gates` rows were populated before runtime/memory gates were appended.
3. Deployment model/path settings lacked complete fail-fast validation for malformed, non-finite, unavailable-candidate, and path-length cases.
4. The Learning Lab test referenced a missing `nested_fusion` visual.
5. Canonical summary, manifest, source hashes, and dashboard state still described the pre-deployment phase.

The expanded E006 suite passes **20/20** under Python 3.10, 3.12, and 3.13. It covers malformed models/configuration, non-finite diagnostics and weights, equal percentile ties, extreme disagreement, empty test directories, missing/duplicate/degenerate typewells, wrong-length paths, exact all-missing-GR fallback, duplicate/missing/extra/wrong-header sample IDs, deterministic serialization, runtime/memory gate failure, and training/deployment PF equivalence.

Additional property checks pass over 1,000 randomized blend-statistic cases, 72 template lengths, and 20 real wells. Training and deployment PF paths are exactly equal on those 20 wells. The installed coverage launcher is broken because it points to a missing `/usr/bin/python`; branch behavior was therefore exercised through explicit regression and property tests rather than a coverage percentage.

## Reproducibility

- Official scoring run: **42:55.97**, maximum RSS **228,668 KB**.
- Independent clean-output reproduction: **42:49.31**, maximum RSS **228,128 KB**.
- Final audited-code replay: **44:42.73**, maximum RSS **228,548 KB**, exit status `0`.
- Frozen runtime limit: **45 minutes** — passed by the final audited replay.
- Frozen memory limit: **1,536 MB** — passed.
- OOF predictions are byte-identical across canonical and final replay: **200,145,240 bytes**, SHA-256 `9b9f26cd8ad82e32dbc0f9ba1466cb38424cd3bbf7001c1079a9867fb5a324b4`.
- All static scored CSVs are byte-identical. `control_metrics.csv` differs only in measured runtime/RSS; summaries are semantically identical after normalizing commit, runtime, and deployment metadata.
- Independent row audit verifies 3,783,989 unique IDs, all finite values, 773 contiguous wells, candidate RMSE agreement within `4.1e-12`, and zero duplicate-control delta.

## Deployment package and Kaggle parity

The promoted rule applies the unchanged conservative full-fit selector to the frozen full-training E004/PF OOF evidence. An independent implementation of the grid and tail constraints also selects deployment weight **0.5**.

- Model: `experiments/E006/results/model.json`, SHA-256 `bee16499c6361395d556dd5c4ff8555e95739c36f68444b3d0dfcf648d73cddf`.
- Notebook: `notebooks/e006_deployment.ipynb`, SHA-256 `6425de71ecceb9c5819fcc1a9ed6ec4c702ed176232b3be0326226f91d9e5750`.
- Package size: **92,160 bytes**, far below the 100 MB gate.
- Direct, locally executed notebook, separately rebuilt package, and private Kaggle version 3 outputs are raw-byte-identical.
- Submission rows: **14,151** across **3** authoring wells; bytes: **367,933**.
- Submission SHA-256: `e412864a6721ab4e13575355f37188aa9ddf19223bd46fe33224a8b536d81008`.
- Kaggle kernel: private CPU, Python 3.12, internet disabled, official competition source only, no dataset/kernel/model sources.
- Leaderboard submission created: **no**.

## Next action

Retain E006 as the primary deployment-ready surface-free family and E004 as exact fallback. Pre-register E007/H005 horizontal self-correlation as a genuinely independent candidate with template-shuffle, repeated-fold, tail, spatial/typewell, runtime, and residual-correlation controls. Do not start E007 until this E006 audit record and repository validation are committed.
