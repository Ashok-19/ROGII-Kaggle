# E004 Result — Surface-Free Deployment Candidate and Ablations

Status: **completed; deployment-ready; no leaderboard submission**  
Run: `R20260719-0636-e004-deployment`  
Code SHA: `c728a9a267a89c92e5c0c6b018db1a1658874158`

## Outcome

E004 discovered a decisive deployment-contract limitation: the six formation-surface columns used by the strongest E003 OOF model (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, and `BUDA`) exist in training horizontal wells but are absent from competition test horizontal wells. Test typewells also omit the training-only `Geology` annotation.

The full E003 score of **10.9279740918 RMSE** is therefore not a deployable competition inference result as implemented. E004 removed every unavailable field, evaluated eight pre-registered surface-free family combinations, trained a frozen full-data model, generated an exact-order submission, and built a self-contained offline notebook.

The best deployable local candidate is `geometry_prefix` at **15.4913063983 RMSE**, improving last-known TVT by **0.4185464725**. It passes repeated folds, tail limits, and contiguous spatial stress, but is far weaker than the surface-assisted E003 diagnostic result.

All deployment gates now pass. The exact notebook bytes recorded in this repository completed in a private Kaggle CPU kernel with internet disabled and the official competition source attached. Local direct inference, local notebook inference, and Kaggle inference produced a byte-identical 14,151-row submission.

## Actual competition input contract

| Input | Train horizontal | Test horizontal | E004 inference |
|---|---:|---:|---:|
| MD, X, Y, Z | yes | yes | allowed |
| GR | yes | yes | allowed |
| TVT_input visible prefix | yes | yes | allowed |
| TVT hidden truth | yes | no | forbidden |
| Six formation surfaces | yes | no | forbidden |
| Typewell TVT and GR | yes | yes | allowed |
| Typewell Geology | yes | no | forbidden |

This contract is verified directly from the local competition files. The three visible test wells are train-derived notebook-authoring examples and are not hidden-test generalization evidence.

## Surface-free ablation ladder

| Candidate | Families | Pooled RMSE | Gain vs baseline | Registered map wins |
|---|---|---:|---:|---:|
| **geometry_prefix** | geometry + visible prefix | **15.4913063983** | **0.4185464725** | **5/5** |
| no_gr | geometry + prefix + typewell + spatial | 15.5014518809 | 0.4084009898 | 5/5 |
| no_typewell | geometry + prefix + GR + spatial | 15.5127811078 | 0.3970717629 | 5/5 |
| geometry_prefix_gr | geometry + prefix + GR | 15.5143348086 | 0.3955180622 | 5/5 |
| full_deployable | all five deployable families | 15.5251664631 | 0.3846864077 | 5/5 |
| no_spatial | geometry + prefix + GR + typewell | 15.5266596490 | 0.3831932217 | 5/5 |
| no_geometry | prefix + GR + typewell + spatial | 15.5825636374 | 0.3272892334 | 4/5 |
| no_prefix | geometry + GR + typewell + spatial | 15.8215317579 | 0.0883211129 | 0/5 |

Visible-prefix diagnostics are essential. In this fixed ridge formulation, adding GR summaries, typewell summaries, or absolute spatial coordinates does not improve the geometry-plus-prefix core. This does not establish that raw GR alignment is useless; E004 tests summary features, not a path-search or sequence-alignment candidate.

## Repeated-map results

`geometry_prefix` improves every frozen map:

| Map | RMSE | Gain |
|---|---:|---:|
| v1 | 15.6405525662 | 0.2693003045 |
| v2 | 15.4715365686 | 0.4383163021 |
| v3 | 15.3883481018 | 0.5215047690 |
| v4 | 15.6156395235 | 0.2942133472 |
| v5 | 15.5605534049 | 0.3492994658 |

The five-map mean RMSE is **15.5353260330**.

## Spatial and tail stress

Contiguous X-block stress remains positive:

- Spatial RMSE: **15.5767054107**.
- Spatial gain: **0.3331474600**.
- Spatial p90 well RMSE: **22.7724430468**.
- Spatial worst-5% SSE share: **38.3684%**.

Aggregate tail metrics also improve or remain controlled:

| Metric | Last-known baseline | E004 geometry+prefix |
|---|---:|---:|
| Median well RMSE | 10.6651407867 | **10.1016293787** |
| p90 well RMSE | 22.9725365671 | **22.4630210994** |
| p95 well RMSE | **29.0107644333** | 29.3421252812 |
| Maximum well RMSE | 70.6393746197 | **68.8470020296** |
| Worst-5% SSE share | 38.9907% | **38.3095%** |
| Worst-10% SSE share | 52.4763% | **52.1275%** |

The p95 deterioration is recorded, but p90 and registered worst-tail gates pass.

## What remains predictable without surfaces

For the selected surface-free model:

- Datum Pearson correlation: **0.269164**.
- Datum Spearman correlation: **0.228620**.
- Material datum-sign accuracy: **63.23%** over 465 wells.
- Trend Pearson correlation: **0.058043**.
- Trend Spearman correlation: **0.104146**.
- Material trend-sign accuracy: **54.74%** over 570 wells.
- U-slope-delta Pearson correlation: **0.969842**.

The U-slope change remains geometrically predictable, but that does not provide enough information to recover hidden TVT level and trend accurately. The large E003 signed-action result depended primarily on unavailable surface-derived evidence.

## Frozen deployment package

The full-training model uses 48 selected features from only:

- complete-well trajectory and hidden-zone geometry;
- visible TVT/U prefix level, slope, range, and stability;
- visible-prefix pseudo-holdout errors;
- suffix length and direction.

It requires:

- no surface columns;
- no `Geology` annotation;
- no internet;
- no external model dataset;
- no third-party Python package.

Artifacts:

- Model: `experiments/E004/results/model.json`.
- Notebook: `notebooks/e004_deployment.ipynb`.
- Local direct submission: `artifacts/E004/submission.csv`.
- Local notebook submission: `artifacts/E004/notebook_submission.csv`.
- Submission SHA-256: `62ae06575baa647a5e09686bc9369cbc7303ac391e5c028468593dd6b58d5279`.
- Notebook SHA-256: `28e41432422d078628c4bd41688c641e9b97c54a591a998384e638c4c0d81dd7`.
- Remote parity evidence: `experiments/E004/results/remote_parity.json`.

The notebook dynamically locates `sample_submission.csv` under the Kaggle input mount and reconstructs predictions in the exact sample order.

## Remote Kaggle parity

The exact self-contained notebook was executed as private Kaggle kernel `ashok205/e004-deployment-remote-parity-exact`, version 4:

- Worker status: **COMPLETE**.
- Runtime: Kaggle Python 3.12 CPU image.
- Internet: **disabled**.
- Competition source: official `rogii-wellbore-geology-prediction` mount.
- Notebook bytes: **44,774**.
- Output rows: **14,151** across **3** authoring wells.
- Output bytes: **367,933**.
- Local direct, local notebook, and remote Kaggle SHA-256: `62ae06575baa647a5e09686bc9369cbc7303ac391e5c028468593dd6b58d5279`.
- Raw byte parity: **passed**.
- Leaderboard submission created: **no**.

The first successful remote run revealed a cross-runtime serialization edge case: 190 rows differed only in the last printed digit, with maximum absolute numerical difference `1.000444171950221e-10` ft and prediction-delta RMSE `1.1590274320318884e-11` ft. The model and equations were unchanged; submission serialization was frozen at five decimal places, retaining `0.00001` ft resolution. This produced exact byte parity across local and Kaggle runtimes. Deterministic notebook cell IDs were also added, eliminating a future `nbformat` compatibility warning.

All seven local authoring-fixture files were independently downloaded from Kaggle competition endpoints and verified byte-identical before the final remote run.

## Edge cases and controls

Verified behavior includes:

- Missing formation surfaces are detected and never imputed as if available.
- Missing test `Geology` is harmless because it is unused.
- Entirely missing GR is accepted with explicit missingness features.
- Non-contiguous TVT_input visibility is rejected.
- Non-increasing MD is rejected.
- Missing horizontal/typewell pairs are rejected.
- One-row hidden suffixes produce a finite centered prediction.
- Duplicate, missing, and extra sample IDs are rejected.
- Sample order is preserved exactly.
- Inconsistent model arrays and non-finite coefficients are rejected.
- All predictions and model values are finite.
- The three visible authoring wells are explicitly excluded from generalization claims.
- Cross-runtime CSV serialization is fixed to five decimal places and byte-identical on local Python and Kaggle Python 3.12.
- Every generated notebook cell has a deterministic ID, avoiding future `nbformat` hard failures.

## Reproducibility

- Official ablation/package run: **147.08 seconds**, maximum RSS **48,408 KB**.
- Independent clean-root run: **149.76 seconds**, maximum RSS **45,948 KB**.
- Fifteen result, notebook, and submission files were byte-identical.
- Total compared bytes: **990,477**.
- Direct and notebook submissions are byte-identical over **14,151 rows**.
- The exact private Kaggle output is also byte-identical to both local outputs.
- Final submission bytes: **367,933**; SHA-256: `62ae06575baa647a5e09686bc9369cbc7303ac391e5c028468593dd6b58d5279`.
- Visible-test selected-feature maximum absolute standardized value: **2.5019**.
- Visible-test selected features outside the training range: **0 cells**.

## Decision

Retain `geometry_prefix` as a weak but fully deployment-ready deterministic fallback and candidate-bank leg. Do not submit it solely on the basis of 15.49 local CV. Do not describe the E003 10.93 surface-assisted score as deployable.

The next modeling work should reconstruct useful structural evidence from inputs that genuinely exist at test time: explicit typewell/horizontal GR alignment, particle filtering, trellis/dynamic programming, and candidate-path disagreement. E004 has no remaining deployment gate.
