# ROGII Gold Medal Roadmap

Updated: 2026-07-20  
Final deadline: 2026-08-05 23:59 UTC / 2026-08-06 05:29 Asia/Kolkata  
Time remaining at update: 17 days

## Objective and reality check

The goal is a gold-medal-level final placement with an original, reproducible solution. A medal cannot be guaranteed. The current public leader is 4.859; archived score bands are 5.523 at rank 10, 6.199 at rank 25, 6.505 at rank 50, and 6.799 at rank 100. The account's best Kaggle-MCP-verified score is 7.119; the user-reported 6.888 score still needs a submission reference.

The public board is not the final objective. Private-leaderboard robustness, legal provenance, and reproducibility determine whether a high public score is useful.

## Standing public-intelligence loop

At each session start and before major preregistration or promotion, inspect New/Recent discussions and newly run/high-vote notebooks since the timestamp in `tracking/seed.json`. Source-audit exact code, dependencies, outputs and hashes; archive material deltas by evidence class. Participant claims and notebook scores generate hypotheses only.

## Foundation status

Completed before model development:

- Official rules, evaluation, timeline, leaderboard, submission history, and data summary archived.
- All 132 discussion topics and 981 returned posts/comments/replies archived with zero crawl failures.
- Both Working Note Award winners archived and synthesized.
- Copied notebook dependency audit completed: seven named datasets plus one opaque mount; three named datasets have unknown licenses.
- `AGENTS.md`, `MEMORY.md`, control-first workflow, SQLite ledger, auto-sync CLI, validation command, and dashboard created.
- Previous roadmap preserved under `archive/legacy/`.

E001 is promoted and frozen. E002 rejected every naive U continuation. E003 promoted a surface-assisted cross-fitted datum-plus-trend result at 10.9280 RMSE as OOF understanding. E004 then audited the actual test contract, proved those six surface columns are unavailable at inference, and packaged a weaker surface-free geometry-plus-prefix candidate at 15.4913 RMSE. Local and private Kaggle notebook outputs are byte-identical, so E004 is deployment-ready but remains only a fallback. E005 tested clean affine, PF, and trellis GR paths and rejected standalone PF despite aggregate gain. E006 then validated PF placement through strict nested fusion: `nested_conservative_grid` reaches 14.9331 RMSE, wins all 5 maps and 25 outer cells, improves every registered spatial/typewell group, and is deployment-ready after exact private Kaggle parity. E007 found an independent horizontal-only self-correlation signal and improved pooled RMSE to 14.8305, but failed p90 and spatial/typewell transfer gates. E008 then learned split-local residual datum/trend action from E006 plus E007 evidence, reaching 14.7951 and better p90, but failed repeated-map, outer-cell, spatial, and typewell transfer gates. E009 tested six legal datum branches plus fixed and nested model-consensus abstention. Majority consensus reached 14.7935 with 5/5 maps, but only 15/25 cells and repeated shift failures; it was rejected without packaging.

## What the evidence changes

### 1. The target is structural position

Let `U = TVT + Z`. E002 verifies that U is smoother but unsafe to extrapolate directly. E003 shows that low-order datum/trend is highly predictable when training-only formation surfaces are included, reaching 10.9280 RMSE. E004 establishes the deployment boundary: those surfaces are absent from test, and the best surface-free geometry-plus-prefix ridge reaches only 15.4913. The remaining route is not to pretend train-only context exists, but to reconstruct structural evidence from test-available geometry, visible prefix, horizontal GR, and typewell GR through explicit alignment and state/path models.

### 2. The error is concentrated

Both winning notes show that a small number of wells dominates SSE. One reports worst 5% = 52.5% of SSE and worst 10% = 64.5%; another reports the worst 10 of 773 wells = 25.4%. Aggregate RMSE without well-tail analysis is unsafe.

### 3. Validation ranking can invert

A model with 7.623 grouped OOF scored 6.924 publicly, while an older 8.248 OOF model scored 6.675. Public and local rankings can invert. The response is not to abandon CV; it is to strengthen controls, repeated folds, distribution-shift tests, and final-family diversity.

### 4. Diversity matters more than standalone strength

A 13.420 trellis improved a 7.762 ensemble to 7.699 because residual correlation was only 0.488. Conversely, an individually validated third correction damaged a strong stack to 7.446–7.515. Every component must be tested in final placement.

### 5. Nonlinear whole-tail shape is mandatory

Pilkwang's published 773-well oracle ladder is 15.9099 last-known, 9.0354 for an oracle constant, 6.6972 for an oracle line, and 3.1106 for a robust smooth curve. Therefore the gold route cannot be another datum-only or line-only correction. The next experiment separates legal nonlinear **candidate coverage** from **visible-prefix selector regret**.

Amer's exact second-order HMM adds a deterministic posterior candidate over TVT position and dip rate. It joins PF, beam, DTW, smooth-U and jump/fault candidates, but its reported small-sample 4.57 blend remains preliminary until full frozen validation.

### 6. Risk detection is not correction

E003 shows that risk and signed action are both learnable from legal evidence, but with separate models and separate gates. Forest risk reaches 0.5415 Spearman and 0.8125 worst-20% AUC; ridge datum-plus-trend achieves 88.39% material datum-sign accuracy and improves all five maps. A risk score still cannot authorize routing unless the routed action beats the fixed E003 action under repeated and shift validation.

## Score objectives

These are campaign targets, not promises or medal definitions:

- **Foundation target:** independently reproduce a valid whole-well suffix baseline and metric.
- **Competitive target:** repeated-CV system whose public score is below the archived top-100 band of 6.799 without relying on copied artifacts.
- **Gold-contending target:** reach or beat the current top-25 band of 6.199, then build a decorrelated second family with stronger private expectation.
- **Stretch target:** approach the current top-10 band of 5.523 with controlled datum/trend gains.

A single public score does not satisfy a target unless the corresponding local evidence and reproducibility gate pass.

## Core architecture

### Layer A — immutable validation and diagnostics

- Deterministic whole-well folds with hidden-suffix simulation.
- Repeated fold maps.
- Pooled row RMSE plus per-well tails and SSE concentration.
- Datum/trend/shape residual decomposition.
- Regime slices: hidden length, GR missingness/run length, geometry, typewell quality, candidate disagreement, and spatial/typewell groups.
- Positive, no-op, duplicate, shuffle, and leakage-sentinel controls.

### Layer B — independent candidate bank

Every candidate produces a full path and OOF predictions with the same IDs:

1. Last-known-TVT baseline.
2. Constant `U` continuation.
3. Robust linear/quadratic/spline `U` trend.
4. Training-only formation-surface diagnostics, never direct final-inference inputs.
5. Test-available typewell GR alignment candidates with explicit ambiguity scores.
6. Horizontal pre-PS self-correlation candidate.
7. Particle-filter state tracker over structural position and rate.
8. Exact second-order HMM posterior mean/std over position and dip rate.
9. Dynamic-programming/trellis, multiscale/stochastic DTW, constrained smooth-U and explicit jump/fault paths.
10. Learned residual/correction candidates.

A candidate may be weak alone and retained if its residuals are different.

### Layer C — learned evidence, not blind row regression

Tree and sequence models should primarily consume legal, interpretable evidence:

- trajectory geometry and normalized MD;
- known-prefix `U` level, slope, curvature, stability, and extrapolation diagnostics;
- typewell/horizontal alignment scores, margin between modes, stretch, offset and gain calibration;
- candidate path values, slopes, curvature and disagreement;
- GR availability and missing-run structure;
- reconstructed contact/alignment consistency derived only from test-available inputs;
- uncertainty and regime indicators.

Targets to compare:

- direct hidden `U` residual to a structural candidate;
- per-row correction with a whole-well mean constraint;
- per-well datum plus low-order trend coefficients;
- sequence residual with track-consistency solve.

### Layer D — controlled ensemble

- Fit weights only on OOF predictions.
- Report residual correlation matrix and fold-wise blend gain.
- Prefer small weights for weak/diverse legs.
- Re-test all additions after final pipeline placement.
- Candidate routing is rejected unless signed choice beats a fixed blend on repeated folds and harsher splits.

### Layer E — deployment

- Offline notebook, approved inputs only.
- Deterministic seeds and bounded parallelism.
- Artifact hashes and manifest.
- Exact ID/order audit and finite-prediction checks.
- Runtime target below 8 hours; hard competition limit 9 hours.
- Local/Kaggle inference parity test.

## Seventeen-day execution plan

### July 18 — Foundation

Status: completed by this roadmap update.

Deliverables: archive, source/claim registry, workflow, memory, dashboard, database, dependency audit, and current leaderboard/submission snapshots.

### July 19–20 — Validation harness and independent baselines

Status: completed. E001 was promoted; E002 was rejected with all controls passing.

Deliverables:

- Freeze fold map v1 and repeated maps v2–v5.
- Implement one metric module used by every experiment.
- Implement last-known TVT, constant `U`, robust linear `U`, quadratic `U`, and constrained spline `U`.
- Produce OOF rows, per-well metrics, error decomposition, regime tables, and runtime report.
- Verify controls and leakage sentinel.

Exit gate: identical reruns reproduce metrics; pooled and direct RMSE agree; no ID/order leakage; all controls pass.

Measured outcome: E002 retained last-known TVT at 15.9099 RMSE. Robust-linear U was the least-bad challenger at 39.6546, won 0/25 fold cells, and degraded long-hidden RMSE to 48.5525. The candidate matrix is retained only for diagnostics and future disagreement features.

### July 19–22 — Deployment contract and test-available candidate reconstruction

E004 status: completed and deployment-ready; no leaderboard submission created.

Measured E004 outcome:

- Train horizontal wells contain six formation surfaces; test horizontal wells do not.
- E003's 10.9280 surface-assisted result is not deployable as implemented.
- Eight surface-free ablations were evaluated under all five maps and spatial stress.
- Geometry plus visible-prefix evidence is best at 15.4913 RMSE, a 0.4185 gain; removing prefix leaves only 0.0883 gain and 0/5 wins.
- Direct, self-contained local-notebook, and private Kaggle submissions are byte-identical at SHA-256 `62ae0657...5279`; the notebook uses no external artifact or third-party package.
- The final private Kaggle version ran on Python 3.12 with internet disabled and the official competition source attached. Deployment readiness is proven, but the 0.4185 CV gain is too small for an isolated submission.

E005 status: completed and rejected as a standalone family.

Measured E005 outcome:

- All frozen data, E004-reproduction, duplicate, shuffled-GR, axis-confusion, oracle, no-GR, identity, and pooled-SSE controls passed.
- Particle filtering is the best raw path at 15.3502040715 RMSE, a 0.1411023267 gain over E004, with residual correlation 0.8972483902.
- PF reduces worst-5% SSE share from 38.3095% to 33.3199% and maximum well RMSE from 68.8470 to 59.7754, but raises p90 to 23.4852.
- PF wins only 3/5 repeated maps; its worst spatial and typewell-cluster gains versus E004 are -0.7542 and -0.1857.
- The registered decision is rejection. No deployment package, private Kaggle parity run, or leaderboard submission was authorized.
- A pre-registered diagnostic 50/50 PF-E004 blend scores 15.0131171651 with p90 21.7769. It is hypothesis-generating only and must be re-tested with fresh nested/cross-fit placement.

E006 status: completed, promoted, and deployment-ready; no leaderboard submission created.

Measured E006 outcome:

- Strict nested `nested_conservative_grid` scores 14.9331407872 RMSE, gaining 0.5581656110 over E004 and 0.9767120835 over last-known TVT.
- It wins 5/5 repeated maps and 25/25 outer cells, lowers p90 to 21.7856, and improves every nested spatial/typewell group and every special slice.
- Final audited replay completes in 44:42.73 under the 45-minute gate with 228,548 KB maximum RSS.
- The full-fit weight independently recomputes to 0.5. Direct, local-notebook, deterministic package rebuild, and private internet-disabled Kaggle v3 outputs are byte-identical at `e412864a...d81008`.
- E004 remains exact fallback; no leaderboard submission was made.

E007 status: completed and rejected; no deployment package or leaderboard submission created.

Measured E007 outcome:

- The horizontal-only predictor matches hidden GR fingerprints to same-well visible-prefix fingerprints and transfers only bounded local `d(TVT+Z)` evidence; typewell data is excluded from prediction.
- Visible-only pseudo-holdouts improve at all three frozen boundaries. The fixed 0.10 final placement reaches 14.8304784179 RMSE, a 0.1026623694 gain over E006, with 5/5 map wins and 20/25 repeated-cell wins.
- The raw bounded correction is diverse at 0.4675700642 residual correlation to E006, and deterministic template shuffle returns exactly to E006.
- Promotion is rejected because p90 rises from 21.7855500884 to 22.0503957463, narrowly exceeding the 0.25 cap, while spatial groups 1/3 and typewell groups 2/4 regress. Reliability shrink still regresses typewell group 2.
- Long suffix, high-GR-missingness, and poor-visible-pseudo-gain slices expose the same instability. Runtime and memory pass comfortably, but packaging is prohibited after statistical rejection.

E008 status: completed and rejected; no deployment package or leaderboard submission created.

Measured E008 outcome:

- The surface-free split-local datum-only ridge reaches 14.7951380415 RMSE, gaining 0.1380027458 over E006, lowering p90 to 21.4079596409, and lowering worst-5% SSE share to 0.3548957222.
- E007 evidence is selected in all five maps and improves the full datum+trend model by 0.0953021558 RMSE versus the no-E007 ablation. Shuffled targets lose 0.0792 RMSE versus E006 and have near-zero target correlation.
- Datum prediction is weak but real at Pearson 0.1759 and 60.76% material sign accuracy. Trend prediction is weaker and degrades several special slices.
- Datum action wins only 2/5 maps and 14/25 repeated cells and regresses spatial groups 0/1 and typewell groups 2/4. Conservative scaling is nonzero in 18/25 cells but wins only 3 maps by the registered margin and 11 cells, still failing spatial/typewell gates.
- Runtime, memory, parent hashes, leakage, duplicate, shuffle, oracle, E007 ablation, exact fallback, correction-bound, OOF identity, and SSE controls pass. Statistical rejection prohibits packaging.

E009 status: completed and rejected; no deployment package or leaderboard submission created.

Measured E009 outcome:

- Six pre-registered legal datum branches and three fixed consensus rules were completed under immutable folds; no branch was abandoned after one split.
- `consensus_majority` reaches 14.7934532198 RMSE, gains 0.1396875674 over E006, lowers p90 to 21.3853, lowers worst-5% share to 0.35745, and wins all five maps.
- It wins only 15/25 repeated cells and regresses spatial groups 0/1 and typewell groups 2/4. Fixed-rule action sets overlap at Jaccard 0.984–0.996, so stricter consensus does not isolate risky wells.
- Nested abstention acts in 22.66% of repeated placements, reaches only 14.8964, and wins 2/25 cells. The sign-flipped negative control gains 0.03884, above its frozen 0.03 cap.
- The diagnostic 64-feature ridge reaches 14.6352733150, p90 20.8077, and 5/5 maps, but remains ineligible and still fails one spatial and three typewell groups.
- Official and independent OOF artifacts are byte-identical at SHA-256 `3609b06e...052e`; runtime and memory pass. Statistical rejection prohibits packaging.

Immediate deliverables shift to E010/H013: a fresh nonlinear candidate-coverage and visible-prefix selector-regret experiment. Freeze a legal whole-well bank containing E006, the wide-ridge comparator, exact HMM posterior mean/std, PF scales, beam/trellis, multiscale/stochastic DTW, constrained smooth-U curves, and a jump/fault candidate. First measure hidden-label bank-oracle coverage; then evaluate a selector using only multi-cut visible-prefix evidence and target-independent uncertainty. Oracle routing is diagnostic only. H012/T016 remain queued.

Exit gate: candidate-bank oracle below 5 before selector promotion; legal action at least four repeated maps and 17 cells, tail limits, positive every spatial/typewell group, shuffle/sign controls, exact E006 fallback, runtime, final placement, packaging, and parity.

### July 24–26 — Tabular residual and datum models

Deliverables:

- LightGBM/CatBoost/XGBoost residual baselines over legal candidate/evidence features.
- Per-well datum/trend coefficient models.
- Feature controls and shuffled-evidence tests.
- Leave-typewell-out and leave-spatial-out diagnostics for contextual features.

Exit gate: pre-registered improvement on most repeated folds, no tail-risk breach, and harsher-split evidence.

### July 27–29 — Sequence correction and physical fusion

Deliverables:

- Small GRU/TCN or equivalent residual sequence model, not an architecture sweep.
- Track-consistency/quadratic fusion.
- Comparison against tree residual model using identical folds.
- OOF covariance analysis with PF/trellis/tree legs.

Entry/team-merger deadline is July 29. Any team decision must be complete before 23:59 UTC.

Exit gate: keep only sequence components with stable gain or valuable low residual correlation.

### July 30–August 1 — Robust ensemble and uncertainty

Deliverables:

- Fixed-weight OOF ensemble search with nested or cross-fit weight estimation.
- Uncertainty model evaluated separately for detection and signed action.
- Stress tests: worst wells, long hidden tails, high GR missingness, structural-break proxies, and distribution shift.
- Two candidate final families: public-proven and decorrelated private-expectation.

Exit gate: both families pass controls and differ materially in residual structure.

### August 2–3 — Kaggle parity and measured submissions

Deliverables:

- Offline inference notebooks for finalists.
- Runtime/memory profile and artifact audit.
- One submission per discriminating question, all recorded in the dashboard.
- Compare local/public score pairs without changing historical thresholds.

Exit gate: reproducible `submission.csv`, runtime below target, and no unapproved dependency.

### August 4 — Freeze

- Stop broad feature/model exploration.
- Re-run finalists from clean state.
- Verify hashes, fold evidence, score history, and selected final slots.
- Prepare winner documentation outline and environment lock.

### August 5 — Finalize

- Select up to two final submissions: one public-proven family and one decorrelated private-expectation family.
- Confirm Kaggle final selection before 23:59 UTC.
- Archive exact code, notebook version, artifacts, hashes, and decision rationale.

## Promotion thresholds

Thresholds must be filled in each manifest before the run. Default starting policy:

- Full-pool CV improvement at least 0.15 RMSE for a new major component, or at least 0.05 for a low-correlation ensemble leg.
- Improvement on at least 4 of 5 repeated fold maps.
- No p90 per-well RMSE deterioration greater than 0.25 unless pooled gain exceeds 0.30 and the risk is explained.
- No worst-5% SSE share increase greater than 2 percentage points without a specific mitigation.
- Positive control passes; no-op/duplicate changes remain within numerical tolerance; shuffled evidence loses its gain.
- Harsher-split result is directionally consistent or the component is explicitly excluded from the private-expectation family.
- Final notebook parity passes and runtime is below 8 hours target.

These defaults may be revised only through a recorded decision before seeing the result they govern.

## Immediate experiment queue

1. **E001 — Metric and fold harness — completed 2026-07-18.** Evaluator, five fold maps, leakage controls, reports, and hashes are frozen.
2. **E002 — Structural baseline ladder — completed and rejected 2026-07-19.** Sign verified; last-known TVT retained; naive U continuation rejected after 0/25 fold-cell wins.
3. **E003 — Datum/trend and risk learnability — completed and promoted 2026-07-19.** Ridge datum-plus-trend scores 10.9280; forest risk scores 0.5415 Spearman / 0.8125 AUC; both pass repeated and spatial gates.
4. **E004 — Surface-free deployment candidate and ablations — completed and deployment-ready 2026-07-19.** Geometry-plus-prefix scores 15.4913; local-direct, local-notebook, and private Kaggle outputs are byte-identical. Retain it as a fallback; no standalone submission was made.
5. **E005 — Clean PF and trellis candidates — completed and rejected 2026-07-19.** PF reaches 15.3502 but fails p90, repeated-map, spatial, and typewell-cluster gates; no packaging or submission.
6. **E006 — Controlled PF-E004 fusion and disagreement evidence — completed, promoted, and deployment-ready 2026-07-19.** Strict nesting selects 14.9331-RMSE conservative fusion; exact private Kaggle parity passes; no leaderboard submission.
7. **E007 — Horizontal self-correlation and candidate evidence stack — completed and rejected 2026-07-19.** Fixed 0.10 placement reaches 14.8305 with 5/5 maps, but fails p90 and spatial/typewell shift gates; no packaging or submission.
8. **E008 — Cross-fitted legal residual-action model — completed and rejected 2026-07-19.** Datum-only ridge reaches 14.7951 with improved p90 and useful E007 evidence, but wins only 2/5 maps and 14/25 cells and fails spatial/typewell transfer; no packaging or submission.
9. **E009 — Residual-model consensus abstention — completed and rejected 2026-07-19.** Majority consensus reaches 14.7935 and wins all maps but only 15/25 cells, repeats spatial/typewell failures, and fails the sign-flipped negative control; no package or submission.
10. **E010 — Nonlinear candidate coverage and selector regret — next.** Freeze the legal whole-well bank, measure bank-oracle RMSE/catastrophe coverage, then evaluate visible-prefix multi-cut candidate probabilities against oracle regret under all repeated/shift/control gates.
11. **E011 — Learned whole-well selector and shape refinement.** Proceed only if E010 proves below-5 candidate coverage; compare calibrated path selection, learned cost maps/control points, and explicit smooth-versus-jump decoding.
12. **E012 — Final Kaggle offline parity.** Clean package, runtime, exact IDs, and finalist notebooks.

## Explicitly rejected behavior

- Random row CV.
- Treating visible test overlap as hidden-test evidence.
- Reusing opaque/private artifacts or unknown-license inputs.
- Large neural architecture sweeps before target/fold controls are proven.
- Treating train-only formation surfaces as final inference features.
- Per-well oracle routing used as evidence that a legal selector exists.
- Submitting undocumented blends or tuning many tiny public-LB changes.
- Replacing negative results with vague summaries that allow the same experiment to be repeated.

## Definition of foundation complete

Foundation is complete only when:

- archive counts validate at 132 topics / 981 messages;
- SQLite integrity is `ok`;
- dashboard `/api/health` is healthy;
- source and claim registries load;
- dependency audit is present;
- legacy roadmap is preserved;
- no training run is falsely presented as completed.

Foundation is complete. The current exact next action is E010/H013 candidate-coverage and selector-regret preregistration, not another roadmap revision.
