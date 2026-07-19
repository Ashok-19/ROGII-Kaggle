# ROGII Gold Medal Roadmap

Updated: 2026-07-19  
Final deadline: 2026-08-05 23:59 UTC / 2026-08-06 05:29 Asia/Kolkata  
Time remaining at update: 18 days

## Objective and reality check

The goal is a gold-medal-level final placement with an original, reproducible solution. A medal cannot be guaranteed. The current public leader is 4.859; archived score bands are 5.523 at rank 10, 6.199 at rank 25, 6.505 at rank 50, and 6.799 at rank 100. The account's best Kaggle-MCP-verified score is 7.119; the user-reported 6.888 score still needs a submission reference.

The public board is not the final objective. Private-leaderboard robustness, legal provenance, and reproducibility determine whether a high public score is useful.

## Foundation status

Completed before model development:

- Official rules, evaluation, timeline, leaderboard, submission history, and data summary archived.
- All 132 discussion topics and 981 returned posts/comments/replies archived with zero crawl failures.
- Both Working Note Award winners archived and synthesized.
- Copied notebook dependency audit completed: seven named datasets plus one opaque mount; three named datasets have unknown licenses.
- `AGENTS.md`, `MEMORY.md`, control-first workflow, SQLite ledger, auto-sync CLI, validation command, and dashboard created.
- Previous roadmap preserved under `archive/legacy/`.

E001 is promoted and frozen. E002 rejected every naive U continuation. E003 promoted a surface-assisted cross-fitted datum-plus-trend result at 10.9280 RMSE as OOF understanding. E004 then audited the actual test contract, proved those six surface columns are unavailable at inference, and packaged a weaker surface-free geometry-plus-prefix candidate at 15.4913 RMSE. Local and private Kaggle notebook outputs are byte-identical, so E004 is deployment-ready but remains only a fallback.

## What the evidence changes

### 1. The target is structural position

Let `U = TVT + Z`. E002 verifies that U is smoother but unsafe to extrapolate directly. E003 shows that low-order datum/trend is highly predictable when training-only formation surfaces are included, reaching 10.9280 RMSE. E004 establishes the deployment boundary: those surfaces are absent from test, and the best surface-free geometry-plus-prefix ridge reaches only 15.4913. The remaining route is not to pretend train-only context exists, but to reconstruct structural evidence from test-available geometry, visible prefix, horizontal GR, and typewell GR through explicit alignment and state/path models.

### 2. The error is concentrated

Both winning notes show that a small number of wells dominates SSE. One reports worst 5% = 52.5% of SSE and worst 10% = 64.5%; another reports the worst 10 of 773 wells = 25.4%. Aggregate RMSE without well-tail analysis is unsafe.

### 3. Validation ranking can invert

A model with 7.623 grouped OOF scored 6.924 publicly, while an older 8.248 OOF model scored 6.675. Public and local rankings can invert. The response is not to abandon CV; it is to strengthen controls, repeated folds, distribution-shift tests, and final-family diversity.

### 4. Diversity matters more than standalone strength

A 13.420 trellis improved a 7.762 ensemble to 7.699 because residual correlation was only 0.488. Conversely, an individually validated third correction damaged a strong stack to 7.446–7.515. Every component must be tested in final placement.

### 5. Risk detection is not correction

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
8. Dynamic-programming/trellis whole-well posterior.
9. Learned residual/correction candidates.

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

## Eighteen-day execution plan

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

Immediate deliverables now shift to E005:

- Clean typewell/horizontal GR alignment, PF, and trellis candidates using only test-available columns.
- No-GR fallbacks, repeated-motif ambiguity measures, and shuffled-GR controls.
- OOF paths with identical IDs, tail metrics, runtime, and residual correlations against last-known TVT and E004 geometry-prefix.

Exit gate: at least one test-available path candidate materially improves 15.4913 or provides validated low residual correlation for a controlled blend.

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
5. **E005 — Clean PF and trellis candidates — next.** Independent test-available implementations with OOF paths and residual correlations against last-known TVT and E004.
6. **E006 — Horizontal self-correlation candidate.** Template-shuffle control.
7. **E007 — Candidate evidence tree stack.** Extend E003 with independent candidate evidence and strict controls.
8. **E008 — Small sequence residual corrector.** Only after E007 freezes the feature/target frame.
9. **E009 — OOF diversity ensemble and uncertainty placement.** Cross-fit weights; risk cannot route without beating the fixed action.
10. **E010 — Final Kaggle offline parity.** Clean package, runtime, exact IDs, and finalist notebooks.

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

The next action after foundation validation is E001, not another roadmap revision.
