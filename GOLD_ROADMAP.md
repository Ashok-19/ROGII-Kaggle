# ROGII Gold Medal Roadmap

Updated: 2026-07-18  
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

No new model family is considered started until the validation harness and independent baselines are frozen.

## What the evidence changes

### 1. The target is structural position

Let `U = TVT + Z`. The known trajectory contributes most local TVT wiggle through `-Z`. The difficult part is the smooth per-well level/trend of `U`, including datum uncertainty and structural breaks. Direct row-wise TVT prediction is therefore not the only or preferred framing.

### 2. The error is concentrated

Both winning notes show that a small number of wells dominates SSE. One reports worst 5% = 52.5% of SSE and worst 10% = 64.5%; another reports the worst 10 of 773 wells = 25.4%. Aggregate RMSE without well-tail analysis is unsafe.

### 3. Validation ranking can invert

A model with 7.623 grouped OOF scored 6.924 publicly, while an older 8.248 OOF model scored 6.675. Public and local rankings can invert. The response is not to abandon CV; it is to strengthen controls, repeated folds, distribution-shift tests, and final-family diversity.

### 4. Diversity matters more than standalone strength

A 13.420 trellis improved a 7.762 ensemble to 7.699 because residual correlation was only 0.488. Conversely, an individually validated third correction damaged a strong stack to 7.446–7.515. Every component must be tested in final placement.

### 5. Risk detection is not correction

Uncertainty and disagreement can locate expensive wells, but signed datum direction may remain unavailable. Gates need evidence that they choose the right action, not only the right wells.

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
4. Physics/contact candidates from legal surface columns.
5. Typewell GR alignment candidates with explicit ambiguity scores.
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
- legal formation/contact consistency;
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

Deliverables:

- Freeze fold map v1 and repeated maps v2–v5.
- Implement one metric module used by every experiment.
- Implement last-known TVT, constant `U`, robust linear `U`, quadratic `U`, and constrained spline `U`.
- Produce OOF rows, per-well metrics, error decomposition, regime tables, and runtime report.
- Verify controls and leakage sentinel.

Exit gate: identical reruns reproduce metrics; pooled and direct RMSE agree; no ID/order leakage; all controls pass.

### July 21–23 — Candidate-bank reconstruction

Deliverables:

- Clean interfaces for physics/contact, typewell alignment, self-correlation, PF, and trellis candidates.
- No copied pretrained models.
- Candidate OOF matrix and residual-correlation matrix.
- Candidate performance by regime and fold.

Exit gate: at least two candidates show genuine complementarity or one candidate materially improves the best independent baseline.

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
2. **E002 — Structural baseline ladder — next.** Verify transform sign conventions, then compare constant, robust linear, quadratic, and constrained spline formulations.
3. **E003 — Datum/trend oracle decomposition.** Quantify learnable headroom and per-well sign problem.
4. **E004 — Clean PF and trellis candidates.** Independent implementations with OOF paths.
5. **E005 — Candidate evidence tree stack.** Positive/no-op/duplicate/shuffle controls.
6. **E006 — Horizontal self-correlation candidate.** Template-shuffle control.
7. **E007 — Small sequence residual corrector.** Only after E005 establishes the feature/target frame.
8. **E008 — OOF diversity ensemble.** Cross-fit fixed weights and placement tests.
9. **E009 — Uncertainty detection versus signed action.** No actuation unless sign evidence passes.
10. **E010 — Kaggle offline parity.** Clean package, runtime and exact IDs.

## Explicitly rejected behavior

- Random row CV.
- Treating visible test overlap as hidden-test evidence.
- Reusing opaque/private artifacts or unknown-license inputs.
- Large neural architecture sweeps before target/fold controls are proven.
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
