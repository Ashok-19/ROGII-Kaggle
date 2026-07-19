# ROGII Test–Iterate–Validate–Submit Workflow

## 0. Reconcile state and public intelligence

Read the required project files, then verify git status/HEAD/upstream, tracker state, locks/claims, and the exact next action. Check Kaggle New/Recent discussions and newly run/high-vote notebooks since `public_intelligence_last_checked`. Source-audit material deltas and archive them. If new evidence invalidates the queued plan, record a decision and update durable state before implementation.

## 1. Formulate

Create one falsifiable hypothesis about datum, trend, nonlinear shape, candidate coverage, selector regret, or ensemble variance. Link evidence IDs, state legal inputs, and define what would disprove it.

## 2. Pre-register

Create `experiments/E###/manifest.json` before code or hidden-label scoring. Freeze the baseline, fold maps, scored rows, candidates, controls, promotion/rejection gates, final placement, runtime budget, and expected artifacts.

## 3. Implement minimally

Build the smallest version that can falsify the hypothesis. Keep metric, folds, target, and baseline immutable. Every candidate must emit stable row IDs and a complete path; keep generation, selection, and evaluation separate.

## 4. Smoke test

Use a fixed small well set only to find crashes, leakage, ID/order errors, NaNs, nondeterminism, and runtime problems. Smoke scores are never evidence.

## 4A. Place compute deliberately

Keep local execution to at most two CPU threads. Run only source checks, unit/contract tests, deterministic packaging, and tiny smokes locally. Prepare a bounded private Kaggle notebook for workflows expected to exceed roughly 15 minutes or materially stress local memory/I/O. Follow `KAGGLE_NOTEBOOK_RUNBOOK.md`: exact committed SHA, hash-sealed input dataset version, manual exact attachment, internet disabled, fail-closed preflight, fixed output filenames, and independent downloaded-output verification.

## 5. Run controls

Use applicable known-positive, no-op, exact-fallback, duplicate, shuffled/reversed/sign-flipped evidence, leakage-sentinel, deterministic rerun, and final-placement controls. A failed control invalidates the run.

## 6. Full validation

Report pooled RMSE; five repeated maps and outer cells; median/p90/p95/max well RMSE; worst-5%/10% SSE share; datum/trend/shape decomposition; spatial/typewell and preregistered regimes; runtime/memory; candidate oracle coverage; legal selector regret; and residual correlation with retained legs. At least one promotion run must use a preregistered harsher split.

## 7. Decide

Use only frozen thresholds:

- `promoted`: controls, repeated maps, tails, shifts, final placement, and deployment gates pass;
- `rejected`: the hypothesis or exact configuration fails;
- `blocked`: a legal, data, provenance, or compute prerequisite is missing.

Never relabel a failed run, route by observed validation subgroups, or alter history. A changed hypothesis receives a new ID.

## 8. Package and parity-test

Build one canonical offline notebook from approved hash-sealed inputs only. Verify no internet, at most two CPU threads unless a separately approved accelerator workflow requires otherwise, exact sample IDs/order, finite predictions, artifact hashes, deterministic inference, runtime below the 8-hour target / 9-hour hard limit, and local/Kaggle parity. A `COMPLETE` notebook is not accepted until expected outputs are listed, downloaded, hashed, parsed, and independently checked.

## 9. Submit deliberately

Create no submission without explicit user authorization. Each authorized submission must answer one registered question and be recorded immediately with Kaggle reference, run, score, and description.

## 10. Update belief, not history

Add public results and public-intelligence deltas to the ledger. Compare local/public score pairs without retroactively changing folds, gates, or run records. Update `MEMORY.md` with one next action.

## Daily cadence

- Start: reconcile repository/tracker state and scan Kaggle deltas.
- Middle: one primary falsifiable experiment plus one cheap control.
- End: tests, sync, validate, SQLite integrity, diff/status review, durable memory update, and lock/claim release.
