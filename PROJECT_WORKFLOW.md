# ROGII Test–Iterate–Validate–Submit Workflow

## 1. Formulate

Create one falsifiable hypothesis. State why it should alter datum, trend, shape, or ensemble variance. Link evidence IDs. Define legal inputs and what would disprove the idea.

## 2. Pre-register

Create an experiment manifest:

```bash
python tools/rogii.py new-experiment --id E001 --title "Robust U baselines" --track surface
```

Before code changes, fill in baseline, fold map, controls, metrics, promotion margin, tail-risk limit, runtime budget, and expected artifacts.

## 3. Implement minimally

Start with the smallest implementation that can falsify the hypothesis. Keep metric, folds, and baseline immutable. Generate OOF predictions with stable row IDs and well IDs.

## 4. Smoke test

Run on a small fixed well subset only to detect crashes, leakage, ID-order errors, NaNs, or excessive runtime. Smoke scores are never model evidence.

## 5. Run controls

- Known-winner/known-signal positive control.
- Zero/no-op feature control.
- Duplicate feature control.
- Shuffled evidence control.
- Leakage sentinel: intentionally illegal oracle must win strongly, proving the harness can detect signal.
- Pipeline-placement control for additions to an existing stack.

A failed control invalidates the run.

## 6. Full validation

Minimum report:

- pooled row RMSE;
- fold-by-fold RMSE and repeated-fold mean/std;
- median/p90/p95/max well RMSE;
- worst 5% and 10% well SSE share;
- per-well mean-error SSE share and oracle-removed RMSE;
- regime metrics by hidden length, GR missingness, geometry, typewell availability, and candidate disagreement;
- runtime and memory;
- OOF residual correlation against existing candidate legs.

At least one promotion run must use a harsher split: leave-spatial-out, typewell-group-out, or another pre-declared distribution-shift test.

## 7. Decide

Use the pre-registered threshold. Outcomes:

- `promoted`: survives controls, repeated folds, tails, and parity.
- `rejected`: hypothesis falsified or risk exceeds gain.
- `blocked`: missing data, legal ambiguity, license, or compute prerequisite.

Record why. Never relabel a failed exact configuration as promising without a new hypothesis and ID.

## 8. Package and parity-test

Build an offline Kaggle notebook that consumes only approved inputs. Verify:

- exact `sample_submission.csv` ID order;
- finite predictions and sane range;
- no internet;
- all artifacts hashed and found;
- runtime below 8 hours target / 9 hours hard limit;
- local inference and Kaggle notebook inference agree within tolerance.

## 9. Submit deliberately

Every submission must answer one question. Record it before or immediately after submission:

```bash
python tools/rogii.py record-submission --id K123 --run R20260720-1200-u-baseline --score 6.90 --description "U spline, frozen folds"
```

Do not spend submissions on unregistered blends or tiny parameter nudges without a discriminating hypothesis.

## 10. Update belief, not history

Add the public score to the ledger. Compare it with CV and prior score pairs. Do not retune old thresholds or edit run records. Decide the next test using the full evidence matrix.

## Daily cadence

- Start: open dashboard; inspect next tasks, active hypotheses, blockers, and score history.
- Middle: one primary experiment plus one cheap falsification/control.
- End: sync, validate, record decision, update `MEMORY.md`, and set one exact next action.
