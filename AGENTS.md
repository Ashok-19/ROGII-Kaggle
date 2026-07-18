# ROGII Agent Operating Contract

## Mission

Build an original, reproducible, private-leaderboard-robust solution for the ROGII Wellbore Geology Prediction competition before the 2026-08-05 23:59 UTC deadline. Gold is the goal, but no score or medal is guaranteed. Every claimed improvement must be traceable to evidence, code, configuration, artifacts, and validation results.

## Required reading before work

1. `MEMORY.md`
2. `GOLD_ROADMAP.md`
3. `PROJECT_WORKFLOW.md`
4. `archive/writeups/synthesis.md`
5. `archive/discussions/synthesis.md`
6. `archive/competition/notebook_audit.md`

## Evidence hierarchy

Use this order when claims conflict:

1. Official competition pages and rules.
2. Reproduced local measurements on competition data.
3. Controlled experiments with exact code/configuration and OOF predictions.
4. Winning-note experiments with explicit controls or score pairs.
5. Public notebook results with reproducible provenance.
6. Discussion claims and intuition.

Discussion statements are leads, not facts. Record material claims in `archive/claims.csv` or the dashboard with a confidence class.

## Non-negotiable validation rules

- Never use random row splits. Split by whole well and simulate the hidden suffix.
- Primary metric is pooled row-level RMSE. Also report median, p90, p95, maximum per-well RMSE, worst-5%/10% SSE share, mean-error SSE share, and residual correlation with candidate ensemble members.
- Freeze fold maps before model comparison. Use repeated fold maps and at least one harsher split for promotion.
- Every learned feature family must pass a known-positive control, no-op control, duplicate-feature control, and shuffled-evidence control where applicable.
- Candidate routing and uncertainty gates must prove signed action, not merely risk detection.
- Evaluate components in their final pipeline placement. An isolated gain does not authorize stacking.
- Preserve negative results and exact configurations.
- Do not tune on the three visible test examples or use their train overlap as generalization evidence.

## Data, artifacts, and competition compliance

- Competition code or data must not be privately shared outside the official Kaggle team.
- Internet must be disabled in the submission notebook; runtime must remain below 9 hours. Target an 8-hour maximum for safety.
- External artifacts are quarantined until source, license, version, training data, target, folds, code, hash, and accessibility are recorded.
- Never depend on an opaque/private mount for a final solution.
- Unknown-license artifacts may be inspected for ideas but cannot enter a prize-targeting pipeline without resolution.
- No target-derived feature may use hidden suffix labels at inference or validation time.

## Experiment lifecycle

Every experiment has `experiments/<experiment_id>/manifest.json`. The lifecycle is:

`proposed -> designed -> smoke_passed -> cv_running -> evaluated -> promoted | rejected | blocked`

Before implementation, the manifest must state:

- one falsifiable hypothesis;
- baseline and expected mechanism;
- exact legal inputs;
- fold map and scored rows;
- controls;
- promotion and rejection thresholds;
- expected runtime and artifacts.

After execution, append immutable run entries containing configuration path, git SHA, metrics, controls, and artifact paths. Run `python tools/rogii.py sync`; the dashboard also auto-syncs manifests whenever it serves API state.

## Promotion gate

A candidate may be promoted only when all applicable conditions hold:

1. Metric implementation and positive control pass.
2. No-op, duplicate, and shuffled controls behave as expected.
3. Gain is present on most repeated fold maps, not one convenient split.
4. Pooled RMSE improves by the pre-registered margin.
5. Worst-well tails do not deteriorate beyond the registered limit.
6. The gain survives a harsher split or has a documented reason it cannot.
7. Residual correlation shows genuine complementarity for ensemble legs.
8. Kaggle runtime parity and input provenance pass.

Public leaderboard scores update belief; they do not retroactively change a hypothesis or promotion threshold.

## Repository conventions

- Source code: `src/` or `tools/`.
- Experiment definitions and immutable run manifests: `experiments/`.
- Generated models/predictions: `artifacts/` and ignored runtime paths.
- Project ledger: `tracking/rogii.sqlite`, generated from manifests and seeds.
- Evidence archive: `archive/`.
- Dashboard: `dashboard/`.
- Temporary agent work: `scratch/`, never committed.

Use stable IDs: `H###` hypotheses, `E###` experiments, `RYYYYMMDD-HHMM-<slug>` runs, `D###` decisions, `T###` tasks.

## End-of-session duties

- Sync and validate: `python tools/rogii.py sync && python tools/rogii.py validate`.
- Update `MEMORY.md` only with durable state, decisions, verified scores, blockers, and the exact next action.
- Record submissions immediately, including Kaggle reference, run ID, score, and description.
- Do not silently change folds, metric code, target construction, or selected final submissions.
