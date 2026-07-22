# ROGII Agent Operating Contract

## Mission

Build an original, reproducible, private-leaderboard-robust solution for the ROGII Wellbore Geology Prediction competition before 2026-08-05 23:59 UTC. Gold is the target; no score or medal is guaranteed. Evidence, legality, reproducibility, and private robustness outrank persuasive narratives or public-score chasing.

## Start every work session by reconciling state

Read, in order:

1. `AGENTS.md`
2. `MEMORY.md`
3. `GOLD_ROADMAP.md`
4. `PROJECT_WORKFLOW.md`
5. the active hypothesis/task/decision records in `tracking/inbox/`
6. relevant experiment manifests and `RESULT.md` files

Then verify `git status`, `HEAD`, upstream state, active workspace locks/claims, and `python tools/rogii.py state`. Do not continue a stale handoff or remembered plan blindly.

Before implementation, compare the proposed action with current evidence and the latest Kaggle state. When they conflict, stop, identify the stale assumption, record a new decision, and update the durable sources of truth before scoring. Never silently rewrite history, thresholds, folds, targets, or prior decisions.

## Public-intelligence loop

At the start of each new session, and immediately before pre-registering or promoting a major experiment:

- check the competition's New/Recent discussions and newly run/high-vote public notebooks;
- compare against `public_intelligence_last_checked` in `tracking/seed.json` and inspect only material deltas;
- source-audit notebook code, mounted inputs, executed outputs, provenance, and license before using a claim;
- archive material findings with date and evidence class, then update claims/tasks if the research priority changes.

Discussion claims, notebook titles, votes, and leaderboard scores are leads, not promotion evidence. Do not copy opaque artifacts or infer causality from a public score. A full discussion recrawl is required before changing canonical archive counts.

## Evidence and validation

Use this hierarchy: official rules; reproduced local measurements; controlled OOF experiments; source-audited public outputs; winning-note experiments; participant claims; intuition.

Non-negotiable rules:

- split by whole well and simulate the hidden suffix; never use random row splits;
- score pooled row RMSE and report repeated maps, per-well tails, worst-5%/10% SSE share, shift groups, runtime/memory, and residual correlation;
- keep `folds/v1.json` through `folds/v5.json`, metric semantics, and target construction immutable;
- pre-register legal inputs, controls, thresholds, final placement, runtime, and artifacts before hidden-label scoring;
- require positive, no-op, duplicate, shuffle/sign, leakage-sentinel, exact-fallback, and pipeline-placement controls where applicable;
- validate candidate routing as signed action; uncertainty alone cannot authorize movement;
- test every component in final placement and preserve negative results;
- never treat the three visible overlap wells as hidden-test evidence.

A candidate is promoted only if it passes its frozen repeated-map, tail, shift, scientific-control, and deployment-output gates. Public leaderboard results update belief but never retroactively alter a gate.

## Experiment and repository discipline

Every experiment lives in `experiments/E###/manifest.json` and follows:

`proposed -> designed -> smoke_passed -> cv_running -> evaluated -> promoted | rejected | blocked`

Canonical locations:

- code: `src/` and `tools/`;
- immutable experiment evidence: `experiments/`;
- generated predictions/models: `artifacts/`;
- durable ledger inputs: `tracking/seed.json`, manifests, and `tracking/inbox/*.json`;
- generated database: `tracking/rogii.sqlite`—regenerate it; never hand-edit it;
- evidence archive: `archive/`;
- temporary work: agent-specific `scratch/` only.

Use stable IDs (`H###`, `E###`, `R...`, `D###`, `T###`). Acquire a task claim for focused work and the official lock only for serialized durable writes. Stage explicit intended paths only. Never delete canonical artifacts referenced by manifests; remove scratch copies only after confirming they are not needed for provenance or recovery.

Update the dashboard Learning Lab in the same workstream when verified evidence changes the problem framing, next experiment, or an important failure mode; label public and oracle evidence explicitly.

## Compute, packaging, and submissions

Follow `KAGGLE_NOTEBOOK_RUNBOOK.md`.

Choose local or Kaggle compute based on practicality. Avoid stressing the local machine, but do not impose fixed CPU, accelerator, internet, or thread requirements unless an experiment specifically needs them.

For every experiment, prefer the smallest workflow that can answer the registered question. A Kaggle notebook should locate its required inputs, run the model, preserve sample IDs/order, reject non-finite predictions, and save the primary outputs. Do not add repeated hashes, environment inventories, byte-equality requirements, or device checks as default gates.

A completed notebook becomes usable evidence when its primary outputs exist, parse correctly, cover the expected rows/IDs, and support the claimed result. Additional provenance or parity checks are optional and should be added only when they address a demonstrated risk.

The assistant prepares notebooks and inputs but does not run a Kaggle notebook unless the user explicitly authorizes it. No Kaggle submission is sent without explicit user authorization. Record every authorized submission immediately with its reference, run, score, and purpose.

## Finish every work session

Run scoped tests, the full test suite when relevant, `python tools/rogii.py sync`, and `python tools/rogii.py validate`; check SQLite integrity, git diff/status, locks, claims, and push state. Update `MEMORY.md` with only durable state, decisions, blockers, public-intelligence timestamp, and one exact next action. Release locks/claims and report only verified outcomes.
