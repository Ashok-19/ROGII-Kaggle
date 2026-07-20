# ROGII Project Memory

Last updated: 2026-07-20
Public intelligence last checked: 2026-07-20 00:30 Asia/Kolkata

## Mission state

- Deadline: 2026-08-05 23:59 UTC / 2026-08-06 05:29 Asia/Kolkata.
- Current leader snapshot: 4.859. Best Kaggle-MCP-verified account submission: 7.119 (ref 54754431, 2026-07-16). User-reported best: 6.888, reference unverified.
- E001 is frozen; E002, E005, E007, E008, and E009 are rejected; E003 is promoted only as nondeployable surface-assisted understanding; E004 is the exact fallback; E006 is the promoted deployment-ready primary.
- E010 is implemented and `smoke_passed` but has no scored run. Kaggle input v1 failed during preflight before hidden-label scoring because the profile signature depended on the physical mount root. Commit `49ca3995269638026561c5593ad2628f8f08a4e9` fixes this without changing the frozen 773-well signature. Corrected private dataset/notebook v2 is ready. H012/T016 remain queued as comparators.

## Verified project state

- The immutable evaluator covers 773 wells and 3,783,989 hidden rows with five deterministic whole-well fold maps and frozen controls.
- Last-known TVT scores 15.9098528707 RMSE. Baseline SSE is 67.75% datum, 14.53% linear trend, and 17.72% remaining shape; the worst 5%/10% of wells contribute 38.99%/52.48% of SSE.
- E003 reaches 10.9279740918 with formation surfaces, proving strong structural action is learnable, but all six surfaces are absent from test horizontal files.
- E004 surface-free geometry/prefix ridge scores 15.4913063983 and has byte-identical local/private-Kaggle output at `62ae0657...5279`.
- E006 nested PF-E004 fusion scores 14.9331407872, wins 5/5 maps and 25/25 outer cells, improves every registered spatial/typewell group, and has byte-identical local/private-Kaggle output at `e412864a...d81008`.
- E007–E009 produced aggregate gains but failed frozen repeated-cell, shift, tail, or negative-control gates. E009's legal 64-feature diagnostic ridge reaches 14.6352733150 but fails one spatial and three typewell groups and is not promotion evidence.

## New durable understanding from exact public-source audits

- Pilkwang's published 773-well oracle ladder is: last-known 15.9098528707, oracle per-well constant 9.0354099881, oracle line 6.6971947701, and oracle robust smooth curve 3.1105775591. Constant or linear datum/trend recovery cannot reliably reach sub-5; nonlinear whole-tail shape is mandatory.
- Pilkwang's fixed PF/beam selector scores 10.5830774219 pooled RMSE; its worst 10% of wells contribute 61.95% of SSE. The central problem is catastrophic candidate selection on a minority of wells, not only average tracker quality.
- Pilkwang's heel affine GR calibration worsens ±2 ft datum localization from 89.13% to 85.77% and p90 shift error from 3.4 ft to 7.0 ft. Keep calibration parameters as diagnostics, not a default correction.
- Amer's exact second-order HMM is materially different from the rejected first-order smoother: it performs deterministic forward-backward inference over joint TVT position and dip rate and returns posterior mean/std. Its reported PF/HMM/50-50 sample scores are 4.90/5.18/4.57 on only a small sample; full repeated/shift validation is still required.
- The prvsiyan 6.979 and high-vote derivative notebooks are composite artifact/pretrained/contact/overlap systems. Their leaderboard scores are not clean unseen-well evidence. Reusable ideas include robust `U=TVT+Z` projection, heterogeneous path banks, visible-prefix pseudo-holdouts, and bounded late moves.
- A current discussion participant reports grouped whole-well pooled CV in the 4.x range using test-time inputs and complete-well samples; another reports 4.98 CV versus 5.7 LB. These are unverified leads, but they independently support whole-well nonlinear modeling and explicit LB-noise analysis.

## Decisions

- Repository files and verified Kaggle records—not chat recollection—are the source of truth. Every new session must reconcile git, tracker state, locks, current public intelligence, and the exact next action before work.
- Check New/Recent discussions and newly run/high-vote notebooks at every session start and before major preregistration/promotion. Inspect deltas, source-audit exact code/dependencies/outputs, archive material findings, and never promote from claims or votes.
- Preserve E006 as primary and E004 as exact fallback. Rejected E005/E007/E008/E009 actions may supply candidate paths or diagnostics only in a fresh preregistered experiment.
- Queue H012/T016. Do not spend the next experiment optimizing datum-only wide ridges: the oracle line floor shows that even perfect linear action is insufficient for the gold target.
- Candidate generation and candidate selection must be evaluated separately. Hidden-label oracle routing is diagnostic only and can never enter inference.
- No opaque/private or unknown-license artifact enters a prize-targeting pipeline. No Kaggle submission occurs without explicit authorization.

## Exact next action

Run the corrected private Kaggle E010 notebook using `ashok205/rogii-e010-selector-inputs` version 2 and the official competition input. Require the v2 preflight to identify exactly one 773-well competition root, then complete the full run and persist `e010-run-receipt.json`, `e010-output-manifest.json`, and `rogii-e010-results-v2.zip`. Download and independently verify every output before recording any score or verdict.

## Open risks

- Public CV/LB ordering is noisy and hidden test contains roughly 200 wells; current public claims are not verified evidence.
- HMM/DTW/PF can over-count autocorrelated GR, commit to repeated motifs, or smooth across faults. Posterior uncertainty must be calibrated OOF and a jump candidate must remain available.
- Candidate oracle below 5 does not imply a legal selector exists; selector regret and catastrophe probability must be measured explicitly.
- The current public topic list reports 134 topics while the frozen archive contains 132. Canonical counts remain unchanged until a complete delta recrawl verifies additions, removals, and message counts.
- Every future finalist needs its own clean Kaggle runtime, exact-ID, artifact-provenance, and output-parity verification.

## Memory update rule

Keep this file concise. Store detailed evidence in manifests, tracker records, and `archive/`. Record only verified state, durable decisions, blockers, public-intelligence timestamp, and one executable next action.
