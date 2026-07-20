# ROGII Public Intelligence Refresh — 2026-07-20

Checked through Kaggle MCP and exact downloaded notebook sources. This file records research leads and source-visible diagnostics; it does not promote any public solution or artifact.

## Exact notebook audits

### Amer — Exact HMM Smoother

- Kaggle: `amerhu/rogii-wellbore-geology-exact-hmm-smoother`
- Source SHA-256: `2321997c8bcbca442d7d7abfc5b9b7eeed251bac8c671b3554b54c9355c231dc`
- Implements deterministic forward-backward inference over a joint TVT-position and dip-rate state in `U = TVT + Z`, with posterior mean and posterior standard deviation.
- This is a second-order smoother, not the weak first-order residual random walk previously shelved.
- The notebook reports PF 4.90, HMM 5.18, and a 50/50 blend 4.57 on a small sample. The claim is hypothesis-generating only until reproduced on all frozen maps and shift groups.

### Pilkwang corpus

Source-audited notebooks:

- `rogii-eda-target-free-alignment-for-tvt` — `8c6491a59a1409c8ac492b8a90d090785627690acd0f9a0401ef568ba23f69aa`
- `working-note-target-free-tvt-geosteering` — `4052e5a4a29598984124ef04a6dccbc4d218b5510f3943da98bb1c2c9056e900`
- `rogii-v3-heel-calibrated-contact-geosteering` — `a534041a8803d15d8cc949a800e90a7e004f77183128d5743603f70a68702d89`
- `rogii-target-free-tvt-geosteering` — `adc5a813b4faa6a954920564f655af6849da0df2d113d6c74cb24f43c5ad82d0`
- `rogii-dual-track-prefix-calibrated-geosteering` — `5f2eea2cf6e112b7b4eb107ed356df374d768663e55338512347b7c38ceefe62`

Published full-data diagnostics:

| Diagnostic | Pooled RMSE / result |
|---|---:|
| last-known suffix | 15.9098528707 |
| oracle per-well constant | 9.0354099881 |
| oracle per-well line | 6.6971947701 |
| oracle robust degree-6 smooth curve | 3.1105775591 |
| fixed PF/beam selector | 10.5830774219 |
| selector worst-decile SSE share | 61.9455% |

The oracle paths read hidden labels and are ceilings, not deployable candidates. They nevertheless establish that datum and linear trend alone cannot reach a reliable sub-5 result; nonlinear whole-tail shape is necessary.

Heel affine GR calibration was tested as a localization diagnostic over 773 wells. It reduced the fraction localized within ±2 ft from 89.13% to 85.77% and increased p90 absolute shift from 3.4 ft to 7.0 ft. It should not be a default correction.

Reusable ideas are a heterogeneous legal path bank, robust projection in `U`, complete-well models, visible-prefix multi-cut backtests, uncertainty/disagreement features, and bounded late movement. Active high-score profiles rely on pretrained artifacts, model packages, same-well contacts, and global bias; they are not clean generalization evidence.

### prvsiyan — Verified Public 6.979

- Kaggle: `prvsiyan/rogii-verified-public-6-979-visuals`
- Source SHA-256: `5f958a0a02fdba3d00280a3544d4fe08db77733c92ff2d897be5f87214417fdd`
- The score comes from a composite artifact/pretrained/PF/contact/visible-prefix/bias pipeline. Self-contained components are much weaker, so 6.979 cannot be attributed to a clean unseen-well method.

### Yusuke Togashi — Another Approach

- Kaggle: `yusuketogashi/rogii-another-approach`
- Source SHA-256: `a3f1bc42a6fd3155db35b746f5636a608b73b0e0cb5a10979ac525333829a93a`
- Source contained no executed outputs and mounted eight external datasets, including ridge artifacts, model packages, pretrained boosters, and TabICL mirrors.
- It adds a local GR-slope refinement to the 6.979 lineage, but supplies no clean causal or portable evidence. Retain only as an idea lead.

## Discussion and notebook delta scan

- Kaggle's current topic listing reported 134 topics; the frozen archive has 132. New visible topics include `727537`, `727569`, and `727570`, but list-count differences are not sufficient to infer the complete delta. Canonical archive counts remain unchanged pending a full recrawl.
- Topic `727570` contains an unverified report of group-by-well CV 4.98 over five folds and five seeds versus public LB 5.7, with the warning that the LB is noisy.
- Topic `726465` contains an unverified report of pooled five-fold CV below 5 using complete-well samples, whole-well grouping, and only test-time inputs. Another participant reports 5.77 pooled five-fold CV.
- Neighbor-copying, azimuth splitting, kriging failure, and sub-5 architecture claims are mutually incomplete and remain hypotheses. No route or split is authorized from them.
- The DateRun notebook scan surfaced several high-vote artifact/reproduction notebooks and no source-visible clean replacement for a nonlinear whole-well candidate study.

## Late-session delta — 20:33 Asia/Kolkata

- New/Recent topic listings now report 135 topics. Canonical archive counts remain 132 topics and 981 messages until a complete recrawl verifies the delta.
- Topic `727708` reports a visible Save Version success followed by hidden reruns with `COMPLETE` and `totalBytes=0` for a roughly 7.39 GB, 195-feature LightGBM/CatBoost/Ridge workflow. Possible OOM and hidden-well edge cases are self-reported operational hypotheses, not verified performance evidence. This reinforces compact, streaming, fail-closed execution.
- A new comment on topic `722236` self-reports roughly 9.71 four-seed CNN CV and argues that representation, validation, and training matter more than depth. No source or independently downloadable result establishes that score, so it remains a lead and does not alter E011's frozen design or gates.
- The current `prvsiyan/rogii-goal-lowest-public-frontier-lab-visuals` remains version 22. Its unknown-license inputs, unrestricted parallelism, probe/canary logic, score-derived bias constants, and large materialized artifacts keep it quarantined as an implementation source.

## Research consequence

The next experiment must distinguish:

1. **candidate coverage** — the hidden-label oracle of a legal nonlinear path bank; and
2. **selector regret** — loss from a legal visible-prefix selector relative to that oracle.

Candidate generation is inadequate if bank-oracle RMSE remains above 5. Selection is the bottleneck if the oracle is below 5 but legal selection is not. This decomposition supersedes the unstarted datum-only wide-ridge priority; the wide ridge remains a comparator.

## Monitoring rule

At each new work session and before a major preregistration or promotion, query New/Recent discussions and DateRun/VoteCount notebooks, compare with `tracking/seed.json:meta.public_intelligence_last_checked`, inspect only material deltas, source-audit exact code and dependencies, and archive findings by evidence class. Discussion claims and votes never promote a model.
