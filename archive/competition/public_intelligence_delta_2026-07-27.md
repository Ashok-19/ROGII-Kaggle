# ROGII Public-Intelligence Delta — 2026-07-27

Checked through the official Kaggle MCP on 2026-07-27 after a two-day pause. This file records public state only; it does not authorize public-leaderboard tuning or ingestion of opaque artifacts.

## Leaderboard frontier

The official public leaderboard now contains two sub-5 entries:

| Rank | Team | Public RMSE | Submission time UTC |
|---:|---|---:|---|
| 1 | `shu01` | **4.679** | 2026-07-25 23:57:17 |
| 2 | `Yannan Chen` | **4.902** | 2026-07-27 04:32:45 |
| 3 | `N&A&O&A` | 5.064 | 2026-07-26 23:55:35 |
| 4 | `SaintLouis` | 5.108 | 2026-07-24 18:22:27 |
| 5 | `Rishikesh Jani` | 5.237 | 2026-07-26 15:23:16 |

This changes the strategic interpretation: sub-5 is operationally attainable on the public 26% split, not only in hidden-label local oracles. The top methods are not publicly disclosed, so the scores do not identify a legal mechanism and cannot be used as training targets.

## Public notebook frontier

Current broadly copied public notebooks remain above 6:

- `raunakdey07/rogii-stacked-ensemble`: public score **6.461**.
- `prvsiyan/rogii-public-score-frontier-lab-visuals`: best public score **6.622**.
- `yasut0ra/rogii-codex-exact-public-6-768-v1`: public score **6.710**.

The `prvsiyan` version-80 notebook declares seven external dataset inputs, including the already quarantined `phongnguyn23021656/koolbox-offline`, `nina2025/rogii-03`, and `needless090/rogii-tabicl-mirror`. It remains a public-mechanism catalogue, not an ingestible evidence artifact.

## Discussion delta

The competition topic count is now **145**, up from the prior 139-topic snapshot.

### PF GR-noise scale

Topic `728712` reports that multiplying the public PF notebook's GR noise scale `gs` by about 1.3 improved its public score. A July 27 reply explicitly warns that this single public-LB change should not be expected to transfer to the private well set. Treat it only as a public sensitivity observation; do not tune `gs` from public score.

### Visible test overlap

Topic `729837` confirms that the visible `test/` files are example wells copied from training and are replaced by actual hidden test files during scoring. There is no hidden-test/train overlap to exploit. Any train-copy override inherited by public forks is irrelevant to hidden scoring and must remain prohibited.

## Priority decision

1. Complete T036 exactly as frozen because it is the only current legal branch that evaluates evidence conditional on each candidate path while preserving a 4.628457-RMSE candidate oracle.
2. Do not divert into public `gs` tuning, fixed shifts, overlap overrides, or opaque public-stack cloning.
3. If T036 hard-stops, immediately use the result to choose one new mechanism with demonstrated sub-5 capacity; do not reopen global coordinate, residual-profile, analog-identity, or blend searches.

## Official public references

- Competition leaderboard: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/leaderboard
- Recent topics: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion
- PF `gs` topic: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728712
- Hidden-overlap topic: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/729837
