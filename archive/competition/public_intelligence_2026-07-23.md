# ROGII public intelligence refresh — 2026-07-23

Checked at 2026-07-23 14:15 Asia/Kolkata. Previous cutoff: 2026-07-22 14:59 Asia/Kolkata.

This refresh separates four evidence classes:

1. exact Kaggle metadata, source, and saved outputs;
2. participant measurements and self-reports;
3. notebook titles, votes, and leaderboard scores, which are discovery signals only;
4. unsupported or unavailable material, which cannot justify implementation.

## Current leaderboard snapshot

The public leader remains `shu01` at 4.859. The visible top ten are:

| Rank | Team | Score |
|---:|---|---:|
| 1 | shu01 | 4.859 |
| 2 | Yannan Chen | 4.905 |
| 3 | SaintLouis | 5.108 |
| 4 | Rishikesh Jani | 5.237 |
| 5 | Shrey Gandhi | 5.265 |
| 6 | L & J & A & A | 5.325 |
| 7 | yu4u | 5.443 |
| 8 | Tucker Arrants | 5.444 |
| 9 | Nocturne & Adil | 5.502 |
| 10 | Ryo Takaki | 5.511 |

The rank-2 score improved from 4.913 to 4.905 and the rank-10 band from 5.523 to 5.511. The top-five cutoff remains 5.265. These movements do not identify a mechanism.

## Discussion delta

Kaggle New and Recent currently report 138 topics, one more than the prior observed count of 137 and six more than the frozen 132-topic archive. The canonical archive remains unchanged until a complete recrawl verifies additions, removals, and message counts.

### AI coding-assistant clarification

Topic `728256` asks whether Codex or ChatGPT may be used and whether disclosure is required. The only reply says “yes but” and links an unrelated BirdCLEF writeup. It is not an organizer response. No competition-rule change is established by this thread.

### Runtime and hidden-scale inference

Topic `728152` contains an additional participant report that a 14:58 visible-test run timed out and recommends benchmarking on more than the three public wells. Another participant reports about 180 seconds for the visible wells and suggests getting below ten minutes. Combined with the earlier 200-well extrapolation, this supports the existing operational requirement for a pseudo-hidden-scale benchmark; it does not change a model decision.

### PF uncertainty and multimodal hedging

Message `3501604` in topic `727149` reports, as participant evidence:

- heel calibration closes much of an oracle-versus-legal GR-localization diagnostic gap;
- in one fixed protocol, PF improves by about 0.18 ± 0.04 ft across five seeds while DTW is a wash;
- PF posterior spread correlates about +0.23 with actual error;
- branch uncertainty may be useful as a trust feature rather than as a direct predictor;
- formation columns do not supply six independent absolute anchors;
- multimodal errors may be partly unobservable, motivating a posterior mean or bounded hedge;
- nearest-well borrowing helped only at very short spatial distance and hurt at larger distances.

These are self-reported measurements, not verified promotion evidence. The reusable hypothesis is continuous branch/posterior uncertainty. The reported constants and performance values cannot enter an experiment without clean-room reproduction.

## Exact notebook source and output audit

### Shared composite lineage

Exact Kaggle source was pulled for four current frontier notebooks:

| Notebook | Exact source SHA-256 | Pairwise lineage finding |
|---|---|---|
| `arnavsalkade/rogii-full-stack-selector-cv100` | `451442f6d9f06a107a74794e93ec028c9c568f0cd9163065131040b7043a861a` | 99.9472% sequence similarity to the 6.213 “new strategy” source |
| `blacklions/rogii-contact-gated-stratigraphic-alignment` | `66f11958fcc0f7f01171198620e7f5399fa71ac08bdf73126115c0bfcb44ba8b` | same base with stochastic-TTA and a residual gate appended |
| `hityth/ctrl-a27-branch-shape-exact` | `08132add379686c2c7cddd76e8b34f19ed16ad927d439ff1f79221693d39648d` | 98.1380% similarity to the 6.213 source; public-well A27 layer appended |
| `leonidzaporozhets/new-strategy-score-6-213` | `4b4879a6d427422c127a300e09dc763b71ea5e7878eb3639941c75753a23933c` | baseline source for the comparison |

All four activate the same `vp_balanced_modelpkg_005` composite profile, including guarded same-well overlap, visible-prefix calibration, formation-based code, model-package correction, PF seed logic, and the same quarantined artifact lineage. Titles and public scores therefore do not establish independent mechanisms.

The common inputs include unknown-license or otherwise disallowed dependencies such as `phongnguyn23021656/koolbox-offline`, `nina2025/rogii-03`, and `needless090/rogii-tabicl-mirror`. None may enter the prize-targeting pipeline.

### PF seed-branch midpoint hedge

The common source builds likelihood-weighted PF seed medians, divides them into two one-dimensional branches, and applies a capped midpoint shift when both branch masses and their separation pass fixed thresholds.

Exact saved output from `arnavsalkade/rogii-rb594-u-continuity-8` shows:

- `000d7d20`: skipped; branch separation 0.280 ft;
- `00bbac68`: skipped; branch separation 3.594 ft;
- `00e12e8b`: applied; branch separation 29.439 ft, branch masses 0.721/0.279, +2.0 ft on all 4,301 rows.

Thus the public output reproduces a known visible-well-only move. It is not unseen-well evidence for the threshold or shift. The branch masses and separation remain a clean-room feature hypothesis.

### U-continuity fade

The U-continuity notebook source was independently pulled at SHA-256 `a98e718dc8cc3b8e94b74bf76b9d91d3e5af50b59b917d0fef31eb636f46f8ab`. It retains the same composite profile and appends a post-composition boundary fade. Full-stack CV and full-stack ablation are disabled.

Exact output moves are:

- `000d7d20`: maximum 0.0185 ft;
- `00bbac68`: maximum 0.0359 ft;
- `00e12e8b`: maximum 2.0401 ft.

The layer nearly eliminates the visible-to-hidden U boundary gap on the three public wells, but its only material action is again on `00e12e8b`. This supports a generic continuity diagnostic, not the public constants or a deployable correction.

### Full-stack selector CV100

The exact source differs from the 6.213 clone mainly by enabling `RUN_CV_REPORT`, setting 100 wells, and using 32 PF seeds. Its saved summary reports:

- 100 wells and 486,676 rows;
- pooled RMSE 11.3803;
- per-well median 5.8953 and p90 14.7737;
- worst-decile SSE share 68.608%;
- zero bimodal activations;
- zero heel-calibration activations.

This is a small composite sample with no clean ablation and strong remaining tail concentration. It does not establish that the branch detector, overlap layer, model package, or any named subcomponent caused the score.

### Stochastic TTA residual gate

The contact-gated source adds two stochastic test-time replicates, public scores 6.478 and 6.510 embedded in source, and a shallow grouped-OOF residual gate. Exact saved output reports:

- baseline OOF RMSE 10.37225;
- corrected OOF RMSE 10.36846;
- gain 0.00379;
- p90 deterioration 0.06273;
- the gate is inactive with reason `oof_gate_failed`.

The exact 773-well table also shows a slightly larger maximum per-well RMSE after correction. The notebook’s own gate rejects this mechanism. Stochastic disagreement may remain a diagnostic feature, but no correction is justified.

### A27 branch-shape layer

Exact source targets only `00e12e8b`, expects exactly one applied branch well, checks fixed source prediction hashes, restores 10% of a PF-1.3 shape, and caps the final move at 0.40 ft. It explicitly depends on a prior public submission and public-well branch report. This is score-directed visible-well tuning and is excluded.

### Geographic restoration

`paulodmayra/rogii-geologia-v92-geographic-restoration` declares no external datasets, but Kaggle CLI returns a zero-byte source file. No implementation, feature, validation, or causality claim can be audited. The title alone is insufficient.

## Decision

No current public notebook justifies ingestion, cloning, or direct clean-room implementation of its full pipeline.

The only mechanism class worth a local preimplementation screen is **continuous uncertainty-conditioned placement around E011**, because:

- participant evidence independently points to PF posterior spread and branch mass/separation as risk signals;
- hard regimes and local experts were already rejected by E012;
- the public TTA gate itself failed, arguing against another unconstrained residual router;
- exact public branch and continuity corrections act materially only on the visible overlap well.

The next step is therefore not to reproduce the composite public stack. It is to measure, using existing legal OOF artifacts, whether E011 has enough recoverable fallback/blend regret and whether legal uncertainty proxies can predict that regret under repeated whole-well splits. A formal experiment is allowed only if that screen shows material oracle headroom and reproducible cross-fitted learnability.
