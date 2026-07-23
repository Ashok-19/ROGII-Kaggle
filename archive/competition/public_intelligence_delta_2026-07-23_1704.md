# ROGII public-intelligence delta — 2026-07-23 17:04 Asia/Kolkata

Previous cutoff: 2026-07-23 14:15 Asia/Kolkata.

This is a bounded pre-preregistration delta check. Exact Kaggle metadata and pulled source are distinguished from titles, public scores, and unsupported mechanism claims.

## Leaderboard and discussions

The visible top ten are unchanged from the 14:15 snapshot: rank 1 is 4.859, rank 2 is 4.905, rank 5 is 5.265, and rank 10 is 5.511. Kaggle New and Recent still report 138 topics. No newly created topic or new material discussion message after the prior cutoff was found.

## Notebook delta

The DateRun search returned several entries after the prior cutoff. Kaggle notebook-info metadata reported older `last_run_time` values for the same current notebook records, so the DateRun timestamp discrepancy is retained as metadata ambiguity rather than interpreted as a new saved-version result.

Five distinct-looking families were source-audited:

1. `mohamadmatali/rogii-pfcfg-bimg` and `prvsiyan/rogii-public-frontier-blend-research-visuals` use the same seven external datasets, including the three quarantined unknown-license inputs. Their normalized source-line Jaccard similarity is 0.939421. The only meaningful PF-config-only line is `SUBMISSION_PROFILE = 'bimodal_guarded'`; the frontier notebook mainly adds visual/public-score appendix code. This is the already rejected composite lineage, not independent evidence.
2. `bohdanbb2000/rogii-dynamic-our-alignment-submission` is a 1.9 KB inference wrapper around external `rogii-our-alignment-public` joblib models and a saved blend manifest. The underlying training and validation evidence is not in the notebook source, and exceptions fall back to zero predictions. It cannot justify implementation.
3. `yuezhengzhang/rogii-wellbore-geology-prediction-v15` uses no external dataset, but directly looks up training TVT for overlap wells and trains a row-level LightGBM residual model on all available training rows with absolute X/Y features. No grouped whole-well validation is implemented in the source. It is not evidence for unseen-well transfer.
4. `sgy2512/rogii-lgb-v47` is not primarily a LightGBM pipeline despite its title. Exact 50 KB source contains XGBoost, PF/DTW-style logic, formation-column processing, grouped CV, feature pruning, and multiple blend layers. It does not isolate or validate a stability-regularized wide residual ridge.

## Decision

The delta does not change the project queue. No new public source authorizes cloning, artifact ingestion, score-derived constants, or a new mechanism ahead of T016. Proceed only with an E011-relative, preimplementation stability screen using existing legal cross-fitted local artifacts. The old E009 gain over E006 must not be assumed to survive placement on E011.
