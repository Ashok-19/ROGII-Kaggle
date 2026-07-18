# ROGII Competition Official Snapshot

Snapshot date: **2026-07-18**  
Competition slug: `rogii-wellbore-geology-prediction`  
Competition ID: `132265`

## Objective

Predict `tvt` for every hidden evaluation row of each horizontal well. The visible `test/` directory contains only a few training-derived examples for notebook authoring; the submitted notebook is rerun against roughly 200 hidden test wells.

## Evaluation and submission

- Metric: pooled row-level root mean squared error (RMSE).
- Submission columns: `id,tvt`; output file must be named `submission.csv`.
- Notebook-only competition; internet must be disabled.
- CPU or GPU runtime limit: 9 hours.
- Freely and publicly available external data and pretrained models are allowed, subject to accessibility, licensing, and winner reproducibility requirements.
- Maximum 5 submissions per day; up to 2 final submissions may be selected.
- Maximum team size: 5.

## Timeline

- Start: 2026-05-05.
- Entry and team-merger deadline: 2026-07-29 23:59 UTC.
- Final submission deadline: 2026-08-05 23:59 UTC, equivalent to 2026-08-06 05:29 Asia/Kolkata.

## Current public leaderboard snapshot

- Rank 1: `shu01`, RMSE **4.859**.
- Rank 10: RMSE **5.523**.
- Rank 25: RMSE **6.199**.
- Rank 50: RMSE **6.505**.
- Rank 100: RMSE **6.799**.
- Best submission returned for the current Kaggle account/team search: RMSE **7.119** on 2026-07-16.
- The user-reported 6.888 result is retained as unverified until its Kaggle submission reference is recorded.

## Data scale

Kaggle MCP reports 2,327 competition files: 1,553 CSV files, 773 PNG files, and one PPTX. Local EDA reports 773 training wells and 5,092,255 horizontal-well rows, with 3,783,989 hidden-suffix rows in simulated training evaluation.

## Rules that directly affect project design

- Private sharing of competition code outside an official Kaggle team is prohibited.
- Winning solutions must be reproducible and may require complete training/inference code, environment details, and documentation.
- Public external inputs must be reasonably accessible; unknown or incompatible licenses are a material risk.
- Final ranking is determined by the private leaderboard, not the public leaderboard.

## Evidence source

This summary was derived from Kaggle MCP snapshots archived on 2026-07-18. The full official page snapshot was read before this summary was produced.
