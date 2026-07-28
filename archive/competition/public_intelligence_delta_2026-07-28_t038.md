# ROGII public-intelligence mechanism audit — 2026-07-28

This record separates public source visibility from repository-owned evidence.

## Source findings

- `raunakdey07/rogii-ultra-sub-6-rmse` and `evgendvorkin/roggi-physics-lb-7-872-v48` have 0.9296 normalized source-sequence similarity and depend on the same broad pretrained/contact/overlap/model-package stack. Their notebook-level scores are not clean unseen-well evidence.
- `lucifer19/rogii-geoanchor` is partly related to that lineage and retains external model packages, contact logic, and overlap layers.
- `amerhu/rogii-wellbore-geology-exact-hmm-smoother`, source SHA-256 `2321997c8bcbca442d7d7abfc5b9b7eeed251bac8c671b3554b54c9355c231dc`, is genuinely independent and exposes two self-contained algorithmic mechanisms: exact joint TVT/dip-rate HMM smoothing and a sequential likelihood-weighted particle filter.

Public code remains quarantined and cannot be promoted or deployed.

## Repository-owned T038 evidence

The frozen score-blind worth audit evaluated 24 fixed candidates over 20 fold-stratified wells and ten observable edge wells.

- E011 primary-panel RMSE: `12.9126155867`.
- Likelihood-PF standalone: `6.3041182153`.
- Fixed 75% likelihood-PF plus 25% E011: `6.1144320131`.
- Gain: `6.7981835736` RMSE.
- Fold-stratum wins: `5/5`.
- p90: `9.7080123360` versus `18.4191628128`.
- worst-20% SSE share: `0.4780033996` versus `0.5481567638`.
- observable edge-panel gain: `1.1532566573`.
- projected 773-well runtime: `0.5961` hours; observed RSS `0.8553` GB.

Independent rerun exactly reproduced sample memberships, source identity, selected and edge metrics, all branch aggregations, and artifact hashes.

The HMM branches were weaker and edge-unstable. T038 therefore selects the sequential likelihood-PF mechanism for a separate clean-room full-data experiment. T038 predictions and public source code remain ineligible for deployment.
