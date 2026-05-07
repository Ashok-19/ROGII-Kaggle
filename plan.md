# Incremental Plan to Beat 11.247

Competition facts verified from Kaggle on 2026-05-06: metric is row-level RMSE on `tvt`; submissions must be notebook submissions, internet disabled, runtime <= 9 hours; the public leaderboard top score is 11.247 and `NNMax` is third at 12.025.

This plan assumes the current notebook remains the base: `notebooks/training_and_submission/lightgbm-public-modified.ipynb`.

## Ground Rules

- Change one idea at a time and keep a local experiment log with CV score, public score, notebook runtime, and changed cells.
- Use GroupKFold by `well` as the primary local check, but repeat it with at least 3 seeds/splits before trusting small gains.
- Report both row-weighted RMSE and per-well RMSE. Kaggle scores rows, but per-well diagnostics catch brittle behavior before it burns submissions.
- Never validate on `data/test`; the three visible test wells are exact visible-column copies of train wells.
- Avoid direct use of train-only horizontal surface columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) in inference features.
- Treat notes from `yt_video_findings/Geosteering_Technologies_playlist.md` as external domain hypotheses, not competition evidence. They are useful only when translated into features/post-processing that improve leakage-safe CV.

## Validation Status

Not every suggestion below is already validated by model CV. The plan intentionally separates dataset-verified facts from experiments that still need implementation and scoring.

Already verified from `data/`, Kaggle metadata, host PPTX, or generated EDA:

- The local visible `data/test` files are only three train-copy examples and are not a validation set.
- Train hidden/evaluation masks are suffixes for all 773 wells.
- Hidden intervals are large: median 4,840 rows and median 74.0% of each well.
- Hidden GR missingness is substantial: median 30.3%, 95th percentile 64.1%.
- Constant `last_known_tvt` is a strong simple baseline: 15.91 row RMSE on all hidden train rows.
- Raw prefix-slope extrapolation is unsafe: `prefix_step100` row RMSE is 116.36.
- Known-prefix typewell alignment has usable signal: known-prefix vs hidden-oracle typewell alignment correlation is 0.454.
- Typewell `Geology` labels are available at inference and cover 66.6% of typewell rows.
- Direct horizontal formation-surface features are unsafe because visible test horizontal files do not include the train-only surface columns.
- Simple nearest-neighbor offset-well target priors are weak globally: nearest-neighbor end-delta correlation is 0.124.

Hypotheses that are grounded in EDA/domain evidence but still need leakage-safe CV before submission:

- GR missingness reliability features.
- Richer typewell residual/recent-window features.
- Apparent-dip proxy and dip-instability features.
- Horizontal GR self-correlation/template features.
- Reversed-window GR matching.
- Azimuth/location/neighbor-prefix features.
- Additional beam parameter sets and jump-capable beam variants.
- Fault-aware smoothing and change-point guards.
- Hidden-fraction or apparent-dip-regime specialist models.
- LightGBM seed/model ensembles and hyperparameter changes.

## Priority 0: Reproducible Local Scoreboard

Add a lightweight experiment table around the current notebook before changing features:

- Current notebook CV folds and mean/std RMSE.
- Public LB score.
- Runtime.
- Feature count.
- Post-processing on/off.

Proof: the train hidden target has 3,783,989 rows across 773 wells, so single-fold or public-only feedback can be noisy. The public/private split is hidden, and the visible local test set is not validation.

## Priority 1: GR Missingness Reliability Features

Add these incremental features to `build_hidden_features`:

- Distance since previous real GR and distance until next real GR.
- Current missing-run length and current missing-run position.
- Local valid-GR fraction for windows 11, 51, 101, 301.
- Prefix GR missing rate, hidden cumulative missing rate, and missingness interactions with `frac_hidden`.

Proof: hidden GR missingness is high: median 30.3%, 75th percentile 49.8%, 95th percentile 64.1%. Long missing runs matter too: 95th percentile longest run is 144 rows.

Expected impact: low-risk feature gain. Current notebook only has `gr_missing` plus interpolated rolling features, so it loses information about interpolation reliability.

Validation tier: dataset-verified problem, unvalidated feature gain.

## Priority 2: Typewell Reliability, Residual, and Geology Features

Extend existing typewell features rather than replacing them:

- Prefix residual mean/std/median/IQR for last 50, 100, 300 known rows.
- Prefix residual slope over TVT and over MD.
- Prefix correlation between horizontal GR and typewell GR in recent windows.
- Candidate typewell GR differences around `last_known_tvt + offset`, but use denser offsets near zero: `[-60,-40,-25,-15,-10,-5,0,5,10,15,25,40,60]`.
- Typewell `Geology` one-hot/count features at the same offsets.

Proof: known-prefix typewell alignment quality predicts hidden alignment quality: per-well correlation between known-prefix GR-vs-typewell RMSE and hidden oracle RMSE is 0.454. Typewell geology labels cover 66.6% of all typewell rows and are available at inference.

Expected impact: medium. This attacks the geological correlation signal directly while staying compatible with hidden test files.

Validation tier: dataset-verified signal, unvalidated feature gain.

## Priority 3: Residual Target Guardrails and Simple Post-Processing

Keep predicting delta from `last_known_tvt`, but test conservative guardrails before adding complex model families:

- Train and score against residual target `TVT - last_known_tvt`, preserving the current notebook's additive final prediction.
- Clip final delta by hidden-fraction quantiles learned from train; test several quantile strengths.
- Add smooth post-processing per well: rolling median/mean blend over predicted deltas, evaluated with windows 5, 11, 21.
- Limit extreme row-to-row jumps using train TVT step quantiles, but do not force monotonicity.

Proof: constant `last_known_tvt` is a strong simple baseline at 15.91 row RMSE. `TVT - last_known_tvt` is centered near zero, with end-delta median 1.40 ft and 5-95% range -31.43 to 32.16 ft. TVT can move down, up, or remain constant according to the host PPTX, so monotonic-only constraints are invalid. Train TVT step median is 0.0, p05 is -0.03, p95 is 0.92.

Expected impact: low to medium. This is a low-complexity way to reduce pathological predictions.

Validation tier: dataset-verified baseline behavior, unvalidated post-processing gain.

## Priority 4: Hidden-Fraction-Aware Features

Add hidden-position interactions before training separate models:

- Interactions between `frac_hidden`, `hidden_len`, `known_len`, and the strongest existing features.
- Separate summary features for early, middle, and late hidden rows.
- Reliability interactions: GR missingness by `frac_hidden`, typewell residual quality by `frac_hidden`, and beam gap by `frac_hidden`.

Proof: error grows strongly with hidden position. Last-known RMSE rises from 4.05 in the first 5% of hidden rows to 20.25 in the last 5%. Prefix-step extrapolation gets dramatically worse with hidden position, so late-zone behavior needs separate handling.

Expected impact: medium. This uses a verified error pattern without changing model architecture.

Validation tier: dataset-verified error pattern, unvalidated feature gain.

## Priority 5: Horizontal GR Self-Correlation Features

Add features that compare hidden GR against known-prefix horizontal GR, not only against the assigned typewell:

- For each hidden row, compute differences to recent known-prefix GR summaries at offsets in prefix TVT space.
- Build rolling-window descriptors of known-prefix GR signatures: mean/std/gradient over windows 25, 51, 101, 201.
- Add nearest-template features: find known-prefix windows whose GR shape best matches the current hidden local GR window, then use their known `TVT_input` deltas as candidate signals.
- Add shape-only matching features: z-score each local GR window before computing correlation/distance, so the model can use pattern similarity even when absolute GR level shifts.
- Add correlation quality and ambiguity features: best match score, second-best score, score gap, and matched-window distance from PS.
- Test a lightweight self-DTW/beam path from hidden GR into the known-prefix horizontal GR track, with strict reliability features and no raw replacement.

Proof: the host PPTX explicitly states that pre-PS horizontal GR can correlate better with post-PS horizontal GR than the typewell GR does. CSV EDA also shows horizontal MD spacing is exactly 1 ft, making local horizontal GR windows naturally aligned to the prediction cadence.

Expected impact: medium to high. This is the biggest new idea from the host deck and is still incremental because it adds features to the existing LightGBM residual model.

Validation tier: host-verified domain hint and dataset-compatible, not yet CV-validated.

## Priority 6: Beam Features, Calibrated Instead of Trusted

Keep beam as features, but test these small changes:

- Add 2-4 more beam parameter sets with different `move_cost`, `emit_scale`, and smoothing radius.
- Add `beam_delta * frac_hidden`, `beam_gap * frac_hidden`, and `abs(beam_delta)` reliability features.
- Add a small post-model blend only if CV proves it: `pred = pred_model * (1-w) + pred_beam * w`, with `w` learned or binned by `frac_hidden`, GR missingness, and prefix typewell RMSE.

Proof: sampled beam baseline is not a good full replacement: row RMSE 18.69 on 97 sampled wells versus 15.91 for last-known across all wells. But by hidden-fraction bin, conservative beam is competitive early: RMSE 3.90 in the first 5% versus 4.05 for last-known, then degrades later.

Expected impact: medium if gated; risky if used as raw output.

Validation tier: partially dataset-verified on sampled beam diagnostics, unvalidated model gain.

## Priority 7: Apparent-Dip and Azimuth Features

Add domain-grounded apparent-dip features after the directly verified signal families:

- Recent apparent-dip proxies: rolling slope of `TVT_input` versus `MD`, `Z`, and lateral distance over windows 25, 50, 100, 300.
- Dip-instability features: difference between recent apparent dip and longer-window apparent dip, plus rolling slope volatility.
- Hidden segment azimuth from PS to each row/end, plus `sin/cos(azimuth)`.
- Distance from PS and projected distance along the hidden lateral.
- Apparent-dip-by-azimuth interactions: recent prefix dip proxy multiplied by `sin/cos(azimuth)` and projected distance.

Proof: host slides say drilling azimuth affects expected geological dip. External geosteering notes emphasize apparent formation dip and warn that average dip can be misleading. Train EDA confirms hidden lateral azimuth is structured, but it does not yet prove apparent-dip features improve CV.

Expected impact: low to medium. Strong domain plausibility, weaker dataset validation than GR/typewell features.

Validation tier: domain-grounded hypothesis, not yet CV-validated.

## Priority 8: Spatial and Offset-Well Features

Add location/direction features and a cautious neighbor-prior experiment:

- Prediction-start `X/Y/Z`, centered or ranked within train.
- Nearest train-well prefix summaries by PS location: neighbor prefix slope, prefix residual quality, GR missingness, known-length, and typewell geology mix.
- Optional neighbor target priors only for CV experiments: kNN mean end-delta/mean-delta from train neighbors, computed out-of-fold to avoid leakage.

Proof: host slides say offset wells can help. Train EDA shows nearest PS neighbors are close and often parallel: median nearest-neighbor distance 478 ft and median nearest-neighbor azimuth difference 0.43 degrees. But the simple nearest-neighbor end-delta correlation is weak at 0.124, so these should be auxiliary features, not a primary model.

Expected impact: low. The direct target-prior evidence is weak, so use these features only after stronger feature families.

Validation tier: mostly negative/weak dataset evidence, still possible as auxiliary signal.

## Priority 9: Reversed-Signature and Fault-Aware Variants

Test these only after the simpler GR/template/beam/post-processing variants have a stable CV baseline:

- Add reversed-template matching. Domain notes say GR signatures can flip when the well turns back up-section, so compare hidden GR windows against both forward and reversed known-prefix/typewell windows.
- Add a fault-tolerant beam variant that permits occasional larger TVT jumps with a high penalty, instead of only `-1/0/+1` index moves.
- Add change-point guards: identify rows where GR shape or predicted delta slope changes abruptly, then reduce smoothing across those boundaries.
- Evaluate two smoothers separately: one continuous smoother for normal intervals and one fault-aware smoother that allows local discontinuities.

Proof: external geosteering notes describe signature reversal when drilling up-section/down-section and mention faults/abrupt repositioning. These are plausible, but the current dataset EDA has not yet quantified fault labels or proven that jump-aware logic improves RMSE.

Expected impact: unknown. Potentially useful on hard wells, but higher risk than basic GR/typewell features.

Validation tier: external-domain hypothesis, not yet dataset-validated.

## Priority 10: Specialist Models

Test specialist models only after feature additions are evaluated in the single-model baseline:

- Early hidden rows: `frac_hidden <= 0.2`.
- Middle hidden rows: `0.2 < frac_hidden <= 0.7`.
- Late hidden rows: `frac_hidden > 0.7`.
- Stable prefix dip.
- Changing prefix dip.
- Ambiguous/low-quality GR correlation.

Proof: hidden-position error growth is verified, but specialist models add training/inference complexity and can overfit folds. Apparent-dip regimes are domain-grounded but not yet CV-proven.

Expected impact: medium if the single-model feature set plateaus; otherwise premature.

Validation tier: model-complexity hypothesis.

## Priority 11: LightGBM Stability Experiments

Run these only after feature additions have a stable CV harness:

- Seed ensemble of 3-5 LightGBM models.
- Slightly lower learning rate with more trees.
- Compare `num_leaves` 63, 127, 255 and `min_child_samples` 50, 100, 200.
- Try `reg_alpha` and larger `reg_lambda` for smoother predictions.
- Save feature importance and fold residuals after each run.

Proof: the public gap from 12.025 to 11.247 is 0.778 RMSE. That is large enough that feature signal matters more than blind parameter search, but small seed ensembles can reduce variance once features are better.

## Things Not To Do Yet

- Do not use train-only formation surface columns as inference features. Visible test horizontal files do not contain them.
- Do not submit raw prefix-slope extrapolation ideas. Prefix-step100 row RMSE is 116.36 and has catastrophic failures above 1,300 RMSE on individual wells.
- Do not rely on the visible local test examples for model selection; all three are train copies.
- Do not replace the model with beam matching. Beam is useful as gated signal, not as a standalone predictor.
- Do not blindly trust offset-well target priors. The host says neighbors can help, but simple nearest-neighbor end-delta correlation is only 0.124 in train.
- Do not convert real-world geosteering advice directly into rules. In this competition, every such idea must become a feature, validation split, or post-processing option with CV evidence.

## Suggested Submission Order

1. Baseline rerun of current notebook with saved CV/fold logs.
2. Add GR missingness reliability features.
3. Add richer typewell residual/recent-window features.
4. Add conservative residual clipping/smoothing guardrails.
5. Add hidden-fraction interactions.
6. Add horizontal GR self-correlation/template features from the known prefix.
7. Add beam interaction/reliability features, no blend yet.
8. Add apparent-dip and azimuth features.
9. Add spatial/offset-well auxiliary features.
10. Add reversed-window GR matching and fault-aware variants only if earlier CV is stable.
11. Try hidden-fraction and apparent-dip-regime specialist models.
12. Try seed ensemble and final parameter polish.

## EDA Evidence Files

- `eda_findings/README.md`
- `eda_findings/tables/summary_stats.json`
- `eda_findings/tables/per_well_summary.csv`
- `eda_findings/tables/baseline_metrics.csv`
- `eda_findings/tables/baseline_rmse_by_hidden_fraction.csv`
- `eda_findings/figures/baseline_rmse_bar.png`
- `eda_findings/figures/hidden_gr_missingness.png`
- `eda_findings/figures/prefix_vs_hidden_typewell_alignment.png`
- `eda_findings/figures/target_delta_by_hidden_fraction.png`
- `eda_findings/host_pptx_notes.md`
- `eda_findings/tables/spatial_offset_diagnostics.csv`
- `eda_findings/figures/spatial_end_delta_map.png`
- `yt_video_findings/Geosteering_Technologies_playlist.md`
