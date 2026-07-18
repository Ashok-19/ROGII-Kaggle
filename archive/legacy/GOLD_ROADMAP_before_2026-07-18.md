# ROGII Wellbore Gold-Medal Roadmap

This roadmap is the working plan for pushing the ROGII wellbore solution from the current public range around `7.153–7.3` toward the current top range around `5.353`.

The main thesis is:

> Do not treat this as a plain tabular regression problem. Treat it as a constrained geosteering / signal-alignment problem with many plausible TVT paths per well, then train a selector/blender to choose or combine the right path per well.

The older `plan.md` is still useful for historical EDA and earlier feature ideas, but it targets an older leaderboard situation around 11 RMSE. This file is the active roadmap for the final gold push.

---

## 0. Current Repo Context

Important existing files:

- `notebooks/training_and_submission/lightgbm-public-modified.ipynb`
  - Current main training/submission notebook in the repo.
- `notebooks/public_notebook/top-2-rank-10-784-physics-informed-baseline.ipynb`
  - Public physics-informed baseline notebook present in the repo.
- `plan.md`
  - Older incremental plan and EDA-backed notes.
- `eda_findings/README.md`
  - Existing EDA summary and artifact map.
- `eda_findings/tables/*`
  - Existing per-well, baseline, spatial, and hidden-zone diagnostics.
- `yt_video_findings/Geosteering_Technologies_playlist.md`
  - Domain notes from geosteering videos.
- `papers/Sakoe-Chiba-DTW.pdf`
  - DTW reference material.
- `papers/chronolog_paper_preprint.pdf`
  - External related paper/reference.

Important known findings from existing EDA:

- Train has 773 wells and about 3.78M hidden/evaluation rows.
- The `TVT_input` mask is a suffix for every train well.
- The visible local test examples are not validation; they are train-copy examples.
- Hidden intervals are large: median hidden share is about 74% of each horizontal well.
- Hidden GR missingness is high: median around 30%, 95th percentile around 64%.
- Last-known TVT baseline is strong relative to naive extrapolation.
- Prefix-slope extrapolation is extremely unsafe.
- Typewell GR alignment has real but noisy signal.
- Typewell `Geology` labels are available at inference.
- Train-only horizontal surface columns are not directly available in visible test horizontal files.
- Beam/path matching is useful as a gated candidate, not as a raw replacement.
- TVT can go up, down, or stay flat; hard monotonic post-processing is invalid.

---

## 1. Winning Architecture

The target architecture should be:

```text
input well files
    -> common loader / feature cache
    -> visible-prefix replay validation harness
    -> candidate generation bank
        -> last-known / public baseline candidates
        -> constrained DTW / Viterbi alignment candidates
        -> NCC / beam-search candidates
        -> particle-filter candidates
        -> typewell/geology alignment candidates
        -> structural U = TVT + Z candidates
        -> local polynomial / spline / Kalman candidates
        -> guarded exact-overlap/rerun candidates
    -> candidate metadata and diagnostics
    -> candidate selector / blender
    -> residual correction model
    -> U-space smoothing and safety guards
    -> submission.csv
```

The competition should be attacked as:

```text
candidate generation problem + candidate selection problem + small residual correction problem
```

not as:

```text
one giant LightGBM directly predicts TVT
```

Tree models should not be the only intelligence. They should help select, weight, and correct physically meaningful path candidates.

---

## 2. Core Principles

### 2.1 Preserve multiple hypotheses until late

A horizontal well can match several plausible typewell intervals or structural paths. Early averaging can destroy the winning mode.

Keep multiple candidates per well and only select/blend after computing candidate confidence.

### 2.2 Validate exactly like the test setup

The train setup allows a realistic replay:

1. Pick a train well.
2. Cut the known prefix earlier than the real prediction start.
3. Hide the suffix after the artificial cut.
4. Run the full inference pipeline using only the artificial prefix.
5. Score the hidden artificial suffix.

This simulates the Kaggle task better than random rows or simple GroupKFold alone.

### 2.3 Score by well, not only by row

Kaggle uses row-level RMSE, but per-well diagnostics are essential. One catastrophic long well can dominate the LB.

Every experiment must report:

- row-weighted RMSE
- mean per-well RMSE
- median per-well RMSE
- p75 per-well RMSE
- p90 per-well RMSE
- worst 20 wells
- score by hidden-fraction bucket
- score by GR-missingness bucket
- score by candidate-disagreement bucket

### 2.4 Smooth geological structure, not raw TVT

Use:

```text
U = TVT + Z
TVT = U - Z
```

Smooth and regularize `U`, then convert back to TVT. This is usually more geological than smoothing raw TVT because raw TVT is affected by borehole trajectory changes.

### 2.5 Public LB is confirmation, not the main optimizer

Use public LB to confirm large trends only. Do not chase tiny public movements unless replay CV, per-well stability, and candidate diagnostics agree.

---

## 3. Required Repo Organization

Create these directories as implementation proceeds:

```text
src/
  rogii/
    __init__.py
    data.py
    metrics.py
    validation.py
    feature_base.py
    candidates/
      __init__.py
      last_known.py
      dtw_viterbi.py
      ncc_beam.py
      particle_filter.py
      typewell.py
      structural.py
      local_models.py
      overlap.py
    selector/
      __init__.py
      build_meta.py
      train_selector.py
      blend.py
    postprocess.py
    submission.py

experiments/
  exp_000_baseline_replay/
  exp_001_candidate_bank_v1/
  exp_002_dtw_viterbi_v1/
  exp_003_selector_v1/
  ...

artifacts/
  replay_splits/
  candidates/
  candidate_meta/
  selectors/
  submissions/
  reports/

logs/
  experiment_log.csv
  submission_log.csv
```

The notebook can remain as the final Kaggle submission wrapper, but core logic should gradually move into importable Python modules so experiments become reproducible.

If Kaggle notebook submission cannot import local modules directly, copy the final compact versions into the submission notebook after validation.

---

## 4. Phase 0 — Freeze Baselines and Build a Scoreboard

### Goal

Lock the current working solution and create a reliable scoreboard before changing logic.

### Tasks

1. Save/freeze the current best notebook state.
2. Record exact public LB scores from each submitted notebook.
3. Create `logs/experiment_log.csv` with columns:

```text
experiment_id,date,notebook_or_script,main_change,cv_row_rmse,cv_well_mean_rmse,cv_well_p90_rmse,public_lb,runtime_minutes,notes
```

4. Create `logs/submission_log.csv` with columns:

```text
submission_id,date,source_experiment,public_lb,private_lb_if_known,blend_description,notes
```

5. Add a minimal local scorer:

```python
rmse = sqrt(mean((pred - true) ** 2))
```

6. Add per-well scorer:

```python
well_rmse = df.groupby('well_id').apply(lambda g: rmse(g.pred, g.true))
```

### Deliverables

- `logs/experiment_log.csv`
- `logs/submission_log.csv`
- `artifacts/reports/baseline_replay_report.md`

### Done when

- Current notebook can be rerun or at least documented as baseline.
- Baseline score is recorded locally and on LB.
- We know whether current local validation agrees with public LB directionally.

---

## 5. Phase 1 — Visible-Prefix Replay Validation Harness

### Goal

Build the validation system that will decide every later experiment.

### Replay design

For every train well, generate artificial prediction starts at multiple cut fractions.

Recommended cut fractions:

```python
CUT_FRACS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
```

But not every well should use all cuts if the known prefix becomes too small. Guardrails:

```text
minimum prefix rows: 300
minimum hidden rows: 300
minimum real GR rows in prefix: configurable, start with 50
```

### Replay object schema

Each replay case should have:

```text
replay_id
well_id
cut_frac
prefix_start_idx
prefix_end_idx
hidden_start_idx
hidden_end_idx
prefix_len
hidden_len
prefix_gr_missing_rate
hidden_gr_missing_rate
last_known_tvt
last_known_md
last_known_x
last_known_y
last_known_z
```

### Validation split types

Use multiple split views:

1. `all_replay_cases`
   - Most data, best for candidate diagnostics.
2. `one_cut_per_well`
   - Avoids repeated-well overconfidence.
3. `hard_replay_cases`
   - High GR missingness, long hidden intervals, unstable prefix trend.
4. `late_cut_cases`
   - Cuts where most of the well is already known.
5. `early_cut_cases`
   - Cuts with long future horizons.

### Metrics

For each model/candidate:

```text
row_rmse
well_mean_rmse
well_median_rmse
well_p75_rmse
well_p90_rmse
rmse_by_cut_frac
rmse_by_hidden_frac_bin
rmse_by_md_since_bin
rmse_by_gr_missing_bin
rmse_by_prefix_typewell_corr_bin
```

### Deliverables

- `src/rogii/validation.py`
- `src/rogii/metrics.py`
- `artifacts/replay_splits/replay_cases.parquet`
- `artifacts/reports/replay_validation_report.md`

### Done when

- Any candidate generator can be evaluated on the same replay cases.
- We can compare current notebook-style predictions against simple baselines under replay.
- We can identify worst wells and worst regimes.

---

## 6. Phase 2 — Candidate Bank Framework

### Goal

Create a standard interface for many TVT prediction candidates.

### Candidate output schema

Every candidate generator must output:

```text
case_id or well_id
row_id or row_idx
candidate_name
pred_tvt
candidate_family
candidate_version
```

Candidate-level metadata:

```text
case_id or well_id
candidate_name
prefix_rmse
prefix_tail_rmse
prefix_derivative_rmse
prefix_corr
prefix_ncc
path_cost
path_smoothness
path_jump_count
local_step_p95
local_step_max_abs
candidate_mean
candidate_std
candidate_min
candidate_max
candidate_end_delta
candidate_runtime_ms
status
```

### Candidate families

Minimum first bank:

```text
last_known_constant
current_notebook_baseline
public_physics_baseline
simple_U_linear_tail
simple_U_poly2_tail
simple_typewell_nearest
ncc_strict
ncc_loose
beam_strict
beam_loose
dtw_viterbi_strict
dtw_viterbi_balanced
dtw_viterbi_loose
surface_geology_anchor
local_spline_U
kalman_U_constant_slope
```

Later bank:

```text
particle_filter_low_noise
particle_filter_medium_noise
particle_filter_high_noise
particle_filter_derivative_likelihood
dtw_gr_rank
dtw_gr_derivative
dtw_gr_highpass
dtw_gr_multiscale
reversed_signature_match
fault_jump_beam
contact_overlap_guarded
selector_blend_v1
selector_blend_v2
```

### Candidate registry

Create a registry like:

```python
CANDIDATE_REGISTRY = {
    'last_known_constant': generate_last_known,
    'dtw_viterbi_strict': generate_dtw_viterbi_strict,
    ...
}
```

### Deliverables

- `src/rogii/candidates/__init__.py`
- `src/rogii/candidates/last_known.py`
- `src/rogii/candidates/local_models.py`
- `src/rogii/candidates/typewell.py`
- `artifacts/candidates/replay_candidates_v1.parquet`
- `artifacts/candidate_meta/replay_candidate_meta_v1.parquet`

### Done when

- At least 10 candidates can be generated for every replay case.
- Candidate oracle score is computed.
- We know the gap between current final prediction and oracle candidate selection.

---

## 7. Phase 3 — Strong Constrained DTW / Viterbi Alignment

### Goal

Add the most important signal-modeling candidate family: constrained GR-to-typewell alignment.

### Why this matters

The wellbore problem is continuous and sequential. A plain tree model will struggle to infer long-range signal alignment from raw row features. Constrained sequence alignment directly models the problem.

### State definition

For each horizontal row `i`, define possible typewell/TVT states `j`.

```text
i = horizontal hidden row index
j = typewell row or TVT-grid index
```

A path is:

```text
j_0, j_1, ..., j_n
```

Each state maps to:

```text
pred_tvt_i = typewell_tvt[j_i] + learned_offset
```

or directly to a TVT grid value.

### Emission costs

Test multiple cost functions:

```text
abs(GR_horiz - GR_typewell)
abs(zscore_GR_horiz - zscore_GR_typewell)
abs(dGR_horiz - dGR_typewell)
abs(rank_GR_horiz - rank_GR_typewell)
abs(highpass_GR_horiz - highpass_GR_typewell)
local_window_ncc_cost
```

Combined cost:

```text
emission =
    w_gr * abs_gr
  + w_z * abs_zscore
  + w_d1 * abs_derivative
  + w_rank * abs_rank
  + w_ncc * ncc_cost
```

### Transition costs

Use geological/path penalties:

```text
transition =
    w_step * abs(j_i - j_{i-1})
  + w_accel * abs((j_i - j_{i-1}) - (j_{i-1} - j_{i-2}))
  + w_jump * indicator(abs(step) > jump_threshold)
  + w_start * abs(j_0 - last_known_state)
```

Candidate variants:

```text
dtw_viterbi_strict: low max step, high jump penalty
dtw_viterbi_balanced: moderate step, moderate jump penalty
dtw_viterbi_loose: larger step, fault-tolerant
dtw_viterbi_derivative: derivative-heavy emission
dtw_viterbi_rank: rank-normalized emission
dtw_viterbi_highpass: high-frequency GR event matching
dtw_viterbi_multiscale: combined smoothed + derivative + raw
```

### Missing GR handling

If horizontal GR is missing/interpolated:

```text
increase emission uncertainty
lower GR mismatch weight
increase smoothness/path prior weight
record missingness diagnostics
```

Never let interpolated GR dominate the path.

### Prefix calibration

Use known prefix to estimate:

```text
best offset
best cost weights
prefix path error
prefix correlation
prefix ambiguity
```

Candidate confidence should depend heavily on prefix backtest quality.

### Deliverables

- `src/rogii/candidates/dtw_viterbi.py`
- `artifacts/candidates/replay_dtw_candidates.parquet`
- `artifacts/reports/dtw_candidate_report.md`

### Done when

- DTW/Viterbi candidates beat last-known baseline in at least some replay regimes.
- Oracle candidate score improves materially when DTW candidates are added.
- Bad DTW cases are identifiable by metadata.

---

## 8. Phase 4 — NCC, Beam, and Particle Filter Candidates

### Goal

Convert existing notebook-style path logic into diverse candidates with metadata.

### NCC candidates

Generate local normalized-correlation candidates using:

```text
raw GR
smoothed GR windows: 5, 15, 31, 61
z-scored GR
GR derivative
GR second derivative
rank-normalized GR
high-pass GR
```

Metadata:

```text
best_ncc_score
second_best_ncc_score
ncc_gap
ncc_entropy
best_offset
offset_stability
window_valid_gr_fraction
```

### Beam candidates

Beam search variants:

```text
beam_strict_smooth
beam_balanced
beam_loose
beam_fault_jump
beam_derivative_likelihood
beam_rank_likelihood
beam_prefix_calibrated
```

Key metadata:

```text
beam_cost
beam_gap
beam_num_competing_paths
beam_step_p95
beam_jump_count
beam_smoothness
```

### Particle filter candidates

Particle filter variants:

```text
pf_low_noise
pf_medium_noise
pf_high_noise
pf_high_momentum
pf_low_momentum
pf_derivative_likelihood
pf_rank_likelihood
pf_typewell_geology_weighted
```

Key metadata:

```text
pf_mean
pf_median
pf_p10
pf_p90
pf_std
pf_effective_sample_size
pf_log_likelihood
pf_resample_count
```

Use the PF spread as uncertainty; it is valuable for selector features.

### Deliverables

- `src/rogii/candidates/ncc_beam.py`
- `src/rogii/candidates/particle_filter.py`
- `artifacts/candidates/replay_path_candidates_v1.parquet`
- `artifacts/candidate_meta/replay_path_meta_v1.parquet`

### Done when

- Path candidates are stored separately, not directly blended.
- Candidate oracle improves over Phase 2.
- We can tell when beam/PF is trustworthy.

---

## 9. Phase 5 — Structural and Geology-Aware Candidates

### Goal

Use the geology labels and `U = TVT + Z` structure to create robust non-GR candidates.

### U-space candidates

For visible prefix:

```text
U_input = TVT_input + Z
```

Fit local models:

```text
U vs MD
U vs Z
U vs lateral_distance
U vs projected_distance
U vs X/Y/Z
```

Candidate variants:

```text
U_tail_linear_80
U_tail_linear_160
U_tail_linear_320
U_poly2_tail_160
U_poly2_tail_320
U_spline_smooth_low
U_spline_smooth_high
U_kalman_constant_slope
U_kalman_changing_slope
```

Then convert:

```text
pred_tvt = pred_U - Z
```

### Geology/typewell candidates

Use typewell `Geology` labels and GR:

```text
ANCC
ASTNU
ASTNL
EGFDU
EGFDL
BUDA
```

Important: direct train-only horizontal surface columns are not available at inference, but typewell geology is available. Use geology as candidate/feature information from typewell, not as direct hidden horizontal surface columns.

Candidate ideas:

```text
nearest_geology_boundary_anchor
same_geology_interval_center
same_geology_interval_top
same_geology_interval_bottom
geology_transition_match
geology_weighted_typewell_alignment
```

Metadata:

```text
geology_label_at_candidate
geology_boundary_distance
geology_interval_thickness
geology_transition_count
geology_match_confidence
```

### Deliverables

- `src/rogii/candidates/structural.py`
- `src/rogii/candidates/typewell.py`
- `artifacts/candidates/replay_structural_candidates_v1.parquet`
- `artifacts/reports/structural_candidate_report.md`

### Done when

- U-space candidates are competitive in high-GR-missingness cases.
- Geology/typewell candidates improve oracle score or selector stability.

---

## 10. Phase 6 — Guarded Exact-Overlap / Rerun Logic

### Goal

Safely exploit exact duplicate/rerun information if present without causing catastrophic leakage-like mistakes.

### Rule

Only use overlap reconstruction when there is strong evidence that a test well corresponds to a train well along the same MD/trajectory.

### Matching checks

For a candidate train well and test well:

```text
well_id match or strong file identity signal
MD overlap consistency
X/Y/Z consistency
GR visible-prefix consistency
TVT_input visible-prefix consistency if available
row count / MD step consistency
```

Use only if:

```text
prefix_tvt_rmse < strict_threshold
prefix_xyz_rmse < strict_threshold
prefix_md_match_ratio high
```

### Output

This must be a candidate, not an unconditional override:

```text
candidate_name = overlap_guarded_train_lookup
```

Selector can trust it heavily if confidence is near exact.

### Deliverables

- `src/rogii/candidates/overlap.py`
- overlap diagnostics table
- report of all matched wells and confidence

### Done when

- The logic refuses ambiguous matches.
- On replay validation, exact-overlap candidates do not damage non-overlap wells.

---

## 11. Phase 7 — Candidate Oracle and Diagnostics

### Goal

Know the theoretical value of the candidate bank before training selector.

### Candidate oracle metrics

For replay cases where true TVT is known:

```text
best_candidate_per_case = argmin(candidate_rmse)
oracle_blend_score = score(best candidate per replay case)
```

Also compute row-level and segment-level oracle:

```text
best candidate per well
best candidate per hidden-fraction segment
best candidate per row
```

Interpretation:

```text
If best single candidate = 7.0 and oracle per-well = 4.8:
    selector is the bottleneck.

If best single candidate = 7.0 and oracle per-well = 6.7:
    candidate generation is weak.

If oracle per-row = very low but per-well oracle is modest:
    segment-level blending may help.
```

### Diagnostic tables

Produce:

```text
candidate_scoreboard.csv
candidate_win_rate_by_regime.csv
candidate_oracle_gap.csv
worst_wells_by_candidate.csv
candidate_correlation_matrix.csv
candidate_diversity_report.csv
```

### Deliverables

- `src/rogii/selector/build_meta.py`
- `artifacts/reports/candidate_oracle_report.md`
- `artifacts/reports/candidate_scoreboard.csv`

### Done when

- We know the candidate bank's maximum possible gain.
- We know which candidate families are worth improving.

---

## 12. Phase 8 — Candidate Selector / Blender

### Goal

Train the actual winning layer: per-well or per-segment candidate selection/blending.

### Selector levels

#### Level A — Per-well best-candidate classifier

Input:

```text
well-level metadata + candidate metadata
```

Output:

```text
candidate_name
```

Use:

```text
CatBoost classifier
LightGBM multiclass
simple rule baseline
```

#### Level B — Candidate ranker

Input row:

```text
one row per replay_case + candidate
```

Target:

```text
candidate hidden RMSE or rank
```

Use:

```text
LightGBM ranker
CatBoost ranker/regressor
```

Output:

```text
candidate weights from inverse predicted error
```

#### Level C — Soft blender

Input:

```text
candidate predictions + metadata + row features
```

Output:

```text
final_tvt
```

Start with constrained linear/ridge blending before complex models.

### Selector features

Well-level features:

```text
prefix_len
hidden_len
known_fraction
last_known_tvt
last_known_z
hidden_z_span
hidden_md_span
hidden_gr_missing_rate
prefix_gr_missing_rate
prefix_gr_std
prefix_gr_entropy
prefix_tvt_slope_recent
prefix_U_slope_recent
azimuth
lateral_distance
```

Candidate-level features:

```text
candidate_family
prefix_rmse
prefix_tail_rmse
prefix_corr
path_cost
path_smoothness
jump_count
candidate_end_delta
candidate_std
candidate_disagreement_from_median
candidate_disagreement_from_last_known
ncc_gap
pf_std
beam_gap
```

Row/segment-level features:

```text
md_since
frac_hidden
z
dz_dmd
gr
gr_missing
candidate_spread_at_row
candidate_p10_p90_width
```

### Conservative blending rule

Early selector submissions should be conservative:

```text
final = 0.70 * selector_blend + 0.30 * locked_baseline
```

Then reduce baseline weight only if replay and LB both agree.

### Deliverables

- `src/rogii/selector/train_selector.py`
- `src/rogii/selector/blend.py`
- `artifacts/selectors/selector_v1.pkl`
- `artifacts/reports/selector_v1_report.md`

### Done when

- Selector beats best fixed blend in replay.
- Selector reduces p90 per-well error, not only row RMSE.
- Selector does not collapse onto one candidate everywhere.

---

## 13. Phase 9 — Residual Correction Model

### Goal

Use tree models where they are strongest: correcting systematic residuals after candidate selection.

### Target

Do not train raw TVT directly as the main prediction.

Train:

```text
residual = true_tvt - selected_candidate_tvt
```

or:

```text
residual_U = true_U - selected_candidate_U
```

### Features

```text
base_selected_pred
candidate_spread
candidate_rank_margin
frac_hidden
md_since
GR
GR_missing
GR_valid_fraction_local
Z
dZ_dMD
prefix_slope_features
path_confidence_features
geology_features
selector_confidence
```

### Models

Try:

```text
LightGBM residual regressor
CatBoost residual regressor
Ridge residual model
Huber/quantile residual model
```

### Guardrails

Clip residuals by replay-learned quantiles:

```text
clip by hidden-fraction bin
clip by candidate-confidence bin
clip by GR-missingness bin
```

Never allow the residual model to destroy high-confidence exact/path candidates.

### Deliverables

- residual model training script/notebook section
- residual OOF predictions
- residual correction report

### Done when

- Residual model improves replay row RMSE and p90 well RMSE.
- Residual correction is small and stable.
- Large residual corrections are explainable by low candidate confidence.

---

## 14. Phase 10 — Adaptive U-Space Post-Processing

### Goal

Reduce noisy predictions without erasing real formation boundary events.

### Post-processing space

Convert candidate/final predictions to:

```text
U_pred = TVT_pred + Z
```

Smooth `U_pred`, then return:

```text
TVT_pred = U_smoothed - Z
```

### Smoother variants

```text
rolling median U
rolling mean U
Savitzky-Golay U
lowess-like local linear U
Kalman smoother U
edge-preserving smoother using GR change-points
```

### Adaptive rules

Use stronger smoothing when:

```text
candidate disagreement is low
GR is missing/interpolated
path confidence is low but structural trend is stable
```

Use weaker smoothing when:

```text
sharp GR event exists
candidate confidence is high
fault/jump candidate is selected
GR derivative is high
```

### Safety guards

```text
limit extreme row-to-row TVT jumps by replay quantiles
limit extreme U curvature except at change-points
avoid hard monotonicity
preserve first hidden row continuity from last known TVT
```

### Deliverables

- `src/rogii/postprocess.py`
- post-processing sweep report
- final smoothing configuration table

### Done when

- Smoothing improves replay p90 wells.
- It does not worsen high-confidence alignment cases.
- It reduces pathological jaggedness without flattening true changes.

---

## 15. Phase 11 — Ensembling Strategy

### Goal

Build a stable final ensemble from diverse strong systems.

### Ensemble members

Target final ensemble should include:

```text
locked_current_baseline
candidate_selector_v1
candidate_selector_v2
selector_plus_residual_lgbm
selector_plus_residual_catboost
DTW-heavy variant
PF/beam-heavy variant
U-structural-heavy variant
conservative low-risk variant
```

### Blend selection

Use replay OOF predictions to train final blend weights.

Start with non-negative ridge:

```text
minimize replay RMSE
weights >= 0
sum(weights) = 1
```

Also test:

```text
per-regime blend weights
per-hidden-fraction blend weights
per-confidence blend weights
```

### Anti-overfit constraints

Reject blends that:

```text
overfit one cut fraction
improve row RMSE but worsen p90 well RMSE heavily
put too much weight on a public-LB-only variant
collapse to a single risky candidate
```

### Deliverables

- final OOF blend table
- final submission blend recipe
- submission report

### Done when

- Final ensemble improves replay consistently.
- Public LB improvement is not contradicted by replay diagnostics.

---

## 16. Phase 12 — Final Competition Push

### Goal

Use the last weeks efficiently with controlled submissions.

### Daily workflow

1. Pick exactly one main experiment.
2. Run replay validation.
3. Save OOF predictions and report.
4. Compare against locked baseline and previous best.
5. Submit only if local evidence is strong.
6. Record result in `logs/submission_log.csv`.

### Submission decision rule

Submit if at least two of these are true:

```text
row RMSE improves clearly
well-mean RMSE improves
p90 well RMSE improves
hard-case RMSE improves
candidate oracle gap shrinks
public-risk appears low
```

Do not submit if:

```text
only a tiny row RMSE gain exists
p90 well RMSE gets worse
improvement is isolated to one replay cut
variant relies on aggressive hard overrides
```

### Final week strategy

Keep three solution tracks:

```text
Track A: conservative stable blend
Track B: best replay CV blend
Track C: aggressive public-LB blend
```

Final submissions should include at least one conservative stable blend and one best-CV blend. Use the aggressive public-LB blend only if it also has tolerable replay diagnostics.

---

## 17. Implementation Order

This is the exact implementation sequence to follow one by one.

### Step 1 — Scoreboard and replay harness

Files to create/edit:

```text
src/rogii/data.py
src/rogii/metrics.py
src/rogii/validation.py
logs/experiment_log.csv
logs/submission_log.csv
```

Outcome:

```text
We can run visible-prefix replay validation and produce per-well reports.
```

### Step 2 — Baseline candidate framework

Files:

```text
src/rogii/candidates/__init__.py
src/rogii/candidates/last_known.py
src/rogii/candidates/local_models.py
```

Outcome:

```text
Last-known, simple U-linear, U-poly, and current baseline candidates are stored in a common format.
```

### Step 3 — Candidate oracle report

Files:

```text
src/rogii/selector/build_meta.py
artifacts/reports/candidate_oracle_report.md
```

Outcome:

```text
We know whether candidate generation or candidate selection is the bottleneck.
```

### Step 4 — DTW/Viterbi candidate family

Files:

```text
src/rogii/candidates/dtw_viterbi.py
```

Outcome:

```text
Constrained sequence alignment candidates are available and scored.
```

### Step 5 — NCC/beam/PF candidate families

Files:

```text
src/rogii/candidates/ncc_beam.py
src/rogii/candidates/particle_filter.py
```

Outcome:

```text
The path-search candidates from notebook logic become reusable artifacts with confidence metadata.
```

### Step 6 — Typewell/geology and structural U candidates

Files:

```text
src/rogii/candidates/typewell.py
src/rogii/candidates/structural.py
```

Outcome:

```text
High-GR-missingness and geology-driven regimes get stronger candidates.
```

### Step 7 — Selector v1

Files:

```text
src/rogii/selector/train_selector.py
src/rogii/selector/blend.py
```

Outcome:

```text
Per-well candidate selector beats fixed candidate averages.
```

### Step 8 — Residual correction

Files:

```text
selector/residual training script or notebook section
```

Outcome:

```text
Tree models correct candidate errors instead of predicting raw TVT from scratch.
```

### Step 9 — Adaptive post-processing

Files:

```text
src/rogii/postprocess.py
```

Outcome:

```text
U-space smoothing improves stability without invalid monotonic assumptions.
```

### Step 10 — Final ensemble and Kaggle notebook integration

Files:

```text
src/rogii/submission.py
notebooks/training_and_submission/final_submission.ipynb
```

Outcome:

```text
Best validated pipeline is converted into a Kaggle-compatible notebook.
```

---

## 18. Experiment Naming Convention

Use clear experiment IDs:

```text
exp_000_locked_current_baseline
exp_001_replay_harness
exp_002_candidate_bank_v1
exp_003_dtw_viterbi_strict
exp_004_dtw_viterbi_multiscale
exp_005_beam_pf_bank
exp_006_structural_U_bank
exp_007_selector_v1
exp_008_selector_v2_ranker
exp_009_residual_lgbm
exp_010_u_space_postprocess
exp_011_final_blend_v1
```

Each experiment folder should contain:

```text
config.json
metrics.csv
per_well_metrics.csv
candidate_scoreboard.csv
notes.md
```

---

## 19. Metrics That Matter Most

Primary:

```text
replay row-weighted RMSE
public LB RMSE
```

Secondary but critical:

```text
mean per-well RMSE
p90 per-well RMSE
worst-20 well RMSE
hard-case replay RMSE
high-GR-missingness RMSE
long-hidden-interval RMSE
```

Candidate-specific:

```text
best single candidate RMSE
candidate oracle RMSE
selector RMSE
selector-vs-oracle gap
candidate family win rate
candidate family diversity
```

Gold-push diagnostic:

```text
If oracle is near 5 but selector is near 7:
    train selector better.
If oracle is near 7:
    generate better candidates.
If selector is good locally but LB bad:
    validation split or leakage assumptions are wrong.
```

---

## 20. Risk Register

### Risk 1 — Public LB overfitting

Mitigation:

- Use replay validation.
- Track p90 well RMSE.
- Maintain conservative blend.

### Risk 2 — Candidate bank becomes too slow

Mitigation:

- Cache candidates to parquet.
- Precompute typewell arrays.
- Use numba/numpy for Viterbi inner loops.
- Start with fewer candidate variants.

### Risk 3 — DTW overfits noise

Mitigation:

- Constrain transitions.
- Penalize jumps.
- Downweight missing/interpolated GR.
- Require prefix calibration metadata.

### Risk 4 — Tree selector overfits replay cases

Mitigation:

- Group by well.
- Use one-cut-per-well validation view.
- Compare rule selector vs ML selector.
- Avoid high-cardinality leakage features.

### Risk 5 — Smoothing erases true faults/boundaries

Mitigation:

- Smooth U, not raw TVT.
- Use GR change-point guards.
- Allow fault/jump candidate to bypass heavy smoothing.

### Risk 6 — Exact-overlap logic causes catastrophic wrong overrides

Mitigation:

- Treat as candidate only.
- Require strict prefix identity checks.
- Log all matched wells.
- Never use weak fuzzy match as hard replacement.

---

## 21. What Not To Do

Do not:

- Chase public LB with random blend weights.
- Add more LightGBM seeds before fixing candidate generation/selection.
- Trust the visible local test files as validation.
- Use train-only horizontal surface columns directly at inference.
- Force monotonic TVT.
- Use raw prefix-slope extrapolation.
- Average all candidates too early.
- Let interpolated/missing GR dominate alignment.
- Submit a variant that only improves row RMSE while worsening p90 well RMSE heavily.

---

## 22. First Implementation Target

The first implementation should be:

```text
Phase 1: visible-prefix replay validation harness
```

This is the foundation. Without it, every later change will be public-LB guessing.

Minimum first deliverable:

```text
python script or notebook cell that:
    loads train wells
    creates replay cases at cut fractions
    evaluates last-known baseline
    outputs row RMSE + per-well RMSE + hidden-fraction bucket RMSE
```

After that, implement candidate bank v1.

---

## 23. Final Target State

A realistic gold-medal pipeline should look like this:

```text
1. Load train/test wells.
2. Build replay validation cases.
3. Generate 30–100 candidate TVT paths per well.
4. Compute candidate metadata and confidence.
5. Train selector/ranker/blender on replay OOF data.
6. Apply selector to test candidates.
7. Apply small residual correction.
8. Smooth/guard in U = TVT + Z space.
9. Blend conservative + best-CV + DTW-heavy + structural-heavy variants.
10. Export Kaggle-compatible notebook submission.
```

The main performance jump should come from:

```text
candidate oracle improvement
+ selector improvement
+ safe residual correction
+ U-space post-processing
```

not from blindly adding more tree models.
