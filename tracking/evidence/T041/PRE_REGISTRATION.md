# T041 preregistration — public-frontier visible-prefix arbitration worth audit

Status: **frozen before T041 hidden-suffix metrics**  
Hypothesis: `H033`  
Task: `T041`  
Decision: `D045`

## Question

Can the strongest legally reusable mechanisms exposed by high-scoring public notebooks—likelihood PF, exact second-order HMM diversity, and visible-prefix self-verification—produce a robust uplift beyond the full-data 50/50 E011–PF result, while reducing catastrophic minority-well errors?

This is a research-only source audit. Public source code may execute only to decide whether a separate clean-room experiment is worth implementing. Public predictions, mounted models, artifact packages, contacts, formation columns, overlap rules, fixed public-well shifts, score-derived constants, and unknown-license datasets are forbidden from deployment.

## Frozen sample

Use exactly 100 label-independent wells: in each `folds/v1.json` fold, sort well IDs and take 20 equally spaced indices including endpoints. Original hidden targets are not read during prediction-cache generation. They are opened only once during finalization.

## Frozen tracker bank

Run exactly three trackers from the independently identified public mechanism family:

- likelihood PF: 384 particles, 16 seeds, likelihood temperature 5, initial spread 4.5;
- stable HMM: Gaussian/std, step 0.7, emission tempering 0.5, zero-centered rate grid;
- edge HMM: Student-t/MAD, step 0.7, emission tempering 1.0, zero-centered rate grid.

E011 is the immutable fallback. The fixed reference is `0.5 × E011 + 0.5 × PF`, which scored 9.9839055870 on all 773 wells in E013.

## Visible-prefix self-verification

For every sampled well, create two fixed pseudo-cut profiles from the actually visible TVT prefix:

- conservative: 50%, 65%, 75%;
- balanced: 55%, 70%, 84%.

At each cut, hide all later TVT_input rows, run each tracker using the complete test-available horizontal GR sequence, and score only the held-out portion of the original visible prefix. Every cut contributes equally through its MSE. A cut with fewer than 20 held-out visible rows is invalid; every well must retain at least one valid cut per profile.

The primary tracker mixture is permanently fixed before hidden scoring:

`weight_j ∝ 1 / (pseudo_MSE_j + 16)` using the conservative profile.

The primary final path is `0.5 × E011 + 0.5 × weighted_tracker`. Hard selection, balanced-profile inverse-MSE, softmax temperatures 4 and 8, and uniform tracker averaging are diagnostics only and cannot replace the primary after metrics.

## Controls

- Shuffle complete pseudo-score vectors across wells with a deterministic seed.
- Rotate candidate labels in every pseudo-score vector while retaining each well’s predictions.

Each must lose at least 0.25 RMSE versus the legal primary. Exact row identity, finite predictions, deterministic cache reuse, source identity, input immutability, and visible-prefix preservation are mandatory.

## Frozen worth decision

PASS requires every gate in `config.json`, including:

- at least 2.0 RMSE gain versus E011;
- at least 0.25 RMSE gain versus the fixed 50/50 E011–PF reference;
- at least 4/5 fold and 18/25 cell wins versus that reference;
- no p90 deterioration and no more than 0.02 worst-20% SSE-share increase;
- mean within-well tracker rank Spearman at least 0.20 and tracker hit rate at least 40%;
- hidden-label component oracle at or below 6.0 RMSE, proving the bank contains target-level capacity. The oracle set is frozen to E011, the three raw trackers, and the three fixed 50/50 E011–tracker blends; it may choose one complete candidate per well but may not interpolate or tune weights;
- both destructive arbitration controls lose at least 0.25 RMSE;
- projected 773-well runtime at most four hours and RSS below 4 GB.

A pass authorizes only a clean-room E014 implementation and all-773-well validation. A failure rejects this exact tracker bank and self-verification rule without threshold, cut, temperature, or weight retuning.

No Kaggle notebook save, run, submission, or repository push is authorized.
