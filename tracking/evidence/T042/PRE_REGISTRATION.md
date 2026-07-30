# T042 preregistration — posterior-state public-core candidate selector

Status: **frozen before T042 model metrics**  
Hypothesis: `H034`  
Task: `T042`  
Decision: `D046`

## Question

Can target-free diagnostics from the complete public-core paths identify the best whole-well candidate under strict cross-fitting, where direct visible-prefix pseudo-loss ranking failed?

T041's labels and outcome are known and define this 100-well development screen. The remaining 673 wells are untouched by HMM candidate generation and are reserved for a later fixed-model holdout. Nothing from T042 may be promoted directly.

## Frozen inputs and candidate set

Use the exact T041 score-blind caches and the fixed seven-candidate set:

1. E011;
2. likelihood PF;
3. stable Gaussian HMM;
4. heavy-tail HMM;
5. 50/50 E011–PF;
6. 50/50 E011–stable-HMM;
7. 50/50 E011–edge-HMM.

The reference is the 50/50 E011–PF path. Targets are per-well `log(1 + MSE)` for all seven candidates. Every prediction for a v1 fold is fitted only on the other four folds.

## Frozen legal features

Generate one deterministic score-blind feature row per well from:

- known/hidden row counts, MD/Z extent and steps, GR missingness and robust scale;
- typewell TVT/GR support and scale;
- visible-prefix GR residual scale and terminal U-rate;
- each complete tracker path's level, slope, roughness, and disagreement from E011;
- pairwise PF/HMM path disagreement;
- rowwise spread across PF/stable-HMM/edge-HMM;
- HMM posterior mean/p90 standard deviation and normalized log-likelihood.

No T041 pseudo-cut MSE, pseudo ranks, cut counts, or held-out visible-prefix outcomes may enter any feature.

## Frozen model and placement bank

Fit exactly four multi-output predictors from `config.json`: ridge alpha 10, ridge alpha 100, ExtraTrees, and random forest. Evaluate exactly three placements for each:

- hard minimum predicted loss;
- guarded hard selection, reverting to fixed E011–PF unless predicted log-MSE gain is at least 0.10;
- soft complete-path mixture at temperature 0.50.

The lowest-RMSE registered candidate is the screen selection. This selection authorizes only the later untouched holdout if every gate passes.

## Controls

After selecting the legal model/placement by the frozen rule, repeat its outer-fold fitting with:

- deterministic training-target row shuffling;
- cyclic candidate-target column rotation.

Each corrupted control must lose at least 0.25 RMSE versus the legal selection.

## Frozen decision

PASS requires all gates in `config.json`: at least 0.50 RMSE gain versus fixed E011–PF, 4/5 fold and 18/25 cell wins, no p90 deterioration, at most 0.02 worst-20% share increase, mean candidate-rank Spearman at least 0.25, top-candidate hit rate at least 25%, component oracle at most 6.0, and both controls losing at least 0.25 RMSE.

A pass freezes the selected model, feature schema, and placement for an untouched 673-well public-source holdout. A failure closes this exact diagnostic/model family without feature, depth, threshold, target, or placement retuning.

No public artifact, Kaggle run, submission, or repository push is authorized.
