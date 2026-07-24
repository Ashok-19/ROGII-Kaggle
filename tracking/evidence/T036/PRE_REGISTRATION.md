# T036 Preregistration — Candidate-path-conditioned GR evidence ranking

## Question

Can test-available hidden horizontal GR and typewell GR identify a sub-5 residual action **when the evidence is recomputed under each proposed TVT path**, rather than predicting one global coefficient or coordinate vector from the well alone?

T035 closes global rank-8 coordinate identification: the best legal model reaches 12.267294 RMSE and retains only 2.73% of the T033 oracle gain. This does not test candidate-conditioned evidence. E010 predicts family coefficients and RMSE from well-level features, while T023 predicts one residual profile using mismatch channels evaluated only around E011/E006. Neither compares competing high-capacity paths by the GR likelihood induced by each path.

## Frozen candidate bank

For every inner or outer training partition, fit the exact T033 32-center residual-profile KMeans dictionary on training wells only. The candidate set contains the 32 centers plus every distinct two-center mixture at weights 0.25, 0.50, and 0.75, yielding exactly 1,520 unique 128-node actions. Add one exact zero action as the E011 fallback, for 1,521 candidates.

The five-map hidden-label oracle of the nonzero dictionary must reproduce T033 `kmeans_pair_k032` at 4.628457367010 RMSE within numerical tolerance. Candidate truth is used only for training labels, inner selection, oracle diagnostics, and held-out scoring. Query hidden TVT never enters candidate construction or features.

## Path-conditioned evidence

For each candidate, add its action to the exact E011 path, sample at most 128 finite hidden horizontal-GR rows, and interpolate typewell GR at the candidate TVT values. Visible-prefix calibration is the only calibration. The frozen feature vector contains support, robust mismatch, correlation, derivative, four-segment mismatch/correlation, action-shape, path-support, and legal well-context channels listed in `config.json`.

This evidence is materially different from a global coordinate model because the typewell reference values change for every candidate path.

## Frozen models and selection

The registered branches are:

- fixed robust-alignment score;
- pointwise ridge regression;
- pointwise histogram gradient boosting;
- pointwise ExtraTrees;
- pairwise logistic ranking.

Pointwise models predict within-well log loss relative to exact E011. The pairwise model learns candidate preference from feature differences. Each outer context uses one frozen inner validation fold for model, hyperparameter, hard/soft output, top-k, and temperature selection by actual-row TVT RMSE. The outer model is then refit on all outer-training wells. Five map actions are averaged before final scoring.

Training uses exactly 64 deterministic candidate examples per well: zero action, eight best, eight worst, eight median-near, and 39 hash-selected remaining candidates. Pairwise training uses the frozen 64-pair construction. Held-out inference scores all 1,521 candidates.

## Controls

The full selected pipeline is repeated with reversed typewell GR, circularly shifted hidden horizontal GR, action/well priors only, and shuffled within-well training losses. A sign-flipped selected action, candidate-order permutation, duplicate-action invariance, exact zero fallback, all-missing GR, invalid calibration, short support, long suffix, and finite/cap contracts are also tested.

## Decision

- **BREAKTHROUGH:** legal RMSE <=5.0, all five maps and every legacy group improve, tail gates pass, and all destructive controls fail.
- **GO:** legal RMSE <=8.0, at least 55% oracle-gain retention, 5/5 map wins, at least 20/25 outer-cell wins, every legacy group improves, path-conditioned evidence beats action-prior-only by at least 0.50 RMSE, tail gates pass, and all destructive controls fail.
- **RESEARCH ONLY:** RMSE 8.001-10.0 with valid controls.
- **STOP:** best legal RMSE >10.0, oracle-gain retention <0.25, or path-conditioned evidence fails to beat the action-prior control.

Passing T036 authorizes a separate packaging/parity decision only. It does not authorize a Kaggle notebook run, package, or competition submission.
