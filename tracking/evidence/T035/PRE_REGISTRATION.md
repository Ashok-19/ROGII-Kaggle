# T035 Preregistration — Actual cross-fitted PCA action-coordinate identifiability

## Question

Can legal observations identify the exact rank-8 PCA action coordinates that produced T033's 2.1613289429 RMSE outer-training capacity?

T034 is not a test of this target: it predicts fixed spline7 coefficients. T035 fits the rank-8 PCA basis inside every inner and outer training partition, derives actual-row least-squares action coordinates in that basis, and predicts those coordinates without hidden held-out labels.

## Frozen boundary

- Residual action: true hidden TVT minus exact E011 OOF path, represented on the same 128-node grid and clipped to plus/minus 160 ft.
- Outer basis: full-SVD rank-8 PCA fitted on outer-training action profiles only.
- Inner selection basis: separately fitted on the inner-training partition only.
- Coordinate target: exact actual-row least-squares coordinates for each well in the partition-specific PCA mean/components.
- Legal inputs: frozen 142 E011 summaries and the frozen 4,544-dimensional T031 legal raw tensor.
- Privileged inputs: six train-only hidden formation-offset changes plus finite masks, diagnostic only.
- Aggregation: reconstruct one action profile per map and average five profiles per well before final TVT scoring.

The exact model grids, preprocessing, controls, stress systems, compute cap, and GO/STOP thresholds are in `config.json`. No branch, target, PCA rank, hidden width, tree grid, or gate may be added after the first T035 branch metric.

## Decision logic

- **GO:** legal TVT RMSE <=8, at least 50% of the rank-8 oracle gain retained, coordinate R2 >=0.35, median coordinate Pearson >=0.55, at least 65% positive wells, every legacy group positive, and controls fail.
- **BREAKTHROUGH:** legal TVT RMSE <=5 with every transfer/control gate passing. This authorizes a separate full-fit inference/package preregistration, not an automatic Kaggle run.
- **PRIVILEGED MEDIATION:** privileged RMSE <=5.5 with legal failure is causal diagnostic evidence only.
- **STOP:** best legal RMSE >10 or coordinate R2 <0.20. Close the compact-manifold identification route without further model/seed/basis search.
