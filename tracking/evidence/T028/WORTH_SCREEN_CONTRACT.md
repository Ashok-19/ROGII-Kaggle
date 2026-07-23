# T028 preimplementation worth-screen contract

Frozen before the first valid diagnostic result.

## Question

Does the positive T027 original-view coefficient direction have enough objective-level, non-routing headroom to justify implementing domain-balanced or distributionally robust training?

## Legal boundary

- Exactly one original 142-feature E011 vector and one original-boundary four-control target per well.
- No T027 masks, duplicated views, view indicators, or mask-local targets.
- Spatial and typewell groups are diagnostic/training-only; they are forbidden as inference features and routing labels.
- Per-domain and per-well oracle placements use hidden labels and are opportunity diagnostics only.

## Frozen gates

Proceed to implementation only if all four pass:

1. Per-well oracle placement adds at least 0.05 RMSE beyond the T027 original-only candidate.
2. Reoptimizing one global placement adds no more than 0.02 RMSE, proving the missed opportunity is not an already-covered scalar placement issue.
3. The weaker of spatial-domain and typewell-domain oracle-placement gains is at least 0.05 RMSE beyond the current candidate.
4. At least three of ten domain-optimal weights differ from the global optimum by at least 0.10.

Any failed gate closes H021/T028 without model implementation or hyperparameter search.

## Correction history

The first executable attempt failed before loading data because the dynamic module wrapper did not register the imported module in `sys.modules`; this was corrected without observing a metric.

The first completed calculation incorrectly summed both complete spatial and complete typewell partitions and divided by one population, double-counting every well. This invalid diagnostic was discarded. The corrected audit reports the two disjoint partition systems separately and applies gate 3 to the weaker gain. The corrected output was then reproduced byte-for-byte.
