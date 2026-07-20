# E011 Pre-registration — Memory-safe nonlinear coefficient and control-point learning

Frozen at 2026-07-20T12:10:00Z, before any learned E011 hidden-label score was computed.

## Question

Can legal split-local features predict a few nonlinear correction coefficients or spline control points well enough to recover materially more of E010's proven candidate coverage, while preserving exact E006 fallback and remaining below the low-memory execution budget?

E011 does **not** add another path bank. It converts the whole-well problem into supervised prediction of 3–7 coefficients around the frozen E006 path. Hidden truth supplies coefficient labels only for outer-training wells. Held-out-well coefficients, oracle paths, spatial groups, and typewell groups never enter fitting, feature selection, hyperparameter choice, shrinkage, confidence gating, or fallback decisions.

## Evidence before implementation

- E010's legal bank oracle is 4.7511, but its legal selector is 14.7803. Candidate coverage is sufficient; selection is not.
- Giving E010's chosen families hidden-label-optimal coefficients yields 5.7075. Family identity explains 9.54% of the selector gap; coefficient estimation and aggregation explain 90.46%.
- The preimplementation representation audit on all 773 persisted E010 OOF wells gives oracle RMSE 8.1360 for one coefficient, 5.9417 for two, about 4.82 for three, 4.0810 for four spline controls, 3.4391 for five, and 2.6251 for seven. One- and two-dimensional learned paths are excluded from promotion before implementation.
- E010 used 17.5 GB RSS. A recent public notebook also reports hidden-rerun failure after materializing a roughly 7.39 GB table. E011 must use sufficient statistics and streaming output, not row-by-candidate matrices.

## Frozen representations

`shape3` and `spline3` are compact learned controls. `spline4` and `spline5` are primary. `spline7` is a high-capacity stress branch. Every spline is piecewise linear through fixed normalized-suffix knots and is exactly zero at the visible/hidden boundary. `linear1` and `quadratic2` remain capacity-negative diagnostics only.

Spline knot predictions are clipped to ±80 ft and final emitted correction to ±120 ft. The preimplementation audit puts spline 99% coefficient quantiles below about 56 ft and maximum fitted correction near 110.4 ft, so these bounds are fixed from evidence rather than E011 outcomes.

## Frozen model branches

For every learned representation, run equal-well multi-output ridge. For spline4/5/7 also run row-weighted ridge, deterministic ExtraTrees, a fixed 50/50 ridge-tree blend, and the blend with a training-only disagreement/magnitude fallback to exact E006. Ridge feature count and alpha are selected only by immutable inner-map OOF on the outer-training wells. ExtraTrees uses 128 trees, depth 8, leaf size 8, half the features, one thread, and seed 11011.

Every predicted correction is evaluated at fixed shrinkages 0.25, 0.50, 0.75, and 1.00. The confidence fallback chooses among fixed disagreement and magnitude quantiles using only outer-training inner OOF. If no legal gate passes, it emits exact E006.

## Two-stage complete evaluation

Stage 1 runs every frozen branch across all 25 repeated map/cell contexts. A branch advances to all five spatial and five typewell holdouts only if repeated OOF gains at least 0.30 RMSE, wins at least 4/5 maps and 17/25 cells, and retains at least 3% of its representation's available oracle gain. Every branch receives a complete Stage-1 result; failed branches are closed, not abandoned.

Every passing branch advances. If more than eight pass, preserve diversity by retaining the best passing branch per representation and per model family, then fill in frozen candidate order to eight. This rule is immutable before scoring.

## Promotion

Promotion requires at least 0.50 RMSE gain versus E006, 1.0 versus last-known, and 5% retention of the representation oracle gain; 5/5 map wins with at least 0.02 each; at least 17/25 outer-cell wins; no p90 deterioration; worst-5% SSE-share increase at most 0.005; positive every spatial and typewell group; positive long-suffix, high-GR-missingness, and E006-catastrophe slices; all controls; deterministic reproduction; final placement; packaging; and parity.

The 0.50 gain corresponds approximately to the frozen 5% oracle-gain-retention floor for the 3D–5D representations. It is intentionally stronger than E010's 0.153 aggregate selector gain.

## Controls and edge cases

The run is invalid unless parent hashes, data signature, feature schema, forbidden-name scan, split membership, target isolation, zero-coefficient E006 identity, basis reconstruction, sufficient-statistic versus direct-row scoring, pooled SSE, OOF uniqueness, finite/bounded predictions, exact fallback, shuffled/sign-flipped negatives, duplicate-model identity, deterministic reproduction, runtime, memory, and two-thread limits pass.

Tests cover one-row and two-row suffixes, duplicate MD, singular and rank-deficient bases, zero-span suffixes, missing/constant legal features, all-missing columns, constant coefficient targets, tiny train partitions, empty advancement sets, tied inner scores, coefficient clipping, fallback-all and fallback-none, malformed IDs, repeated motifs, high GR missingness, long suffixes, catastrophic wells, safe gzip output, and interrupted finalization.

## Resource contract

All row scoring uses per-well `BᵀB`, `Bᵀy`, and `yᵀy` sufficient statistics. The full E010 bank and row-by-candidate matrices are forbidden. Only the final repeated OOF path is streamed to disk. Local execution is capped at two threads, 2 GB RSS, and 30 minutes. If a pre-score projection exceeds 15 minutes or 2 GB, the official run moves to a hash-sealed private Kaggle CPU notebook.

No Kaggle submission is authorized.
