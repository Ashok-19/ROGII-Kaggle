# E006 Pre-registration

## Question

Can a strictly nested PF–E004 fusion preserve the useful, decorrelated portion of E005 particle-filter evidence while eliminating the standalone PF p90, repeated-map, spatial, and typewell-cluster failures?

## Evidence boundary

E006 may use only horizontal `MD`, `X`, `Y`, `Z`, `GR`, the contiguous visible `TVT_input` prefix, paired typewell `TVT`/`GR`, the frozen E004 geometry-prefix algorithm, the frozen E005 PF algorithm, and target-free PF diagnostics. Hidden training `TVT` is evaluator-only except inside the inner-training portion of a nested outer cell. The outer fold is forbidden from every E004 inner fit, fusion-weight fit, route fit, reliability calibration, and threshold calculation.

The observed E005 fixed 50/50 diagnostic score of 15.0131171651 is motivation only. It cannot select an E006 candidate, weight, threshold, or deployment decision. The fixed 50/50 path is reported as a diagnostic reference and is ineligible.

## Strict nested frame

For each immutable E001 map and each of its five outer folds:

1. Hold out the outer fold completely.
2. Fit E004 geometry-prefix on the other four folds and predict the outer fold.
3. Within those four folds, use each original fold label once as inner validation and the other three labels as E004 training data.
4. Combine the resulting inner E004 OOF paths with deterministic PF paths and inner truth to choose only the frozen fusion rule.
5. Apply the chosen rule to the untouched outer fold using target-free diagnostics calibrated on outer-training wells only.

Every well therefore receives one untouched outer prediction per map. Five-map predictions are averaged for the full OOF summary, while every map and outer cell is also scored separately.

The PF algorithm is recomputed around the E004 path produced for the same inner or outer cell. E005 PF rows were anchored to a five-map averaged E004 path and are never reused as nested predictions. E005's 15.3502040715 PF score is reproduced only in a separate parent-algorithm audit, not as an E006 outer-validation comparator.

Nested leave-spatial-group-out and leave-typewell-cluster-out runs use the same process: the stress group is the outer test set, E004 never trains on it, and inner folds come from frozen `v1` labels restricted to the remaining wells.

## Frozen candidate bank

Comparators and diagnostics:

- `last_known_tvt`.
- `e004_geometry_prefix`.
- `pf_gr_path` standalone.
- `fixed_50_50_reference`, diagnostic only.
- `nested_rmse_grid`, unconstrained inner RMSE selection and diagnostic only.
- `duplicate_conservative`, exact duplicate control.
- `shuffled_pf_conservative`, deterministic well-level shuffle control.
- `oracle_rowwise_best`, evaluator-only leakage sentinel.

Promotion-eligible candidates:

1. `nested_conservative_grid`: choose from the frozen convex grid using inner pooled RMSE, requiring at least 0.02 inner gain, no more than 0.25 p90 deterioration, and no more than 0.01 worst-5% SSE-share increase. Choose the smallest weight within 0.02 RMSE of the best passing weight. Fall back to zero.
2. `nested_reliability_shrink`: multiply the conservative base weight by the arithmetic mean of four favorable outer-training empirical percentiles: GR coverage, PF effective fraction, ambiguity margin, and reversed PF–E004 disagreement.
3. `nested_disagreement_cap`: multiply the conservative base weight by one until outer-test disagreement exceeds the outer-training 80th percentile, then shrink inversely with disagreement.
4. `nested_reliability_gate`: apply the conservative base weight only when reliability is at least 0.5 and disagreement is no higher than the outer-training 90th percentile; otherwise use E004 exactly.

Weights are limited to `[0, 1]`. All ties prefer the lower PF weight and then the frozen candidate order.

## Controls

The averaged nested E004 comparator must reproduce 15.491306398267565 RMSE within `1e-9`. A separate parent audit must reproduce E005 PF at 15.350204071523788 within `1e-9`, but those audit predictions cannot enter E006 nested scoring. Zero weight must reproduce E004 within `1e-12`. The duplicate candidate must be identical within `1e-12`. Direct and well-aggregated SSE must agree within `1e-12` relative error.

The shuffled control circularly permutes whole-well PF-minus-E004 correction paths in a SHA-256 well order, linearly resamples each source correction to the recipient hidden length, and adds it to the recipient E004 path. This preserves correction shape and scale but destroys well-specific GR evidence. It must be at least 0.05 RMSE worse than the corresponding unshuffled candidate and may not improve E004 by more than 0.02.

The rowwise oracle chooses the closer of E004 and PF using hidden truth and is permanently ineligible. It must beat the best legal candidate by at least 0.05, proving that the evaluator can detect useful candidate placement.

At least 15 of 25 repeated outer cells and at least four maps must use nonzero median PF weight. The selected outer base-weight range may not exceed 0.75.

## Promotion

An eligible candidate is promoted only when all frozen gates pass:

- at least 0.45 RMSE gain versus last-known and 0.05 versus E004;
- at least four of five repeated maps improve E004 by at least 0.02;
- at least 17 of 25 outer cells improve E004;
- p90 deterioration versus either comparator is at most 0.50;
- worst-5% SSE-share increase versus either comparator is at most 0.01;
- every nested spatial group and every nested typewell group has positive gain versus E004;
- long-suffix, high-GR-missingness, and ambiguous-alignment deterioration is at most 0.10;
- all controls, weight-stability, runtime, memory, exact-ID, and independent reproduction checks pass.

Among candidates passing every gate, select the lowest mean repeated-map RMSE, breaking ties by frozen candidate order. If none pass, reject E006 and retain E004 as the deployment fallback. Candidate definitions and thresholds cannot change after official scoring begins.

## Deployment

A promoted candidate is fitted for deployment by applying its unchanged selection rule to full-training E004 OOF and PF evidence, then embedding the frozen E004 model, PF configuration, reliability calibration, and standard-library inference code in an offline notebook. Exact sample IDs/order, finite values, no-GR fallback, package size, local byte parity, and private internet-disabled Kaggle parity are mandatory. Deployment readiness does not authorize a leaderboard submission.

## Reproducibility

The design is committed before official scoring. The scoring-code commit is recorded separately. Results include outer and inner membership audits, selected weights, repeated-map and cell metrics, nested spatial/typewell stress, special slices, per-well metrics, controls, runtime/RSS, OOF predictions, artifact hashes, and a second independent output-root reproduction. Unrelated pre-existing files remain untouched.
