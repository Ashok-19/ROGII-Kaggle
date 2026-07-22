# E011 Result — Memory-Safe Nonlinear Coefficient Learning

Status: **statistically and operationally promoted; deployment-ready primary; no Kaggle submission**
Official run: `R20260722-1537-e011-local-reproduction`  
Frozen source: `c326a812b0f663bb7a095a0d435fd302abc0bf6f`

## Decision

Promote `spline4_ridge_equal_s075` as both the verified statistical winner and the primary deployment model. Saved Kaggle notebook version 1 completed and its three outputs passed the D019 correctness review. Retain E006 as the secondary fallback and E004 as the exact fallback.

The complete-looking directory discovered after the timed-out first call was not accepted on static consistency alone. Its ownership, timestamps, source/config/fold identities, parent hashes, and manifest were consistent, but no successful invocation receipt, process log, task log, Codex receipt, or operation receipt could be linked to that run. The original process provenance is therefore incomplete. Its runtime-dependent manifest, controls, and summary are archived under `experiments/E011/provenance/original_unregistered/`.

The official decision is based on a new isolated reproduction from the frozen commit with two-thread limits and a persistent receipt. The reproduced OOF and all eleven substantive CSV tables are byte-identical to the discovered output. The only expected differences are runtime and RSS fields in `controls.csv` and `summary.json`, plus their dependent manifest hashes.

## Verified metrics

| Metric | E011 | E006 | Gain / change |
|---|---:|---:|---:|
| Pooled RMSE | **12.550756295690** | 14.933140787238 | **2.382384491548** |
| Median well RMSE | 9.068494 | 9.759889 | 0.691396 |
| p90 well RMSE | 17.831024 | 21.785550 | 3.954526 |
| p99 well RMSE | 31.347572 | 44.602342 | 13.254770 |
| Worst-5% SSE share | 0.315322 | 0.368506 | 0.053184 |
| RMSE catastrophes >=12 / >=20 / >=30 | 233 / 59 / 11 | 291 / 95 / 31 | -58 / -36 / -20 |

The frozen E011 representation-oracle retention is **0.219531752751** because the preregistration uses the spline4 oracle RMSE of 4.0810227602. Relative to E010's broader 4.7510768278 path-bank oracle, the same gain fraction is **0.233978543156**. The earlier handoff value near 0.234 referred to the latter denominator, not the frozen E011 gate.

## Repeated and stress validation

All five repeated maps and all 25 outer cells improve over E006. Every spatial and typewell holdout improves. The special-slice gains are:

- long suffix: 2.707447 RMSE
- high GR missingness: 1.549099 RMSE
- E006-catastrophe wells: 4.591898 RMSE

All 21 controls pass. Shuffled-target gain is -0.034298; sign-flipped-target gain is -0.299514. Exact fallback coefficient delta is zero.

## Identity, legality, and structure

The independent review verified 3,783,989 rows, 773 wells, 3,783,989 unique IDs, no duplicates, exact row/order/parent identity, five maps, 25 repeated cells, ten stress contexts, three slices, finite values, unique table keys, and deterministic block ordering. The selected-context reconstruction matches the shipped metrics to serialization tolerance; feature-frequency counts match exactly.

The legal feature table contains 773 wells and 142 features. No forbidden hidden-target/label fields, absolute coordinates, spatial/typewell group IDs, formation surfaces, public-overlap indicators, leaderboard scores, score-decoded constants, paid data, unknown-license artifacts, or validation-label routing entered the legal system. Coefficient targets are created only for split-local training and inner-validation IDs; outer-test targets are used only for scoring after prediction.

## Reproduction resources

- Model wall time: **1059.980 seconds**
- External receipt wall time: **1069.385 seconds**
- Model maximum RSS: **399.438 MB**
- Child-process maximum RSS: **402.328 MB**
- Numerical thread limit: **2**
- Return code: **0**; stderr empty

## Deployment and submission

The packaged E011 inference path reconstructs all 142 legal features across 773 wells, passes 13 edge-case groups and the 167-test repository suite, and processes the 200-well pseudo-hidden benchmark in about 75–76 seconds using about 353 MB RSS.

The canonical notebook is `notebooks/training_and_submission/e011_spline4_deployment_kaggle.ipynb`. It now contains two code cells: one small contract cell and one inference cell. It locates the attached bundle and competition data, runs inference, and writes `submission.csv`, `e011-run-receipt.json`, and `e011-well-predictions.json`. It does not check internet state, CPU/device/thread settings, hashes, training signatures, or local/Kaggle byte parity. Private dataset `ashok205/rogii-e011-deployment-inputs` version 1 is READY.

The canonical notebook was executed and saved as `ashok205/e011-spline4-deployment-kaggle`, version 1, `scriptVersionId=337242364`. Kaggle reported `COMPLETE`; the receipt records a 3.540805-second inference run. The three expected outputs were retrieved and reviewed: `submission.csv` has 14,151 unique IDs in exact sample order with finite TVT values, the receipt is readable and complete, and the well diagnostics contain three records. E011 is deployment-ready and primary. E006 remains the secondary fallback, E004 remains the exact fallback, and no Kaggle competition submission has been made.

## Next action

Preregister H015 regime-specialized coefficient experts with legal cross-fitted routing, minimum regime support, bounded actions, exact E011 fallback, and frozen negative and distribution-shift controls before implementation or scoring.
