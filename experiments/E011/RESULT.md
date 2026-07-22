# E011 Result — Memory-Safe Nonlinear Coefficient Learning

Status: **statistically promoted; exact package prepared and uploaded; awaiting authorized user Kaggle run; not deployment-ready; no Kaggle submission**
Official run: `R20260722-1537-e011-local-reproduction`  
Frozen source: `c326a812b0f663bb7a095a0d435fd302abc0bf6f`

## Decision

Promote `spline4_ridge_equal_s075` as the verified E011 statistical winner. The exact inference runtime is now committed and sealed, but do not promote it to the deployment primary yet. E006 remains the deployment-ready primary and E004 remains the exact deployment fallback until the user-run private-Kaggle outputs are independently verified.

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

The canonical notebook has **not** been executed. The user is the only authorized notebook runner. E011 remains not deployment-ready until the saved outputs are retrieved and checked for complete rows, IDs/order, finite predictions, and readability. E006 remains primary, E004 remains exact fallback, and no Kaggle competition submission has been made.

## Next action

The user imports the canonical notebook, attaches private dataset version 1 and the ROGII competition data, runs all cells, saves a notebook version, and supplies the owner/slug/version. Then retrieve and inspect the three expected outputs. Do not start H015 specialist routing before that result is reviewed.
