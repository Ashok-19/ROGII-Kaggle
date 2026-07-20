# E010 Result — Nonlinear Candidate Coverage and Visible-Prefix Selector Regret

Status: **completed; rejected; no deployment package; no leaderboard submission**  
Run: `R20260720-1059-e010-nonlinear-selector`  
Kaggle: `ashok205/e010-candidate-selector-kaggle`, version 1, session `COMPLETE`  
Design commit: `b9fb275f5df7d99bcf39f08a66c891742a3fa922`  
Frozen implementation: `fa21e2951a700aeeff355e1cf439c55abc3ab408`  
Runtime finalization overlay: `437d83a03df429657850546cc2d572b62df0681b`

## Decision

Reject every E010 legal selector as a finalist. Preserve the candidate bank and oracle diagnostics as strong positive research evidence, but retain E006 as the deployment-ready primary and E004 as the exact fallback.

The candidate-generation hypothesis succeeded: the legal fixed bank reaches **4.7510768278 RMSE** under hidden-label oracle routing, below the preregistered 5.0 coverage threshold. The selection hypothesis failed: `hybrid_confidence_fallback` reaches **14.7802528528 RMSE**, improving E006 by only **0.1528879345**. No selector passed every frozen gate.

## Candidate coverage

The bank contains **130,252** bounded paths. Its oracle improves E006 by **10.1820639594 RMSE** and reduces worst-well concentration materially. Sixteen distinct families win at least one well, led by E006 hinge families at 25%, 50%, and 75% of the suffix.

| Family | Family oracle RMSE | Oracle wins |
|---|---:|---:|
| E006 quadratic | 5.8401 | 8 |
| E006 hinge 25% | 5.9633 | 200 |
| E006 hinge 50% | 5.6458 | 134 |
| E006 hinge 75% | 5.9704 | 130 |
| PF quadratic | 5.8433 | 44 |
| Wide64 quadratic | 5.8432 | 35 |
| E006 jump 50% | 6.4802 | 85 |
| DTW derivative | 30.4340 | 18 unique wins |

The registered HMM screen produced zero unique oracle wins and every HMM variant was materially worse than E006 on the 48-well screen. It was correctly stopped. DTW derivative added rare coverage and advanced, but its standalone full-data RMSE is 30.4340; it is a rare diagnostic candidate, not a safe default path.

## Legal selector result

| Candidate | RMSE | Gain vs E006 | Map wins | Outer-cell wins | p90 well RMSE | Worst-5% SSE share |
|---|---:|---:|---:|---:|---:|---:|
| `hybrid_confidence_fallback` | **14.7802528528** | **0.1528879345** | **5/5** | **17/25** | 21.0839 | 0.3710 |
| `hybrid_ridge_trees` | 14.7802660756 | 0.1528747116 | 5/5 | 17/25 | 21.0839 | 0.3710 |
| `ridge_path_f48` | 14.8039242043 | 0.1292165829 | 3/5 | 13/25 | 20.8018 | 0.3608 |
| `ridge_all_f64` | 14.8563749290 | 0.0767658583 | 1/5 | 10/25 | 21.2664 | 0.3567 |
| E006 | 14.9331407872 | — | — | — | 21.7856 | 0.3685 |
| bank oracle | 4.7510768278 | 10.1820639594 | diagnostic only | diagnostic only | 6.5542 | 0.1964 |

The reported selector meets the aggregate 0.15 gain threshold, all five repeated-map wins, 17/25 outer-cell wins, p90 and worst-5% limits, and every typewell holdout. It still regresses spatial group 1 by **0.2589909065 RMSE** and the high-GR-missingness slice by **0.2394504920 RMSE**. It also violates the frozen memory cap.

## Regret attribution

This is the decisive E010 result.

- Total selector-to-bank-oracle gap: **10.0291760250 RMSE**.
- Mean per-well regret: **7.8022**; median **5.6203**; p90 **16.6234**; p99 **38.4981**; maximum **57.6048**.
- Exact oracle-family identification in repeated held-out contexts: **16.43%**.
- Counterfactual RMSE if the selector kept its chosen families but received each family's hidden-label-optimal coefficients: **5.7075190658**.
- Family choice accounts for only **9.54%** of the total RMSE gap.
- Coefficient estimation and path aggregation account for **90.46%**.

Therefore another flat family classifier or a larger candidate bank is not the next evidence-backed action. The next experiment must target low-dimensional path coefficients/control points directly.

## Controls, runtime, and edge cases

Twenty of twenty-one registered controls pass. Parent hashes, data identity, candidate count, basis reconstruction, no-op, duplicates, exact fallback, membership, OOF identity, oracle positive control, SSE consistency, runtime, HMM/DTW screen completeness, shuffled target, sign-flipped target, and unique-family controls all pass.

The only failed control is memory: maximum RSS is **17,528,436 KB** against a frozen **3,072 MB** limit. Model wall time is **2199.34 seconds**, within the 60-minute budget, with all native thread pools capped at two.

The independent review verifies all **3,783,989** hidden rows and **773** wells, finite predictions, unique IDs, all 35 membership contexts, exact parent hashes, every output-manifest hash, every ZIP member, safe/unique archive paths, and direct RMSE recomputation within `4.3e-11`.

## Kaggle artifact integrity

- Output archive: **218,524,517 bytes**, SHA-256 `e2e12bec89ccdce4a35dff2fcacfa555fae1432eb260fa32659a863e57e09622`.
- OOF artifact: **228,267,787 bytes**, SHA-256 `80bae7649c783b1fa1536546e0485f1357cd59abf4a641ad26d01f90490d8f46`.
- Output manifest SHA-256: `0a9ad021322584ad69fc53a3adb6b3522deb4d1d1153358de7c7ff44572312b9`.
- Every committed notebook code cell appears exactly once in Kaggle's executed virtual source.
- No submission file was created and no leaderboard submission was made.

## Deployment decision

Statistical authorization is false. E010 is not packaged for inference and is not eligible for submission. E006 remains the primary deployment-ready surface-free model; E004 remains the exact fallback.

## Next action

Pre-register E011 as a memory-safe, split-local coefficient-learning experiment. It must compare multiple low-dimensional formulations—orthogonal correction coefficients, three control points, and family-specific coefficient experts—under the frozen repeated/spatial/typewell/slice controls. It must preserve E006 fallback, avoid 130k-candidate materialization, and stop any branch that cannot demonstrate coefficient learnability on a registered screen.
