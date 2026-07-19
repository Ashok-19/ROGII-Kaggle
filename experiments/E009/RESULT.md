# E009 Result — Residual-Model Consensus Abstention

Status: **completed; rejected; no deployment package; no leaderboard submission**  
Run: `R20260719-1706-e009-consensus-abstention`  
Design commit: `9818888f5527f57856b31bf6af63f3a9bf7313d8`  
Implementation and frozen scoring commit: `ed3a8a1c7c4e09a01fc3bf3d0e316228de55fade`

## Decision

Reject every E009 consensus or nested-abstention candidate as a finalist. The strongest eligible aggregate result, `consensus_majority`, reaches **14.7934532198 RMSE**, improving E006 by **0.1396875674** and last-known TVT by **1.1163996509**. It lowers p90 from `21.7855500884` to `21.3853200589`, lowers worst-5% SSE share from `0.3685061112` to `0.3574498491`, improves all four registered special slices, and improves all five immutable maps by more than the frozen 0.01 margin.

The transfer contract still fails. `consensus_majority` wins only **15/25** repeated outer cells, regresses spatial groups 0 and 1 by `0.1973596457` and `0.3256765316` RMSE, and regresses typewell groups 2 and 4 by `0.2169185014` and `0.3712272832`. The sign-flipped-model negative control improves E006 by `0.0388441284`, exceeding its pre-registered maximum of `0.03`; therefore the all-controls gate also fails. No candidate passes every frozen gate, and E006 remains the primary deployment-ready candidate.

## Legal model and uncertainty boundary

E009 verifies SHA-256 identities for the E008 legal feature table, residual-datum targets, E008 OOF parent rows, and E008 summary before fitting. It uses **142** legal features and trains all statistics strictly inside each applicable training split. Formation surfaces, direct typewell predictors, absolute spatial coordinates, hidden outcomes at inference, leaderboard feedback, external artifacts, and observed E008 subgroup outcomes/scales/thresholds are forbidden. Spatial and typewell information is evaluator-only.

Six datum-only branches were frozen before scoring:

1. all legal features, ridge `alpha=25`, top 32;
2. all legal features, ridge `alpha=100`, top 32;
3. all legal features, ridge `alpha=25`, top 64;
4. no-E007 features, ridge `alpha=25`, top 32;
5. path/evidence features, ridge `alpha=25`, top 24;
6. visible/relative-geometry features, ridge `alpha=25`, top 32.

The three fixed rules use only model predictions: median action, sign-agreement fraction, median absolute deviation, relative MAD, and action magnitude. Failed rules emit exactly zero correction and return exactly to E006. `nested_consensus` selects one fixed rule or E006 from inner OOF evidence only; no action scale is tuned.

## Eligible candidate ladder

| Candidate | RMSE | Gain vs E006 | p90 | Worst-5% SSE share | Map wins | Repeated-cell wins | Repeated action fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `consensus_majority` | **14.7934532198** | **0.1396875674** | 21.3853200589 | 0.3574498491 | **5/5** | 15/25 | 69.81% |
| `consensus_tight_absolute` | 14.8278919773 | 0.1052488100 | 21.4815982603 | 0.3583174298 | 2/5 | 14/25 | 63.67% |
| `consensus_strict` | 14.8329284362 | 0.1002123510 | 21.5386189082 | 0.3585751198 | 2/5 | 12/25 | 61.14% |
| `nested_consensus` | 14.8964410056 | 0.0366997817 | 21.8890510896 | 0.3646525116 | 0/5 | 2/25 | 22.66% |
| E006 nested fusion | 14.9331407872 | 0.0000000000 | 21.7855500884 | 0.3685061112 | — | — | 0% |

Map gains for `consensus_majority` are positive on v1–v5: `+0.1377947819`, `+0.0285630792`, `+0.0593582115`, `+0.0378252183`, and `+0.0550040445`. This improves E008's map stability, but it does not reach the 17/25 outer-cell requirement and does not repair shift transfer.

## Branch exploration

All six frozen branches were completed and persisted; no branch was abandoned after the first unfavorable fold.

| Branch | RMSE | Gain vs E006 | p90 | Map wins | Repeated-cell wins | Worst spatial gain | Worst typewell gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ridge_all_a25_f64` | **14.6352733150** | **0.2978674723** | **20.8076874949** | **5/5** | 15/25 | -0.1176817962 | -0.4363660342 |
| `ridge_path_evidence_a25_f24` | 14.7463752881 | 0.1867654992 | 21.0923199416 | 4/5 | 13/25 | -0.1232545611 | -0.1842050396 |
| `ridge_all_a25_f32` | 14.7951380415 | 0.1380027458 | 21.4079596409 | 2/5 | 14/25 | -0.4788483433 | -0.5931493901 |
| `ridge_all_a100_f32` | 14.8568894227 | 0.0762513646 | 21.5485229652 | 2/5 | 12/25 | -0.5161339670 | -0.5253930039 |
| `ridge_no_e007_a25_f32` | 14.9808346143 | -0.0476938271 | 21.7630450624 | 0/5 | 9/25 | -0.2848589525 | -0.5709502537 |
| `ridge_visible_geometry_a25_f32` | 15.1954842405 | -0.2623434532 | 22.7779419186 | 0/5 | 0/25 | -0.5876476834 | -0.7663620852 |

The 64-feature ridge is the strongest legal residual branch seen so far and improves all five maps, p90, and worst-5% concentration. It is **diagnostic and ineligible in E009** because E009 pre-registered consensus/abstention—not direct branch promotion—as the tested hypothesis. It also remains unstable: spatial group 1 regresses by `0.1176817962`; typewell groups 2, 3, and 4 regress by `0.4059118284`, `0.0922913283`, and `0.4363660342`; only 15/25 outer cells improve. It may justify a fresh stability-regularized experiment, but its E009 score cannot be relabeled as promotion evidence.

## Why consensus did not identify risk

- The fixed rule action sets overlap almost completely: majority-versus-strict Jaccard is `0.9841479524`, majority-versus-tight is `0.9881109643`, and strict-versus-tight is `0.9959893048`.
- `consensus_majority` acts in 69.81% of repeated held-out placements. Tightening sign or dispersion thresholds reduces action modestly but does not isolate the failing shift groups.
- Majority datum prediction is weak but real: Pearson `0.1613177596`, Spearman `0.1448601388`, and 57.85% material-sign accuracy over 446 wells.
- `nested_consensus` selects an action rule in only 8/25 repeated cells and exact E006 fallback in 17/25. It retains only `0.0366997817` RMSE gain and loses on four maps.
- All registered special slices improve under majority consensus: long suffix `+0.2793634316`, high GR missingness `+0.0591590594`, low consensus margin `+0.1531955552`, and high model dispersion `+0.1648931161`. These global slices therefore do not explain the spatial/typewell transfer failures.

The evidence shows that model agreement largely measures shared model bias, not safe signed action. More agreement is not equivalent to more transferability.

## Controls and edge cases

Fifteen of sixteen registered controls pass. The sole failure is the sign-flipped-model negative control:

- Parent hashes, data signature, well set, E006 RMSE, and last-known RMSE reproduce.
- Exactly 3,783,989 unique ordered hidden-row IDs across 773 contiguous wells are emitted; all predictions are finite.
- Legal feature scan finds zero forbidden columns.
- Zero action reproduces E006 exactly.
- Duplicate-model-invariant consensus changes action by exactly `0.0`.
- Deterministically shuffled model predictions improve E006 by only `0.0064177830`, below the `0.03` cap.
- Alternating sign-flipped model outputs improve E006 by `0.0388441284`, exceeding the `0.03` cap by `0.0088441284`; this control fails.
- Oracle residual datum reaches `8.9768336628` RMSE and remains ineligible.
- Direct row SSE and sufficient-statistic SSE agree within `1.4879e-13` relative error.
- Maximum emitted legal correction is `21.5237619250` ft under the 30 ft cap.
- All 35 placement-membership contexts pass; runtime and memory pass.

The 12-test E009 suite passes under Python 3.10, 3.12, and 3.13. It covers all six branches, training-only feature preparation, rule precedence, zero/small/split-sign/relative-dispersion/absolute-dispersion/non-finite fallback, duplicate invariance, deterministic shuffle/sign flip, nested action/fallback, one-row suffixes, malformed parent rows, full miniature repository runs, parent-hash failure, deterministic outputs, and official artifact hashes. A separate 5,000-case randomized audit passes with maximum sufficient-statistic RMSE delta `1.5988e-14`, zero duplicate-action delta, and no finite/fallback/cap failures.

## Reproducibility

- Official frozen-code run: **5:27.04**, maximum RSS **725,808 KB**, exit status `0`, empty stderr.
- Independent clean-output run: **5:35.69**, maximum RSS **725,340 KB**, exit status `0`, empty stderr.
- OOF artifact: **182,723,863 bytes**, SHA-256 `3609b06eb5c38ebbdecbd951e925e31b8633ba7c904d508230a8e5bc07c0052e`, byte-identical across runs.
- Twenty static result files are byte-identical. Summary/control files are semantically identical after runtime/RSS normalization; artifact manifests match after runtime-bearing files and output-root paths are normalized.
- Independent row scoring reproduces all candidate RMSEs within `1.38e-10`, attributable to fixed eight-decimal OOF serialization.
- Official internal wall time is `325.8496336710` seconds; reproduction internal wall time is `334.647862889004` seconds as recorded in its summary. Both are far below the 30-minute gate.

## Deployment decision

Statistical authorization is false. No E009 package, local notebook, Kaggle kernel, parity artifact, submission file, or leaderboard submission was created. E006 remains the primary deployment-ready surface-free candidate and E004 remains its exact fallback.

## Next action

Do not retune E009 consensus thresholds or route by its observed spatial/typewell outcomes. Pre-register a fresh wide-residual stability experiment. It should treat the 64-feature ridge as a comparator, derive feature stability only from inner training folds, test stable-core and path-evidence branches plus training-only fixed blending, and require the same repeated-map, outer-cell, tail, spatial/typewell, negative-control, runtime, and deployment gates. E009 held-out subgroup outcomes may be used only as evaluation evidence, never as inference thresholds.
