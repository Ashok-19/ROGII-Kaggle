# E008 Result — Cross-Fitted Surface-Free Residual Action

Status: **completed; rejected; no deployment package; no leaderboard submission**  
Run: `R20260719-1605-e008-residual-action`  
Design commit: `60a217a96b4c692ff715e037eb10973eabb8f90d`  
Implementation commit: `4f693e68e059d3a6bbd098f7352e3f639f1b536a`  
Frozen scoring commit: `ac2acc15c94afb16b2cb5da79c6e19b66e9e52f1`

## Decision

Reject every E008 direct or conservatively scaled residual action as a finalist. The strongest aggregate result, `ridge_residual_datum`, scores **14.7951380415 RMSE**, improving E006 by **0.1380027458** and last-known TVT by **1.1147148293**. It lowers p90 from `21.7855500884` to `21.4079596409`, lowers worst-5% SSE share from `0.3685061112` to `0.3548957222`, and improves all four registered special slices.

Those aggregate and tail gains do not satisfy the frozen transfer contract. Datum-only ridge wins only **2/5** immutable maps by at least 0.01 RMSE and **14/25** repeated outer cells. It regresses spatial groups 0 and 1 and typewell groups 2 and 4; the worst spatial and typewell gains are **-0.4788483433** and **-0.5931493901**. No eligible candidate passes every repeated, shift, and placement gate. No model package, notebook parity run, private Kaggle execution, submission artifact, or leaderboard submission is authorized.

## Legal predictor and target boundary

E008 trains only inside each validation training split. It predicts two low-order residual coefficients around the already cross-fitted E006 path:

1. `residual_datum_ft`, an additive hidden-zone shift;
2. `residual_trend_ft`, an end-to-end centered linear trend over normalized hidden index.

The frozen feature table contains **142** surface-free columns derived from relative trajectory geometry, visible TVT/U/Z summaries, hidden supplied Z geometry, horizontal GR summaries/missingness, visible-prefix pseudo-backtests, E006 path summaries, and target-independent E007 self-correlation summaries. Feature screening selects at most 32 columns using training-split target correlations only. Ridge regularization is fixed at `alpha=25`. Datum and trend actions use 30 ft and 60 ft soft caps.

Formation surfaces, typewell-derived predictor features, absolute spatial coordinates, hidden TVT, oracle labels, leaderboard feedback, and external artifacts are forbidden. Typewell and spatial information enter only evaluator grouping. The legal-feature audit finds zero forbidden columns.

## Candidate ladder

| Candidate | Role | RMSE | Gain vs E006 | p90 | Worst-5% SSE share | Map wins | Repeated-cell wins |
|---|---|---:|---:|---:|---:|---:|---:|
| `ridge_residual_datum` | eligible / reported | **14.7951380415** | **0.1380027458** | 21.4079596409 | 0.3548957222 | 2/5 | 14/25 |
| `ridge_residual_conservative` | eligible | 14.8647314584 | 0.0684093289 | 21.9436966741 | 0.3648432190 | 3/5 | 11/25 |
| `ridge_residual_datum_trend` | eligible | 14.8664838148 | 0.0666569725 | 21.7211761747 | 0.3551600069 | 0/5 | 11/25 |
| `ridge_no_e007_ablation` | diagnostic | 14.9617859705 | -0.0286451833 | 21.4928555117 | 0.3580596026 | — | — |
| `mean_residual_datum` | no-evidence control | 14.9450614970 | -0.0119207097 | 21.9521914260 | 0.3684323459 | — | — |
| E006 nested fusion | comparator | 14.9331407872 | 0.0000000000 | 21.7855500884 | 0.3685061112 | — | — |
| shuffled-target ridge | negative control | 15.0123454725 | -0.0792046853 | 22.1501201463 | 0.3687188819 | — | — |
| oracle residual datum+trend | ineligible oracle | 6.6971947701 | 8.2359460171 | 9.5963114989 | 0.2411956667 | — | — |

Map wins use the frozen minimum gain of 0.01. Conservative placement has nominal positive gain on all five maps, but v1 and v5 improve by only `0.00598` and `0.00673`, so only three maps pass the registered margin.

## Evidence and failure anatomy

- Datum target Pearson: **0.1759287774**; material-sign accuracy: **60.76%** over 446 wells.
- Trend target prediction is weaker; adding trend worsens the final score relative to datum-only action.
- E007 evidence is selected in all five maps. The registered full-model versus no-E007 ablation gain is **0.0953021558 RMSE**.
- Datum-only special-slice gains versus E006 are positive: long suffix `+0.2694020404`, high GR missingness `+0.0170880132`, poor visible pseudo-gain `+0.0902539484`, and low match margin `+0.2859605109`.
- Datum-only map v3 regresses by `0.0592342304`; v2 and v4 also fail to improve by the registered margin.
- Datum-only spatial groups 0/1 regress by `0.3487282398` and `0.4788483433`; typewell groups 2/4 regress by `0.2217250387` and `0.5931493901`.
- Conservative inner scaling uses nonzero median scale in 18/25 repeated cells, has median scale 0.25 and range 0.50, but wins only 11 repeated cells. It still regresses spatial groups 0/1 and fails strict positive typewell transfer because at least one group abstains exactly to E006.

## Controls and edge cases

All **15/15** registered controls pass:

- E006/E007 parent SHA-256 values and E006/E004/last-known RMSEs reproduce within `1e-9`.
- Parent E006 and E007 rows align with maximum value delta `0.0` over 3,783,989 rows.
- Exactly 3,783,989 unique ordered hidden-row IDs across 773 contiguous wells are emitted; all predictions are finite.
- Zero action reproduces E006 exactly.
- Regularization-preserving duplicate-feature action delta is `1.0836e-13`.
- Shuffled targets have datum/trend correlations `-0.0468`/`-0.0160` and worsen E006 by `0.0792` RMSE.
- Direct and well-aggregated SSE agree within `1.35e-13` relative error.
- Maximum emitted legal correction is `40.9599` ft under the 60 ft cap.
- E007 feature evidence, feature leakage, oracle, membership, scale stability, runtime, and memory controls all pass.

The 13-test E008 suite passes under Python 3.10, 3.12, and 3.13. It covers split-local statistics, target derangement, duplicate-feature invariance, correction caps, malformed parent IDs/values, abstention diagnostics, context overlap, candidate order, deterministic fixture outputs, and official artifact hashes. A separate 1,000-case sufficient-statistic property test agrees with direct scoring within `9.49e-13` RMSE.

## Reproducibility

- Official frozen-code run: **6:37.23**, maximum RSS **96,948 KB**, exit status `0`, empty stderr.
- Independent clean-output run: **6:46.22**, maximum RSS **96,796 KB**, exit status `0`, empty stderr.
- OOF artifact: **263,014,805 bytes**, SHA-256 `fe1f05f6dd6dc91e94a07658637d318c2b311b7f470bae24cc86f51c9cc9d94d`, byte-identical across runs.
- Fourteen static result files are byte-identical. Summary/control files are semantically identical after normalizing runtime/RSS; artifact manifests match after normalizing runtime-bearing files and output-root paths.
- Independent row scoring reproduces candidate RMSEs within `2.1e-10`; the small delta is from fixed decimal OOF serialization.
- All 35 membership contexts pass: 25 repeated cells, five spatial holdouts, and five typewell evaluator holdouts.

The initial implementation run at commit `4f693e68...` failed during E007 abstention-diagnostic loading before scoring or writing a result JSON. The diagnostic handling fix was committed as `ac2acc15...`; only the subsequent clean run from that SHA is official evidence.

## Deployment decision

Statistical authorization is false. No E008 inference package, notebook, Kaggle kernel, submission artifact, or leaderboard submission was created. E006 remains the primary deployment-ready surface-free candidate and E004 remains its exact fallback.

## Next action

Do not retune E008 scales, features, or subgroup thresholds. Pre-register E009/H011 as a multi-model datum consensus-abstention experiment. Disagreement and sign consensus must be target-independent at inference, learned strictly inside folds, and return exactly to E006 when evidence is weak. Observed E008 subgroup outcomes are not reusable routing thresholds.
