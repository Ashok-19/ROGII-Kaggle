# E007 Result — Horizontal GR Self-Correlation and Controlled E006 Placement

Status: **completed; rejected; no deployment package; no leaderboard submission**  
Run: `R20260719-1502-e007-self-correlation`  
Final design commit: `e0e0e5283ba3d8951cd927097cefd6f1ba8dc782`  
Frozen scoring commit: `92a9496202d18d6437845e2ac36b5660907b4814`

## Decision

Reject every E007 placement as a finalist. The strongest registered aggregate result, `visible_fixed_0p10`, scores **14.8304784179 RMSE**, improving E006 by **0.1026623694** and last-known TVT by **1.0793744528**. It wins all **5/5** repeated maps and **20/25** repeated outer cells, while the raw bounded self-correlation correction has residual correlation **0.4675700642** to E006.

Those positive results do not satisfy the frozen contract. P90 rises from `21.7855500884` to `22.0503957463`, a deterioration of **0.2648456580** against a `0.25` cap. Spatial groups 1 and 3 and typewell groups 2 and 4 regress. Every eligible cross-fitted placement also fails at least one spatial and typewell stress gate. No candidate is statistically authorized, so no model package, notebook parity run, private Kaggle execution, or leaderboard submission is permitted.

## Predictor and evidence boundary

The E007 leg is independent of typewell matching. It uses only same-well horizontal `GR`, `Z`, and the contiguous visible `TVT_input` prefix:

1. Build deterministic multiscale GR fingerprints at radii `0`, `6`, `18`, and `48` rows.
2. Match each hidden query anchor to five visible-prefix fingerprint states.
3. Transfer only the clipped local visible structural slope `d(TVT+Z)`; absolute TVT residual transfer was rejected before scoring.
4. Integrate the slope into a raw TVT path and apply a 40 ft soft cap around the frozen averaged E006 path.
5. Evaluate fixed and strictly cross-fitted weights. Typewell summaries are used only to define evaluator stress groups, never by the predictor.

The visible-only design audit used pseudo-boundaries at 0.65, 0.75, and 0.85 of the visible prefix. The actual implementation improves the visible structural prior by approximately `0.3412`, `0.1229`, and `0.0585` RMSE at those boundaries. The local slope window is required to remain wholly inside the earlier visible segment.

## Candidate ladder

| Candidate | Eligible | RMSE | Gain vs E006 | p90 well RMSE | Worst-5% SSE share | Map wins | Outer-cell wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`visible_fixed_0p10`** | yes | **14.8304784179** | **0.1026623694** | 22.0503957463 | 0.3567221937 | 5/5 | 20/25 |
| `crossfit_conservative` | yes | 14.8512754016 | 0.0818653856 | 21.5518797841 | 0.3633432094 | 5/5 | 18/25 |
| `crossfit_rmse_grid` | diagnostic | 14.8575079369 | 0.0756328504 | 21.8009637088 | 0.3607070645 | — | — |
| `crossfit_reliability_shrink` | yes | 14.8818004356 | 0.0513403517 | 21.7512628919 | 0.3662082308 | 5/5 | 19/25 |
| `crossfit_positive_gate` | yes | 14.9049530600 | 0.0281877272 | 21.7855500884 | 0.3667737989 | 4/5 | 18/25 |
| E006 nested fusion | comparator | 14.9331407872 | 0.0000000000 | 21.7855500884 | 0.3685061112 | — | — |
| raw bounded self-correlation | diagnostic | 25.9078871044 | -10.9747463172 | 38.5790028539 | 0.2326645152 | — | — |

No eligible candidate passes every gate. The reported fixed candidate is retained for transparent comparison only and is not selected or promoted.

## Stability and failure anatomy

- Fixed 0.10 map gains are identical at **0.1026623694** because the fixed path is the same legal OOF placement under each immutable partition; it wins every map.
- Fixed 0.10 wins **20/25** repeated outer cells.
- Worst fixed spatial gain versus E006: **-0.1454598846**; spatial group 3 is worst.
- Worst fixed typewell-group gain versus E006: **-0.0968325010**; typewell group 2 is worst.
- Reliability shrink reduces the worst typewell regression to approximately `-0.02128`, but does not make every group positive.
- Long-suffix gain: **-0.0858814164**.
- High-GR-missingness gain: **-0.0944899681**.
- Poor-visible-pseudo-gain slice gain: **-0.0804089860**.
- Low-match-margin slice gain: **0.2798715408**.
- The fixed path lowers worst-5% SSE share from `0.3685061112` to `0.3567221937`, showing useful tail concentration despite subgroup instability.

## Controls and edge cases

All **14** registered controls pass; none fail:

- Parent E006 OOF SHA-256 and E006/E004/last-known RMSE reproduce within `1e-9`.
- Exactly 3,783,989 unique ordered hidden-row IDs are emitted, all finite across 773 contiguous wells.
- Zero weight is an exact E006 no-op; duplicate placement has maximum delta `0.0`.
- Deterministically shuffled self-correlation removes all gain and returns exactly to E006 RMSE.
- Direct and well-aggregated SSE agree within `1.1824e-13` relative error.
- The raw correction passes the frozen diversity gate at residual correlation `0.4675700642`.
- Maximum observed correction is `39.9999581674` ft under the 40 ft soft cap.
- The rowwise oracle remains ineligible and scores `11.4682809263` RMSE.
- Runtime and memory are within the frozen 30-minute and 1,536 MB limits.

The 13-test E007 edge suite passes under Python 3.10, 3.12, and 3.13. It covers malformed configuration, boundary leakage, deterministic fingerprints, missing/constant/repeated GR, one-row and short suffixes, noncontiguous visibility, duplicate MD, non-finite parent OOF rows, conservative fallback, reliability gating, and soft-cap saturation.

## Reproducibility

- Official frozen-code run: **5:26.68**, maximum RSS **98,844 KB**, exit status `0`, empty stderr.
- Independent clean-output reproduction: **5:31.93**, maximum RSS **99,228 KB**, exit status `0`, empty stderr.
- OOF artifact: **204,668,270 bytes**, SHA-256 `9b90d44797950f9c08fe873fe42438edae8cd92de16f49beadbe1584d3b826b7`, byte-identical across runs.
- Eleven static scored CSVs are byte-identical. Summary and control files are semantically identical after normalizing measured runtime/RSS; artifact manifests match after normalizing path-sensitive and runtime-bearing entries.
- Independent row recomputation agrees with every candidate RMSE within `9.4e-13`.
- All 35 membership contexts pass: 25 repeated cells, five spatial holdouts, and five typewell evaluator holdouts.

## Deployment decision

Statistical authorization is false. No E007 model, inference notebook, Kaggle kernel, submission artifact, or leaderboard submission was created. E006 remains the primary deployment-ready surface-free candidate and E004 remains its exact fallback.

## Next action

Do not retune E007 weights or thresholds. Pre-register E008/H010 as a compact cross-fitted legal residual-action model. E007 fingerprints and visible-only diagnostics may be used only as input evidence within each training fold; the rejected fixed path and observed E007 outcome cannot be reused as an untuned promotion candidate.
