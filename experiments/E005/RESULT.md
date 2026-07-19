# E005 Result — Clean Typewell-Horizontal GR Candidate Bank

Status: **completed and strictly rejected**
Run: `R20260719-1018-e005-gr-path`
Frozen design commit: `b55bfd542391a5e308ba6e5d02923e2a1d372b1f`
Frozen scoring-code commit: `8077526a6eab514ff08247cec5b19d41c9592317`

## Decision

E005 does **not** promote a standalone affine-alignment, particle-filter, or trellis candidate. The best raw candidate, `pf_gr_path`, improves pooled RMSE but fails the pre-registered p90, repeated-map, spatial, and typewell-cluster gates. E004 `geometry_prefix` remains the deployment-ready fallback.

No E005 deployment package was built. No private internet-disabled Kaggle parity run was started. No submission file was created for leaderboard use and no leaderboard submission was made. This follows the frozen rule that Kaggle parity is authorized only after statistical promotion.

## Exact score ladder

| Candidate | Eligible | Pooled RMSE | Gain vs last-known | Gain vs E004 | p90 well RMSE | Worst-5% SSE share | Residual corr. to E004 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pf_gr_path` | yes | **15.3502040715** | **0.5596487992** | **0.1411023267** | 23.4851836927 | **0.3331989190** | 0.8972483902 |
| `e004_geometry_prefix` | comparator | 15.4913063983 | 0.4185464725 | 0.0000000000 | **22.4630210994** | 0.3830949846 | 1.0000000000 |
| `last_known_tvt` | comparator | 15.9098528707 | 0.0000000000 | -0.4185464725 | 22.9725365671 | 0.3899072385 | — |
| `trellis_gr_path` | yes | 15.9165018153 | -0.0066489446 | -0.4251954170 | 22.5562922737 | 0.3729740210 | 0.9694920838 |
| `no_gr_typewell_affine` | no | 18.2067307296 | -2.2968778589 | -2.7154243313 | 26.2193163455 | 0.3520083096 | — |
| `align_visible_path` | yes | 19.6484954349 | -3.7386425641 | -4.1571890366 | 30.4935974603 | 0.3091321229 | 0.7398426205 |
| `align_affine` | yes | 23.3078052753 | -7.3979524045 | -7.8164988770 | 36.8963505721 | 0.2919348815 | 0.6274015469 |

The exact E004 comparator reproduced at `15.49130639826753`, differing from its frozen reference by about `3.6e-14` RMSE.

## Why PF is rejected despite aggregate improvement

The frozen promotion gate was conjunctive. `pf_gr_path` passed aggregate-gain, diversity, control, and worst-5% concentration gates, but failed five required stability gates:

| Gate | Registered requirement | Observed result | Status |
|---|---|---:|---:|
| Gain vs last-known | at least 0.35 | 0.5596487992 | pass |
| Gain vs E004 | at least 0.10 | 0.1411023267 | pass |
| Repeated maps | at least 4 of 5 | **3 of 5** | **fail** |
| p90 vs last-known | deterioration at most 0.50 | **+0.5126471256** | **fail** |
| p90 vs E004 | deterioration at most 0.50 | **+1.0221625933** | **fail** |
| Worst-5% share vs both | increase at most 0.01 | improved vs both | pass |
| Every spatial block | positive gain vs E004 | minimum **-0.7542149375** | **fail** |
| Every typewell cluster | positive gain vs E004 | minimum **-0.1856813071** | **fail** |
| Residual diversity | correlation at most 0.995 | 0.8972483902 | pass |

PF lowers maximum well RMSE from `68.8470020296` to `59.7753757843`, p95 from `29.3421252812` to `28.0885407324`, and worst-5% SSE share from `0.3830949846` to `0.3331989190`. Those improvements do not override the registered failures.

## Repeated maps and stress

PF wins a majority of fold cells on maps `v1`, `v2`, and `v4`, but not `v3` or `v5`. Its per-map winning-fold counts are `3, 3, 2, 3, 2`.

Spatial-block gains versus E004 are `-0.4745`, `-0.7542`, `+0.2917`, `+1.1059`, and `+0.5497` RMSE. Typewell-cluster gains are `+0.4568`, `-0.1857`, `-0.0746`, `+0.3287`, and `+0.1769`.

PF is directionally useful on the registered long-suffix, high-GR-missingness, and ambiguous-alignment slices, improving E004 by `0.8252`, `0.5630`, and `0.3103` RMSE respectively. The group instability still blocks promotion.

## Frozen controls

All model and evaluator controls passed:

- Data signature `6ebe65b403f80fefd97dcd7bbfce7314252e779a1837c97364cf55761590fe77` and all 773 wells matched.
- Exactly 3,783,989 hidden-row IDs were emitted.
- Direct pooled SSE and aggregated well SSE agreed within `1.1861e-13` relative difference.
- Duplicate alignment predictions differed by exactly `0.0`.
- Shuffled typewell GR scored `179.9124753271` versus `19.6484954349` for the corresponding unshuffled visible-alignment candidate.
- Wrong-axis control scored `187.4785008887`.
- Oracle leakage sentinel scored exactly `0.0` and remained ineligible.
- E004/no-GR fallback reproduced exactly. Synthetic all-missing and invalid-calibration cases abstain byte-for-byte to E004.
- Nine synthetic suites cover missing GR, repeated motifs, flat/short/no-overlap typewells, clipping, single/short/long suffixes, nonuniform and duplicate MD, missing pairs, noncontiguous visibility, malformed arrays, non-finite outputs, and sample-ID contract failures.

## Diagnostic blend: evidence, not promotion

The pre-registration allowed fixed PF-E004 blends only as diagnostics. The 50/50 blend has the best diagnostic aggregate result:

- RMSE: **15.0131171651**.
- p90 well RMSE: **21.7769037293**.
- Worst-5% SSE share: **0.3638317937**.

This blend is not an E005 promoted candidate. Selecting or deploying it after observing the score would violate the frozen candidate and placement boundary. It is recorded only as evidence for H009/E006, which must estimate placement or weights with fresh nested/cross-fit validation and repeat every stability gate.

## Runtime, memory, and reproducibility

- Official run: 432.38 seconds; maximum RSS 316,400 KB.
- Independent clean-output run: 409.97 seconds; maximum RSS 316,080 KB.
- Runtime limit: 45 minutes — passed.
- Memory limit: 1,536 MB — passed.
- Ten scored result files plus the 163,956,757-byte OOF gzip reproduced byte-for-byte.
- The path-sensitive manifest differed in the first comparison only because the independent output root appeared in its external-artifact path. After rebuilding under the canonical `artifacts/E005/...` relative layout, it was byte-identical at SHA-256 `011363023c3f8cf958b0154b9ced8d798c1c2e4c5aab81ce81802b8e373c9fb8`.
- Total compared bytes: 164,337,107.
- OOF SHA-256: `5115f7ce80f70c827c6df6edc572657253495a44333a2bb35bc46a7547e13ae5`.

The 163.96 MB OOF file is a local training-evidence artifact under the ignored `artifacts/` tree, not a deployment package. Because E005 was rejected, the 100 MB deployment-package cap was never engaged.

## Key hashes

- `summary.json`: `f7c6cdf3d7cccf7419395464627a1c5c440d85e4220bf375891fa534a795976a`.
- `candidate_metrics.csv`: `a549598e50aec9d9362dddf107b85c7744e0641d7b4f288a9724eb532b277864`.
- `fold_metrics.csv`: `c862ac376bdd8786ecf288499a907a74d107d51fd50a620020a16e953ac8c71e`.
- `stress_metrics.csv`: `58503003afa8e9abd9ea0d1933f82d185b6a0a8a6bbbb6e7c6e9debd182cd45d`.
- `selected_well_metrics.csv`: `93aae700d7d0bb34ea6eac205568593efb87cba2d31c877f58d0a7670765a8e0`.
- `artifact_manifest.json`: `011363023c3f8cf958b0154b9ced8d798c1c2e4c5aab81ce81802b8e373c9fb8`.
- OOF predictions: `5115f7ce80f70c827c6df6edc572657253495a44333a2bb35bc46a7547e13ae5`.

## Next action

Pre-register E006/H009 before scoring: validate PF-E004 fusion and disagreement evidence through nested or cross-fit placement under immutable E001 folds and the same p90, worst-tail, repeated-map, spatial, typewell-cluster, shuffled-evidence, runtime, exact-ID, and deployment gates. Do not treat the observed 50/50 diagnostic blend as validated deployment evidence.
