# ROGII Public Intelligence Refresh — 2026-07-22

Checked through Kaggle MCP at 2026-07-22 14:59 Asia/Kolkata. This record separates exact Kaggle metadata and source-visible mechanisms from participant claims, public-test overlap behavior, and inference. It does not promote any public artifact or score.

## Current public leaderboard

| Rank | Team | Public score |
|---:|---|---:|
| 1 | shu01 | 4.859 |
| 2 | Yannan Chen | 4.913 |
| 3 | SaintLouis | 5.108 |
| 4 | Rishikesh Jani | 5.237 |
| 5 | Shrey Gandhi | 5.265 |
| 6 | L & J & A & A | 5.325 |
| 7 | yu4u | 5.443 |
| 8 | Tucker Arrants | 5.444 |
| 9 | Ryo Takaki | 5.511 |
| 10 | Nocturne & Adil | 5.523 |

The solo top-five target is currently below about 5.265. Second place improved from 5.083 in the 2026-07-18 archive to 4.913, so the competitive frontier is moving materially.

## New and recently updated discussions

### Topic 728152 — hidden scoring runtime

- A participant reports a visible three-well notebook runtime near 15 minutes, followed by more than nine hours in scoring and eventual timeout.
- A reply scales 15 minutes over three visible wells to roughly 17 hours over about 200 hidden wells and recommends a visible runtime below roughly nine minutes, preferably below five minutes after allowing overhead and length effects.
- This is participant guidance rather than an official runtime formula, but the arithmetic and the competition's hidden-well scale make it a strong deployment warning.
- Consequence: every finalist needs a pseudo-hidden inference benchmark over roughly 200 held-out wells. Validation runtime and inference runtime must be measured separately.

### Topic 728022 — commercial external subsurface data

- A participant asks whether paid Enverus/IHS-style databases violate the equal-access/no-cost external-data rule and whether public government records are acceptable.
- No organizer answer was present at the audit time.
- Consequence: commercial data remains blocked. Free public records are not authorized for the prize pipeline until rules/provenance/reproducibility are resolved.

### Topic 727570 — CV/LB and hardware variance

- The earlier self-report remains 4.98 grouped whole-well CV over five folds and five seeds versus 5.7 public LB.
- New comments report that exact code, folds, and seeds can still differ across GPU hardware; another participant reports approximately CV 6/7/8 mapping to LB 8/9/10, while a GRU near CV 6 reportedly reached LB 7.35.
- These are participant claims, not reproduced measurements.
- Consequence: deterministic CPU or tightly controlled accelerator reproduction, simple-family baselines, and explicit CV/LB uncertainty remain mandatory.

### Topic 727708 — hidden rerun creates no output

- The reported stack materializes about 7.39 GB, 195 features, three LightGBM models, two CatBoost models, and Ridge, then reruns full OOF work.
- The visible save succeeds, but the hidden rerun reports COMPLETE with zero output bytes.
- A reply recommends pseudo-test execution over about 200 training wells. The latest additional comment was deleted and adds no technical resolution.
- Consequence: compact streaming, fail-closed output receipts, and hidden-scale smokes are required.

### Other deltas

- Topic 721549 still reports that pattern matching can produce attractive paths but damages flat wells; no new solution was supplied.
- Topic 697400 asks about an offline `hmmlearn` wheel. A participant says an uploaded wheel or Kaggle dependency may work, but this is operational advice rather than an official rules ruling.
- Kaggle now lists 137 competition topics. The canonical local archive remains 132 topics and 981 messages until a complete recrawl updates canonical counts.

## Exact public-notebook audits

### Shared composite lineage

The current P100/ANCC, F594, public 6.40, public 6.768, MHA 6.832, high-vote public TVT, and Yusuke second-approach notebooks are mostly forks of one composite architecture rather than independent methods. Common source-visible components include:

- pretrained LightGBM/CatBoost/Ridge or static submissions;
- 128-seed likelihood-weighted particle filtering and beam paths;
- robust projection in `U = TVT + Z`;
- visible-prefix multi-cut calibration;
- guarded same-well contact reconstruction using train formation columns;
- optional model-package corrections;
- public overlap probes, fixed public-well branch shifts, or score/canary-derived bias constants;
- external datasets already quarantined for unknown licensing: `phongnguyn23021656/koolbox-offline`, `nina2025/rogii-03`, and `needless090/rogii-tabicl-mirror`.

Their public scores are composite references, not clean causal evidence for unseen-well generalization.

### prvsiyan — P100 Confirmed ANCC Seed Frontier, version 25

- Source SHA-256: `607b234d9cea9e30d594724192587f71752c20f0e23f8b15818c710cd9087f41`.
- Active profile: `vp_balanced_modelpkg_010`, with guarded overlap and visible-prefix calibration enabled.
- A later explicit override raises the model-package gate to 0.020 based on twelve preregistered route partitions.
- The 128-seed PF cloud is summarized as two weighted level branches. On the three visible wells, only `00e12e8b` triggers the branch hedge: branch centers differ by 29.439 ft and the notebook moves all 4,301 rows by +2.0 ft. The other two wells move zero.
- Its model-package trajectory differs from the base by 16.264 RMSE and 26.701 ft p95 absolute difference, so the notebook's own difference guard disables that correction.
- The final audit identifies its locked base as inherited from `beicicc/rogii-mha160-sep3-r2-20260720`; it does not establish an unseen-well score for the P100 delta.

### pi500phys — Public 6.40 Guard Repro

- Source SHA-256: `92fc5934f04fb6a62b6e9ee89268a5e57f309f352018314653b2731ad3fee2f5`.
- The exact public-overlap calibration uses weight 0.055 and activates all three known visible wells only.
- Per-well mean absolute moves are 0.385, 0.489, and 0.287 ft; the declared fallback is unchanged upstream prediction for unseen or mismatched wells.
- Therefore the 6.40 title is not evidence that the 0.055 layer improves hidden unseen wells.

### pi500phys — Public 6.768 Likelihood Slope

- Source SHA-256: `1c68827ff35fa702577aa0698f86ac1c918f3993f580c35c1a3d2ad2c69e0e8a`.
- The source claims nested well-level OOF improves a weighted base from 10.3859 to 10.1858, but the public output report applies zero correction to all three visible wells because contact override already controls them.
- The preceding continuity layer is bounded and target-free in form, but it is composed on top of public-tuned artifact and overlap layers. It is a mechanism lead, not portable score evidence.

### muelsyse111 — MHA250SEP2 WellBias Measured 6.832

- Source SHA-256: `5b90fe55e8a94df8bc038387240d916e07c95244198f1f69687846f3c3a5c801`.
- The notebook uses 128-seed PF, train-only formation surfaces, pretrained models, guarded overlap, a probe/canary section, and a global -0.40 ft correction explicitly decoded from a public probe score.
- Its midpoint hedge shifts one public well, and its embedded well-bias random forest changes many output variants. The well-bias model is an interesting legal-feature hypothesis only after clean reconstruction and frozen OOF validation; the canary-derived global bias is ineligible.

### Yusuke — Another Approach 2nd, version 4

- Source SHA-256: `ee70216f5a5926df631c71cf2b8949f883128ccd9f0d523a30d272b689912590`.
- A23 changes only public well `00e12e8b`: +0.5 ft on 4,301 rows, extending an existing +2.0 ft public branch shift to +2.5 ft relative to an earlier 6.809 lineage.
- Its experiment summary explicitly uses source public score 6.594. This is score-directed public-well tuning, not unseen-well evidence.

### blacklions — Well-Level GBDT Gate, version 3

- Source SHA-256: `aaa4f621b794ffd7fb8fec4df6ad14a1e3623cf6f9db09752c40837e8e6b0ba1`.
- It aggregates 391 features over 773 wells from the composite precomputed pipeline.
- Executed OOF: base 10.3722527, gate 10.3823964, gain -0.0101438; p90 worsens from 14.6850 to 14.8066.
- Its own gate remains inactive. This is useful negative evidence against a shallow well-level residual gate on that feature representation.

### evgendvorkin — Single CatBoost, version 47

- Source SHA-256: `22cc728ef9b6bc6645cfe3b56a2cdab63a83cec77d6369b7d4ea31ef5104da6e`.
- Despite the title, the executed source contains no CatBoost model. It is a compact PF/beam selector with special handling for the three train-overlap visible wells.
- It is comparatively self-contained, but it supplies no full 773-well grouped CV result and no clean high-score evidence.

## Reusable, clean-room hypotheses

The source audit yields four ideas worth testing without copying artifacts or public-well constants:

1. **PF seed-cloud branch uncertainty:** summarize likelihood-weighted seed levels into branch separation, branch mass, entropy, and stability features. Learn or validate action strictly OOF; do not hard-code the public `00e12e8b` shift.
2. **Bounded boundary/continuity correction:** measure `U` gap, visible-prefix slope, and candidate endpoint disagreement; apply only through a preregistered split-local model with E006 fallback.
3. **Prefix-GR well-bias features:** reproduce the legal prefix diagnostics independently and test a small correction model under repeated spatial/typewell splits. The public embedded forest and canary-derived -0.40 ft correction are not reusable.
4. **Hidden-scale inference benchmark:** run final inference logic over roughly 200 pseudo-test wells and require fixed IDs, bounded memory, deterministic output, and a conservative visible-runtime target around five minutes.

## Strategic consequence for a solo top-five attempt

- The current cutoff is about 5.265. Public composite notebooks around 6.4–6.8 do not close that gap and do not supply clean unseen-well causality.
- The project should not spend time reproducing the complete public stack. It should verify the newly appeared E011 output set, then use its errors to decide whether branch-uncertainty and bounded continuity features deserve a fresh preregistered experiment.
- Gold still requires converting the verified sub-5 path coverage into a legal selector/coefficient model while avoiding catastrophic minority wells. Public-test overlap optimization is orthogonal to that hidden-test problem.
- No public artifact, train-only formation column, commercial data, score-derived bias, probe/canary constant, or exact visible-well override is authorized for the prize pipeline.
