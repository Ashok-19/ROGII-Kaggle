# ROGII public-intelligence delta — 2026-07-23 19:34 Asia/Kolkata

Cutoff before this scan: 2026-07-23 18:35 Asia/Kolkata.

## Leaderboard

The first 20 public leaderboard rows are unchanged at the decision-relevant bands:

- rank 1: 4.859
- rank 2: 4.905
- rank 5: 5.265
- rank 10: 5.511

## New discussion

Topic `728477`, “The public LB is a precise ruler and a biased one,” is participant analysis rather than organizer guidance. It argues from five published CV/LB pairs that one participant had correlation 0.999, slope 1.01, offset +0.32 ft, and residual scatter 0.028 ft, while other pipelines report 0.2–0.5 ft seed/run variation. The author explicitly does not verify the roughly 200-hidden-well estimate. This is hypothesis-generating only. It reinforces existing policy: use whole-well CV for model selection, deterministic local reproduction, and do not infer private performance from absolute public score.

## Newly surfaced notebook records

Exact source was pulled for all four records surfaced after the prior audit.

### `nikitagajbhiye30/rogii-0001`

- Source SHA-256: `47511037d8dd5c2954b3510a8be465eead9ab6969031bcdc0215a48ae063fa24`
- Code SHA-256: `f32ab97cdab420ddd77e8a7c43f7a5c00ae8c274e96b5b75811a2c440a1446f0`
- Uses the same seven external datasets as the existing composite stack, including quarantined unknown-license inputs.
- Normalized-line Jaccard similarity is 0.941321 versus `iaztec/top-pf-config-branch-conservative-visuals` and 0.990597 versus `zhexinjiang/rogii-shift-275`.
- Retains `vp_balanced_modelpkg_005`, precomputed submission discovery, model-package layers, train-only formations, guarded same-well overlap, and artifact models.
- Conclusion: another composite-lineage derivative, not independent unseen-well evidence.

### `zhexinjiang/rogii-shift-300-v1`

- Source SHA-256: `5b32857abba4b77dc38bc8bb0ede333ea82e70e11cb47f11c6c01bae53c173bb`
- Code SHA-256: `89a181948a6d1a25c8237a74c298a8357447279784060b5c873f889089777525`
- Normalized-line Jaccard similarity is 0.999054 versus `rogii-shift-275`.
- The only substantive source difference is changing the visible well `00e12e8b` extra shift from +0.75 ft to +1.0 ft, producing a stated total shift of 3.0 ft.
- Conclusion: score-directed visible-well tuning only.

### `fleongg/rogii-yuanzhe-full5-public-submit`

- Source/code SHA-256: `278321a96ea01f1caf7096896f659cb1fc53dbb7c956e7620b8fee83271e2989`
- The 58-line script does not contain the model. It locates an external dataset bundle and executes `code/kaggle_final_lt8_infer.py` from that mount.
- Conclusion: opaque external-bundle launcher; no mechanism or validation claim is accepted from the wrapper.

### `fleongg/rogii-yuanzhe-blend-public-submit`

- Source/code SHA-256: `049104911b3a744039e87544a3422ec12fccaf53e8d6706d2dd8442cc895cb9e`
- Launches the same external full5 bundle and blends its output with `baseline_opencv411.csv` from a public prediction-bank dataset at weights 0.15, 0.25, 0.35, 0.50, and 0.65.
- Conclusion: artifact/prediction-bank blend, not clean causal evidence.

## Decision impact

No new public source changes the legal mechanism queue. T026 remains the exact next action. Public notebooks, scores, fixed shifts, opaque bundles, and participant LB calculations do not enter T026 features, targets, thresholds, or branch selection.
