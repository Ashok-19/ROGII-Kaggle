# ROGII public-intelligence delta — 2026-07-23 22:15 Asia/Kolkata

Previous cutoff: 2026-07-23 19:34 Asia/Kolkata.

## Leaderboard and discussions

- Public rank 1 remains 4.859.
- Public rank 2 moved from 4.905 to 4.904 on a 2026-07-23 15:15 UTC submission.
- Rank 5 remains 5.265 and rank 10 remains 5.511.
- Competition topic count remains 139.
- Topic 727570 received one new validation comment. It recommends whole-well and spatial/field holdouts, reports a roughly 0.3 ft gap between simple well-CV and field-CV baselines, and confirms that the three visible test wells are exact training placeholders. This supports the project's existing immutable whole-well and spatial stress policy; it does not introduce a new predictor.

## Exact source audit

Seven distinct post-cutoff notebook records were pulled as source only. No notebook was executed.

### Composite frontier lineage

The following notebooks mount the same seven-dataset composite stack containing pretrained models, formation features, overlap/contact logic, and model-package or precomputed outputs:

- `kaiwalyaatulraut/rogii-solution`
- `losist/rogii-fork-frontier-lab-639`
- `losist/rogii-fork-codex-640-guard`
- `zoli800/rogii-public-score-frontier-lab-visuals-7ab570`

Normalized-line overlap is 0.999560 between the Kaiwalya and Losi frontier sources and 0.987325–0.987760 against the Zoli source. These are not independent clean unseen-well evidence and do not alter the legal queue.

### Dip-aware HMM + GBM

`tiktoktrendz/rogii-dip-aware-hmm-gbm` is a 681-line source with PF, beam, second-order HMM, and a row-level HistGradientBoosting meta-learner. It declares no mounted external dataset in metadata but hardcodes optional paths to Ravaghi artifacts, Fleongg learned predictions, and a model package. Its final postprocessing includes formation/Geology contact logic and same-well overlap handling for visible test wells. It constructs GroupKFold predictions but reports no grouped OOF RMSE, no spatial/typewell transfer, and no negative-control result. The HMM family is already covered by E010 and subsequent screens. This source does not justify replacing T027.

### Honest-CV notebook

`souldrive/rogii-tvt-identity-and-honest-cv-design` is a clean validation/identity notebook with no external mounts. It verifies whole-well and field splits and the visible-test overlap issue. It supplies validation-policy support, not a new prediction mechanism.

### Embedded clean-room alignment bundle

`atharvasoundankar/notebook09f05fb39b` embeds eleven reviewed modules for beam alignment, structural alignment, legal features, final training, inference, and spatially balanced fold assignment. Its gate receives out-of-fold well predictions, but its final sampled row-residual model is fit on all suffix rows and the notebook reports no OOF RMSE, spatial/typewell transfer, or negative controls. It therefore remains an unmeasured implementation proposal rather than evidence of a superior path.

## Decision

No new source changes the next evidence question. Proceed with a bounded target-preserving causal feature-view screen: every auxiliary view must retain the same source well's original competition-boundary spline target, use the frozen 142 legal E011 feature schema, preserve exact outer-well isolation, and prove gain over both E011 and an original-view-only comparator before any formal or neural implementation.
