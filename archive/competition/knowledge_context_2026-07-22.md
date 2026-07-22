# Verified knowledge context — 2026-07-22

## ROGII public/private leaderboard split

The exact split was verified from Kaggle's official Meta Kaggle dataset rather than inferred from the visible leaderboard.

- Kaggle competition ID: `132265`
- Competition slug: `rogii-wellbore-geology-prediction`
- Official file: `kaggle/meta-kaggle`, `Competitions.csv`
- Snapshot file creation time reported by Kaggle: `2026-07-22T07:27:53.964Z`
- Downloaded byte size: `153329115`
- Downloaded SHA-256: `acdc004d8f532ff7913c1b76b4ffd97bad3657c0651b77422ba819956e4a3576`
- Exact row field: `LeaderboardPercentage = 26`
- Therefore: public leaderboard = `26%` of the scored test data; private leaderboard = `74%`.

The normal competition pages describe the public leaderboard only as a representative sample. The exact percentage comes from the official Meta Kaggle competition record.

## BirdCLEF+ 2026 first-place writeup

Official Kaggle MCP discovery and message retrieval verified the following source identity:

- Competition: `birdclef-2026`, Kaggle competition ID `129329`
- Author: Nikita Babych, Kaggle user ID `6343664`
- Title: `1st Place Solution: Noisy Student Meets Distillation`
- DOI supplied for the writeup: `10.34740/KAGGLE/W/86265`
- Kaggle writeup topic ID: `704752`
- First forum message ID: `3467343`
- Post time: `2026-06-05T23:37:20.710Z`
- Official topic URL: `/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati`

### Exact structural lessons

The winning system was a diverse 5-second ensemble, not a single specialist model. It combined SED-head CNNs, an MLP-head CNN, an Amphibia/Insecta specialist, a genus-level specialist, and native Perch v2 predictions.

Training used two phases. Backbones were first distilled from teacher embeddings with cosine loss, primarily from Perch v2 and once from AudioProtoPNet for diversity. Distillation was then disabled for low-learning-rate end-to-end fine-tuning, except for the stated MLP variation. Fine-tuning used supervised data followed, where enabled, by Multi-Iterative Noisy Student self-training.

Pseudo-labeling was tightly controlled. The pseudo-label sum was capped below the focal-label sum; labeled-soundscape and pseudo-labeled-unlabeled-soundscape injectors operated on separate focal samples and were never allowed to overlap; labeled-soundscape injection remained present to counter pseudo-label noise. Two pseudo-label iterations improved the tracked result, while a third slightly regressed it.

The Amphibia/Insecta specialist had a narrow, explicit role. It used a `tf_efficientnet_b3.ns_jft_in1k` SED model over an extended Amphibia/Insecta label space, was trained supervised-only for 40 epochs, and used extra public multi-taxon samples. Its restricted predictions were scattered back to the full 234-class width with a mask before blending. The source does not support the simplified claim that specialists alone won the competition.

Diversity was deliberately introduced through different CNN families, heads, label spaces, mel inputs, distillation teachers, and pseudo-label iteration counts. The author states that distillation plus self-training produced highly correlated models and that design diversity was needed to add useful nonlinearity.

Validation used two complementary labeled-soundscape splits: a site-oriented split for unseen-site generalization and a greedy split for species coverage. Post-processing was tuned separately for fine-tuned models and native Perch predictions, mostly on those validation splits rather than by leaderboard probing. The final blend rank-transformed each class before combining the fine-tuned ensemble at weight 0.8 with native Perch at weight 0.2.

### Transfer rule for ROGII

The transferable principle is conditional specialization, not "specialists win." A regime-specialized coefficient expert is admissible only when:

1. the regime is defined from legal test-time inputs and routed cross-fitted;
2. the regime is demonstrably distinct and has sufficient training support;
3. the specialist's action is bounded and uncertain cases fall back exactly to the strongest legal global path;
4. the complete routed system improves pooled, repeated-map, outer-cell, spatial, typewell, missingness, tail, and catastrophe metrics;
5. shuffled, sign-flipped, duplicate, random-routing, and regime-label negative controls lose the gain;
6. source, artifacts, licenses, runtime, and deterministic reproduction pass independently.

Public notebooks and writeups remain hypothesis generators. Their mechanisms may be reconstructed cleanly, but their pipelines, scores, thresholds, artifacts, and routing rules are not trusted project evidence until independently preregistered, reproduced, and validated under the ROGII contract.
