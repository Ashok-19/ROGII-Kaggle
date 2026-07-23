# ROGII public-intelligence delta — 2026-07-23 18:35 Asia/Kolkata

Previous cutoff: 2026-07-23 17:04 Asia/Kolkata.

This bounded T024 refresh separates exact Kaggle metadata/source from titles, votes, public scores, and unsupported mechanism claims.

## Leaderboard and discussions

The visible top ten remain unchanged: rank 1 is 4.859, rank 2 is 4.905, rank 5 is 5.265, and rank 10 is 5.511. Kaggle New still reports 138 topics. No new material discussion message after the prior cutoff was found.

The DateRun notebook search showed six records after the prior cutoff, while notebook-info metadata returned older July 20 `last_run_time` values for the same records. This timestamp discrepancy is retained as metadata ambiguity; it is not treated as proof of a new saved execution.

## Exact source audit

### Composite and fixed-shift lineage

- `iaztec/top-pf-config-branch-conservative-visuals` uses the same seven external datasets as the quarantined public composite stack, including `phongnguyn23021656/koolbox-offline`, `nina2025/rogii-03`, and `needless090/rogii-tabicl-mirror`. Its normalized source-line Jaccard similarity to the two new shift notebooks is 0.941676.
- `zhexinjiang/rogii-shift-225` and `zhexinjiang/rogii-shift-275` have 0.999054 normalized-line Jaccard similarity. Their only substantive source difference is an extra +0.25 versus +0.75 ft applied to visible overlap well `00e12e8b`, producing total shifts of 2.25 and 2.75 ft. This is score-directed visible-well tuning, not unseen-well evidence.

### Same-well target lookup

`yuezhengzhang/rogii-kaggle-inference-notebook` and `yuezhengzhang/rogii-wellbore-geology-prediction-v16` are byte-identical. They require a same-well training horizontal file, fit a degree-2 polynomial to the complete training target surface `TVT + Z`, calibrate it on the known zone, and fall back to direct training-TVT lookup. The source supplies no legal unseen-well mechanism or grouped validation.

### Second-order HMM derivative

`evgendvorkin/roggi-physics-lb-7-872-v48` has 0.872340 normalized-line Jaccard similarity to the already audited `amerhu/rogii-wellbore-geology-exact-hmm-smoother` source whose raw SHA-256 is `2321997c8bcbca442d7d7abfc5b9b7eeed251bac8c671b3554b54c9355c231dc`. It retains the same joint position/rate HMM, PF baseline, defaults, and small-sample comparison. The main additions are defensive plotting and a test submission loop. The comparison remains capped at 12 wells and the multi-cut harness at six wells. E010 already found zero unique full-data oracle wins from this HMM family.

## Decision

No new public source supplies independent, legal evidence that changes the project queue. Public composite/shift variants, same-well target lookup, and the HMM/PF derivative remain excluded.

The only mechanism selected by T024 for a bounded local screen is within-well multi-cut nonlinear coefficient dynamics: use several legal pseudo-cuts inside each observed prefix, fit the actual spline coefficient trajectory over those observed pseudo-tails, and test whether that trajectory forecasts the final hidden-tail coefficients. This differs materially from E011's 15 aggregate pseudo-cut summary features and directly targets the verified sub-5 spline-oracle capacity.
