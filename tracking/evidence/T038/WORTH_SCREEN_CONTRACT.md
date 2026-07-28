# T038 preregistration — public HMM/PF mechanism worth audit

Status: **frozen before implementation and before any additional T038 scoring**  
Hypothesis: `H030`  
Task: `T038`  
Decision: `D042`

## Question

Do self-contained trajectory mechanisms visible in valuable public notebooks—exact second-order HMM smoothing and likelihood-weighted particle filtering—contain enough complementary legal signal to justify a fresh clean-room experiment aimed at sub-6 RMSE?

This is a bounded source audit, not a promotion experiment. The downloaded public notebook remains quarantined. No public source code, model artifact, contact file, formation file, overlap rule, pretrained model, score-derived constant, or generated prediction may enter a deployment candidate from this task.

## Fixed evidence panels

The primary panel contains exactly 20 wells: within each `folds/v1.json` fold, sort well IDs and take four equally spaced indices including both endpoints. This selection is independent of hidden labels and model errors.

The observable edge panel adds at most ten wells selected without hidden labels: two extremes each for hidden-row count, known-row count, hidden-GR missing fraction, hidden-GR robust scale, and maximum MD step. Duplicate wells are removed. Edge wells are reported separately and cannot choose a branch.

E011 OOF predictions are the immutable baseline. Scored rows are exactly the registered hidden suffix rows with finite target and baseline prediction.

## Frozen branches

Run all four HMM configurations in `config.json`, one likelihood-PF branch, every fixed E011/tracker blend at weights 0.25, 0.50, and 0.75, a 50/50 HMM/PF blend, and the two registered E011/HMM/PF three-way blends. No continuous weight optimization, per-well routing, subgroup routing, or post-score branch addition is allowed.

The HMM state is joint TVT position and dip rate in `U = TVT + Z`; it uses exact forward-backward posterior means. The PF is the source-visible likelihood-weighted seeded particle ensemble. Both use only visible-prefix TVT, horizontal GR, typewell GR, MD, and Z.

## Controls and edge cases

On the first ten primary wells, repeat the selected legal HMM configuration with:

- emissions removed (`lam=0`);
- reversed typewell GR;
- hidden horizontal GR circularly shifted by one third of the suffix;
- hidden horizontal GR permuted with a fixed well-derived seed.

Also require:

- byte/numeric identity for a duplicate configuration and deterministic rerun;
- complete row/order coverage and finite predictions;
- all-missing hidden GR behavior;
- short visible-GR support behavior;
- no mutation of input frames;
- exact E011 no-op fallback identity;
- pooled RMSE/SSE algebra identity;
- every registered branch completes or the task is invalid.

## Frozen worth decision

A fixed global candidate passes only if all are true:

1. primary-panel gain versus E011 is at least 1.0 RMSE;
2. it improves at least four of five fold strata;
3. p90 well RMSE does not worsen;
4. worst-20% well SSE share increases by at most 0.02;
5. edge-panel pooled RMSE is no more than 0.25 worse than E011;
6. the legal HMM candidate beats every destructive control by at least 0.50 RMSE on the control panel;
7. structural and missing-data controls pass;
8. projected 773-well runtime is at most four hours and observed RSS is below 4 GB.

A pass authorizes only a new E013 clean-room reimplementation and full five-map outer-isolated validation. A failure rejects H030 for this exact HMM/PF/fixed-blend family. No T038 repair, branch pruning, threshold change, weight tuning, or public-leaderboard probing is allowed after metrics.
