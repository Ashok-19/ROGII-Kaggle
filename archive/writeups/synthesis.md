# Winning Working-Note Synthesis

## Shared conclusion

Both award-winning notes independently identify the dominant problem as **whole-well structural position**, not row-local wiggle prediction. Let `U = TVT + Z`. Much of the high-frequency TVT movement is inherited from known trajectory `-Z`; the difficult term is the smooth level/trend of `U`, including a per-well datum and occasional structural breaks.

## Evidence from “The Wiggle Is Free, the Trend Is the Wall”

- Flat-last-known baseline: about 15.9 RMSE on the leaderboard and 15.1 on the authors' whole-well holdout.
- A smooth-surface oracle reaches roughly 3.0–3.9 ft, showing that most avoidable error lives in the trend/datum.
- The worst 10 wells in a 773-well proxy contribute 25.4% of pooled squared error.
- A decorrelated particle-filter ensemble improved 7.230 to 7.096; a higher-seed robustness variant reached 7.091.
- A decorrelated neural correction plus physics post-processing produced real-LB gains 7.080 to 6.836 to 6.794.
- Individually plausible additions can destroy the stack: `pf_z` produced 7.446–7.515, and shrink-to-last-known hedges produced 7.252/7.304.
- The note treats changes below roughly 0.07 inside its PF family as potentially seed/public-split noise; that threshold is family-specific, not a universal rule.

## Evidence from “When Better CV Scores Worse”

- A 7.623 grouped-OOF model scored 6.924 publicly, while an older 8.248 OOF model scored 6.675. Aggregate CV ordering inverted.
- Worst 5% of wells contributed 52.5% of SSE; worst 10% contributed 64.5%.
- A constant mean error per well explained 58.3% of SSE; oracle removal reduced RMSE to 5.014.
- A weak trellis model scored 13.420 alone but had only 0.488 error correlation with the neural ensemble; a small blend improved 7.762 to 7.699.
- A spatial/tree member improved ordinary CV but worsened public score and failed leave-spatial-out evidence, so it was rejected.
- Risk can often be detected, but signed correction direction is difficult: detection does not justify actuation.

## Project rules derived from both notes

1. Optimize pooled row RMSE, but always report per-well RMSE distribution and SSE concentration.
2. Validate hidden suffixes by whole well; never use random row splits.
3. Track the datum/trend target explicitly through `U = TVT + Z` and residual decompositions.
4. Require positive, no-op/duplicate, and shuffled-evidence controls.
5. Evaluate every candidate on repeated fold maps and at least one harsher split, such as spatial or typewell-group holdout.
6. Keep weak components only when their OOF residual correlation is low and the blend improves consistently.
7. Separate uncertainty detection from signed action; a gate needs evidence that it chooses the right direction.
8. Preserve negative results and exact configurations to prevent cycling.
9. Keep one public-proven final family and one decorrelated private-expectation family.
