# E013 preregistration — clean-room sequential likelihood-PF

Status: **frozen before implementation scoring**  
Hypothesis: `H031`  
Task: `T039`  
Decision: `D043`

## Scientific question

Does an independently implemented sequential likelihood-weighted particle filter retain the T038 multi-RMSE gain over all 773 wells when placed at the already frozen weight `0.25 × E011 + 0.75 × PF`?

## Legal inputs and provenance

The scoring implementation may read only competition train/test-available columns: visible `TVT_input`, horizontal `MD`, `Z`, and `GR`, and paired typewell `TVT` and `GR`. E011 OOF is the legal frozen baseline. Hidden `TVT` is evaluator-only.

The algorithmic idea was identified in a public notebook, but E013 may not import, execute, package, or copy that notebook or any public artifact. Before hidden-label scoring, the clean-room implementation must match the source-visible algorithm on three score-blind wells for finite output and numerical parity. Public code remains quarantined.

Forbidden: contacts, formations, Geology, same-well target lookup, overlap rules, pretrained/model-package artifacts, public scores, score-derived constants, hidden routing, per-well blend weights, and post-score parameter changes.

## Frozen mechanism

For each well and each seed 0–15:

1. estimate visible-prefix GR residual scale against typewell GR and clip it to 10–60;
2. initialize particles over structural coordinate `U = TVT + Z` and dip rate from the visible heel;
3. propagate dip rate and U with the fixed process noise in `config.json`;
4. weight particles by hidden horizontal GR likelihood under typewell GR;
5. systematically resample below the fixed effective-sample threshold;
6. output the weighted TVT path;
7. combine the 16 seeded paths by their complete hidden-suffix likelihoods at temperature 5.

Evaluate PF standalone and fixed global blends at PF weights 0.50, 0.75, and 1.00. The primary candidate is permanently fixed at 0.75 from T038. The full-data result may not choose another weight.

## Validation

- exactly 773 wells and 3,783,989 hidden rows;
- pooled row RMSE and SSE identity;
- all five registered maps and 25 cells;
- legacy spatial/typewell groups, legal-covariate, spatial-2D, and horizon groups using the exact T036 membership artifact;
- median, p90, p95, maximum well RMSE;
- worst-5% and worst-10% SSE share;
- long suffix, high hidden-GR missingness, and E011-catastrophe slices;
- runtime, RSS, finite rows, and deterministic output.

## PF-specific controls

Use a fixed 100-well label-independent panel: 20 equally spaced sorted IDs from each `folds/v1.json` fold.

Destructive controls repeat the same PF and fixed 0.75 placement with:

- all GR emissions removed;
- reversed typewell GR;
- hidden horizontal GR circularly shifted by one third;
- hidden horizontal GR deterministically permuted.

The legal candidate must beat every destructive control by at least 0.50 RMSE on that panel.

Structural controls require:

- clean-room/public parity on three score-blind wells before metrics;
- duplicate deterministic rerun;
- reversed seed-order invariance;
- all-missing hidden GR and short visible-GR support finite behavior;
- input immutability;
- exact E011 no-op identity;
- full row and group-membership identity.

## Frozen decision

BREAKTHROUGH requires primary RMSE `<= 6.0` plus every GO gate.

GO requires:

- primary RMSE `<= 8.0`;
- 5/5 map wins and at least 20/25 cell wins versus E011;
- every legacy spatial and typewell group improves;
- p90 does not worsen;
- worst-5% SSE share increases by at most 0.005;
- every destructive and structural control passes;
- runtime at most four hours and observed RSS below 4 GB.

A result above 8.0 or any failed scientific/control gate rejects the exact frozen family. No repair or retuning of weight, particle count, seed count, seed temperature, process noise, GR scale, resampling, control panel, or groups is allowed after metrics.

No Kaggle run, save, or submission is authorized.
