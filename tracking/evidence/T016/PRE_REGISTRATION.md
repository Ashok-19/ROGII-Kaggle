# T016 preregistration — E011-relative stable wide residual datum worth screen

Frozen before computing any T016 placement score.

- Source commit: `b47d620378d4b56eac7fd33d682ee03826440e25`
- Baseline: exact E011 `spline4_ridge_equal_s075`
- Population: 773 wells and 3,783,989 hidden rows
- Inputs: existing cross-fitted E009 per-well legal datum actions plus exact E011 OOF predictions
- Full contract: `tracking/evidence/T016/config.json`

## Why this screen comes first

H012 was originally motivated by a 64-feature ridge that improved E006. E011 is now materially stronger, so the old gain cannot be assumed to remain useful. T016 first tests whether any saved E009 action retains signed residual value when placed on E011. No feature or model is refit in this stage.

## Branching

The screen completes all six saved E009 action families, mean/median robust ensembles, a stable three-action median, and three fixed wide64/path-evidence blends. Every family is tested across frozen positive, zero, and negative scales and four action caps. For each repeated or stress context, family, scale, and cap are selected only on complement wells.

## Gates

A full stability refit is allowed only if the nested placed system gains at least 0.03 RMSE over E011, wins at least 4/5 maps and 17/25 cells, preserves tails, improves every spatial and typewell holdout and all three special slices, defeats three negative controls, reproduces exactly, and passes all edge groups.

Failure closes H012/T016 for this action space. It does not authorize scale retuning, subgroup routing, or relabeling E009's old E006-relative result.
