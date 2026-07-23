# T023 Preregistration — Residual sequence-state worth screen

Frozen before any T023 residual diagnostic or candidate score was computed.

- Source commit: `f2a295606132a8a2f067c064823c71907d33c0bb`
- Base: exact E011 `spline4_ridge_equal_s075`
- Population: 773 wells and 3,783,989 hidden rows
- Validation: 25 immutable repeated whole-well contexts; ten spatial/typewell stress contexts for every preliminary passer
- Full machine-readable contract: `tracking/evidence/T023/config.json`

## Question

Does E011 leave legally predictable within-well residual state that is not merely another well-level regime or E011/E006 router?

## Sequence representation

The screen constructs 16- and 32-control piecewise-linear residual profiles around E011. Correction is fixed to exactly zero at the suffix boundary. Features use only test-available row sequence evidence: MD, Z, observed GR, the visible TVT prefix, E011/E006 paths, and typewell TVT/GR forward mismatch. Formation surfaces, hidden labels, absolute coordinates, evaluator group IDs, overlap identities, public scores, and external artifacts are forbidden.

## Branching

The screen completes linear profile regression, local block regression, nonlinear histogram gradient boosting, nearest-profile regression, a typewell-only physical branch, and a disagreement-only branch. It also completes three distinct negative controls: shuffled well profiles, reversed residual profiles, and shuffled GR/typewell sequence evidence. Both 16- and 32-control resolutions are tested where registered.

## Gates

Oracle capacity must first show at least 0.50 RMSE headroom for 16 controls. Legal preliminary advancement requires at least 0.05 RMSE gain, 4/5 map wins, 17/25 cell wins, bounded p90 and worst-5% risk, and no negative-control status. A new full experiment is authorized only at at least 0.10 RMSE gain with positive every-spatial, every-typewell, and all special-slice transfer, failed negative controls, exact independent reproduction, and all edge groups.

No neural or new state/path implementation is authorized by this preregistration alone.
