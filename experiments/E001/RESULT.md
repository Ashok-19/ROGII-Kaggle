# E001 Result — Metric, Fold, and Control Harness

Status: **promoted and frozen**  
Run: `R20260718-1720-e001-validation`  
Code SHA: `d2b0066c36988c5f8dcb77c54350bc641e167f82`

## Outcome

E001 passed every pre-registered gate. The evaluator, suffix contract, five repeated whole-well fold maps, controls, reports, and artifact hashes are now the immutable validation substrate for later experiments.

No model training occurred in E001.

## Validated data contract

- 773 horizontal wells.
- 5,092,255 total rows.
- 1,308,266 visible-prefix rows.
- 3,783,989 scored hidden-suffix rows.
- Every well has one contiguous visible prefix followed by one contiguous hidden suffix.
- Visible `TVT_input` matches `TVT` exactly; maximum observed absolute delta is 0.0.
- Data signature: `6ebe65b403f80fefd97dcd7bbfce7314252e779a1837c97364cf55761590fe77`.

## Last-known-TVT baseline

| Metric | Result |
|---|---:|
| Pooled row RMSE | 15.9098528707 |
| Median well RMSE | 10.6651407867 |
| p90 well RMSE | 22.9725365671 |
| p95 well RMSE | 29.0107644333 |
| Maximum well RMSE | 70.6393746197 |
| Worst 5% wells' SSE share | 38.9907% |
| Worst 10% wells' SSE share | 52.4763% |
| Fold RMSE range across 25 fold/map cells | 13.7492106667–17.5505922388 |

## Error decomposition

The baseline confirms that smooth per-well level and trend dominate error:

- Per-well mean/datum component: 67.7475% of SSE.
- Linear suffix-trend component: 14.5329% of SSE.
- Remaining shape component: 17.7196% of SSE.
- Removing the per-well mean oracle reduces RMSE to 9.0354099881.
- Removing per-well mean and linear trend oracles reduces RMSE to 6.6971947701.

These oracle values quantify headroom only. They are not legal predictors.

## Controls

All controls passed:

- Reference baseline: observed 15.9098528707 within the registered 15.91 ± 0.02 gate.
- Direct pooled SSE versus per-well aggregation: relative difference `1.1861e-13`, below `1e-12` tolerance.
- No-op control: unchanged RMSE.
- Duplicate control: unchanged RMSE and residual correlation 1.0.
- Synthetic legal-signal positive control: RMSE 0.0 versus null RMSE 44.2258637486.
- Shuffled-evidence control: degraded to RMSE 65.2990272953.
- Explicit target oracle sentinel: RMSE 0.0 and correctly marked target-using/ineligible.
- Five repeated fold maps have unique fingerprints; maximum-to-minimum hidden-row load ratio is 1.00803.

The synthetic positive control validates signal transport and evidence association in the harness. It is not evidence that a competition model improved.

## Determinism and runtime

Three full runs from committed code produced byte-identical copies of all 11 fold/result artifacts. The timed full run completed in 53.06 seconds on the local CPU environment. Six unit/integration tests, tracker sync, project validation, and SQLite integrity all passed.

## Decision

Freeze `folds/v1.json` through `folds/v5.json`, the metric definitions, data signature, and control gates. Later experiments may add diagnostics but must not silently alter these files or their semantics.

## Next experiment

Proceed to **E002 — Structural baseline ladder** using these frozen folds and metrics: constant/robust-linear/quadratic/constrained-spline formulations, including both direct TVT and explicitly verified structural transforms before assuming a sign convention.
