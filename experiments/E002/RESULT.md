# E002 Result — Structural Baseline Ladder

Status: **rejected; last-known-TVT retained**  
Run: `R20260718-1916-e002-structural`  
Code SHA: `4c59d85c384a70c5892f76c25fda084d80556360`

## Outcome

E002 verified the structural transform sign but falsified the pre-registered hypothesis that a simple continuation of `U = TVT + Z` from the visible heel would beat direct last-known-TVT continuation.

All controls passed. The failure is therefore a valid negative modeling result rather than an evaluator, leakage, aggregation, or determinism failure. No E002 structural challenger is promoted. The retained baseline remains last-known TVT at **15.9098528707 RMSE**.

## Transform verification

On all 1,307,493 visible-prefix row differences:

| Visible-prefix diagnostic | RMS row change |
|---|---:|
| Correct structural coordinate `U = TVT + Z` | 0.1879834871 |
| Wrong-sign coordinate `TVT - Z` | 1.1897215249 |
| Correct/wrong RMS ratio | 0.1580062923 |

`U = TVT + Z` is therefore substantially smoother and is the correct structural representation for this dataset. Smoothness alone, however, does not make its future level or slope safely extrapolatable.

## Candidate ladder

| Candidate | Pooled RMSE | Change versus last-known | Improved frozen fold cells |
|---|---:|---:|---:|
| Last-known TVT | **15.9098528707** | 0.0000 | retained |
| Robust linear U | 39.6545765336 | -23.7447 | 0/25 |
| Constrained spline U | 69.2416365101 | -53.3318 | 0/25 |
| Constant U | 107.4948242582 | -91.5850 | 0/25 |
| Quadratic U | 115.8430874839 | -99.9332 | 0/25 |

The best challenger, robust-linear U, also failed every performance promotion gate:

- p90 well RMSE: 60.5327 versus 22.9725 baseline.
- Worst-5% SSE share: 42.3854% versus 38.9907% baseline.
- Long-hidden RMSE: 48.5525 versus 16.3900 baseline.
- Pooled improvement: negative 23.7447 RMSE.
- Frozen fold-cell wins: 0 of 25.

## Why the apparently correct structural idea failed

The post-hoc trend-transfer diagnostic uses hidden labels and is explicitly oracle-only. It explains the failure; it is not a legal feature.

- Visible-to-hidden U-slope correlation: 0.928291.
- Visible/hidden U-slope sign agreement: 97.93%.
- Median visible U slope: 0.016472 ft per row.
- Median hidden U slope: 0.024744 ft per row.
- Median hidden Z slope: 0.024022 ft per row.
- Median hidden TVT slope: 0.000136 ft per row.
- Median absolute heel-to-toe U-slope error: 0.008192 ft per row.

The visible U slope usually points in the correct direction, but a small slope error accumulates over hidden suffixes that are commonly about 5,000 rows long. A median error of 0.00819 ft per row implies roughly 41 ft of accumulated displacement over 5,000 rows.

In the hidden region, U generally moves almost with Z, leaving TVT nearly flat. Last-known TVT implicitly encodes that cancellation. Constant U instead forces TVT to follow `-Z`, while heel-linear U applies a slightly incorrect structural slope for thousands of rows. Both create large whole-well shifts.

## Error structure and diversity

Constant U and the low-order continuations often have small residual shape error after oracle removal of their per-well level and trend, but their inferred level/trend is wrong. This confirms that the remaining challenge is signed structural placement, not local smoothness.

The rejected structural paths are decorrelated from last-known TVT:

- Last-known versus constant U residual correlation: 0.0122.
- Last-known versus constrained spline U: 0.0987.
- Last-known versus quadratic U: 0.0227.
- Last-known versus robust-linear U: 0.3052.

Low correlation does not justify blending a candidate that is this weak. These paths may be retained only as candidate evidence or disagreement features in a later controlled model.

## Controls

All eleven controls passed:

- Data signature and 773-well integrity.
- Visible-prefix transform-sign verification.
- Algebraic boundary round trip.
- Direct versus per-well pooled-SSE consistency.
- No-op equality.
- Duplicate constant-U equality and correlation 1.0.
- Wrong-sign transform degradation.
- Synthetic legal-slope positive control.
- Shuffled-evidence degradation.
- Hidden-target leakage sentinel.
- Runtime budget.

## Reproducibility

- Official full run: 241.49 seconds, maximum RSS 35,504 KB.
- Independent full verification: 247.31 seconds, maximum RSS 35,160 KB.
- Twelve result files were byte-identical.
- The compressed full OOF candidate matrix was byte-identical.
- OOF artifact: `artifacts/E002/structural-ladder/oof_predictions.csv.gz`.
- OOF SHA-256: `9052488ff964092e716e1045cc0f9bd69b6af63993984021d5697e68859f58cd`.
- OOF size: 110,694,385 bytes. It is intentionally ignored by Git.

## Decision

Reject unconditional constant, heel-linear, quadratic, and damped-spline continuation of U. Retain `U = TVT + Z` as an interpretable representation, diagnostic coordinate, and candidate-feature space.

The next experiment must separate:

1. predicting the small heel-to-toe U-slope change;
2. predicting hidden TVT drift directly;
3. detecting risk magnitude; and
4. choosing a signed action.

No Kaggle submission is warranted from E002.
