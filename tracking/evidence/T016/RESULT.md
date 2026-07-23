# T016 Result — E011-relative stable wide residual datum worth screen

## Decision

**Reject H012 and close T016 without a feature/model refit.**

Existing cross-fitted E009 datum actions retain a small real signal on top of E011, but the nested placed system fails the frozen every-spatial and every-typewell transfer requirements. No scale/cap retuning, new wide-ridge fit, feature-count search, or subgroup routing is authorized.

## Frozen boundary

- Scientific implementation commit: `8627eb7ce119dc66109b6908e8a2ff5d1e45f030`
- Serialization-only amendment: `9e0b2128b85b3e14ce6e31ba262dcb8b30f68a73`
- Baseline: exact E011 at 12.5507562957 RMSE
- Wells: 773
- Hidden rows: 3,783,989
- Registered action families: 12
- Registered scale/cap placements: 480
- Repeated cells: 25
- Spatial/typewell stress contexts: 10

The first sealed invocation completed calculations but emitted no files because a NumPy boolean could not be serialized. The only amendment converted NumPy scalar values to Python scalars during JSON writing; scoring, selection, branch, and gate logic were unchanged.

## Result

| Candidate | RMSE | Gain versus E011 | p90 | Worst-5% SSE share |
|---|---:|---:|---:|---:|
| E011 | 12.5507562957 | — | 17.8310239265 | 0.3153217243 |
| Nested placed E009 actions | 12.4929660562 | 0.0577902395 | 17.8620634070 | 0.3138931388 |

The system wins all 5 maps and 21/25 outer cells. The selected placement is overwhelmingly a negative scale on `ridge_visible_geometry_a25_f32`, showing that the old E009 action direction reverses when placed on the materially stronger E011 baseline.

## Transfer failures

Spatial gains versus E011:

```json
{
  "0": 0.12424887741429202,
  "1": 0.10987246910503856,
  "2": 0.08678478853958893,
  "3": -0.09101634726864383,
  "4": 0.05933080451189987
}
```

Typewell gains versus E011:

```json
{
  "0": -0.0972045259714136,
  "1": 0.08680019124571103,
  "2": 0.1277441017668881,
  "3": 0.04534177930240091,
  "4": 0.13391096903546362
}
```

Spatial group 3 regresses by 0.0910163473 RMSE and typewell group 0 regresses by 0.0972045260. These are frozen hard failures.

All registered special slices improve:

```json
{
  "e011_catastrophe": 0.17980247084086187,
  "high_gr_missingness": 0.05201861063916802,
  "long_suffix": 0.08835221846844377
}
```

## Controls and edge cases

All three negative controls lose:

```json
{
  "permuted_families": -0.007399089656205504,
  "shuffled_wells": -0.01780202891680105,
  "sign_flipped_selected": -0.19263533345835882
}
```

All 12 edge groups pass, including malformed action/OOF files, missing and duplicate wells/IDs, non-finite inputs, constant/zero actions, empty partitions, negative and zero scales, clipping boundaries, deterministic ties, fold/stress membership, five-map averaging, and exact E011 fallback.

The independent run reproduces every substantive table byte-for-byte and the normalized summary exactly. Runtime is 14.770 seconds for the main run and 14.850 seconds for reproduction.

## Interpretation

The old E009 wide-action family is not obsolete in the sense of containing zero signal: a small decorrelated datum correction remains. However, its safe direction is not stable under spatial/typewell shift, and the strongest selected action is the inverse of the original visible-geometry correction. This does not justify a new stability-selection experiment over the same feature space.

E011 remains unchanged as deployment primary. No Kaggle notebook was executed and no competition submission was made.
