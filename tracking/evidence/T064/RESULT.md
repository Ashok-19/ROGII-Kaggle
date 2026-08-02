# T064 result — exact T055 plus tail-safe T063 full-pool notebook

Status: **PRIVATE VALIDATION PASS; NOTEBOOK-LINKED SUBMISSION 55191306 SCORING PENDING**

## Frozen composition

`verified 6.494 parent stack -> exact T055 all-reference spatial correction -> exact T063 25% U=TVT+Z projection -> final audit -> final summary -> hidden-test contract`

No additional scoring mechanism was added.

## Source and notebook identity

- T055 parent notebook SHA-256: `c8d6907cc27f91e8fb1820d3abb30dcb5280dc69a1483e5521d4799b71047ee0`
- Exact T063 scoring-cell SHA-256: `ba0f3a3fe94563d540183f28b150b50558454cd69bfb29438f805a3f6c115528`
- T064 notebook SHA-256: `3337a049e1e44e2e893eb9fabe2cfc2190f8783dab15df142e8573f2593d49bc`
- Kaggle notebook: `ashok205/rogii-t055-t063-best-local-fullpool`
- Notebook version: `1`
- scriptVersionId: `339740133`
- Private execution status: `COMPLETE`

## Private validation

- Required outputs: all present.
- Pre-T063 IDs and values versus prior T055 private output: exact.
- Prior T055 private output SHA-256: `3004a8a906e671ee9c3ffe47ce34ef7e2ee82dc05656e2a32b9ee8501d5b2d59`
- T055 applied/fallback wells: `3 / 0`
- T063 applied/fallback wells: `3 / 0`
- T063 moved rows: `14,151`
- T063 mean absolute action: `0.4503434245094659 ft`
- T063 maximum absolute action: `2.740377222570867 ft`
- Frozen cap: `25 ft`
- Final Kaggle CSV SHA-256: `6b49a3becac34406cb8d1989c561dc08995c1059ebf734d0a823cdea4e45f3dc`
- Final Kaggle parsed prediction SHA-256: `4ba4b59ed512648e2626c48f938c88da72e8fb415c95ec786cde787efbfd2bc4`
- Local/Kaggle maximum absolute difference: `5.4569682106375694e-12 ft` against frozen tolerance `1e-9 ft`
- Exact sample ID order: pass.
- Finite predictions: pass.
- Hidden-test contract: pass.
- Public-specific IDs, row count or output hashes used by the T063 inference layer: none.
- True `Traceback` or standalone `ERROR` log markers: none.

Two validation-only corrections were recorded before submission: the pandas-written pre-T063 audit copy is checked by exact IDs and values rather than byte identity, and in-memory versus parsed-CSV prediction hashes are recorded separately. Neither correction changed notebook source or predictions.

## Competition submission

- Submission ref: `55191306`
- URL contract: `/code/ashok205/rogii-t055-t063-best-local-fullpool?scriptVersionId=339740133`
- Status at recording: queued/unscored.
- Public RMSE: unknown.

The account best remains submission `55140317` at `6.494`. T064 is not a verified sub-6 or sub-5 result unless Kaggle posts such a score.
