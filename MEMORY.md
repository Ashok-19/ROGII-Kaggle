# ROGII Project Memory

Last updated: 2026-07-18

## Mission state

- Final deadline: 2026-08-05 23:59 UTC / 2026-08-06 05:29 Asia/Kolkata.
- Current leader snapshot: 4.859.
- Best Kaggle-MCP-verified submission for this account/team history: 7.119 on 2026-07-16, ref 54754431.
- User-reported best: 6.888; submission reference not yet verified.
- Current public top-100 cutoff in the archived snapshot: 6.799.
- E001 validation infrastructure is complete; no competition model training has started.

## Foundation completed

- All 132 competition discussion topics and 981 returned messages archived with zero crawl failures.
- Both Working Note Award winners archived and synthesized.
- Official rules, timeline, evaluation, leaderboard, submission history, and data summary archived.
- Copied notebook audited: seven named external datasets plus one opaque mount; three named datasets have unknown licenses.
- SQLite experiment ledger, auto-sync CLI, dashboard, manifests, validation protocol, and roadmap are installed.
- E001 is promoted: five deterministic whole-well fold maps, one shared evaluator, controls, reports, and hashes are frozen.

## Durable understanding

- Score is pooled row-level RMSE on hidden suffix rows.
- Use whole-well suffix CV; random rows are invalid.
- The visible three test wells are authoring examples derived from training data.
- `U = TVT + Z` separates known trajectory wiggle from the difficult smooth structural level/trend.
- Per-well datum/mean error and a small number of expensive wells dominate SSE.
- Aggregate CV may invert against the public leaderboard.
- Risk detection does not prove signed correction direction.
- Weak but decorrelated models may improve an ensemble; stacking individually good corrections may still fail.
- E001 reproduced last-known-TVT RMSE 15.9098528707 on 3,783,989 hidden rows.
- Baseline SSE decomposes into 67.75% per-well mean/datum, 14.53% linear trend, and 17.72% remaining shape.
- Worst 5% and 10% of wells contribute 38.99% and 52.48% of baseline SSE.

## Decisions

- The copied notebook is an idea catalogue, not an approved baseline.
- No private/opaque or unknown-license artifact enters the final solution.
- No model is promoted from a single aggregate CV number.
- One final slot should represent a public-proven family; the other should be a decorrelated, control-validated private-expectation family.
- `folds/v1.json` through `folds/v5.json`, data signature `6ebe65b4...fe77`, and E001 metric/control semantics are immutable.
- Heavy training, large OOF generation, and accelerator workflows should use Kaggle MCP notebook sessions after committing exact code/configuration.

## Exact next action

Run E002: reproduce a structural baseline ladder on frozen folds, verifying transform sign conventions before constant, robust-linear, quadratic, and constrained-spline candidates are compared.

## Open risks

- Public leaderboard is a small/noisy ranking sample and may reward the wrong family.
- Hidden test has about 200 wells and may differ strongly from local fold composition.
- Datum correction sign may be weakly identifiable from legal inputs.
- Runtime and artifact packaging must remain under the 9-hour offline notebook limit.
- A Kaggle bearer credential appeared in a local application log during discovery and should be rotated; it is not stored in this repository.

## Memory update rule

Keep this file concise. Add only verified state, durable decisions, blockers, and the next executable action. Detailed experiments belong in manifests and the database.
