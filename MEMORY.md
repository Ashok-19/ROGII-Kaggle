# ROGII Project Memory

Last updated: 2026-07-18

## Mission state

- Final deadline: 2026-08-05 23:59 UTC / 2026-08-06 05:29 Asia/Kolkata.
- Current leader snapshot: 4.859.
- Best Kaggle-MCP-verified submission for this account/team history: 7.119 on 2026-07-16, ref 54754431.
- User-reported best: 6.888; submission reference not yet verified.
- Current public top-100 cutoff in the archived snapshot: 6.799.
- E001 validation infrastructure is complete. E002 is complete and rejected; no competition model training has started.

## Foundation completed

- All 132 competition discussion topics and 981 returned messages archived with zero crawl failures.
- Both Working Note Award winners archived and synthesized.
- Official rules, timeline, evaluation, leaderboard, submission history, and data summary archived.
- Copied notebook audited: seven named external datasets plus one opaque mount; three named datasets have unknown licenses.
- SQLite experiment ledger, auto-sync CLI, dashboard, manifests, validation protocol, and roadmap are installed.
- E001 is promoted: five deterministic whole-well fold maps, one shared evaluator, controls, reports, and hashes are frozen.
- E002 is rejected: the transform sign is verified, but every naive low-order structural continuation lost to last-known TVT on all 25 frozen fold cells.
- The dashboard Learning Lab provides a beginner-first visual guide, real-well interactive playground, feature glossary, error demonstrations, idea prompts, and a breakthrough timeline.

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
- E002 verified that visible `U = TVT + Z` row changes are much smoother than `TVT - Z` (RMS ratio 0.158), but smoothness does not make heel trend safely extrapolatable.
- Robust-linear U was the best structural challenger at 39.6546 RMSE versus 15.9099 for last-known TVT; it won 0/25 frozen fold cells.
- Oracle-only diagnostics show visible and hidden U slopes correlate 0.928 with 97.93% sign agreement, yet the median absolute slope error is 0.00819 ft/row and compounds across long suffixes.
- Median hidden U slope nearly equals hidden Z slope, leaving median hidden TVT slope near zero; last-known TVT captures this cancellation better than naive U continuation.

## Decisions

- The copied notebook is an idea catalogue, not an approved baseline.
- No private/opaque or unknown-license artifact enters the final solution.
- No model is promoted from a single aggregate CV number.
- One final slot should represent a public-proven family; the other should be a decorrelated, control-validated private-expectation family.
- `folds/v1.json` through `folds/v5.json`, data signature `6ebe65b4...fe77`, and E001 metric/control semantics are immutable.
- Heavy training, large OOF generation, and accelerator workflows should use Kaggle MCP notebook sessions after committing exact code/configuration.
- Material breakthroughs and newly understood failure modes must also be added to the Learning Lab in evidence-labeled visual form.
- Naive constant/heel-linear/quadratic/spline U continuation is rejected. U remains useful only as a representation, diagnostic, and evidence space until signed future-trend change is legally predictable.

## Exact next action

Run E003: quantify datum and trend oracle headroom, then test whether legal visible-prefix, trajectory, GR-availability, surface, and whole-well features predict heel-to-toe U-slope delta, hidden TVT drift, risk magnitude, and signed action separately.

## Open risks

- Public leaderboard is a small/noisy ranking sample and may reward the wrong family.
- Hidden test has about 200 wells and may differ strongly from local fold composition.
- Datum correction sign may be weakly identifiable from legal inputs.
- Runtime and artifact packaging must remain under the 9-hour offline notebook limit.
- A Kaggle bearer credential appeared in a local application log during discovery and should be rotated; it is not stored in this repository.

## Memory update rule

Keep this file concise. Add only verified state, durable decisions, blockers, and the next executable action. Detailed experiments belong in manifests and the database.
