# ROGII Project Memory

Last updated: 2026-07-19

## Mission state

- Final deadline: 2026-08-05 23:59 UTC / 2026-08-06 05:29 Asia/Kolkata.
- Current leader snapshot: 4.859.
- Best Kaggle-MCP-verified submission for this account/team history: 7.119 on 2026-07-16, ref 54754431.
- User-reported best: 6.888; submission reference not yet verified.
- Current public top-100 cutoff in the archived snapshot: 6.799.
- E001 is frozen, E002 is rejected, E003 is promoted as surface-assisted OOF understanding, E004 is the exact deployment fallback, E005 and E007 are rejected candidate families, and E006 remains the promoted Kaggle-verified primary deployment-ready surface-free fusion.

## Foundation completed

- All 132 competition discussion topics and 981 returned messages archived with zero crawl failures.
- Both Working Note Award winners archived and synthesized.
- Official rules, timeline, evaluation, leaderboard, submission history, and data summary archived.
- Copied notebook audited: seven named external datasets plus one opaque mount; three named datasets have unknown licenses.
- SQLite experiment ledger, auto-sync CLI, dashboard, manifests, validation protocol, and roadmap are installed.
- E001 is promoted: five deterministic whole-well fold maps, one shared evaluator, controls, reports, and hashes are frozen.
- E002 is rejected: the transform sign is verified, but every naive low-order structural continuation lost to last-known TVT.
- E003 is promoted as analysis: surface-assisted cross-fitted ridge datum-plus-trend scores 10.9279740918 RMSE, and a separate forest risk probe reaches 0.5415 Spearman / 0.8125 worst-20% AUC.
- E004 is completed: the actual test schema was audited, eight surface-free ablations were evaluated, and the deterministic offline notebook produced a byte-identical submission locally and in a private internet-disabled Kaggle Python 3.12 run.
- E005 is completed: frozen affine, visible-calibrated, particle-filter, and trellis paths were evaluated from test-available inputs only; all controls passed, but the best standalone PF candidate failed the registered stability gates and was rejected without Kaggle packaging or submission.
- E006 is completed and deployment-ready: strict nested PF-E004 fusion reaches 14.9331407872 RMSE, wins all 5 maps and 25 outer cells, improves every spatial/typewell group, passes final-code reproduction, and produces raw-byte-identical local and private Kaggle output.
- E007 is completed and rejected: same-well horizontal GR self-correlation is independently useful but not robust enough. The fixed 0.10 placement reaches 14.8304784179 RMSE, wins 5/5 maps and 20/25 repeated cells, but fails the frozen p90, spatial, and typewell gates; no package or submission is authorized.
- The dashboard Learning Lab provides a beginner-first visual guide, real-well playground, feature glossary, error demonstrations, idea prompts, and a breakthrough timeline.

## Durable understanding

- Score is pooled row-level RMSE on hidden suffix rows; use whole-well suffix CV.
- The visible three test wells are train-derived authoring examples, not hidden-test evidence.
- `U = TVT + Z` separates known trajectory wiggle from difficult structural level/trend, but naive U continuation is unsafe.
- Baseline last-known-TVT RMSE is 15.9098528707 over 3,783,989 hidden rows.
- Baseline SSE is 67.75% datum, 14.53% linear trend, and 17.72% remaining shape; the worst 5% and 10% of wells contribute 38.99% and 52.48% of SSE.
- E002 shows a median U-slope error of only 0.00819 ft/row can compound to roughly 41 ft over 5,000 rows.
- E003 proves strong signed action is learnable when supplied formation surfaces are included: datum Pearson 0.7909, trend Pearson 0.8668, and 10.9280 RMSE.
- The six formation-surface columns `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, and `BUDA` exist in train but are absent from test horizontal files. Test typewells also omit `Geology`.
- Therefore E003's 10.9280 result is valid OOF understanding but is not deployable as implemented.
- E004's best test-deployable ridge uses only geometry plus visible-prefix evidence and scores 15.4913063983 RMSE, a 0.4185464725 gain over last-known TVT.
- E004 improves all five maps and retains 15.5767054107 RMSE under contiguous spatial blocks, but trend Pearson falls to 0.0580 and datum Pearson to 0.2692 without surfaces.
- Prefix evidence is essential: removing it leaves only 0.0883 gain and 0/5 registered map wins.
- GR, typewell summaries, and absolute spatial context do not improve the frozen geometry-prefix ridge. This does not test explicit GR alignment, PF, or trellis paths.
- E004 local direct, local notebook, and private Kaggle submissions are byte-identical over 14,151 authoring-example rows at SHA-256 `62ae0657...5279`; deployment-ready is true.
- E005 particle filtering reaches 15.3502040715 RMSE, improving E004 by 0.1411023267 with residual correlation 0.897248, but p90 rises to 23.4851836927, only 3/5 maps pass, and the worst spatial/typewell-cluster gains are -0.754215/-0.185681; standalone promotion is rejected.
- The pre-registered E005 diagnostic 50/50 blend was not promotion evidence. E006 independently validates the family through strict nesting: selected RMSE 14.9331407872, p90 21.7855500884, worst-5% share 0.3685061112, and positive gain in every registered spatial/typewell group.
- E006 deployment weight independently refits to 0.5; direct, local-notebook, package-rebuild, and private Kaggle outputs are byte-identical at SHA-256 `e412864a...d81008`.
- E007's raw bounded correction has residual correlation 0.4675700642 to E006, and deterministic template shuffle removes all gain. Fixed 0.10 placement improves E006 by 0.1026623694 RMSE and lowers worst-5% SSE share to 0.3567221937, but p90 rises by 0.2648456580, spatial groups 1/3 regress, typewell groups 2/4 regress, and long/high-missing/pseudo-poor slices are unsafe.

## Decisions

- The copied notebook is an idea catalogue, not an approved baseline; no private/opaque or unknown-license artifact enters the final solution.
- No model is promoted from a single aggregate CV number.
- `folds/v1.json` through `folds/v5.json`, data signature `6ebe65b4...fe77`, and E001 metric/control semantics are immutable.
- Heavy training, large OOF generation, and accelerator workflows should use Kaggle MCP notebook sessions after committing exact code/configuration.
- Material breakthroughs and failure modes must be added to the Learning Lab in evidence-labeled visual form.
- E003's surface-assisted coefficient action remains a diagnostic/candidate-learning result, not a competition inference candidate.
- Retain deployment-ready E004 geometry-prefix only as a weak packaged fallback and possible ensemble leg; do not submit it solely from 15.49 local CV.
- Reject raw E005 affine, PF, and trellis paths as standalone finalists. Preserve E004 as exact no-GR fallback, and promote only the strictly nested E006 conservative fusion for surface-free deployment.
- Reject every fixed or cross-fitted E007 self-correlation placement as a finalist. Preserve its horizontal-only fingerprints, visible pseudo-holdout diagnostics, and bounded raw correction only as evidence for a fresh cross-fitted residual model.

## Exact next action

Pre-register E008/H010 before any new scoring: build a compact cross-fitted legal residual-action model over test-available geometry, visible-prefix evidence, E006 predictions, and E007 diagnostics. Fit every signed correction strictly inside training folds, preserve E006 as the zero-action fallback, and require repeated-map, tail, spatial/typewell, shuffle, runtime, and final-placement gates against E006.

## Open risks

- Hidden test has about 200 wells and may differ strongly from local fold composition.
- Public/local ordering can invert; E004's 0.42 gain is too small to justify an isolated submission.
- GR has repeated motifs and substantial missingness; E005 confirms that aggregate PF gain can coexist with worse p90 and subgroup regressions.
- E007 confirms that even a low-correlation horizontal-only signal with 5/5 aggregate map wins can fail long-horizon, high-missingness, spatial, and typewell transfer; a visible pseudo-gain score is not sufficient routing evidence.
- E004 parity is closed, but every future finalist notebook still requires its own clean Kaggle runtime, exact-ID, and output-parity verification.
- Runtime and artifact packaging must remain under the 9-hour offline notebook limit.

## Memory update rule

Keep this file concise. Add only verified state, durable decisions, blockers, and the next executable action. Detailed experiments belong in manifests and the database.
