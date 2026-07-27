# T037 Preregistration — Full-emission structured path decoding

Status: **preregistered; no implementation, scoring, Kaggle execution, packaging, or submission authorized**
Hypothesis: `H029`
Task: `T037`
Decision: `D041`

## Scientific question

Can the complete row/node hidden-horizontal-GR and typewell-GR emission field identify a high-capacity TVT path when decoded jointly with continuity, even though T036 candidate-level summary ranking retained only 0.983% of oracle gain?

## Why this is materially different

T036 reduced every complete candidate path to 39 aggregate features and then used pointwise or pairwise candidate rankers followed mostly by soft top-k averaging. T037 never predicts a global PCA coordinate and never ranks complete candidates from aggregate loss summaries. It constructs a node-by-state emission volume and performs one hard structured decode over the union graph of legal action paths. Exact duplicate profiles are removed by action-profile hash **before** graph construction, score normalization, or decoding.

## Frozen capacity identity

- Start from the exact split-local T033 32-center pair-mixture bank used by T036.
- Preserve all 1,520 unique nonzero 128-node actions plus one exact zero E011 fallback.
- Build a directed union graph from the 1,521 complete profiles; every original complete path remains admissible.
- The hidden-label complete-path oracle must reproduce `4.628457367010144` RMSE before any legal model is scored.
- A graph oracle may improve on that value by splicing admissible edges, but promotion is judged against the frozen complete-path oracle denominator to avoid moving the gate.
- Action cap remains ±160 ft.

## Legal emission inputs

At each hidden support node and proposed TVT state, use only test-available quantities:

- hidden horizontal-well GR and finite/missing support flags;
- typewell GR interpolated at the proposed TVT state;
- visible-prefix-only robust GR calibration;
- local robust mismatch, signed mismatch, derivatives, and short-window support;
- horizontal MD spacing and trajectory derivatives available at inference;
- path-state support and boundary indicators.

Forbidden: hidden TVT, hidden validation residuals, oracle path identity, oracle coefficients, formation surfaces, `Geology`, absolute X/Y, evaluator group IDs, public scores, score-derived constants, same-well target lookup, or opaque/unknown-license artifacts.

## Frozen mechanism

1. Fit the exact candidate bank inside each outer-training partition.
2. Deduplicate exact action profiles before graph construction.
3. Build the node/state union graph while proving that all original complete paths remain present.
4. Fit every calibration and emission model inside the outer-training partition only.
5. Build row/node emission potentials for legal states.
6. Decode one hard path with Viterbi dynamic programming on the frozen union graph.
7. Select emission family and transition strength only by actual-row pooled inner TVT RMSE.
8. Refit the selected decoder on all outer-training wells and score untouched outer wells.
9. Average the five repeated-map decoded profiles before final per-well scoring, matching prior evaluation semantics.

Frozen legal branches:

- fixed robust GR emission;
- split-local Ridge-calibrated emission;
- split-local HistGradientBoosting-calibrated emission;
- transition strength in `{0.0, 0.25, 1.0, 4.0}`;
- exact E011 zero fallback;
- hard Viterbi output only; no soft top-k averaging.

## Validation

- 25 outer contexts from the five registered whole-well maps;
- inner fold `(outer_fold + 1) % 5`;
- every transform, action bank, graph, target, emission model, and transition choice split-local;
- same-well exclusion mandatory;
- exactly 773 wells and 3,783,989 hidden rows;
- report pooled row RMSE, map/cell wins, all legacy and broader transfer groups, special slices, p90, worst-5% SSE share, datum/trend/shape SSE, complete-path oracle coverage, graph-oracle coverage, decoded path identities, and transition usage.

## Frozen destructive controls

Each control repeats full inner emission-family and transition selection:

- reversed typewell GR;
- circularly shifted hidden horizontal GR;
- within-well emission-row permutation;
- transition-only decoder with all GR emissions removed;
- shuffled within-well training state losses;
- randomized graph-edge identity preserving degree counts;
- sign-flipped selected action.

A destructive control fails only when its gain versus E011 is at most `0.05` RMSE. The legal decoder must beat transition-only by at least `0.50` RMSE.

## Frozen structural controls

- genuine state/path-order permutation invariance;
- genuine duplicate-path invariance after exact pre-graph deduplication;
- exact zero fallback;
- same-well exclusion;
- complete-path capacity identity;
- graph contains every original complete path;
- pooled RMSE/SSE identity;
- finite and bounded paths within ±160 ft;
- all-missing and short-support behavior;
- deterministic repeated decode;
- exact legacy and broader group membership identities.

## Frozen decisions

BREAKTHROUGH requires:

- legal RMSE `<= 5.0`;
- 5/5 map wins;
- every legacy group improves;
- no p90 deterioration;
- worst-5% SSE-share increase `<= 0.005`;
- every destructive control fails;
- every structural control passes.

GO requires:

- legal RMSE `<= 8.0`;
- at least 55% oracle-gain retention;
- 5/5 map wins;
- at least 20/25 outer-cell wins;
- every legacy group improves;
- at least `0.50` RMSE advantage over transition-only;
- tail gates pass;
- every destructive control fails;
- every structural control passes.

RESEARCH ONLY requires legal RMSE above 8.0 and at most 10.0.

STOP triggers if best legal RMSE is above 10.0, oracle-gain retention is below 25%, or the full-emission decoder has no positive advantage over transition-only.

No gate, branch, transition grid, control, input, output rule, graph rule, or denominator may change after metric observation.

## Resource contract

- maximum wall runtime: 4 hours;
- maximum observed RSS: 4 GB;
- numerical worker cap: 2;
- scientific artifacts written atomically at completion.

A STOP closes full-emission union-graph decoding and does not authorize retuning T036 summaries, T023 residual profiles, T032 analog identity, global coordinates, public `gs`, fixed shifts, overlap logic, or opaque public stacks.
