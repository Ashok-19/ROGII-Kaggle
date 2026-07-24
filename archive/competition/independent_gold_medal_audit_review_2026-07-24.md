# Independent gold-medal audit — owner-side review and decision

Reviewed at: 2026-07-24T16:36:08+05:30  
Source archive: `ROGII_GOLD_MEDAL_INDEPENDENT_AUDIT_DELIVERABLES_2026-07-24.zip`  
Archive bytes: 1845934  
Archive SHA-256: `bde68147baec87063796b99dc78446dc0653e1c1a09e578cc713e2398df0290d`

## Completeness and integrity

- ZIP CRC passes and no unsafe extraction paths exist.
- All 48 declared internal SHA-256 entries match exactly.
- All 49 extracted files were read using their native structure: Markdown, DOCX XML, PDF text/rendered pages, every CSV/CSV.GZ row and cell, every JSON object, every reproducibility script, and every PNG figure.
- The 43-page PDF has no blank page; the DOCX/PDF preserve every decision-critical report value and identifier.
- All 52 archive citations resolve; all weighted strategy scores recompute; all 22 closed-mechanism entries and all three preregistrations contain their required fields.
- The audit scripts are not portable without editing because they hard-code the original `/mnt/data` paths. The owner repository contains the missing `src/rogii_validation` package and source artifacts, so this is an audit handoff limitation rather than a scientific-score defect.

## Evidence accepted

1. E011 remains the strongest independently reproduced legal deployable model at 12.550756295690 RMSE.
2. T030 completes at 12.368913795098 but has no substantive passer and remains rejected.
3. Sub-5 E010 and coefficient results remain hidden-label oracles, not legal models.
4. The E011-to-5 gap requires removing 501,459,836.18 SSE, 84.1291% of current SSE. Error is broad: idealized repair must affect at least 608 wells at RMSE 5 or 352 perfect wells.
5. Perfect datum correction alone leaves 8.1883 RMSE; perfect datum plus trend leaves 6.5102. Nonlinear shape recovery is mandatory.
6. Existing 142 legal summaries cannot reliably identify risk, action, or oracle winner. Routing AUCs remain near 0.5.
7. The six E003 hidden formation displacement components form one target: pairwise correlation is 0.98385–1.0 and PC1 explains 99.5626% of variance.
8. Legal smooth coordinate reconstruction cannot predict that hidden displacement (mean R² below zero for the hidden-offset category), so a coordinate-only surface model is closed.
9. Full raw inference channels needed by the state test exist locally across all 773 wells and all public test wells: contiguous TVT_input boundary, complete suffix MD/X/Y/Z, partial GR, and typewell TVT/GR. Test typewell Geology is absent and prohibited.
10. Current spatial/typewell evaluator groups are one-dimensional quantile proxies, not geological domains. Stronger target-free 2D spatial and legal-covariate cluster holdouts must be frozen before scoring a new mechanism.

## Crucial decision

**Close T030. Select H023/T031 as the immediate task: a bounded auxiliary-state identifiability gate.** Do not start a joint TVT sequence model until raw legal sequences predict the common hidden formation displacement at the frozen correlation/MAE and transfer thresholds.

Maintain H024/T032 as a genuinely independent second path, but run only the registered analog-tail oracle-coverage gate before any legal retrieval selector.

Defer P3 switching state-space, GP, learning-to-rank, FPCA, test-time adaptation, foundation models, GroupDRO, and additional blends unless T031 or T032 supplies new causal state evidence.

## Why this ordering

- T031 asks the highest-value unresolved question: whether the train-only formation information responsible for multi-RMSE E003 headroom is identifiable from raw legal sequence evidence.
- The gate is cheap relative to a complete system and has explicit stop rules. A failure materially lowers the probability of sub-5 and prevents architecture search.
- T032 independently asks whether registered analog tails have enough oracle coverage before selector work. It does not share T031's neural/state assumptions.
- T030 and all nearby summary/routing/objective/view/blend families are already closed.
