# T030 Result — Outer-isolated complementary coefficient–datum ensemble

## Decision

**Reject H022 and close T030 without promotion.**

The completed corrected run reports `fixed_pair_box_046_090` at **12.368913795098 RMSE**, a gain of **0.181842500592** versus E011 and **0.056013813397** versus T025. It wins **5/5 maps** and **19/25 repeated cells**.

The branch is not promotion-eligible. It fails the frozen p90, spatial-transfer, and typewell-transfer gates. No substantive candidate passes. Independent reproduction was required only for a substantive promotion passer; therefore a second full run cannot change the rejection decision and is not authorized.

## Exact failed gates for the reported branch

- p90 gate: **False**; candidate p90 17.895692435571 versus E011 17.831023926499.
- every-spatial-group gate: **False**.
- every-typewell-group gate: **False**.
- independent-reproduction gate: **False**, inapplicable after the substantive gates fail.

The gain, map, cell, horizon, special-slice, worst-5%, parent-reproduction, nondegeneracy, control, and correction-bound gates pass.

## Controls

- 773 wells and 3,783,989 unique hidden rows covered.
- 275 repeated-context rows and 110 stress-context rows completed.
- 555 isolation audits pass.
- T016/T025/T027 parent actions reproduce to numerical tolerance.
- all 23 synthetic edge groups pass.
- maximum corrupted-control gain is -0.401970048973; all controls lose.
- actual reconstructed row corrections remain within the frozen 160 ft audit bound.
- exact E011 fallback remains 12.550756295690 RMSE.

## Interpretation

T029's covariance finding was genuine: the coefficient and datum actions combine additively. The failure is not parent identity, implementation, or negative-control contamination. The failure is that the same fixed combination does not transfer uniformly across the registered spatial/typewell domains and slightly worsens p90. This closes final-placement blending of these saved actions and does not authorize routing, domain-specific weights, or further scale/cap searches.

No package, Kaggle execution, submission file, or competition submission was produced.
