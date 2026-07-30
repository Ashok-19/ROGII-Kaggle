# T041 result — public-frontier visible-prefix arbitration worth audit

Decision: **REJECT_H033_EXACT_ARBITRATION**

- E011 RMSE: `12.526884618114`
- Fixed 50/50 E011–PF RMSE: `9.276141513244`
- Primary arbitration RMSE: `11.154147760459`
- Gain versus E011: `1.372736857655`
- Gain versus fixed PF blend: `-1.878006247215`
- Fold wins: `0/5`; cell wins: `6/25`
- Mean rank Spearman: `0.160000`; tracker hit rate: `0.430000`
- Component oracle RMSE: `5.883054150583`
- Minimum control loss: `-0.428243477968`
- Projected 773-well runtime: `10.927` hours
- Observed max RSS: `0.802` GB

Scientific gates:
- gain_vs_e011: FAIL
- gain_vs_fixed_pf_blend: FAIL
- fold_wins: FAIL
- cell_wins: FAIL
- p90: FAIL
- worst20_share: FAIL
- rank_spearman: FAIL
- tracker_hit_rate: PASS
- component_oracle: PASS
- controls: FAIL
- runtime: FAIL
- memory: PASS

Structural gates:
- all_caches_complete: PASS
- source_identity: PASS
- input_immutability: PASS
- finite_predictions: PASS
- complete_rows: PASS
- rmse_sse_identity: PASS
- valid_pseudo_profiles: PASS

This research source audit cannot promote public code or predictions. No Kaggle run or submission was made.
