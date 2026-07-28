# T038 result — public HMM/PF mechanism worth audit

Decision: **PASS_AUTHORIZE_CLEAN_ROOM_E013**

- Selected fixed candidate: `blend_e011_likelihood_pf_384x16_w0.75`
- Primary E011 RMSE: `12.912615586736`
- Selected primary RMSE: `6.114432013120`
- Gain versus E011: `6.798183573616`
- Fold-stratum wins: `5/5`
- Edge gain versus E011: `1.153256657258`
- Primary p90: `9.708012335955` versus `18.419162812784`
- Worst-20% SSE share: `0.478003399588` versus `0.548156763792`
- Minimum advantage over destructive controls: `5.849808909804`
- Projected 773-well runtime: `0.596` hours
- Observed max RSS: `0.855` GB

Scientific gates:
- minimum_primary_gain: PASS
- fold_stratum_wins: PASS
- p90_not_worse: PASS
- worst20_share: PASS
- edge_panel: PASS
- destructive_controls: PASS
- runtime: PASS
- memory: PASS

Structural gates:
- deterministic_rerun: PASS
- missing_data_controls: PASS
- input_immutability: PASS
- exact_fallback: PASS
- rmse_sse_identity: PASS
- finite_complete_outputs: PASS
- all_registered_branches_completed: PASS

This source-audit task cannot promote public code or predictions. A pass only authorizes a separate clean-room E013 implementation and full outer-isolated validation.
