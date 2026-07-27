# T036 Result — Candidate-path-conditioned GR evidence ranking

Decision: **STOP_CLOSE_PATH_CONDITIONED_RANKING**

E011 baseline RMSE: 12.550756295690
Candidate oracle RMSE: 4.628457367010
Nested legal ranker RMSE: 12.472864734861
Gain versus E011: 0.077891560829
Oracle-gain retention: 0.983194%
Path-conditioned advantage versus action-prior control: 0.092333091879 RMSE
Map/cell wins: 0/5 and 7/25

## Breakthrough gates
- FAIL — maximum_legal_rmse
- FAIL — minimum_map_wins
- FAIL — positive_every_legacy_group
- FAIL — tail_gates
- PASS — destructive_controls_fail

## GO gates
- FAIL — maximum_legal_rmse
- FAIL — minimum_oracle_gain_retention
- FAIL — minimum_map_wins
- FAIL — minimum_outer_cell_wins
- FAIL — positive_every_legacy_group
- FAIL — path_conditioned_advantage
- FAIL — tail_gates
- PASS — destructive_controls_fail

This is a legal diagnostic only. It does not authorize a Kaggle run or submission.
