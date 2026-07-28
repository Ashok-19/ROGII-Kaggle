# E013 result — clean-room sequential likelihood-PF

Decision: **REJECT_E013**

## Frozen primary result

- E011 RMSE: `12.550756295690`
- Fixed 75% PF / 25% E011 blend RMSE: `10.105331947939`
- Gain versus E011: `2.445424347750`
- Map wins: `5/5`
- Cell wins: `25/25`
- Every legacy spatial/typewell group improves: **yes**
- p90: `14.690626996116` versus `17.831023926499`
- Worst-5% SSE share: `0.463097447509` versus `0.315321724264`
- Minimum destructive-control advantage: `6.458743039828`
- Accumulated compute plus finalization runtime: `1.419` hours
- Maximum RSS: `0.642` GB

The primary fails the frozen GO contract because RMSE is above `8.0` and worst-5% SSE concentration increases by `0.147776`, far above the allowed `0.005`.

## Registered diagnostics

- Sequential likelihood-PF standalone: `11.201088179170` RMSE
- Fixed 50% PF / 50% E011 diagnostic: `9.983905587039` RMSE

The 50% diagnostic is the lowest all-well RMSE observed in E013, but it was not the preregistered primary and still increases worst-5% SSE share to `0.393573974402`. It is retained as research evidence only and is not promoted or packaged.

## Controls and reproduction

All PF evidence-destruction controls, determinism, reversed-seed-order, missing-data, exact-fallback, row identity, finite-output, input-immutability, runtime, memory, and source-parity controls pass. Independent recomputation reproduces all four candidate RMSEs, map/cell/group/control decisions, and every pre-serialization-fix scientific artifact hash exactly.

No Kaggle notebook was run and no submission was made.
