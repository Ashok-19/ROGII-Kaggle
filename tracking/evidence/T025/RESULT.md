# T025 Result — Multi-cut spline coefficient dynamics worth screen

Status: **completed; rejected; independently reproduced; no formal experiment; no Kaggle execution; no submission**

Implementation commit: `281f4eb4da15928a45b74ff6c6a688fa3861d9e4`

## Decision

Reject H018 as tested. Do not open a formal experiment by retuning the four cut fractions, ridge alphas, placement weights, tree settings, or KNN neighborhood used here. The mechanism contains real signal, but it fails the frozen complete-system authorization contract. Exact E011 remains the deployment primary.

## Verified nonlinear capacity

The last-known plus four-control hidden-label spline oracle reaches **4.1137431295 RMSE** over all 773 wells and 3,783,989 hidden rows. This independently confirms that the selected target representation has sub-5 capacity.

## Complete branch and placement result

T025 completed all 13 registered branches and all four fixed placements, for 52 legal candidates across 25 repeated outer cells. It also completed 72 pseudo-only negative-control placements, all qualifying spatial/typewell holdouts, three special slices, and 16 edge groups.

The strongest aggregate candidate is `ridge_pseudo_e011_a1__w0.50`:

- RMSE: **12.4249276085**
- gain versus E011: **0.1258286872**
- map wins: **5/5**
- outer-cell wins: **16/25**
- p90: 17.9345316289 versus E011 17.8310239265
- worst-5% SSE share: 0.304356048812 versus E011 0.315321724264

It misses the frozen 0.15 gain floor and the 17/25 cell floor.

## Transfer audit

Only `ridge_pseudo_e011_a1__w0.25` passed the preliminary repeated gates. It reaches RMSE **12.4519560877**, gains **0.0988002080**, wins 5/5 maps and 20/25 cells.

Its stress gains are:

```text
spatial:0    -0.1323834614
spatial:1    +0.0303204644
spatial:2    +0.0607692421
spatial:3    +0.2647682106
spatial:4    +0.1184442442
typewell:0   +0.0508901238
typewell:1   +0.1055539445
typewell:2   +0.0504214714
typewell:3   +0.1170935003
typewell:4   -0.2452747498
```

It regresses spatial group 0 by 0.1323834614 and typewell group 4 by 0.2452747498. Therefore the every-spatial and every-typewell gates fail.

All registered global slices improve:

```text
e011_catastrophe     +0.2240777049
high_gr_missingness  +0.0748140357
long_suffix          +0.0553975542
```

## Comparator and controls

The best E011-only comparator is `ridge_e011_only_a10__w0.25` at RMSE 12.6526586787; it is worse than exact E011 by 0.1019023830. The conservative pseudo+E011 candidate has **0.2007025910** incremental RMSE value over that same-target comparator.

Every corrupted pseudo control loses. The strongest negative-control gain is **-0.2958407299**, safely below the +0.03 cap. This establishes that the observed gain is specific to the legal within-well pseudo trajectory rather than generic bounded movement.

## Edge cases and reproduction

All 16 registered edge groups pass. A clean second run reproduces every substantive CSV and edge file byte-for-byte, and the normalized summaries are exactly equal. Official runtime is 149.164 seconds and reproduction runtime is 147.512 seconds.

## Durable interpretation

T025 proves a unique, legal coefficient-dynamics signal and reconfirms sub-5 target capacity. It also proves that four fixed pseudo-cuts summarized into one shallow well-level vector do not transfer uniformly across domains.

The next mechanism must change the training unit, not tune this vector. Use many randomized mask tasks from outer-training wells, include long horizons matching the real hidden suffix, and learn a mask-conditioned coefficient predictor under explicit spatial/typewell transfer controls. Do not start with a neural architecture until a bounded randomized-task screen proves that the expanded task distribution improves transfer.
