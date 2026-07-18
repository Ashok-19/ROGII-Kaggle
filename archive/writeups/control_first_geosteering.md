<!-- Archived from https://www.kaggle.com/writeups/radiantallomancer/when-better-cv-scores-worse-a-control-first-geost via Kaggle MCP on 2026-07-18. Copyright remains with the author. Stored for competition research and source-reference use. -->

## Executive summary

My best public score so far is **6.675**. My best well-grouped OOF system improved from **8.248 to
7.623**, but the public leaderboard did not preserve that ordering: the 7.623 model scored 6.924,
while the older 8.248 model scored 6.675. That inversion changed how I worked on this competition.

The main lesson is not that validation is useless. It is that this problem has a small number of
very expensive wells, a large per-well datum component, and a public subset too small to reliably
rank nearby systems. A trustworthy experiment therefore needs more than a lower aggregate RMSE. I
required a known-winner positive control, no-op/duplicate controls for learned features, a shuffled
evidence control where applicable, well-level fold consistency, and a pre-registered promotion bar.

The current system has three physically interpretable layers:

1. a particle filter over structural position and rate, using GR/typewell agreement as the
   observation likelihood;
2. a six-model bidirectional GRU ensemble that refines tracker, trajectory, and alignment features,
   followed by a quadratic track-consistency solve;
3. a separate whole-well posterior over TVT and dip, decoded by sum-product on a trellis and blended
   only because its errors are genuinely different.

The most useful model insight was that a weak standalone model can still be a valuable ensemble
member. The classical trellis scores 13.420 alone but has only 0.488 error correlation with the
neural ensemble; adding it improves debiased OOF from 7.762 to 7.699. Replacing its emission with a
lightly calibrated learned map produces a 12.576 standalone leg and improves the blend again to
7.623. The weight is small because the model contributes evidence, not authority.

The most useful data insight was harsher: on the 7.762 OOF ensemble, the worst 5% of wells account
for **52.5% of all squared error**, and the worst 10% account for **64.5%**. A constant per-well
mean error explains 58.3% of SSE; an oracle correction of only that mean would reduce RMSE to 5.014.
Many signals can detect these risky wells. I found almost none that can predict the sign of the
needed correction on a held-out well.

This note documents what worked, what failed, and how I separated real evidence from a validation
mirage.

## 1. Physical frame: predict a surface, not an isolated row

Let

\[
S(MD) = TVT(MD) + Z(MD).
\]

`Z` is known throughout the well. In this frame the target is a structural surface sampled along
measured depth. The revealed heel gives an exact starting point,

\[
S_0 = TVT_{\text{last known}} + Z_{\text{last known}},
\]

and the hidden task is to continue a slowly changing surface through the lateral.

This framing immediately gave two useful facts:

- a bare last-known-TVT continuation scores 15.91 RMSE over all 773 training wells;
- a piecewise-linear fit to the *true* surface needs a median of about 19 segments per well and can
  reproduce it at roughly 0.8 RMSE.

So representation capacity is not the bottleneck. A low-dimensional path can express the answer.
The hard part is choosing the correct datum and local rate from legal evidence.

The GR log provides alignment evidence against the typewell, but it is ambiguous. The lateral GR
has gaps and a different amplitude distribution from the typewell. Repeated motifs create several
plausible alignments, and a locally good match is not necessarily the correct absolute TVT. That is
why all of my successful methods combine a GR likelihood with path persistence, an anchor prior,
and explicit uncertainty.

## 2. Validation: the row metric hides a well lottery

I used five-fold `GroupKFold` by well and always scored the pooled hidden rows, matching the
competition metric. The core OOF table has 773 wells and 3,783,989 hidden rows.

Pooled scoring is important, but it also means that long, difficult wells dominate. On the frozen
7.762 neural ensemble:

| Diagnostic | Value |
|---|---:|
| Median per-well RMSE | 4.188 |
| 90th percentile per-well RMSE | 10.600 |
| Worst per-well RMSE | 51.124 |
| SSE share of worst 5% of wells | 52.5% |
| SSE share of worst 10% of wells | 64.5% |
| SSE explained by one mean error per well | 58.3% |
| RMSE after oracle mean-error removal | 5.014 |

The public leaderboard behaves like a small composition sample from that heavy-tailed well
distribution. Here is my actual progression; every row is a real submission, not a leaderboard
projection:

| System | Well-grouped OOF | Public LB |
|---|---:|---:|
| pseudo-typewell particle filter | 10.325 | 8.306 |
| early sequence stack (v3) | 8.248 | **6.675** |
| augmented three-GRU stack (v4) | 7.943 | 6.877 |
| dual generation-4 stack (v7) | 7.788 | 6.927 |
| six-GRU stack (v8) | 7.762 | 6.886 |
| neural stack + classical trellis (v10) | 7.699 | 6.765 |
| v10 + spatial/tree member (v11) | 7.635 | 7.002 |
| neural stack + calibrated trellis (v12) | **7.623** | 6.924 |

Across the real pairs I analyzed, local-to-public ranking had Spearman correlation -0.243. This is
not permission to ignore CV or chase the board. It is the opposite: nearby public differences are
not a safe optimization target. I use OOF for expected private performance and keep a second,
decorrelated final candidate as composition insurance.

### My experiment controls

After several attractive but false gains, I adopted the following gate:

- **Known-winner positive control.** On the exact rows and fold map, a previously real-winning
  component must beat its parent. If it does not, that slice cannot validate a new claim.
- **No-op and duplicate controls.** Adding zero columns or an exact duplicate must not create a
  material gain. One early stride-sampled tree gate moved 0.039 merely from a duplicate column and
  up to 0.141 on a fold; every feature conclusion from that unstable adapter was invalidated.
- **Shuffled evidence.** When adding matching or same-well evidence, I permute complete evidence
  blocks across wells. A candidate has to beat this deranged input, not only a weaker baseline.
- **Fold and bootstrap checks.** A pooled gain concentrated in one fold does not promote. For final
  gates I use paired well bootstraps and multiple deterministic fold maps.
- **Pre-registered thresholds.** A cheap screen may kill an exact configuration, but it cannot
  promote a submission. Promotion requires the full 773-well scale.

These controls made the experiment ledger less exciting and much more useful.

## 3. The model that survived

### 3.1 Adapt the emission with the revealed heel

The raw typewell is not the same instrument response as the horizontal well. I first calibrate GR
amplitude from the known heel. I also build a pseudo-typewell by blending the horizontal well's
known `(TVT_input, GR)` samples into the supplied typewell curve.

On a fixed 100-well screen, particle-filter RMSE changed as follows:

| Heel blend weight | PF RMSE |
|---:|---:|
| 0.0 | 11.02 |
| 0.5 | 10.31 |
| **0.8** | **9.93** |
| 1.0 | 10.19 |

The interior optimum matters physically. The original typewell supplies broad stratigraphic
coverage; the heel supplies local tool response. Replacing either one completely is worse.

### 3.2 Particle filter over structural position and rate

Each particle carries structural position `S` and rate `dS/dMD`. The transition has high rate
momentum plus small process noise. For each hidden row I convert a particle back to TVT,
interpolate typewell GR at that TVT, and apply a robust Gaussian-like observation likelihood.
Effective-sample-size resampling preserves multiple plausible tracks.

I run multiple deterministic seeds and combine them by sequence log likelihood. Their posterior
mean is a prediction; their spread and likelihood spread become uncertainty features.

The particle filter is not the final answer. Full-pool pseudo-typewell PF scores 10.325 locally and
8.306 publicly. Its value is that it turns a raw matching problem into a physically constrained
proposal distribution for later models.

### 3.3 Sequence refinement and track fusion

At 4-ft spacing, a two-layer bidirectional GRU receives:

- raw and smoothed lateral GR, gap indicators, and local variability;
- trajectory slope plus legal look-ahead summaries of future `Z` and GR;
- particle-filter means, alternative likelihood temperatures, posterior spread, and GR mismatch
  at fixed TVT offsets;
- a coarse node-dip estimate and distance-to-end features.

It predicts `TVT - anchor` for the entire hidden tail. Artificial prefix cuts remain in the same
fold as their parent well, and validation uses only the original hidden zones.

The first GRU scored 8.404. Prefix-cut augmentation improved a comparable model to 8.159. Three
different GRUs plus a quadratic path fusion reached 7.943. Six models from two feature generations
reached 7.762; the third model in the later generation added only 0.026, showing that seed and
feature-family diversity was saturating.

The fusion is a banded quadratic solve. It keeps the point prediction close to the neural output
while encouraging adjacent TVT changes to agree with a separately predicted structural dip. This
is more stable than integrating dip predictions directly: small rate errors accumulate over a
5,000-ft tail.

### 3.4 A weak, different whole-well posterior

The second leg does not reuse neural point predictions. It places a grid over `(TVT, dip)` at
50-ft nodes, scores every state with a robust Cauchy GR/typewell cost, and applies priors for dip
persistence, near-zero dip, and increasing datum uncertainty with distance. Forward-backward
sum-product returns a posterior mean path.

This state-space direction was motivated by Shrey Gandhi's public
[STRIDE writeup](https://www.kaggle.com/writeups/shreygandhi/stride-a-joint-posterior-over-depth-and-distance).
My implementation and validation protocol differ, but that note was the primary-source prompt to
stop treating the tail as independent rows.

Standalone, this trellis scores 13.420—far worse than the neural ensemble. But its error
correlation with the ensemble is only 0.488. A small fixed weight improves ten out of ten fold maps:

| Blend | Debiased OOF |
|---|---:|
| six-GRU ensemble | 7.762 |
| ensemble + classical trellis | **7.699 ± 0.028** |

I then trained a 19k-parameter map to calibrate local emission sharpness. The map is not trusted to
produce TVT directly: its negative log probability is added at weight 0.005 to the classical cost,
and the same physical trellis performs the decode. Fold maps are trained out of fold, and the
emission weight is chosen leave-one-fold-out.

The learned leg scores 12.576 alone. Averaged across two seeds, it replaces the classical leg and
improves the blend to **7.623 ± 0.021**, a paired -0.076 at 10/10 fold maps. This was the rare neural
experiment whose weakness was useful because it supplied genuinely different evidence.

The calibrated cost-map experiment was prompted by Vishal Kishore's public
[SEGN writeup](https://www.kaggle.com/writeups/kishorevishal/segn-an-autoregressive-cost-map-view-of-tvt).
I retained the classical decoder and used the learned map only as a small emission correction.

## 4. What did not work—and what each failure taught me

### 4.1 Candidate routing: the oracle is real, the selector is not

Choosing the best family independently for each well gives an oracle RMSE near 6.9, roughly 1.4
better than the early ensemble. That made routing look like the obvious next step.

However, a selector trained on legal known-zone backtests scored 8.61–8.71 versus a 7.997 fixed
ensemble. A meta-model over six neural predictions plus uncertainty scored 8.39. Candidate-state
and contrastive path rankers also failed full five-fold confirmation.

Why? Disagreement reliably identifies hard wells, but it does not identify the corrective
direction. The oracle uses the target to answer a question that the legal features often do not.
This separated **detection** from **actuation**, and stopped me from treating oracle headroom as
harvestable model headroom.

### 4.2 Spatial features: a clean CV gain that did not transfer

A LightGBM member with spatial and per-well tracker features improved the v10 blend from 7.699 to
7.635 in ordinary grouped CV. It passed reproduction and kernel-parity checks. Public score then
worsened from 6.765 to 7.002.

A retrospective leave-spatial-block-out audit found two mechanisms:

1. the member helped only the hardest well tercile and hurt easy/medium wells, so a friendly public
   sample was expected to regress;
2. its apparent -0.053 blend gain collapsed to -0.013 when spatial blocks were removed, with
   boundary wells reversing to +0.314 harm.

The lesson is not “spatial data is bad.” It is that holding out wells is insufficient when the
training set still contains their close spatial neighbors. Every cross-well feature now requires a
spatial-block gate before it can ship.

### 4.3 More expressive sequence models did not cross the information wall

I tested wider GRUs, dense artificial cuts, a TCN, stabilized pre-LN transformers, future-target
auxiliary losses, multi-hypothesis heads, direct increments, and typewell cross-attention.

Dense prefix cuts gave a real -0.187 on a matched screen, mostly on the hard fold, but the full
five-fold version did not reproduce it. The best stabilized transformer scored 8.504 and added
0.006 to the production ensemble; a larger batch scored 9.057. The apparently decorrelated
diverged transformer was just decorrelated noise: once stabilized, error correlation returned to
0.87.

This is why I did not continue architecture tuning. Stability was fixed; the missing evidence was
not.

### 4.4 Full-well images and pseudo-cuts learned a prior, not the mode

A legal 2-D image of horizontal/typewell GR mismatch around v10 improved v10 by only 0.026 on a
calibrated fold. A more radical model learned the entire anchor-relative path from three late
artificial cuts in each training well. On 155 untouched wells it scored 15.775; even a fixed 25%
blend worsened v10 by 1.032. Its predictions had only 4.76-ft standard deviation against 15.44-ft
targets and just 0.091 row correlation.

The network learned the safe local prior present in short late cuts. It did not learn which datum
mode should persist through a much longer hidden tail. Making this U-Net larger would be tuning the
symptom, so I closed the adapter rather than the entire image family.

### 4.5 Same typewell does not mean same interpretation

Exact duplicate typewell groups were rare, and transferring TVT between wells with an identical
typewell gave about 117 RMSE versus an 18.1 anchor baseline in that diagnostic. The typewell is an
observation template, not a shared absolute geological solution. This killed a whole class of
apparently tempting cross-well shortcuts.

## 5. Uncertainty: knowing where is easier than knowing which way

The model contains several natural uncertainty measures: particle spread, likelihood-temperature
spread, disagreement among six GRUs, and disagreement between the neural and trellis families.

GRU disagreement correlates 0.278 with absolute residual, and its top quintile contains 44% of the
total SSE. In a separate frozen ensemble/trellis diagnostic, the highest-disagreement decile had
11.60 RMSE versus 7.07 for the rest. These are useful risk indicators.

But the signed mean disagreement between the neural and trellis paths has only 0.016 correlation
with the signed per-well error. The uncertainty can say, “this well is dangerous,” but usually not,
“move the path up 12 ft.” That is why confidence-gated correction repeatedly failed even when
confidence-gated detection worked.

For deployment I would report at least two quantities per well:

- a local posterior band from particle/trellis mass, describing ambiguity conditional on the
  chosen datum;
- a well-level model-disagreement risk, describing the larger chance that the datum itself is
  wrong.

The second should not be converted into a directional correction without independent evidence.

I also bootstrapped wells at the approximate private-set scale. For 150–200 wells, the 10th–90th
percentile RMSE span is still roughly 2.1–2.7 ft, much narrower than a 35-well simulation but large
enough to move ranks. This is another reason to communicate uncertainty at the well level and to
select final models for expected private performance, not the most favorable public draw.

## 6. Contribution ledger

The table below separates material ideas from variations:

| Idea | Measurement | Decision |
|---|---:|---|
| pseudo-typewell heel adaptation | PF 11.02 → 9.93 on fixed screen | keep |
| prefix-cut GRU augmentation | 8.404 → 8.159 | keep |
| multi-GRU ensemble + physical fusion | 8.159 standalone family → 7.943, then 7.762 | keep |
| classical TVT/dip trellis | 13.420 alone; blend 7.762 → 7.699 | keep as small diverse leg |
| calibrated emission map | 12.576 alone; blend 7.699 → 7.623 | keep for private expectation |
| spatial/tree member | 7.699 → 7.635 CV; public 6.765 → 7.002; LSO gain -0.013 | reject |
| learned routing | oracle ~6.9; learned selectors 8.39–8.71 | reject: sign unavailable |
| stabilized transformers | 8.504 / 9.057; no production blend gain | reject |
| pseudo-cut full-path U-Net | 15.775; best blend +1.032 | reject exact adapter |
| duplicate/no-op feature gate | duplicate moved +0.039 pooled, +0.141 fold | invalidate adapter |

## 7. Where I draw the line

The final pipeline is an ensemble, but its components have physical jobs:

- the particle filter proposes geologically continuous structural paths;
- the GRU learns repeatable residual behavior and uses the legally known future trajectory;
- the quadratic solve prevents noisy point predictions from violating plausible track changes;
- the trellis preserves multiple TVT/dip hypotheses and contributes only a small posterior-mean
  correction.

I allow OOF-tuned blend weights because RMSE rewards a posterior mean under squared loss. I do not
allow a feature because it improves one convenient fold, a selector because its target oracle is
strong, or a spatial member because ordinary grouped CV likes it. The line is crossed when the
experiment can no longer distinguish geological evidence from adapter instability or sample
composition.

The best public model is still the older 6.675 sequence stack. The strongest private-expectation
model is the 7.623 calibrated-trellis blend, despite its 6.924 public score. At the final deadline I
intend to keep one public-proven family and one decorrelated, control-validated family rather than
pretend that the public subset can resolve their ordering.

## 8. Reproducibility checklist

- Five-fold grouping is by base well; artificial cuts inherit the parent's fold.
- OOF scoring is pooled over all hidden rows, exactly matching RMSE.
- Every shipped test path uses only competition inputs available for that well. Cross-well spatial
  inputs were removed from the durable final family.
- Particle seeds are deterministic from the well ID; neural seeds and fold maps are recorded.
- Trellis grids, priors, blend weights, and fallback order are fixed before test inference.
- Submission kernels count successful model loads and per-well fallbacks, then run exact schema,
  row-order, uniqueness, finiteness, and coverage checks.
- Negative results are retained with their controls; a failed exact configuration does not become
  a claim that the whole model family is impossible.

## Closing thought

This competition looks like a rowwise regression problem, but the decisive unit is the well. Most
rows are easy once the well's datum and rate are right; a few wells dominate the metric when they
are wrong. The most productive change I made was to stop asking only, “did RMSE improve?” and start
asking, “what evidence moved it, would a shuffled version also move it, and did the experiment
recover something already known to be real?”

That discipline did not reveal the missing sign on every hard well. It did reveal which gains I
could trust.

## Acknowledgements

Thanks to the competition host and to participants who shared methods while the competition was
still active. In addition to STRIDE and SEGN, Georgy Mamarin's
[“Fork the ruler, not the model”](https://www.kaggle.com/writeups/georgymamarin/fork-the-ruler-not-the-model)
helped sharpen my distinction between path representation and per-well selection. The early public
particle-filter notebooks provided a valuable reproducible baseline. All measurements and failure
claims in this note are from my own reruns and artifacts; links above identify the ideas that
changed the direction of those experiments.
