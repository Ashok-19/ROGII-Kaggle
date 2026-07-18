# Competition Discussion Synthesis

Archive coverage: **132/132 topics**, **981 messages**, **0 failed topics**. Every root post, comment, and nested reply returned by Kaggle MCP was normalized into the JSONL chunks beside this file.

Participant statements are not treated as established facts unless they are independently supported by official pages, local data, reproducible code, or controlled score evidence.

## Recurrent high-signal themes

- **Validation mismatch is central.** Multiple participants report CV/LB inversions or public-score variance; whole-well suffix simulation and repeated fold maps are mandatory.
- **Per-well datum/trend dominates.** Threads on the ±15 ft datum, the “ruler,” worst wells, and line-oracle limits consistently focus attention on signed whole-well structural error.
- **Row-local tabular modeling is not the only viable formulation.** Participants report strong tabular, non-tabular, and pure-physics results, but most details are withheld; these are hypothesis signals, not reproducible evidence.
- **Visible test wells are train-derived examples.** Same-well overlap/contact reconstruction is useful as a parity test but must not be mistaken for hidden-test validation.
- **Typewell and horizontal GR matching is ambiguous.** Discussions repeatedly describe self-similarity, bimodal paths, missing GR, and poor toe localization.
- **Spatial/context features are tempting but dangerous.** Their legality, coordinate meaning, split design, and transfer must be demonstrated with leave-spatial/typewell-out controls.
- **Weak but decorrelated models can help.** Several threads and both winning notes favor small ensemble weights based on residual diversity rather than standalone rank.

## Highest-priority discussion threads

| Topic | Why it matters |
|---|---|
| [716699](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716699) How to submit your Writeup for the Working Note Award |  |
| [712037](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037) Fork the ruler, not the model — where the error actually lives |  |
| [699853](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853) multi-trajectory prediction (MTP) with deep CNN for welllog inversion |  |
| [702474](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702474) ML to learn generator for forward simulation |  |
| [711878](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711878) The ±15 ft datum: why some wells are unsolvable |  |
| [700424](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424) Share an UI visualizer  |  |
| [717573](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573) Score Without Tabular Models |  |
| [714514](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/714514) What wrong with TVT |  |
| [707613](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613) PF baseline got LB 8.863 - any ideas to make it learnable with NN? |  |
| [726465](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726465) Where does the top-team signal come from below the per-well line-oracle? |  |
| [697418](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697418) Diagram of the problem |  |
| [701691](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701691) cv and lb correlations ..... |  |
| [698644](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698644) Over fitting on well |  |
| [699289](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699289) Paradigm Shift: Why pure Tabular Models might be hitting a wall (Spatial & Sequential Context) |  |
| [708367](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367) Problem Breakdown |  |
| [719389](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/719389) Does CV correlates with LB？ |  |
| [697431](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431) besides regression, also dwt (time warping)!  |  |
| [711308](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711308) Cluster wells by dz offset |  |
| [704273](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704273) How much should we trust the LB score? |  |
| [698282](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698282) Defintion of tvt |  |
| [708167](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708167) Formation Columns Are Derived from Typewell, Not Independent 3D Surfaces |  |
| [697329](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697329) Submission Scoring Error — Is the scorer live yet? |  |
| [716289](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716289) Pointwise GR Makes No Sense |  |
| [698449](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449) Duplicate type wells for different horizontal wells |  |
| [697416](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416) Welcome to ROGII - Wellbore Geology Prediction! |  |
| [699326](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699326) stage.1 : global search using linear prior tvt = linear(md,z) |  |
| [699207](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699207) Can a single model achieve LB/CV below 10.0? |  |
| [709495](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/709495) New: Working Note Awards! Submit by July 6 |  |
| [705210](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/705210) PNG files don't match the data |  |
| [720701](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/720701) Final Gold and Silver Medal Cutoff Predictions |  |

## Claims requiring direct local tests

1. Reframing the target as smooth `U = TVT + Z` should materially reduce modeling complexity.
2. A candidate bank with fixed blending may outperform learned per-well routing because the sign of candidate advantage is weakly identifiable.
3. Horizontal-well pre-PS self-correlation may be more reliable than typewell matching for some wells.
4. Ordinary tree ensembles may reach the 5.x regime only when fed trajectory/alignment/candidate-path features rather than raw rows.
5. Pure physics can reach the mid-6 range under a correct hidden-suffix CV, but the exact observation model is not public.
6. Per-well risk is detectable; signed datum correction may remain unavailable.

Each item above is entered as a hypothesis in the dashboard and is not considered true until the prescribed controls pass.
