# GraphMamba next-innovation literature and code audit

Date: 2026-08-13

## Scope and current backbone

The accepted GraphMamba uses a period-constrained local patch and a complete-
period patch, scans them independently with a scale-conditioned shared Mamba,
adds a shared variable-graph branch, flattens both resolutions, and finally adds
an independently projected moving-average trend. CMRHM remains a separate,
frozen old-history module.

The audit asks which remaining bottleneck has both local empirical support and
a defensible gap relative to published work. No test result was used and no
third-party code was copied.

## Backbone bottleneck ranking

| Direction | Local evidence | Prior-art density | Decision |
|---|---|---|---|
| Scale-specific graph | LagGraph/GF/TIRGE fail the 1% gate | high: MAGNN, ESG, MillGNN | reject as main novelty |
| Graph-conditioned/ordered Mamba | graph and temporal states strongly cancel | high: Graph-Mamba, STG-Mamba, SpoT-Mamba, MambaTS, GPS-Mamba | reject as standalone claim |
| Channel/bidirectional scan | no matched local bottleneck | high: S-Mamba, Bi-Mamba+, MambaTS, FSMamba | reject |
| Longer/adaptive lookback | strong in general | already solved locally by CMRHM; ALW/IRPA adjacent | reject as next GraphMamba module |
| Periodic contrastive pretraining | plausible, no local bound | RCL, CoST, CLeaRForecast adjacent | supporting technique only |
| Seasonal--trend component fusion | 1.38--5.63% frozen split-validation upper bound | generic fusion crowded, period-reliability condition not located | **diagnose next** |

## Key external sources

| Source | What it establishes | Relationship to this project | Code status |
|---|---|---|---|
| DLinear, AAAI 2023 | moving-average seasonal/trend specialization | source of the established additive decomposition pattern | official code Apache-2.0 |
| FEDformer, ICML 2022 | frequency modeling and mixture-of-experts decomposition | adaptive decomposition is not novel by itself | public code; no code reused |
| TimeMixer, ICLR 2024 | scale-dependent seasonal/trend processing and multi-predictor fusion | generic multiscale component fusion is occupied | official code Apache-2.0 |
| MAGNN, IEEE TKDE 2022 | scale-specific variable graphs | rules out a simple local-graph/period-graph claim | paper reference only |
| ESG, KDD 2022 | evolutionary scale-specific graph structure | rules out dynamic multiscale graph as primary novelty | paper reference only |
| MambaTS, 2024 | variable-aware scanning and selective-parameter regularization | rules out generic variable serialization | official repository; no code reused |
| STG-Mamba, 2024 | graph selective state-space blocks | rules out broad graph-conditioned SSM claim | official repository; no code reused |
| RCL, Neural Networks 2026 | contrastive pretraining of Mamba selectivity | period contrast can only be supporting novelty | no code reused |
| ALW, AAAI 2026 | wavelet-conditioned adaptive lookback | adjacent to, but redundant with, local CMRHM evidence | official code not located |
| AdaFusionNet, withdrawn ICLR 2026 submission | adaptive decompose--specialize--fuse formulation | novelty warning only; low evidence weight | promised code unavailable |

Primary links:

- DLinear: https://github.com/cure-lab/LTSF-Linear
- FEDformer: https://proceedings.mlr.press/v162/zhou22g.html
- TimeMixer: https://github.com/kwuking/TimeMixer
- MAGNN: https://arxiv.org/abs/2201.04828
- ESG: https://arxiv.org/abs/2206.13816
- MambaTS: https://arxiv.org/abs/2405.16440
- STG-Mamba: https://arxiv.org/abs/2403.12418
- RCL: https://arxiv.org/abs/2504.09185
- ALW: https://doi.org/10.1609/aaai.v40i31.39797
- AdaFusionNet: https://openreview.net/forum?id=yDqrGP4w2E

## Selected hypothesis: PCRF

**Periodic Component Reliability Fusion (PCRF)** separates component geometry
from component reliability:

1. Keep the accepted period and local patch sizes unchanged.
2. Keep seasonal and trend forecasters unchanged.
3. Estimate causal, per-window reliability observables from the input only:
   adjacent-cycle seasonal consistency at lag `P`, and normalized trend
   roughness.
4. Use those observables only to calibrate the marginal seasonal/trend forecast
   contributions, with a zero initialization that exactly reproduces the
   accepted backbone.

The potential novelty is not "adaptive fusion." It is the narrower claim that
the physically detected period supplies an observable reliability measure for
when the seasonal Mamba branch should be trusted relative to the trend branch.
This claim remains provisional until it beats a static per-variable component
rescaling control.

## Why diagnosis precedes implementation

The old frozen ETTm upper bound is strong but belongs to a pre-periodic
GraphMamba. A simple learned gain may merely repair optimization bias and would
not justify a module claim. The current periodic ETTh checkpoints must therefore
show all of the following before `models/GraphMamba.py` is edited:

- static component recalibration is useful on both datasets;
- observed component reliability adds value beyond static recalibration;
- seasonal-only and trend-only controls cannot explain the joint result;
- shuffled reliability removes most of the incremental gain;
- a moving-block bootstrap excludes zero in the positive direction;
- no test split is accessed.

Until then, PCRF is a research route, not an accepted module.
