# GraphMamba same-phase cross-period scan design

Date: 2026-08-14

## Status

Retired candidate: structural contract passed, predictive validation failed on
both preregistered datasets, and the active implementation was removed.
PSSC is retired from the active route and remains only as an archived hypothesis.

See `GraphMamba_cross_period_phase_scan_validation.md` for the complete result.

## Tensor contract

Given normalized seasonal history `X` shaped `[B,N,L]`, period `P`, and
`C=floor(L/P)` complete cycles:

1. retain the latest `C*P` observations;
2. reshape to `[B,N,C,P]` in chronological order;
3. transpose to `[B,N,P,C]` so each phase owns a chronological cycle sequence;
4. embed each scalar observation into `D` dimensions;
5. fold phase into batch, yielding `[B*P,N,D,C]`;
6. invoke the same shared Mamba used by the local branch, scanning only `C`;
7. mean-pool all bidirectional cycle states and unfold to `[B,N,D,P]`.

The 24 phase sequences never share recurrent state. A Mamba transition always
means one complete period has elapsed.

No explicit sine/cosine phase embedding is added. Phase identity is preserved
by the invertible fold/unfold position and by the phase-indexed head weights.
This keeps the candidate focused on scan topology rather than adding a second
phase-feature mechanism already crowded by prior work.

## Controlled changes

- Unchanged: local patch4/stride2 branch, decomposition, graph mixer, shared
  Mamba parameters, delta/B/C/A equations, bidirectionality, and final forecast
  loss.
- Replaced only in candidate mode: overlapping length-24/stride-12 period patch
  tokens become 24 same-phase cross-cycle summary tokens.
- ETT-96 candidate head width becomes `48+24` tokens instead of `48+8`; parameter
  and runtime differences must be reported.
- `periodic_aligned` remains the accepted default and exact comparison control.

## Prior-art ceiling

PhaseFormer already establishes phase-driven time-series forecasting and cannot
be ignored. TimesNet establishes period-based 2D reshaping, and multi-scale
Mamba models are also crowded. The provisional differentiator is limited to:

> independent same-phase cross-cycle state evolution using the same Mamba core
> as a separate local continuous-time patch branch.

This is a combination claim pending deeper paper-level verification and
validation evidence, not a claim that phase modeling or period reshaping is new.

Primary sources:

- PhaseFormer: https://openreview.net/forum?id=Lk9SqMQzhX
- TimesNet: https://openreview.net/forum?id=ju_Uqw384Oq
- ms-Mamba: https://arxiv.org/abs/2504.07654
- TimeMachine: https://arxiv.org/abs/2403.09898

## Structural gate

- Exact index audit proves each sequence contains one fixed phase across cycles.
- Encoder calls are local `[B,N,D,48]` and phase-folded `[B*24,N,D,4]` for ETT-96.
- No state transition crosses phase boundaries.
- The same encoder object and parameters serve both calls.
- Cycle pooling uses every position because endpoint selection would discard
  most backward-scan context in the bidirectional shared encoder.
- Forecast shape is unchanged and backward gradients are finite.
- Candidate mode does not change accepted-mode outputs or state dictionaries.

## Validation gate

- Paired seed-2021 ETTh1/ETTh2-192 validation only; test forbidden.
- Both datasets must improve MSE, macro MSE gain at least `0.5%`.
- Neither MAE may worsen by more than `0.2%`.
- Period-branch removal must remain material on both datasets.
- If stage 1 passes, repeat a second seed. Otherwise retire candidate without
  tuning phase pooling, period, or patch geometry on these outputs.
