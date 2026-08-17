# PSSC: Period-Scaled State Conjugacy design

Date: 2026-08-14

## Status and claim boundary

PSSC is the next candidate, not an accepted innovation. The only provisional
claim is the following structural combination:

> Period-normalized physical patch scale selects an inverse-tied coordinate
> system around one shared Mamba, producing scale-conjugate realizations of a
> common latent state dynamics.

Coordinate transforms, dynamical conjugacy, invertible adapters, low-rank
adapters, and multi-scale Mamba are established ideas individually. A targeted
search did not locate their exact use as inverse-tied physical-scale coordinates
around a shared Mamba forecaster, but absence from this search is not proof of
global novelty.

## Mechanism

For scale `s` with patch stride `stride_s` and detected period `P`, define a
fixed centered log-scale coordinate `c_s`. Let

`K = U V^T - V U^T`

be a low-rank skew-symmetric generator. Then

`T_s = exp(c_s K)` and `T_s^{-1} = exp(-c_s K)`.

The temporal path becomes

`E_s(x) = T_s^{-1} Encoder_shared(T_s x)`.

Because `K` is skew-symmetric, `T_s` is orthogonal: it changes coordinates but
does not arbitrarily amplify features. `K=0` gives exact identity and reproduces
the accepted model. The graph path and prediction head remain unchanged.

To avoid the zero-factor gradient trap, initialize `U` as a small orthonormal
basis and `V=0`, with both trainable. This gives `K=0` and exact identity while
allowing a nonzero first-step gradient through `V`; initializing both factors to
zero is forbidden. Center `c_s` across the configured scales so the shared core
is the geometric reference rather than privileging either branch.

## Why this locus

- Whole-core local/period gradients conflict on both ETTh1 and ETTh2.
- Input/gate, convolution, and output interfaces conflict on both.
- The state generator `A_log` does not meet the conflict threshold on either.
- Therefore duplicating the full Mamba discards evidence of shared dynamics,
  while changing delta repeats an already failed and occupied route.

## Required controls

1. Accepted shared Mamba with the existing scale adapter.
2. PSSC with `K=0` frozen: exact baseline equality.
3. Parameter-matched input-only scale adapter: tests whether any extra input
   capacity is enough.
4. Parameter-matched untied input/output adapters: tests whether inverse tying
   and coordinate interpretation matter.
5. PSSC with scale labels swapped: tests physical-scale conditioning.
6. Two fully independent Mambas under a matched or reported parameter budget:
   upper-capacity control, not a novelty baseline.

## Staged stop gates

### Structural gate

- Identity initialization maximum error `<=1e-6` in evaluation mode.
- `T_s^T T_s` and `T_s^{-1}T_s` maximum error `<=1e-5`.
- Finite nonzero first-step gradient for `V` at identity initialization and for
  both factors after leaving identity.
- No changes to graph, head, data split, or Mamba `delta/B/C/A` equations.

### Validation gate

- ETTh1 and ETTh2 MSE improve on the first seed; macro improvement `>=0.5%`.
- No dataset MAE worsens by more than `0.2%`.
- PSSC beats both parameter-matched adapter controls in macro MSE.
- Re-measured local/period gradient cosine moves toward zero by at least `0.10`
  on both datasets without collapsing either branch gradient norm.
- Only after this pass, repeat on a second seed. Test remains forbidden.

Failure retires PSSC and preserves the model/result artifacts.

## Prior-art boundary sources

- Mamba: https://arxiv.org/abs/2312.00752
- ms-Mamba: https://arxiv.org/abs/2504.07654
- TimeMixer: https://openreview.net/forum?id=7oLshfEIC2
- Recon conflict-layer specialization: https://arxiv.org/abs/2302.11289
- PCGrad: https://arxiv.org/abs/2001.06782
- Continuous scale-conditioned Hyper-LoRA: https://arxiv.org/abs/2605.07562
- Learned dynamical conjugacy: https://doi.org/10.1016/j.chaos.2021.111151
- Semigroup consistency diagnostic: https://arxiv.org/abs/2605.26324
- MS-Temba cross-scale alignment: https://arxiv.org/abs/2501.06138
