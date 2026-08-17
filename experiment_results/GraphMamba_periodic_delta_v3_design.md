# GraphMamba periodic-normalized delta V3 design

Date: 2026-08-13

## Hypothesis

Patch tokens at different resolutions represent different elapsed physical
times. A shared Mamba should therefore share its continuous dynamics while
calibrating the positive discretization step used by selective scan.

For scale `s`, define the period-normalized stride

`r_s = stride_s / period`.

Relative to the local branch, V3 uses

`delta_scale_s = exp(tanh(alpha) * log(r_s / r_local))`.

`alpha` is one learned scalar initialized to zero. Thus both branches start at
exactly unit delta; when `tanh(alpha)=1`, the period branch uses the complete
physical stride ratio. For the ETT setup, the normalized strides are `2/24`
and `12/24`, so the full ratio is 6.

## Controlled modes

- `legacy`: accepted V1 using the installed Mamba fused/default path.
- `unit`: explicit Mamba-1 selective scan with delta scale fixed to 1.
- `physical`: the same explicit scan with the fixed normalized stride ratio.
- `learned`: the same explicit path with the learned period-normalized scale.

The primary comparison is `unit` versus `learned`, not `legacy` versus
`learned`, because the former holds the scan kernel and initialization fixed.

## Implementation boundary

- Mamba-1 only for V3; Mamba2 requests are rejected explicitly.
- Delta is multiplied after its projection and softplus and before selective
  scan. Patch-token amplitude is not used as a proxy for elapsed time.
- Local and period states remain isolated and share all Mamba parameters.
- No retired alignment, confidence router, or LagGraph path is restored.
- The accepted V1 `legacy` mode remains the default until validation supports a
  change.

## Validation gate

Primary validation-only pairs use ETTh1-192 and ETTh2-192, seed 2021, with all
settings held fixed except `periodic_delta_mode=unit|learned`.

- Advance to a second seed only if learned scaling improves MSE on both tasks
  and macro MSE by at least 0.5%, without worsening MAE on both tasks.
- Stop V3 if macro MSE is non-positive or MAE worsens on both tasks.
- Treat a mixed or sub-0.5% result as insufficient rather than tuning `alpha`.
- Do not access the test split.

## V3a outcome and exploratory V3b amendment

This amendment was recorded after V3a and before V3b training. Learned scaling
was effectively neutral on both primary tasks: the best checkpoints produced
period-branch scales of approximately `1.0005` (ETTh1) and `1.0279` (ETTh2).
It therefore failed the preregistered gate and will not receive another seed.

V3b directly tests the physical hypothesis with a fixed scale of
`(12/24)/(2/24) = 6`, using the same `unit` controls already run. This is an
exploratory, post-V3a comparison and must be reported as such. It advances only
if it improves MSE on both tasks with macro improvement of at least 0.5%; no
tuning or test access follows a failed result.

## Structural acceptance

- Unit and learned modes must share identical common initialization.
- Their initialization-time outputs must be exactly equal.
- Explicit unit delta must match Mamba-1's original non-fused path within
  floating-point tolerance.
- Active scale must change output and give the exponent a finite nonzero
  gradient.
- A real CUDA forward/backward must pass.
