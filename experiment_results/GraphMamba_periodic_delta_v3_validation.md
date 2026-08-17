# GraphMamba periodic-normalized delta V3 validation

Date: 2026-08-13

## Protocol

- Forecast horizon: 192; input length: 96.
- Datasets: ETTh1 and ETTh2.
- Seed: 2021.
- Baseline: `unit`, the explicit Mamba-1 selective-scan kernel with delta scale 1.
- V3a: zero-initialized learned exponent over the normalized stride ratio.
- V3b: exploratory fixed physical ratio `(12/24)/(2/24) = 6`.
- Same periodic adapter backbone and initialization; validation only; no test access.

## Structural verification

- Common unit/learned initialization: identical.
- Unit/learned initialization output: exact equality.
- Original non-fused Mamba-1 versus explicit unit path: maximum CUDA output
  difference `1.19e-7`.
- Active scale changed output and produced finite nonzero exponent gradients.
- V3 adds only one scalar parameter; V3b adds none.

## Results

| Dataset | Mode | Validation MSE | Validation MAE | MSE gain vs unit | MAE gain vs unit |
|---|---:|---:|---:|---:|---:|
| ETTh1 | unit | 0.986667838 | 0.652019216 | — | — |
| ETTh1 | learned | 0.986668306 | 0.652019657 | -0.000047% | -0.000068% |
| ETTh1 | physical | 0.985869653 | 0.651806606 | +0.080897% | +0.032608% |
| ETTh2 | unit | 0.273022348 | 0.351866308 | — | — |
| ETTh2 | learned | 0.273022385 | 0.351866589 | -0.000014% | -0.000080% |
| ETTh2 | physical | 0.273026957 | 0.351936476 | -0.001688% | -0.019942% |

V3a's best learned period scales were approximately `1.0005` on ETTh1 and
`1.0279` on ETTh2. The mechanism effectively remained at the unit control.

V3b's macro MSE gain was approximately `0.0396%` and its direction was mixed.
Macro MAE gain was approximately `0.0063%`. Training duration was essentially
unchanged within the explicit-kernel comparisons.

## Decision

Both V3a and exploratory V3b fail the preregistered gate. No second seed,
hyperparameter tuning, or test evaluation is justified. The exact-delta
implementation is structurally valid, but one global stride-conditioned delta
scale is not supported as a forecasting contribution by these experiments.

The active model returns to the accepted periodic adapter-only V1. V3 design,
diagnostic, checkpoints, raw logs, and this result remain archived. Any future
return should use a materially different hypothesis, such as token-dependent or
channel-dependent physical discretization, rather than tuning this global
scalar on the same validation tasks.
