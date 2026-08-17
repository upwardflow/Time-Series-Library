# GraphMamba PCRF D0--D4 diagnostic result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-13
- Verification Status: ANALYZED
- Version Label: `pcrf_diagnostic_v0`
- Source: frozen accepted periodic GraphMamba seed-2021 checkpoints
- Data use: ordered train fit, ordered validation evaluation, test untouched

## Outcome

PCRF fails the preregistered diagnostic gate and must not be added to
`models/GraphMamba.py`. The causal reliability observables contain a small
validation signal, but neither their magnitude nor uncertainty supports a model
candidate.

## Structural audit

- D0 reproduces checkpoint validation MSE to relative errors `2.30e-10`
  (ETTh1) and `1.23e-9` (ETTh2).
- Explicit seasonal-plus-trend reconstruction is exactly equal to the normal
  model forward result (`max_abs = 0`).
- Train and validation loaders are explicitly rebuilt with `shuffle=False`.
- Reliability features use input history only: adjacent period-24 seasonal
  cosine and normalized trend second-difference roughness.
- No test loader is constructed.

## Validation results

| Dataset | D0 accepted | D1 static | D2 cycle/seasonal | D3 roughness/trend | D4 joint | D4-perm |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 MSE | 0.988814 | 0.981974 | **0.979928** | 0.980719 | 0.979941 | 0.981412 |
| ETTh2 MSE | **0.272637** | 0.273658 | 0.273535 | 0.272583 | 0.272782 | 0.273367 |

| Dataset | D1 vs D0 | D4 vs D1 | D4 vs D0 | D4-vs-D1 block-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| ETTh1 | +0.692% | +0.207% | +0.897% | `[-0.023%, +0.474%]` |
| ETTh2 | -0.375% | +0.320% | -0.054% | `[-0.369%, +1.005%]` |

The macro D4-over-D1 improvement is `0.264%`, below the preregistered `0.5%`.
Both uncertainty intervals include zero. D4 also fails the main-effect control:
D2 is marginally better on ETTh1 and D3 is clearly better on ETTh2. Although
permuting training reliabilities removes more than half of D4's incremental
point gain on both datasets, that condition alone is insufficient.

MAE agrees with the no-go decision. ETTh1 D4 MAE (`0.650173`) improves over D0
but is worse than D1 (`0.649100`); ETTh2 D4 MAE (`0.352003`) is slightly worse
than D0 (`0.351920`).

## Interpretation

- Static seasonal/trend recalibration is dataset-dependent: useful on ETTh1,
  harmful on ETTh2. It therefore does not reproduce the stronger older ETTm
  upper bound on the accepted periodic ETTh backbone.
- Cycle consistency and trend roughness each carry weak predictive association,
  but their joint correction does not exceed the best single reliability on
  either dataset.
- The evidence supports using component reliability as a diagnostic observable,
  not as an accepted architectural contribution.
- No post-result feature, ridge-grid, period, or formula search is permitted on
  these validation outputs.

## Fallacy scan

- Coverage: 11/11 checked.

| Fallacy | Severity | Assessment |
|---|---|---|
| Simpson's paradox | NOTE | Macro and dataset-specific results are both shown; mixed D1 directions are not hidden. |
| Ecological fallacy | NOTE | Inference remains at dataset/task forecasting level. |
| Berkson's paradox | CAUTION | Two ETT datasets are a selected benchmark family; broad external validity is not claimed. |
| Collider bias | NOTE | No post-outcome conditioning variable is used. |
| Base-rate neglect | NOTE | No diagnostic-classification probability is reported. |
| Regression to the mean | NOTE | Datasets were not selected by extreme PCRF performance. |
| Survivorship bias | NOTE | Both preregistered datasets and all D0--D4 controls are reported. |
| Look-elsewhere effect | NOTE | Fixed controls and the failed gate are reported together. |
| Garden of forking paths | CAUTION | Reliability definitions were fixed before output; subsequent tuning is prohibited. |
| Correlation implies causation | CAUTION | Reliability effects are predictive associations, not causal mechanisms. |
| Reverse causality | NOTE | Reliability is measured from earlier observed history, but no causal claim is made. |

## Decision

`NO-GO`: do not implement PCRF, do not access test, and do not combine it with
CMRHM. Preserve the diagnostic script and JSON outputs as negative evidence.
The accepted periodic dual-patch GraphMamba remains unchanged.
