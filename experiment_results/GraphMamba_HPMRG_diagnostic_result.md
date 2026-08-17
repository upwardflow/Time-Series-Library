# GraphMamba HPMRG D0--D4 diagnostic result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-13
- Verification Status: POINT ESTIMATES VERIFIED; DEPENDENCE INTERVAL INVALID
- Version Label: `hpmrg_diagnostic_v1`
- Source: frozen periodic GraphMamba seed-2021 checkpoints
- Data use: train fit, validation evaluation, test untouched

## Outcome

The preregistered diagnostic gate fails. HPMRG must not be implemented on the
current hypothesis. No GraphMamba or CMRHM model file was changed.

V0 was invalidated during self-review because it divided already standardized
residuals by per-window standard deviations. V1 removed only this duplicated
scaling and added a fatal checkpoint-MSE reproduction test. Both valid V1 D0
metrics reproduce their frozen checkpoint records to approximately `1e-9`
relative error.

## Validation results

All metrics below are in the data loader's training-standardized units.

| Dataset | D0 MSE | D1 shared | D2 horizon | D3 phase | D4 interaction | D4 vs D1 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.988814 | **0.976598** | 0.983001 | 0.976420 | 0.983604 | **-0.717%** |
| ETTh2 | **0.272637** | 0.302697 | 0.315285 | 0.303001 | 0.316217 | **-4.467%** |

| Dataset | D1 MAE | D4 MAE | D4-vs-D1 bootstrap 95% interval | Phase permutation behavior |
|---|---:|---:|---:|---|
| ETTh1 | 0.646178 | 0.648435 | `[-0.812%, -0.599%]` | D4 remains similarly harmful |
| ETTh2 | 0.377586 | 0.386490 | `[-4.777%, -4.167%]` | D4 remains similarly harmful |

Macro D4-over-D1 MSE change is `-2.592%`; zero of two datasets reaches the
required `+1%`. D4 is worse than the horizon-only and phase-only controls on
both datasets. The phase-permuted negative control retains roughly 94--97% of
the magnitude of the already-negative interaction result, providing no evidence
that true future phase is being exploited.

Coefficient banks also vary strongly across condition cells: D4 mean pairwise
cosines are only 0.058 (ETTh1) and 0.110 (ETTh2), with sign agreement near
0.54--0.58. This is consistent with unstable group-specific fits rather than a
stable horizon--phase relation law.

## Interpretation

- ETTh1 contains a small validation signal for one shared cross-variable
  residual correction: D1 improves over D0 by about 1.24%. Splitting that
  relation by horizon and phase destroys the gain.
- ETTh2 shows a train-to-validation failure even for D1: its correction worsens
  the frozen baseline by about 11.0%. Additional conditioning worsens it further.
- Therefore the current history summaries do not support a generalizable joint
  horizon--phase relation correction on the accepted GraphMamba backbone.
- This result rejects the planned HPMRG implementation. It does not establish
  that variable relations are universally horizon-invariant; it establishes
  that the preregistered observable interaction is absent or non-generalizing
  under this diagnostic and these two datasets.

## Self-review and corrections

| Revision | Finding | Correction | Model impact |
|---|---|---|---|
| V0 | Duplicate per-window residual normalization produced extreme weights | Archived V0; removed duplicate scaling | none |
| V1 | Needed proof that current code reproduces old frozen checkpoint | Added fatal D0 MSE comparison at `1e-5` relative tolerance | none |
| V1 result | D4 failed magnitude, consistency, bootstrap, main-effect, and permutation gates | Stop before HPMRG model implementation | none |

Invalid V0 and valid V1 outputs are both preserved:

- `logs/graphmamba_hpmrg_diagnostic/v0_invalid_window_rescaling/`
- `logs/graphmamba_hpmrg_diagnostic/summary.json`

## Reproducibility verification

V1 was rerun unchanged with only its output directory redirected to
`/tmp/graphmamba_hpmrg_repro`. After excluding that path string, the rerun's
summary and both per-dataset JSON files are exactly equal to the archived V1
files. The rerun again reports macro D4-over-D1 change `-2.5919473963%`, zero
datasets above `+1%`, and `implementation_gate_passed: false`.

## Statistical warnings

- A later loader audit found that `data_provider` shuffles validation by default.
  The point metrics are order-invariant and reproduce exactly, but the reported
  moving-block interval was applied after shuffling and is therefore invalid as
  a serial-dependence interval. The no-go decision does not rely on it: D4's
  point MSE is already worse than D1 on both datasets and fails the magnitude,
  main-effect, and permutation gates.
- Ridge linearity is an upper-bound approximation to the proposed nonlinear
  graph, not an architectural equivalence proof.
- Only two related hourly ETT datasets were evaluated because the primary gate
  already failed. The preregistered non-ETT confirmation condition is unmet by
  construction and no external test was justified.
- The inference is associational/predictive, not causal.

## Fallacy scan

- Coverage: 11/11 checked.

| Fallacy | Severity | Assessment |
|---|---|---|
| Simpson's paradox | NOTE | Aggregate and both dataset directions agree; no reversal observed. |
| Ecological fallacy | NOTE | Claims remain at dataset/task predictive level. |
| Berkson's paradox | CAUTION | ETT is a selected benchmark family; external generalization is not inferred. |
| Collider bias | NOTE | No post-outcome covariate adjustment is used. |
| Base-rate neglect | NOTE | No diagnostic-classification probabilities are reported. |
| Regression to the mean | NOTE | Tasks were not selected for extreme D4 scores. |
| Survivorship bias | NOTE | Both preregistered datasets and all D0--D4 runs are reported. |
| Look-elsewhere effect | NOTE | All fixed controls are reported; no favorable subset is selected. |
| Garden of forking paths | CAUTION | V0 required a justified estimator correction, but both versions and the unchanged gate are preserved. |
| Correlation implies causation | CAUTION | Results support only out-of-sample predictive association. |
| Reverse causality | NOTE | No directional causal interpretation is made. |

## Decision

`NO-GO`: do not create `GraphMambaHPMRG.py`, do not train a candidate, do not
access test, and do not combine this rejected route with CMRHM. The accepted
periodic dual-patch GraphMamba and frozen CMRHM remain unchanged.
