# GraphMamba frozen selective-dynamics diagnostic result

Date: 2026-08-14

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-14
- Verification Status: VERIFIED
- Version Label: `GraphMamba-SDYN-D0-v1`

## Experiment Result

- **ID:** `GraphMamba_frozen_selective_dynamics_branch_audit_v0`
- **Type:** analysis
- **Status:** completed
- **Command:** `.venv/bin/python -u scripts/diagnose_graphmamba_selective_dynamics.py`
- **Working directory:** `/home/cwh/Time-Series-Library`
- **Exit code:** `0`
- **Scope:** ordered validation only, 2,689 origins per dataset; test was never constructed
- **Model mutation:** none; accepted `models/GraphMamba.py` remained unchanged
- **Anomalies:** none

## Structural verification

| Dataset | Fused vs explicit first-batch max abs | E0 checkpoint-MSE relative error | Status |
|---|---:|---:|---|
| ETTh1 | `3.5763e-7` | `1.1533e-9` | pass |
| ETTh2 | `2.3842e-7` | `1.0393e-9` | pass |

The intervention therefore operates on an exact-enough explicit Mamba-1 path.
For each branch it replaces one realized selective sequence by its own temporal
mean, preserving its learned level and all other model operations.

## Primary branch-normalized results

Values are mean encoder relative RMS. The branch distinction threshold was
`25%`, but a family also required at least `2%` response and the same result on
both datasets.

| Dataset | Family | Local | Period | Relative distinction | 95% block CI, local-period | Material | Dataset pass |
|---|---|---:|---:|---:|---:|---|---|
| ETTh1 | delta | `3.787%` | `0.112%` | `97.05%` | `[3.447%, 3.911%]` | yes | yes |
| ETTh1 | B | `2.868%` | `0.112%` | `96.10%` | `[2.620%, 2.896%]` | yes | yes |
| ETTh1 | C | `3.102%` | `0.170%` | `94.53%` | `[2.802%, 3.064%]` | yes | yes |
| ETTh2 | delta | `0.078%` | `0.054%` | `30.31%` | `[0.0227%, 0.0243%]` | no | no |
| ETTh2 | B | `0.310%` | `0.152%` | `50.89%` | `[0.150%, 0.165%]` | no | no |
| ETTh2 | C | `0.411%` | `0.235%` | `42.83%` | `[0.172%, 0.181%]` | no | no |

The direction is local greater than period everywhere, but the practical effect
is dataset-specific. With 2,689 overlapping origins, a narrow interval around a
tiny ETTh2 difference is not evidence of architectural importance.

## Secondary frozen forecast effects

| Dataset | Local flattening delta/B/C: forecast relative RMS | Period flattening delta/B/C: forecast relative RMS |
|---|---:|---:|
| ETTh1 | `3.315% / 3.131% / 2.874%` | `0.0289% / 0.0188% / 0.0311%` |
| ETTh2 | `0.0159% / 0.0894% / 0.0871%` | `0.0038% / 0.0107% / 0.0184%` |

These are frozen counterfactual sensitivities, not expected gains from training.
They are secondary because the prediction head assigns unequal dimensions and
weights to 48 local versus 8 period tokens.

## Gate and decision

No family satisfies cross-dataset materiality. The preregistered gate returns an
empty eligible set.

**NO-GO: do not design or train a branch-specific delta/B/C mechanism from this
evidence.**

The ETTh1-only signal is retained as a checkpoint-specific observation, not
generalized into a model contribution. Direct delta/B/C variation control is
also prior-art-adjacent to MambaSL, selective dropout is covered by MambaTS,
multi-rate delta by ms-Mamba, and generic selectivity enhancement/metrics by
RCL. Because the empirical gate failed, no post-result candidate search was
opened and no overlap risk was converted into code.

## Validation Report

- **Source:** `GraphMamba_frozen_selective_dynamics_branch_audit_v0`
- **Overall Confidence:** SOLID for the preregistered no-go; CAUTION for any
  broader claim about why ETTh1 and ETTh2 differ
- **Multiple comparisons:** three preregistered families were inspected. No
  family passed the practical cross-dataset gate; the decision does not rely on
  uncorrected p-values or selecting a favorable family.
- **Dependence:** validation origins overlap. A 24-origin moving-block bootstrap
  was used, but its interval is interpreted only alongside the fixed effect-size
  threshold.

### Fallacy scan

- **Coverage:** 11/11 statistical fallacy types checked

| Fallacy | Severity | Assessment |
|---|---|---|
| Simpson's paradox | NOTE | Dataset-stratified results are shown; their magnitude differs and is not hidden by a macro average. |
| Ecological fallacy | NOTE | Claims remain at dataset/checkpoint level; no per-variable or individual inference is made. |
| Berkson's paradox | NOTE | Checkpoints are validation-selected, so sensitivity is not claimed for unselected models. |
| Collider bias | NOTE | No learned covariate adjustment or conditioning variable was added. |
| Base-rate neglect | NOTE | No classification probability or diagnostic base rate is involved. |
| Regression to the mean | NOTE | No extreme-origin subgroup or pre/post improvement claim is used. |
| Survivorship bias | NOTE | All 2,689 ordered validation origins were retained for each dataset. |
| Look-elsewhere effect | NOTE | Families and thresholds were fixed before running; no favorable ETTh1 family was selected afterward. |
| Garden of forking paths | NOTE | Plan, interventions, normalization, thresholds, and stop rule were archived before results. |
| Correlation != causation | CAUTION | The frozen intervention measures functional dependence of this checkpoint, not the causal source of dataset differences or retrained performance. |
| Reverse causality | NOTE | No directional observational relation between external variables is asserted. |

## Reproducibility

- **Method:** full deterministic rerun to `/tmp/graphmamba_selective_dynamics_repeat`
- **Verdict:** REPRODUCIBLE
- **Comparison:** structured summaries are exactly equal after removing only
  the configured output-directory string.

## Artifacts

- Preregistered plan: `experiment_results/GraphMamba_selective_dynamics_diagnostic_plan.md`
- Diagnostic: `scripts/diagnose_graphmamba_selective_dynamics.py`
- Summary: `logs/graphmamba_selective_dynamics/summary.json`
- Dataset records: `logs/graphmamba_selective_dynamics/ETTh1_p192.json` and
  `logs/graphmamba_selective_dynamics/ETTh2_p192.json`

## Primary prior-art sources

- Mamba: https://arxiv.org/abs/2312.00752
- MambaSL: https://openreview.net/forum?id=YDl4vqQqGP
- MambaTS: https://arxiv.org/abs/2405.16440
- ms-Mamba: https://arxiv.org/abs/2504.07654
- Repetitive Contrastive Learning: https://arxiv.org/abs/2504.09185
- Bi-Mamba+: https://arxiv.org/abs/2404.15772
- MambaMixer: https://arxiv.org/abs/2403.19888
- Chimera: https://arxiv.org/abs/2406.04320
