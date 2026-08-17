# GraphMamba scale-bottleneck attribution result

Date: 2026-08-14

## Material Passport

- Origin Skill: academic-research-suite + experiment-agent
- Verification Status: VERIFIED WITH NUMERICAL TOLERANCE
- Version: `GraphMamba-SCALE-ATTR-D0-v1`
- Scope: frozen ETTh1/ETTh2-192 checkpoints, 32 evenly spaced ordered-validation batches
- Test access: none
- Accepted-model edit: none

## Decision

The next bottleneck is the shared temporal core, not simple period-head
suppression. Local and period computational paths produce opposing gradients on
the same Mamba parameters on both datasets.

| Dataset | Local/period gradient cosine | 95% bootstrap interval | Period/local gradient norm | Conflict gate |
|---|---:|---:|---:|---|
| ETTh1 | `-0.282` | `[-0.379,-0.193]` | `0.181` | pass |
| ETTh2 | `-0.374` | `[-0.448,-0.295]` | `0.352` | pass |

The inputs are the same validation examples for both paths, so the comparison
does not conflate branch gradients with different sample distributions.

## Head and branch checks

| Dataset | Period/local marginal forecast RMS | Total head-weight norm ratio | Per-token head-weight RMS ratio | Remove period MSE change |
|---|---:|---:|---:|---:|
| ETTh1 | `0.200` | `0.410` | `1.005` | `+1.566%` |
| ETTh2 | `0.278` | `0.414` | `1.013` | `+0.571%` |

The total period norm is smaller because there are 8 period versus 48 local
tokens. Per token, the head weights are almost equal. The period path is weaker
but is not trivially suppressed by the linear head.

## Component localization

| Parameter group | ETTh1 cosine | ETTh2 cosine | Common conflict? |
|---|---:|---:|---|
| Input/gate projection | `-0.242` | `-0.395` | yes |
| Temporal convolution | `-0.269` | `-0.488` | yes |
| Selective projection | `+0.131` | `-0.400` | no |
| State generator `A_log` | `-0.092` | `+0.033` | **no** |
| Skip `D` | `-0.165` | `-0.562` | no under strict two-dataset rule |
| Output projection | `-0.320` | `-0.410` | yes |
| Norm/FFN | `-0.325` | `-0.357` | yes |

The result supports sharing the memory-mode generator while adapting the
coordinates in which different physical patch scales enter and leave the core.
It does not support generic gradient surgery as a model contribution.

## Reproducibility and limitations

- A complete repeat returned the same attribution and identical forward
  quantities.
- CUDA backward summaries differed only at relative `4.1e-10` or less; the run
  is tolerance-reproducible, not bitwise identical.
- Gradients are measured at accepted validation-selected checkpoints, not over
  the full training trajectory. Phase 24 must show that reducing conflict also
  improves held-out forecasting; conflict alone is not a quality metric.
- Frozen branch removal is not interpreted as retrained ablation performance.

## Artifacts

- Plan: `experiment_results/GraphMamba_scale_bottleneck_diagnostic_plan.md`
- Script: `scripts/diagnose_graphmamba_scale_bottleneck.py`
- Raw result: `logs/graphmamba_scale_bottleneck/summary.json`

