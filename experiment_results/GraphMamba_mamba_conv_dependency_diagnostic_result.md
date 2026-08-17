# GraphMamba frozen Mamba-convolution dependency diagnostic

Date: 2026-08-14

## Scope

- Accepted periodic GraphMamba checkpoints: ETTh1-192 and ETTh2-192, seed 2021.
- Ordered validation only; 2,689 origins per dataset; test was never built.
- Every learned parameter remained frozen.
- Mamba-1 was evaluated through an explicit selective-scan path so the temporal
  part of its depthwise convolution could be intervened on separately for the
  local and period patch branches.

## Interventions

For `d_conv=2`, the past-lag tap is the first kernel coefficient and the current
tap is the last. Each removal keeps the learned current tap, convolution bias,
SiLU, delta/B/C/A/D, bidirectional scans, graph branch, and prediction head.

| ID | Local branch | Period branch |
|---|---|---|
| E0 | full convolution | full convolution |
| EL0 | current tap only | full convolution |
| EP0 | full convolution | current tap only |
| EB0 | current tap only | current tap only |

## Structural reproduction

| Dataset | Fused vs explicit first-batch max abs | E0 checkpoint-MSE relative error |
|---|---:|---:|
| ETTh1 | `3.5763e-7` | `1.0546e-9` |
| ETTh2 | `2.3842e-7` | `1.3619e-9` |

Both are comfortably below the fatal limits (`2e-5` output and `1e-5` MSE), so
the explicit path is an exact-enough counterfactual implementation.

## Internal normalized convolution dependency

The temporal-component RMS is divided by the full activated convolution RMS
per origin, then averaged over forward/backward Mamba modules.

| Dataset | Local | Period | Relative difference | 95% ordered-block CI, local minus period | 10% distinction gate |
|---|---:|---:|---:|---:|---:|
| ETTh1 | `0.574671` | `0.548336` | `4.583%` | `[0.026136, 0.026539]` | fail |
| ETTh2 | `0.532158` | `0.524645` | `1.412%` | `[0.007159, 0.007864]` | fail |

The differences are statistically stable but too small for the preregistered
effect-size condition. Statistical detectability across 2,689 overlapping
origins is not treated as architectural importance.

## Frozen forecast counterfactuals

| Dataset | Mode | MSE | MAE | MSE change vs E0 | Forecast perturbation / E0 forecast RMS |
|---|---|---:|---:|---:|---:|
| ETTh1 | E0 | `0.988814139` | `0.652795541` | — | — |
| ETTh1 | EL0 | `1.016163995` | `0.659011450` | `+2.766%` | `16.325%` |
| ETTh1 | EP0 | `0.988785918` | `0.652403370` | `-0.0029%` | `1.524%` |
| ETTh1 | EB0 | `1.019623359` | `0.660295527` | `+3.116%` | `17.308%` |
| ETTh2 | E0 | `0.272636507` | `0.351919733` | — | — |
| ETTh2 | EL0 | `0.271578928` | `0.352208862` | `-0.388%` | `2.486%` |
| ETTh2 | EP0 | `0.272872749` | `0.351754350` | `+0.0867%` | `0.788%` |
| ETTh2 | EB0 | `0.271697964` | `0.351950727` | `-0.344%` | `2.428%` |

These frozen removals are deliberately not interpreted as trained-candidate
performance. Local-branch output perturbation is larger partly because the head
receives many more local tokens than period tokens. The primary distinction
gate therefore uses the per-origin internal normalized dependency, not raw
forecast perturbation.

## Gate and decision

The temporal convolution is material on both datasets, but its normalized
relative contribution does not differ by the required 10% between patch
scales. ETTh1 reaches 4.583% and ETTh2 only 1.412%. Therefore:

**NO-GO: do not implement or train the proposed physical-time continuous Mamba
convolution on this evidence.**

The physically different token strides remain a conceptual fact, but the
accepted model does not exhibit the hypothesized scale-dependent internal
convolution behavior strongly enough to justify a new module. This result also
prevents using the much larger local forecast perturbation as post-hoc support.

## Self-audit

1. Test split was not constructed.
2. Validation order was rebuilt explicitly with `shuffle=False`.
3. Accepted checkpoints and their recorded MSEs were reproduced.
4. Full explicit Mamba reproduced the fused path before intervention.
5. Past-lag removal retained current tap, bias, and activation.
6. Delta, B, C, A, D, graph, head, and all parameters stayed unchanged.
7. Forward and backward Mamba directions received the same branch intervention.
8. Internal dependency was normalized per origin before branch comparison.
9. The 10% effect threshold and block bootstrap were fixed before execution.
10. Raw output sensitivity was not used as the primary gate because branch token
    counts and head weights differ.
11. Frozen ablation was not presented as evidence that a retrained no-conv model
    would improve.
12. A corrected full rerun and a temporary rerun were exactly equal after only
    excluding the output-directory field.

## Artifacts

- Diagnostic: `scripts/diagnose_graphmamba_mamba_conv_dependency.py`
- Summary: `logs/graphmamba_mamba_conv_dependency/summary.json`
- Per-dataset JSON: `logs/graphmamba_mamba_conv_dependency/ETTh1_p192.json`,
  `logs/graphmamba_mamba_conv_dependency/ETTh2_p192.json`
- Literature/code audit:
  `experiment_results/GraphMamba_internal_mamba_literature_code_audit.md`
