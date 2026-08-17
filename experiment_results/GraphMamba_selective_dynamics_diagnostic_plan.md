# GraphMamba frozen selective-dynamics diagnostic plan

Date: 2026-08-14

## Material Passport

- ID: `GraphMamba-SDYN-D0`
- Type: preregistered code-experiment plan
- Status: planned before diagnostic output inspection
- Scope: accepted ETTh1/ETTh2-192 checkpoints, seed 2021, ordered validation only
- Forbidden scope: model training, test construction, accepted-model edits, novelty claims from the diagnostic itself

## Question

After branch-normalizing the response, does the accepted shared Mamba use one of
its three selective quantities (`delta`, `B`, or `C`) differently on the local
patch grid and the complete-period patch grid, consistently across ETTh1 and
ETTh2?

This is a bottleneck diagnostic. It is not a proposed architecture.

## Prior-art exclusion contract

The following claims and corresponding direct implementations are excluded:

| Occupied mechanism | Prior work | Consequence here |
|---|---|---|
| Input-dependent `delta`, `B`, and `C` | Mamba | Vanilla selection is not new. |
| Time-varying versus time-invariant `delta/B/C` switches | MambaSL | Freezing or flattening a selective family is not a module contribution. |
| Dropout on selective parameters | MambaTS | Selective dropout is not a candidate. |
| Multiple sampling-rate/discretization multipliers | ms-Mamba | Scale-specific `delta` is excluded; GraphMamba V3 also failed locally. |
| Generic selectivity enhancement and selectivity metrics | RCL | A focus/selectivity loss or metric is excluded without a distinct mechanism. |
| Mamba forget gate | Bi-Mamba+ | A new forget gate is excluded. |
| Token/channel or two-dimensional selection | MambaMixer and Chimera | Another scan axis is excluded as the next claim. |

No third-party code will be copied. The diagnostic uses the installed official
Mamba-1 equations and the repository's own frozen-checkpoint loader.

## Counterfactuals

For a branch and one selective family, compute the normal explicit Mamba
quantities, then replace only their token axis by the per-sequence temporal mean:

- `delta_flat`: replace positive `softplus(delta + bias)` by its temporal mean;
- `B_flat`: replace `B_t` by its temporal mean;
- `C_flat`: replace `C_t` by its temporal mean.

The replacement preserves each sequence's learned level and removes only its
within-sequence timing. The other two selective families, convolution, `A`, `D`,
gate `z`, bidirectionality, residual/FFN, graph branch, and head remain frozen.
Each family is intervened on local and period branches separately.

## Outcomes

Primary per-origin outcome:

`encoder_relative_RMS = RMS(E_intervened - E0) / max(RMS(E0), 1e-8)`

where `E` is the corresponding branch output of the complete shared Mamba
encoder before graph fusion and the prediction head. RMS is taken over variables,
features, and branch tokens, so unequal branch token counts do not directly sum
into the comparison.

Secondary outcomes are frozen forecast perturbation, MSE/MAE, and correlation
between per-origin internal response and baseline forecast error. These cannot
override the primary gate.

## Preregistered gate

A selective family advances to a post-result novelty search only when all hold:

1. Structural reproduction: first-batch fused/explicit maximum absolute error
   is at most `2e-5`, and explicit validation MSE relative error is at most
   `1e-5`.
2. Materiality: at least one branch has mean encoder relative RMS of `2%` or
   more on each dataset.
3. Branch distinction: `abs(local-period)/max(local,period) >= 25%` and the
   paired moving-block-bootstrap 95% interval for `local-period` excludes zero.
4. Cross-dataset consistency: the same family passes on both ETTh1 and ETTh2
   with the same sign of `local-period`.
5. Repeatability: an independent repeat is equal apart from output paths.

Failure means no candidate module, no training, and an archived no-go. Passing
does not establish novelty: it only authorizes an exact operation-level search.

## Expected artifacts

- Diagnostic: `scripts/diagnose_graphmamba_selective_dynamics.py`
- Raw JSON: `logs/graphmamba_selective_dynamics/`
- Final decision: `experiment_results/GraphMamba_selective_dynamics_diagnostic_result.md`

