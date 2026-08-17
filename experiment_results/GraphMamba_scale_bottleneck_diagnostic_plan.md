# GraphMamba scale-bottleneck attribution plan

Date: 2026-08-14

## Material Passport

- ID: `GraphMamba-SCALE-ATTR-D0`
- Type: preregistered frozen-checkpoint diagnostic
- Status: fixed before output inspection
- Scope: accepted ETTh1/ETTh2-192 checkpoints; evenly spaced ordered validation batches
- Forbidden: test access, training, architecture edits, treating gradient diagnostics as innovation

## Competing explanations

1. **Shared-core conflict:** local and period paths ask the same Mamba weights to
   move in opposing directions.
2. **Period-role under-identification:** gradients are not opposed, but the
   unconstrained `48 local + 8 period -> FlattenHead` solution gives the period
   branch little marginal predictive responsibility.
3. **Neither:** both paths are used with aligned, comparable optimization
   pressure; the next bottleneck lies elsewhere.

## Exact measurements

- Decompose the full validation-loss gradient with respect to shared Mamba
  parameters into its local-encoder and period-encoder computational paths.
- Per batch, record gradient cosine and period/local gradient-norm ratio.
- Decompose the linear head exactly into local and period marginal outputs after
  temporal-plus-graph fusion; remove its shared bias from both marginals.
- Record period/local marginal-output RMS, head Frobenius norms, and frozen
  branch-removal MSE. Branch removals are sensitivity tests, not trained models.
- If whole-core conflict passes, decompose the same path gradients into fixed
  parameter groups: input/gate projection, temporal convolution, selective
  projection (`x_proj` + `dt_proj`), state generator (`A_log`), skip (`D`),
  output projection, and block norm/FFN.

## Fixed attribution rules

- **Core conflict** requires, on both datasets: mean gradient cosine `< -0.10`,
  its 95% batch-bootstrap interval below zero, and period/local gradient norm
  ratio in `[0.10, 10]`.
- **Period-role under-identification** requires, on both datasets: gradient
  cosine not meeting conflict, period/local gradient norm ratio `< 0.25`, and
  period/local marginal-output RMS `< 0.25`.
- Otherwise the result is mixed and no locus is selected.
- A shared-generator/scale-interface candidate is admissible only if `A_log`
  does not meet the `< -0.10` conflict criterion on either dataset, while at
  least one interface group meets it on both. This component rule is fixed
  before component outputs are inspected.

The thresholds are practical attribution heuristics, not statistical laws. A
selected locus only authorizes operation-level prior-art search and a new
candidate diagnostic.

## Expected artifacts

- Script: `scripts/diagnose_graphmamba_scale_bottleneck.py`
- Raw results: `logs/graphmamba_scale_bottleneck/summary.json`
- Decision: `experiment_results/GraphMamba_scale_bottleneck_diagnostic_result.md`
