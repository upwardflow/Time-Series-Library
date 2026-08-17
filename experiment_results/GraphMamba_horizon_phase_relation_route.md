# GraphMamba prediction-horizon and periodic-phase conditioned relation route

Date: 2026-08-13

## Status and scope

This document is a preregistered research and modification route, not an
implemented model. The working name is **HPMRG**: Horizon--Phase Marginal
Relation Graph. No architecture is to be added until the diagnostic gate in
Stage 1 passes.

CMRHM-v1 remains frozen. The first HPMRG candidate is evaluated on GraphMamba
without CMRHM so that long-history memory and future-conditioned variable
relations are not changed in the same experiment.

## Research question

Does the marginal predictive contribution of variable `j` to variable `i`
change jointly with:

1. forecast distance `h`; and
2. the known periodic phase at target time `t+h`?

The null is that one relation graph shared by every future step is sufficient.
The alternative is not merely "a dynamic graph": it is that the useful
cross-variable residual after the current GraphMamba forecast has a stable,
out-of-sample horizon--phase interaction.

## Conditions without target leakage

For forecast step `h = 1, ..., H`, define the normalized distance

`r_h = h / H`.

Define future phase from the known decoder timestamp, never from future target
values. For a daily period:

`phi_(t+h) = 2 pi * time_of_day(t+h) / one_day`.

The condition is

`c_h = [r_h, sin(phi_(t+h)), cos(phi_(t+h))]`.

ETT hourly data recover time of day from `HourOfDay`; ETT minute data recover
it from `HourOfDay` and `MinuteOfHour`. These features are already present in
`x_mark_dec`. Dataset-specific phase extraction must be explicit and tested;
an ambiguous frequency must fail rather than silently assume a daily phase.

## Stage 1: read-only counterfactual diagnosis

### Data split

- Use training data to fit all diagnostic regressions and graphs.
- Use validation data for the preregistered comparison.
- Do not access a test split.
- Primary datasets: ETTh1 and ETTh2.
- External diagnostic confirmation: Electricity and Weather if their timestamp
  frequency and dominant-period record pass the same training-only checks.
- ETTm1/ETTm2 test results already consumed by CMRHM must not influence model
  selection. They may be used only in a later frozen validation study if needed.

### Factorial controls

Fit the same regularized residual predictor with five relation conditions:

| ID | Cross-variable information | Conditioning |
|---|---|---|
| D0 | none | own-variable history only |
| D1 | yes | one graph shared over all future steps |
| D2 | yes | horizon distance only |
| D3 | yes | future periodic phase only |
| D4 | yes | horizon distance and future phase interaction |

All models use identical lag features, regularization selection, training rows,
and validation rows. Hyperparameters are selected within the training split by
chronological folds. D4 is compared primarily with D1, not with D0, because the
claim concerns conditional relations rather than the generic value of other
variables.

### Diagnostic outputs

- Per-horizon and per-phase-bin validation MSE/MAE.
- D1--D4 aggregate validation differences.
- Edge-weight similarity across horizon and phase bins.
- Edge sign/direction stability across four chronological training blocks.
- Moving-block bootstrap confidence intervals over validation origins.
- A label-permutation negative control for the phase condition.

### Go/no-go gate

Implement HPMRG only if all are true:

1. D4 improves validation MSE over D1 by at least 1% on at least two datasets.
2. The macro MSE improvement is at least 1%, with no dataset worse than 0.5%.
3. The 95% moving-block-bootstrap interval for the macro improvement excludes
   zero.
4. D4 improves over both D2 and D3, showing an interaction rather than only a
   horizon or phase main effect.
5. Phase-label permutation removes at least half of D4's gain.
6. At least one non-ETT dataset confirms the direction before any test access.

Failure means stop this route and preserve the diagnostic as a negative result.

## Stage 2: minimal model candidate

### Existing backbone remains the control

Let the accepted GraphMamba produce its unchanged base forecast `Y_base`.
Expose the pre-graph temporal state only as a read source:

`Z in R^(B x N x D)`.

For periodic dual-patch GraphMamba, `Z` is obtained by pooling the concatenated
local and period temporal states after their independent shared-Mamba scans.
The existing static/adaptive graph branch and prediction head remain unchanged,
so a zero HPMRG gate gives exact baseline behavior.

### Conditioned residual graph

Use a low-rank condition-modulated node representation:

`q_i(h) = Q(z_i) * (1 + gamma_q(c_h))`

`k_j(h) = K(z_j) * (1 + gamma_k(c_h))`

and

`A_ij(h) = masked_softmax_j(q_i(h)^T k_j(h) / sqrt(r))`.

The mask is the union of the training-derived static top-k graph and self edges.
This keeps complexity `O(B H N K r)` rather than dense `O(B H N^2 r)` on large
datasets. The first implementation uses no free per-horizon adjacency tensor.

The horizon-specific message is

`m_i(h) = sum_j A_ij(h) V(z_j)`.

### Marginal paired decoder

Following the identification principle that made CMRHM defensible, graph
influence is expressed as a paired marginal difference through one shared
decoder:

`DeltaY_i(h) = D(z_i + m_i(h), c_h) - D(z_i, c_h)`.

The candidate forecast is

`Y_hat_i(h) = Y_base_i(h) + tanh(g_i) DeltaY_i(h)`.

`g_i` is a zero-initialized variable gate. At initialization, enabled and
disabled models must have identical shared parameters, train-mode outputs,
dropout RNG trajectories, and shared gradients. HPMRG is a decoder-side
marginal correction; it does not inject graph context into Mamba state updates,
which separates it from graph-conditioned selective-state approaches.

## Stage 3: mandatory ablation order

One mechanism is enabled at a time, with identical seeds and validation-only
selection:

1. `G0`: accepted GraphMamba baseline.
2. `G1`: paired marginal graph with no horizon/phase condition.
3. `G2`: horizon-conditioned paired graph.
4. `G3`: phase-conditioned paired graph.
5. `G4`: joint horizon--phase paired graph (HPMRG).
6. `G5`: G4 with shuffled phase labels (negative control).
7. `G6`: G4 with separate unshared with/without decoders (identification
   ablation; expected to be less interpretable).

G4 must beat G1, G2, and G3. Beating only G0 would support an extra graph head,
not the horizon--phase hypothesis.

## Stage 4: validation sequence

### Structural gate

- Exact zero-gate equivalence in train and eval modes.
- Nonzero finite gradients for condition modulation and the graph gate.
- Future timestamps change adjacency; future target values are never inputs.
- Shifting all timestamps by one full period leaves the adjacency unchanged.
- Holding phase fixed while changing `h/H` changes only the horizon path.
- Holding `h/H` fixed while changing phase changes only the phase path.
- Sparse complexity and memory accounting match the implementation.

### Predictive gate

- First seed: two development datasets, G0--G4 only.
- Continue only if G4 improves both datasets and macro MSE by at least 1% with
  MAE not worsening by more than 0.2%.
- Second seed plus negative control only after the first gate.
- Confirm on at least one non-ETT dataset.
- Report mean, standard deviation, parameter count, peak GPU memory, training
  time, graph entropy, gate magnitude, and condition sensitivity.
- Test is accessed once only after architecture, seed count, and all
  hyperparameters are frozen.

## Integration with CMRHM

CMRHM integration is a later factorial experiment, not part of HPMRG selection:

| Backbone | HPMRG | CMRHM | Purpose |
|---|---|---|---|
| GraphMamba | off | off | base |
| GraphMamba | on | off | relation contribution |
| GraphMamba | off | on | memory contribution |
| GraphMamba | on | on | complementarity/interaction |

The combined model is retained only if its gain is not explained by one module
alone and neither learned gate collapses toward zero.

## Planned code boundaries after Stage 1 passes

- `scripts/diagnose_horizon_phase_relation_bound.py`: D0--D4 diagnostic,
  bootstrap, permutation control, and machine-readable output.
- `layers/GraphMambaHPMRG_Layers.py`: sparse condition graph and paired decoder.
- `models/GraphMambaHPMRG.py`: opt-in candidate that preserves GraphMamba as the
  unchanged control.
- `scripts/run_graphmamba_hpmrg_validation.py`: strict paired experiment matrix.
- `experiment_results/GraphMamba_HPMRG_*`: preregistered design, diagnostics,
  validation, and retirement/acceptance records.

Do not initially place HPMRG directly inside `models/GraphMamba.py`. A separate
candidate file avoids changing the accepted baseline before the gate passes.

## Novelty boundary

Not supportable:

- first horizon-aware graph;
- first dynamic graph forecaster;
- first periodic graph or graph--Mamba model.

Potentially supportable after evidence:

> A sparse decoder-side relation graph whose edge utility is jointly
> conditioned on prediction distance and known future periodic phase, with the
> graph contribution identified by a shared paired marginal decoder rather than
> entangled with the temporal state transition.

The paired marginal identification and the demonstrated horizon--phase
interaction must both be present; otherwise this is only another conditional
dynamic graph.
