# CMRHM-v1 strict evidence audit plan

Date: 2026-08-14

## Status and interpretation

This is a post-hoc robustness audit requested after the original ETTm test was
consumed. CMRHM-v1 is frozen. All new runs are validation-only and cannot make
the historical test split unconsumed again. Results may strengthen or weaken
the evidence, but the experiment is not designed to guarantee a positive result.

Frozen mechanism: input 336, recent backbone window 96, old history 240,
average-pool 16 (15 memory tokens), hidden size 32, paired decoder difference,
and variable-wise bounded gate. No result-dependent retry or tuning is allowed.

## Group A — multi-seed paired stability

- Tasks: ETTm1/ETTm2 × horizons 96/192/336/720.
- Seeds: 2021, 2022, 2023.
- Pair: `GraphMambaRecent` vs frozen `GraphMambaCMRHM`.
- Seed 2021 records are reused; seeds 2022/2023 require 32 new runs.

Pass gate:

1. CMRHM wins MSE on at least 20/24 paired tasks;
2. task-level macro MSE improvement is at least 1%;
3. neither dataset has a negative seed-level four-horizon macro MSE improvement;
4. MAE wins on at least 18/24 tasks and macro MAE does not worsen.

Report paired mean, sample standard deviation across seeds, win counts, and all
individual records. No test evaluation follows this audit.

## Group B — long-input capacity control

- Tasks: ETTm1/ETTm2 × horizons 96/192/336/720, seed 2021.
- Same loader window and forecast target for all models.
- `Recent336`: receives 336 but backbone uses only the most recent 96.
- `Raw336`: standard GraphMamba directly processes all 336 observations.
- `CMRHM336`: recent-96 backbone plus frozen compressed old-history branch.
- Eight Raw336 runs are new; the other records are reused.

Pass gate for the stronger efficiency claim:

1. CMRHM beats Raw336 MSE on at least 6/8 tasks;
2. macro MSE improves by at least 1%;
3. macro MAE does not worsen;
4. parameters and runtime are reported rather than hidden.

Failure does not erase the proven value over Recent336; it limits the claim to
an efficient recent-backbone enhancement rather than superiority to raw context.

## Group C — mechanism ablation

- Tasks: ETTm1/ETTm2 × horizons 96 and 720, seed 2021.
- Full CMRHM records are reused; 12 new runs cover three ablations.

Ablations:

1. `Concat`: concatenate recent context and compressed memory, decode a residual,
   retaining the zero-initialized variable gate. This is a higher-parameter naive
   old-history integration control.
2. `NoDiff`: use `D(gelu(z+m))` without subtracting `D(gelu(z))`; parameters and
   initialization match full CMRHM.
3. `GlobalGate`: replace variable-specific gate values by their shared mean;
   parameters and initialization match full CMRHM.

Component support gate: full CMRHM must beat each ablation on at least 3/4 MSE
tasks, improve each ablation's macro MSE by at least 0.5%, and not worsen macro
MAE. Any failed component is described as unsupported rather than silently kept
in the novelty claim.

## Integrity rules

- Validation-only: every new command has `test_after_train=0`.
- One implementation revision before results; no result-driven edits.
- Sequential GPU execution, atomic JSON records, per-run logs, resumable tmux.
- Continue after isolated run failure and preserve failure records.
- Final report separates previously consumed test evidence from this new
  validation robustness audit.

