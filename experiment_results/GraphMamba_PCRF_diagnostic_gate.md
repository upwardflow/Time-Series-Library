# GraphMamba PCRF preregistered diagnostic gate

Date fixed: 2026-08-13, before running PCRF outputs.

## Frozen scope

- Backbones: accepted periodic GraphMamba ETTh1/ETTh2, prediction length 192,
  seed 2021.
- Fit: training split only, with the chronological last 20% of training origins
  used to select ridge strength.
- Evaluation: validation split only.
- Test: prohibited.
- Period: frozen at 24; no period, reliability, bin, or window search.

## Reliability observables

For every input window and variable, calculate from the already normalized,
decomposed history:

- `cycle_consistency`: cosine similarity between the latest seasonal cycle and
  the preceding cycle, using exactly period 24;
- `trend_roughness`: mean squared second difference of the trend, divided by
  trend first-difference energy plus epsilon, then `log1p` transformed.

Both are causal because they use observed history only. Features are
standardized with training statistics and applied unchanged to validation.

## Factorial controls

- D0: accepted forecast, no correction.
- D1: static per-variable seasonal/trend contribution correction.
- D2: D1 plus seasonal contribution conditioned on cycle consistency.
- D3: D1 plus trend contribution conditioned on trend roughness.
- D4: D1 plus both reliability-conditioned contributions and their cross
  reliability terms.
- D4-perm: D4 with training-origin reliabilities shuffled before fitting,
  evaluated with true validation reliabilities.

All candidates use the same fixed ridge grid
`{1e-4, 1e-3, 1e-2, 1e-1, 1, 10}`. No neural module is trained at this stage.

## Go/no-go rule

PCRF model implementation is permitted only if all conditions pass:

1. D1 improves D0 MSE on both datasets and macro improvement is at least 1%.
2. D4 improves D1 MSE on both datasets and macro incremental improvement is at
   least 0.5%.
3. D4 improves D0 by at least 1% on both datasets.
4. D4 beats D2 and D3 on both datasets.
5. The 95% moving-block-bootstrap interval for D4 versus D1 is positive on both
   datasets; block length is 24 origins and repetitions are 1,000.
6. D4-perm removes at least half of D4's positive incremental improvement over
   D1 on both datasets.
7. D0 reproduces the frozen checkpoint validation MSE within `1e-5` relative
   error.

Failure means archive the diagnosis and do not add PCRF to GraphMamba. Passing
permits one separate zero-initialized model candidate; it does not permit test
access or combination with CMRHM.
