# GraphMamba HPMRG revision log

Date opened: 2026-08-13

## V0: frozen-backbone counterfactual diagnostic

Status: executed, invalid estimator; preserved as a negative audit artifact.

### Model changes

None. V0 does not modify `models/GraphMamba.py`, add trainable model parameters,
or train a checkpoint. It loads the accepted seed-2021 periodic adapter-only
GraphMamba checkpoints for ETTh1/ETTh2-192 with `strict=False` only to discard
keys belonging to previously retired alignment/router modules. Missing active
keys or any other unexpected key are fatal.

### Diagnostic estimator

- Target: frozen GraphMamba forecast residual divided by each input window's
  standard deviation.
- Source features per variable: last normalized value, local-4 mean,
  period-24 mean, and last-minus-period-lag difference.
- Cross-variable design excludes the target variable's own four summaries.
- D1: one coefficient bank shared over horizons/phases.
- D2: four chronological forecast-distance bins.
- D3: six known future daily-phase bins.
- D4: the 4 x 6 interaction cells.
- D4-permutation: D4 fitted after shuffling training-origin phase anchors, then
  evaluated against true validation phases.
- One ridge coefficient is selected from a fixed six-value grid using the last
  20% of training origins; final coefficients refit on all training origins.
- Evaluation uses validation only. Moving-block bootstrap uses 24-origin blocks
  and 1,000 fixed-seed replicates.

### Preregistered interpretation

This is a representational upper-bound diagnostic, not an end-to-end HPMRG
result. Passing permits a separate model candidate; failure forbids model
implementation on this route. No test split may be read.

### V0 result and audit failure

V0 completed both datasets, but its D0 normalized residual MSE was 4095.84 on
ETTh1 and 11.14 on ETTh2. Self-review traced this to a duplicated normalization:
the data loader already standardizes each channel using training statistics,
then V0 divided the frozen forecast residual by each individual input window's
standard deviation. Nearly constant windows were therefore given unbounded
weight. The D1--D4 comparisons from V0 are invalid and cannot be used for a
go/no-go decision.

The complete invalid output is retained under
`logs/graphmamba_hpmrg_diagnostic/v0_invalid_window_rescaling/`.

## V1: globally standardized residual correction

Status: executed, independently reproduced, and valid; diagnostic gate failed.

### Change from V0

- Removed only the second, per-window residual division.
- Residuals remain in the data loader's training-standardized channel units.
- Added a fatal reproducibility check requiring D0 MSE to match the frozen
  checkpoint's recorded element-weighted validation MSE within `1e-5` relative
  error before any D1--D4 result is accepted.
- No model, checkpoint, feature, condition, ridge grid, split, bootstrap, or
  preregistered decision threshold changed.

### V1 results

- D0 reproduces frozen validation MSE with relative error `2.40e-10` on ETTh1
  and `1.22e-9` on ETTh2.
- ETTh1 D4 versus D1: `-0.7173%` MSE; block-bootstrap interval
  `[-0.8120%, -0.5995%]`.
- ETTh2 D4 versus D1: `-4.4666%` MSE; block-bootstrap interval
  `[-4.7767%, -4.1671%]`.
- D4 loses to D2 and D3 on both datasets; the phase permutation control does
  not remove its behavior.
- Macro D4-versus-D1 change: `-2.5919%`; implementation gate: failed.

### Reproducibility rerun

The identical V1 command was rerun with only `--output-dir` redirected to a
temporary directory. Ignoring that path field, `summary.json`,
`ETTh1_p192.json`, and `ETTh2_p192.json` are byte-for-value identical after
JSON parsing. The point-estimate no-go decision is reproducible; the later
ordering audit separately withdraws the serial-dependence interval.

### Decision

Stop HPMRG before model implementation. There is no V2 architecture, no new
checkpoint, and no test access. Complete interpretation and statistical audit
are recorded in `GraphMamba_HPMRG_diagnostic_result.md`.

## Post-audit ordering correction

While preparing the later PCRF diagnostic, inspection of `data_provider` showed
that both train and validation loaders shuffle unless the split is test. HPMRG's
point MSE/MAE values are order-invariant and exactly reproducible, but its
claimed chronological training holdout and moving-block bootstrap ordering were
not valid. The dependence interval is withdrawn. HPMRG remains a no-go because
D4 is pointwise worse than D1 on both datasets and independently fails the
magnitude, D2/D3, and phase-permutation gates. Future diagnostics explicitly
rebuild loaders with `shuffle=False`.
