# TimeXer-336 vs TimeXer+DHC: Three-Seed Test Summary

## Protocol

- Datasets: ETTm1 and ETTm2
- Forecast horizons: 96 and 720
- Seeds: 2021, 2022, and 2023
- Baseline: native TimeXer directly processing all 336 input points
- Proposed variant: TimeXer+DHC, with a 96-point TimeXer path and a 240-point distant-history path
- Selection: one validation-best-MSE checkpoint per dataset, horizon, seed, and model; the test split was accessed only after all 24 validation records and checkpoints existed
- Metrics below are test-set mean +/- sample standard deviation over three seeds; lower is better

## Accuracy

| Dataset | Horizon | TimeXer-336 MSE | TimeXer+DHC MSE | MSE reduction | TimeXer-336 MAE | TimeXer+DHC MAE | MAE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTm1 | 96 | 0.3103 +/- 0.0095 | **0.2980 +/- 0.0019** | **3.88%** | 0.3544 +/- 0.0042 | **0.3505 +/- 0.0011** | **1.10%** |
| ETTm1 | 720 | 0.4346 +/- 0.0062 | **0.4310 +/- 0.0020** | **0.82%** | 0.4331 +/- 0.0026 | **0.4293 +/- 0.0012** | **0.87%** |
| ETTm2 | 96 | 0.1691 +/- 0.0009 | **0.1654 +/- 0.0022** | **2.20%** | 0.2550 +/- 0.0007 | **0.2541 +/- 0.0020** | **0.33%** |
| ETTm2 | 720 | 0.3802 +/- 0.0102 | **0.3754 +/- 0.0028** | **1.20%** | 0.3972 +/- 0.0081 | **0.3905 +/- 0.0040** | **1.67%** |

TimeXer+DHC wins all four dataset-horizon tasks on both three-seed mean MSE and mean MAE. Averaged across the four task-level relative reductions, DHC lowers MSE by 2.02% and MAE by 0.99%. At the individual-seed level, it wins 10/12 MSE comparisons and 8/12 MAE comparisons. With only three seeds, these results should be presented as consistent mean improvements rather than as a claim of statistical significance.

## Efficiency diagnostics

| Dataset | Horizon | Parameters: TimeXer-336 / DHC | DHC parameter reduction | Peak CUDA memory: TimeXer-336 / DHC | Mean inference ms/batch: TimeXer-336 / DHC |
|---|---:|---:|---:|---:|---:|
| ETTm1 | 96 | 2.212M / 1.789M | 19.14% | 35.7 / 24.9 MB | 4.85 / 5.47 |
| ETTm1 | 720 | 4.939M / 2.140M | 56.68% | 39.4 / 25.0 MB | 4.41 / 5.52 |
| ETTm2 | 96 | 2.212M / 1.789M | 19.14% | 120.6 / 53.5 MB | 6.10 / 6.45 |
| ETTm2 | 720 | 12.500M / 6.874M | 45.01% | 324.2 / 136.6 MB | 7.66 / 7.71 |

Within each matched task, DHC uses fewer parameters and lower peak allocated CUDA memory. Its per-batch inference latency is slightly higher, ranging from approximately 0.6% to 25.3%, so the defensible efficiency claim is parameter/memory efficiency rather than universal latency acceleration.

## Reproducibility checks

- Test records: 24/24 completed
- Matched pairs: 12/12 contain exactly one TimeXer-336 and one TimeXer+DHC record
- Checkpoint selection: validation-best MSE for every test record
- Pairwise command audit: no mismatches in data, history length, horizon, patch length, dimensions, optimization settings, batch size, worker count, or GPU; only the intended model/history-routing identity differs
- Detailed per-seed results: `paired_test_results.csv`
- Three-seed aggregate results: `mean_std.csv`
