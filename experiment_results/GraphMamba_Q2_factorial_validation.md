# GraphMamba Q2 periodic × CMRHM factorial validation

## Protocol

- Validation only; no test evaluation.
- ETTh1/ETTh2 × horizons 192/720 × seeds 2021/2022/2023.
- All variants load 336 points; all backbones process only the recent 96 points; only CMRHM reads the old 240 points.
- `b`: independent dual patches; `p`: periodic multi-resolution backbone; `c`: `b` + CMRHM; `pc`: `p` + CMRHM.

## Paired results

| Dataset | H | Seed | B MSE | P MSE | C MSE | PC MSE | P→PC | C→PC | Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 192 | 2021 | 1.005031 | 0.985599 | 0.967495 | 0.957221 | +2.879% | +1.062% | -0.009158 |
| ETTh1 | 192 | 2022 | 0.999177 | 0.991084 | 0.968737 | 0.963396 | +2.794% | +0.551% | -0.002753 |
| ETTh1 | 192 | 2023 | 0.998083 | 0.994693 | 0.965605 | 0.965807 | +2.904% | -0.021% | -0.003592 |
| ETTh1 | 720 | 2021 | 1.584068 | 1.561051 | 1.478304 | 1.463210 | +6.268% | +1.021% | -0.007923 |
| ETTh1 | 720 | 2022 | 1.634904 | 1.595974 | 1.538957 | 1.513851 | +5.146% | +1.631% | -0.013824 |
| ETTh1 | 720 | 2023 | 1.610749 | 1.561997 | 1.516282 | 1.504248 | +3.697% | +0.794% | -0.036718 |
| ETTh2 | 192 | 2021 | 0.271975 | 0.274951 | 0.285617 | 0.293617 | -6.789% | -2.801% | -0.005024 |
| ETTh2 | 192 | 2022 | 0.274377 | 0.270571 | 0.293027 | 0.281253 | -3.948% | +4.018% | +0.007968 |
| ETTh2 | 192 | 2023 | 0.269957 | 0.272869 | 0.277268 | 0.291955 | -6.995% | -5.297% | -0.011775 |
| ETTh2 | 720 | 2021 | 0.613978 | 0.602762 | 0.638779 | 0.614428 | -1.936% | +3.812% | +0.013134 |
| ETTh2 | 720 | 2022 | 0.620414 | 0.615834 | 0.649174 | 0.648837 | -5.359% | +0.052% | -0.004243 |
| ETTh2 | 720 | 2023 | 0.606799 | 0.601642 | 0.627805 | 0.609994 | -1.388% | +2.837% | +0.012654 |

## Aggregate readout

- `periodic_without_memory_mse_pct`: mean +1.047750; positive 10/12.
- `periodic_with_memory_mse_pct`: mean +0.638281; positive 9/12.
- `memory_without_periodic_mse_pct`: mean +0.148963; positive 6/12.
- `memory_with_periodic_mse_pct`: mean -0.227218; positive 6/12.
- `factorial_interaction_mse`: mean -0.005104; positive 3/12.
- `full_vs_best_single_mse_pct`: mean -1.781294; positive 5/12.
- `periodic_without_memory_mae_pct`: mean +0.819347; positive 11/12.
- `periodic_with_memory_mae_pct`: mean +0.620383; positive 10/12.
- `memory_without_periodic_mae_pct`: mean -1.112072; positive 6/12.
- `memory_with_periodic_mae_pct`: mean -1.311296; positive 6/12.
- `factorial_interaction_mae`: mean -0.000950; positive 4/12.
- `full_vs_best_single_mae_pct`: mean -1.640922; positive 6/12.

## Preregistered gate

- `memory_compatible_with_periodic`: FAIL
- `periodic_adds_with_memory`: PASS
- `interaction_not_materially_negative`: FAIL
- `memory_mae_not_systematically_worse_with_periodic`: FAIL

## Boundary

This factorial determines whether the two frozen contributions coexist under one protocol. It does not authorize model tuning from consumed test results and does not establish cross-dataset periodic generality.
