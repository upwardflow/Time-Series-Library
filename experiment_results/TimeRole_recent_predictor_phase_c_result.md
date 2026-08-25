# TimeRole 近期预测器简化 Phase C 结果

更新时间：2026-08-26T00:38:19.056135+08:00

- 仅使用验证集；seed 2021 复用 Phase B，seeds 2022/2023 来自 Phase C。
- 比较 R0、R2、R4 的 TimeRole 与严格匹配 Recent-only，共 108 个记录。
- 通过全部九项门槛的候选：R2, R4。

| 候选 | 宏 MSE Δ | 宏 MAE Δ | R0 不劣胜场 | 历史收益 | 历史胜场 | 参数下降 | 延迟下降 | Phase C |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0 | +0.000% | +0.000% | 18/18 | +2.364% | 12/18 | +0.0% | +0.0% | FAIL |
| R2 | -1.084% | -0.429% | 14/18 | +2.388% | 12/18 | +65.0% | +40.0% | PASS |
| R4 | -0.791% | -0.271% | 14/18 | +2.301% | 12/18 | +33.2% | +36.5% | PASS |

## 九项门槛

### R2

- PASS — `mse_nonlosses_at_least_11_of_18`
- PASS — `macro_mse_within_0_5pct`
- PASS — `macro_mae_within_0_5pct`
- PASS — `parameter_reduction_at_least_20pct`
- PASS — `latency_reduction_at_least_15pct`
- PASS — `peak_memory_reduction_at_least_15pct`
- PASS — `every_dataset_has_nonloss_horizon`
- PASS — `timerole_macro_mse_better_than_recent`
- PASS — `timerole_history_wins_at_least_12_of_18`
### R4

- PASS — `mse_nonlosses_at_least_11_of_18`
- PASS — `macro_mse_within_0_5pct`
- PASS — `macro_mae_within_0_5pct`
- PASS — `parameter_reduction_at_least_20pct`
- PASS — `latency_reduction_at_least_15pct`
- PASS — `peak_memory_reduction_at_least_15pct`
- PASS — `every_dataset_has_nonloss_horizon`
- PASS — `timerole_macro_mse_better_than_recent`
- PASS — `timerole_history_wins_at_least_12_of_18`
