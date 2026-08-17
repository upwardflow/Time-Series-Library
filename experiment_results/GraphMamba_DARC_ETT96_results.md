# GraphMamba DARC：ETT-96 初步验证结果

## 创新机制

在原始 GraphMamba 的固定融合 `temporal + graph` 基础上，加入统一的
**Dual-Domain Adaptive Residual Calibration (DARC)**：

1. **表示域校准**：为每个变量的长、短 Patch 图残差分别学习可靠性
   `r(v,s)`，避免所有变量和尺度共享同一图贡献。
2. **预测域校准**：用零初始化的共享历史到未来映射学习主干未解释的
   预测残差，并用变量/预测步系数校准修正强度。
3. **严格配对初始化**：`r(v,s)=1` 且预测残差为0，因此训练第0步输出与
   原 GraphMamba 完全一致；GPU核验最大输出差为0。

最终模型参数量由 997,382 增至 1,007,380，增加 9,998（+1.00%）。

## 固定实验协议

- 多变量预测多变量（M→M），输入长度96，预测长度96。
- seed=2021，batch=32，lr=5e-4，scheduler=type1，最多100 epochs，patience=6。
- baseline与新模型使用相同数据划分、初始化主干、数据顺序及dropout随机流。
- 只按 element-weighted validation MSE 选择checkpoint；模型冻结后各执行一次test。

## 最终结果

| 数据集 | 划分 | 模型 | MSE | MAE | MSE相对改善 | MAE相对改善 |
|---|---|---|---:|---:|---:|---:|
| ETTh1 | Validation | GraphMamba | 0.695348 | 0.549859 | — | — |
| ETTh1 | Validation | GraphMamba+DARC | **0.692664** | **0.548932** | **0.386%** | **0.169%** |
| ETTh1 | Test | GraphMamba | 0.375652 | 0.393598 | — | — |
| ETTh1 | Test | GraphMamba+DARC | **0.372267** | **0.392251** | **0.901%** | **0.342%** |
| ETTh2 | Validation | GraphMamba | 0.218634 | 0.318064 | — | — |
| ETTh2 | Validation | GraphMamba+DARC | **0.217594** | **0.315999** | **0.476%** | **0.649%** |
| ETTh2 | Test | GraphMamba | 0.293511 | 0.343722 | — | — |
| ETTh2 | Test | GraphMamba+DARC | **0.290317** | **0.341547** | **1.088%** | **0.633%** |

## 验证消融

| 数据集 | Baseline MSE/MAE | 仅变量尺度图校准 | 仅直接残差校正 | 完整DARC |
|---|---:|---:|---:|---:|
| ETTh1 | 0.695348 / 0.549859 | 0.695304 / 0.551253 | 0.692899 / **0.547347** | **0.692664** / 0.548932 |
| ETTh2 | 0.218634 / 0.318064 | 0.218062 / 0.316314 | 0.218365 / 0.316324 | **0.217594 / 0.315999** |

完整模块在两个数据集上取得最低验证MSE；ETTh2的MSE和MAE均优于两个单模块。
ETTh1中直接残差单模块的MAE更低，但完整模块的主选优指标MSE最低。

## 当前结论与边界

DARC已经是一个有实测正收益的候选创新点：ETTh1、ETTh2的验证和测试
MSE/MAE共8项比较全部改善，并且参数开销约1%。但这仍是 seed=2021、两个
数据集、单预测长度的初步证据，尚不足以直接支持SCI二区论文结论。下一阶段
需要覆盖ETT四数据集的96/192/336/720，并以3 seeds均值±标准差报告，同时
补充训练时间、显存和更强基线比较。

复现实验入口：

```bash
.venv/bin/python scripts/run_graphmamba_innovation.py \
  --dataset ETTh1 --model GraphMambaAF --candidate darc_etth1_96 \
  --af-mode variable_scale_residual --final-test
```
