# IRPA–CMRHM 2×2×3 同协议对照报告

## 1. 协议

- 公共主干：PatchTST，仅建模最近 96 点。
- 总历史预算：960；IRPA 与 CMRHM 均可访问相同的前 864 点旧历史。
- 数据集：ETTm1、Weather；预测跨度：96、720；随机种子：2021、2022、2023。
- 训练：batch size 32，最多 10 epochs，patience 3，learning rate 1e-4；验证 MSE 选择 checkpoint，测试集仅在选点完成后访问一次。
- IRPA：revise length 96、top-k 3、moving average 25，完整保留 refined input 与两类 prediction auxiliary。
- CMRHM：旧历史以 pool=16 压缩为 54 个 token，hidden dimension 32。
- 本实验是 `same-budget L=960` 的机制对照，不是 IRPA 原论文数据集特定长输入（ETTm1 1920、Weather 3600）主结果的复现。

采用 960 而不是 336，是因为完整 IRPA 在 `L=336,H=720` 时无法取得匹配 patch 后所需的 8 个未来 patch，候选集合为空。若强行使用 336，只能删减 IRPA 的预测辅助路径，会变成不完整算法对照。

## 2. 预测精度（mean ± sample std，3 seeds）

| Dataset | Horizon | Method | MSE | MAE |
|---|---:|---|---:|---:|
| ETTm1 | 96 | Recent96 | 0.3390 ± 0.0062 | 0.3746 ± 0.0078 |
| ETTm1 | 96 | IRPA | **0.2921 ± 0.0038** | **0.3453 ± 0.0024** |
| ETTm1 | 96 | CMRHM | 0.3147 ± 0.0122 | 0.3635 ± 0.0057 |
| ETTm1 | 720 | Recent96 | 0.4665 ± 0.0012 | 0.4518 ± 0.0022 |
| ETTm1 | 720 | IRPA | **0.4318 ± 0.0017** | **0.4231 ± 0.0008** |
| ETTm1 | 720 | CMRHM | 0.4350 ± 0.0016 | 0.4418 ± 0.0004 |
| Weather | 96 | Recent96 | 0.1747 ± 0.0010 | 0.2166 ± 0.0017 |
| Weather | 96 | IRPA | 0.1662 ± 0.0006 | 0.2163 ± 0.0012 |
| Weather | 96 | CMRHM | **0.1537 ± 0.0004** | **0.2097 ± 0.0003** |
| Weather | 720 | Recent96 | 0.3528 ± 0.0005 | 0.3461 ± 0.0002 |
| Weather | 720 | IRPA | 0.3191 ± 0.0003 | 0.3364 ± 0.0002 |
| Weather | 720 | CMRHM | **0.3107 ± 0.0011** | **0.3354 ± 0.0010** |

CMRHM 相对 Recent96 在四个设置的 MSE 均改善，约为 7.2%、6.8%、12.0% 和 12.0%。与 IRPA 直接比较时，CMRHM 在 Weather 两个跨度的全部 6/6 个 seed 上同时取得更低 MSE 和 MAE；IRPA 则在 ETTm1 两个跨度的全部 6/6 个 seed 上获胜。因此，本实验不支持“CMRHM 全面优于 IRPA”，支持“CMRHM 在高维 Weather 场景更强，且在 ETTm1-H720 以较小 MSE 损失换取更低成本”。

## 3. RTX 4090 推理效率

统一 batch size 32；10 次预热、30 次 CUDA Event 计时。峰值显存包含模型、输入和 forward 激活。

| Dataset | H | Method | Params (M) | Mean latency (ms/batch) | Peak memory (MiB) |
|---|---:|---|---:|---:|---:|
| ETTm1 | 96 | Recent96 | 10.056 | 2.447 | 135.87 |
| ETTm1 | 96 | IRPA | 10.121 | 3.023 | 136.57 |
| ETTm1 | 96 | CMRHM | 10.064 | **2.606** | **136.27** |
| ETTm1 | 720 | Recent96 | 13.891 | 2.354 | 151.56 |
| ETTm1 | 720 | IRPA | 14.706 | 3.256 | 155.61 |
| ETTm1 | 720 | CMRHM | 13.919 | **2.552** | **151.67** |
| Weather | 96 | Recent96 | 6.904 | 5.638 | 239.68 |
| Weather | 96 | IRPA | 6.969 | 6.290 | 239.93 |
| Weather | 96 | CMRHM | 6.912 | **5.714** | **239.72** |
| Weather | 720 | Recent96 | 10.738 | 5.626 | 257.57 |
| Weather | 720 | IRPA | 11.553 | 6.483 | 262.33 |
| Weather | 720 | CMRHM | 10.766 | **5.822** | **257.67** |

相对 IRPA，CMRHM 在四个设置的 forward 延迟分别降低 13.8%、21.6%、9.2% 和 10.2%。H=720 时，参数量分别降低 5.4%（ETTm1）和 6.8%（Weather），峰值显存分别降低 2.5% 和 1.8%。

训练墙钟均值也保留在原始记录中，但受早停轮次影响，不作为纯计算效率的主要证据。36 次训练累计约 1.72 GPU 小时。

## 4. 可用于论文的结论边界

推荐主张：在相同长历史预算和相同近期预测主干下，CMRHM 以固定压缩记忆实现了稳定低于 IRPA 的推理开销；其精度具有数据依赖性，在 Weather 上优于 IRPA，在 ETTm1 上仍落后。该结果说明 CMRHM 的优势不是“无条件精度更高”，而是更有利的效率—精度折中，以及在变量较多数据上的潜在适配优势。

不推荐主张：CMRHM 全面超过 IRPA；CMRHM 已证明普遍优于检索式长历史方法；本实验复现了 IRPA 官方最优结果。两个数据集和三个种子足以构成二区论文的针对性补充实验，但不足以支撑普遍性或统计显著性结论。

## 5. 可复核文件

- 逐 seed 指标：`logs/patchtst_irpa_cmrhm_2x2x3/all_seed_results.csv`
- 均值与标准差：`logs/patchtst_irpa_cmrhm_2x2x3/aggregate_results.csv`
- 效率原始表：`logs/patchtst_irpa_cmrhm_2x2x3/efficiency.csv`
- 单任务命令与元数据：`logs/patchtst_irpa_cmrhm_2x2x3/records/*.json`
- 单任务训练日志：`logs/patchtst_irpa_cmrhm_2x2x3/logs/*.log`
- 可恢复训练入口：`scripts/run_patchtst_irpa_cmrhm_2x2x3.py`
- 效率测量入口：`scripts/benchmark_patchtst_irpa_cmrhm.py`
