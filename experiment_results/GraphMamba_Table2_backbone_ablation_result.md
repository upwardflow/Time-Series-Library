# GraphMambaRecent 表2主干消融实验记录

## 结论摘要

20组训练与完整验证集复评全部成功，未访问测试集。结果不支持“完整主干的所有组件均必要”：`w/o Mamba` 在四个任务上同时降低 MSE 和 MAE，宏平均改善分别为2.336%和1.693%。当前图—Mamba并行融合因此不能作为已验证的独立贡献，较稳妥的论文定位是将其作为 CMRHM 的实验宿主。

## Material Passport

| 项目 | 内容 |
|---|---|
| 数据 | `dataset/ETT-small/ETTm1.csv`、`dataset/ETT-small/ETTm2.csv` |
| 输入协议 | 数据窗口336；`GraphMambaRecent`仅向主干传入最近96点；M设置 |
| 任务 | ETTm1/ETTm2 × 预测长度96/720 |
| 随机种子 | 2021 |
| 选点与评价 | 验证集早停；最佳检查点在完整验证集上按元素数计算MSE/MAE |
| 结构控制 | 每个变体仅切换一个结构开关，其余超参数冻结 |
| 扫描方式 | `independent_shared` |
| 总控脚本 | `scripts/run_graphmamba_backbone_ablation.py` |
| 启动命令 | `.venv/bin/python -u scripts/run_graphmamba_backbone_ablation.py --gpu 0 --variants no_decomp no_patch uni_mamba no_mamba no_graph --datasets ETTm1 ETTm2 --horizons 96 720 --timeout-seconds 1800` |
| 汇总 | `logs/graphmamba_backbone_ablation/summary.csv` |
| 状态 | 20/20 completed；0 failed；`test_accessed=false` |

## 完整结果

| 变体 | ETTm1-96 | ETTm1-720 | ETTm2-96 | ETTm2-720 | 宏平均MSE变化 |
|---|---:|---:|---:|---:|---:|
| 完整近期主干 | 0.4043/0.4233 | 0.9744/0.6612 | 0.1289/0.2479 | 0.2896/0.3699 | 0.000% |
| w/o Decomp | 0.3936/0.4100 | 0.9477/0.6520 | 0.1266/0.2445 | 0.2957/0.3756 | -1.265% |
| w/o Patch | 0.3981/0.4171 | 0.9772/0.6576 | 0.1301/0.2456 | 0.2937/0.3725 | +0.278% |
| Uni-Mamba | 0.4064/0.4197 | 0.9786/0.6614 | 0.1270/0.2464 | 0.3093/0.3847 | +1.584% |
| w/o Mamba | 0.3846/0.4081 | 0.9641/0.6496 | 0.1273/0.2466 | 0.2832/0.3666 | -2.336% |
| w/o Graph | 0.3969/0.4156 | 0.9729/0.6639 | 0.1313/0.2476 | 0.2902/0.3700 | +0.020% |

负变化表示相对完整主干误差降低。完整主干参考值来自既有冻结检查点的同口径验证复评；五个消融变体均从头训练。

## 结果解释与论文边界

- 分解模块在三个任务上被移除后反而改善，不能列为稳定贡献。
- Patch移除使宏平均MSE轻微上升0.278%，表明其有小幅、非普遍的辅助价值。
- 单向扫描主要在ETTm2-720退化，双向扫描的价值具有任务依赖性。
- `w/o Mamba`四格全部改善，是当前主干最关键的反证；需要调整的是分支融合，而不是继续证明每个现有组件都不可缺少。
- `w/o Graph`总体与完整主干持平，但在ETTm2-96明显退化，说明图信息在特定任务有价值；它仍不能证明当前双分支融合成立。

## 完整性核对

- 20个训练JSON、20个验证JSON和20个最佳检查点均存在。
- 所有最终记录均为`status=completed`、`split=val`、`test_accessed=false`。
- 总控遇到失败即停止且不自动重试；本轮没有发生失败或重试。
