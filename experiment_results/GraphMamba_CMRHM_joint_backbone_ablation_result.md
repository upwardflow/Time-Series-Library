# CMRHM开启条件下的主干联合消融实验记录

## 结论摘要

20组单因素训练与完整验证集复评全部成功，未访问测试集。最重要的结果不是“完整固定主干最优”，而是 CMRHM 在五种删减主干、四个任务形成的20组对应比较中全部降低 MSE 和 MAE，宏平均改善分别为4.786%和2.717%。这为 CMRHM 作为核心贡献提供了跨主干配置的稳健性证据。与此同时，`w/o Mamba` 在开启 CMRHM 后仍获得0.991%的宏平均 MSE 改善，说明当前固定图—Mamba融合仍不是得到充分支持的独立创新点。

## Material Passport

| 项目 | 内容 |
|---|---|
| Origin Skill | academic-research-suite / experiment-agent |
| Run | CMRHM + backbone one-factor ablation |
| Date | 2026-08-16 |
| Verification Status | ANALYZED |
| 数据 | `dataset/ETT-small/ETTm1.csv`、`dataset/ETT-small/ETTm2.csv` |
| 完整模型 | `GraphMambaCMRHM`，输入336点（远期历史240点 + 近期96点） |
| 任务 | ETTm1/ETTm2 × 预测长度96/720 |
| 随机种子 | 2021 |
| 选点与评价 | 验证集早停；最佳检查点在完整验证集上按元素数计算 MSE/MAE |
| 结构控制 | CMRHM保持开启且旧历史干预为`intact`；每个变体仅切换一个主干开关 |
| 扫描方式 | `independent_shared` |
| 总控脚本 | `scripts/run_graphmamba_backbone_ablation.py` |
| 启动命令 | `.venv/bin/python -u scripts/run_graphmamba_backbone_ablation.py --model GraphMambaCMRHM --gpu 0 --variants no_decomp no_patch uni_mamba no_mamba no_graph --datasets ETTm1 ETTm2 --horizons 96 720 --timeout-seconds 1800` |
| 运行位置 | tmux会话`cmrhm-backbone`，GPU 0 |
| 运行时间 | 约43分49秒 |
| 汇总 | `logs/cmrhm_backbone_ablation/summary.csv` |
| 状态 | 20/20 completed；0 failed；`test_accessed=false`；无重试 |

## 完整结果

| 变体（CMRHM均开启） | ETTm1-96 | ETTm1-720 | ETTm2-96 | ETTm2-720 | 宏平均MSE变化 | 宏平均MAE变化 |
|---|---:|---:|---:|---:|---:|---:|
| 完整方法 | 0.3676/0.4001 | 0.9410/0.6399 | 0.1232/0.2434 | 0.2780/0.3597 | 0.000% | 0.000% |
| w/o Decomp | 0.3650/0.4004 | 0.9267/0.6365 | 0.1246/0.2420 | 0.2773/0.3615 | -0.347% | -0.137% |
| w/o Patch | 0.3704/0.4030 | 0.9495/0.6435 | 0.1254/0.2421 | 0.2782/0.3620 | +0.859% | +0.342% |
| Uni-Mamba | 0.3745/0.4038 | 0.9383/0.6390 | 0.1219/0.2417 | 0.2823/0.3648 | +0.517% | +0.375% |
| w/o Mamba | 0.3704/0.4029 | 0.9405/0.6365 | 0.1209/0.2396 | 0.2703/0.3565 | -0.991% | -0.578% |
| w/o Graph | 0.3700/0.4014 | 0.9369/0.6395 | 0.1259/0.2437 | 0.2766/0.3583 | +0.469% | -0.005% |

负变化表示相对完整方法误差降低。完整方法参考值来自既有冻结检查点的同口径完整验证集复评；五个删减变体均从头训练。

## 主干—CMRHM交互

- Patch是联合模型中最一致的主干因素：移除后四个任务的MSE全部上升，宏平均退化0.859%。
- 图分支在两个96步任务上均有正向作用，但在两个720步任务上并非必要，表现出跨度依赖性。
- 纯近期主干中的`w/o Mamba`四格均优；加入CMRHM后，ETTm1-96发生反转（0.3676→0.3704），说明历史记忆会改变时间分支的边际价值。
- `w/o Mamba`在其余三格仍优于完整方法，且宏平均MSE改善0.991%，因此不能宣称当前固定图—Mamba融合已达到最优协同。
- 分解和双向扫描的贡献均随数据集与预测跨度变化，不宜写成普遍必要组件。

## CMRHM跨主干稳健性

将每个开启 CMRHM 的删减变体与对应的纯近期删减变体配对，共获得20个“主干变体×任务”比较：

- MSE：20/20改善，宏平均改善4.786%，单任务改善范围1.596%–8.728%；
- MAE：20/20改善，宏平均改善2.717%，单任务改善范围1.027%–5.165%。

因此，CMRHM的收益没有绑定于某一个完整主干配置。这一结果可以作为正文中“CMRHM是核心方法，而近期图增强状态空间结构是实验宿主”的直接证据，但由于本表仅使用一个随机种子和四个代表任务，不能替代已有的多随机种子稳定性实验。

## 完整性核对

- 20个训练JSON、20个最终验证JSON和20个最佳检查点均存在。
- 20条最终记录的键`(variant, dataset, horizon, seed)`互不重复。
- 所有记录均为`model=GraphMambaCMRHM`、`split=val`、`test_accessed=false`。
- 所有变体均保持CMRHM开启并使用`cmrhm_intervention=intact`。
- 总控遇到失败即停止且不自动重试；本轮没有失败、异常或重试。
