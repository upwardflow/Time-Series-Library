# DARC rank-16低秩深化验证记录

## 目的

根据48个DARC checkpoint的谱分析，直接残差映射90%能量集中在约5--15个
奇异方向，因此测试一个且仅一个rank=16低秩候选，以减少复杂度并检验压缩是否
改善泛化。所有实验均为validation-only，未读取新版模型的test指标。

## 结果

| Task | Role | ΔMSE vs full DARC | ΔMAE vs full DARC |
|---|---|---:|---:|
| ETTh1-192 | Development | +0.052% | -0.043% |
| ETTm1-720 | Development | +0.295% | +0.227% |
| ETTm2-720 | Development | -0.103% | -0.103% |
| ETTh1-96 | Protection | +0.092% | +0.151% |
| ETTh2-96 | Protection | -0.038% | -0.032% |
| ETTm2-336 | Protection | -0.262% | -0.301% |

正值代表低秩候选改善。六任务MSE相对改善均值约为+0.006%，实际近似持平；
三个保护任务中两个发生退化。因此该候选未通过晋级门槛。

## 决策

- 不进入测试集。
- 不搜索其他rank。
- 不继续增加预测距离门控、专家或额外损失。
- 保留完整DARC作为已有实验模块，将下一阶段转向GraphMamba核心表示创新。

原始记录与比较表位于`logs/graphmamba_darc_lowrank_validation/`。
