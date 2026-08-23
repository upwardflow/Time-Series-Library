# TimeRole 在 ETTh2 与 ETTm2 上的短板诊断

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-22
- Verification Status: ANALYZED
- Version Label: validation_v1
- Source: TimeRole 三随机种子测试记录、统一表2单种子基线、严格验证集对照、数据切分统计

## 结论摘要

当前证据支持将两个短板分开处理。

1. **ETTh2 的首要问题是验证—测试迁移，而不是一次训练崩溃。** 三个种子在四个跨度上均出现“验证 MSE 越低、测试 MSE 反而越高”的描述性逆序；H=720 最明显，seed 2023 的验证 MSE 最低，但测试 MSE 最高。训练日志无 NaN、无异常退出。
2. **ETTm2 的首要问题是固定采样点尺度缺乏物理时间适配。** 同样的最近96点在 ETTh2 覆盖4天，在 ETTm2 只覆盖1天；完整336点分别覆盖14天和3.5天。TimeRole 在 ETTm2 上稳定优于 Recent96，却在四个验证任务上均落后直接处理336点的 Raw336，说明远期历史有用，但当前压缩修正路径保留的信息仍不足。
3. **这两点都是诊断而不是最终因果结论。** 其他基线尚在补齐2022/2023，任何“稳定优于基线”或“显著落后”的主表结论都应等统一三种子结果完成后再确定。

## 证据一：三随机种子结果

| 数据集 | H | TimeRole MSE mean ± sd | MAE mean ± sd |
|---|---:|---:|---:|
| ETTh2 | 96 | 0.282767 ± 0.001806 | 0.342768 ± 0.001567 |
| ETTh2 | 192 | 0.343898 ± 0.007230 | 0.386472 ± 0.005722 |
| ETTh2 | 336 | 0.375161 ± 0.003656 | 0.412981 ± 0.005470 |
| ETTh2 | 720 | 0.448514 ± 0.027889 | 0.462265 ± 0.014904 |
| ETTm2 | 96 | 0.175718 ± 0.003675 | 0.264075 ± 0.003068 |
| ETTm2 | 192 | 0.230751 ± 0.008973 | 0.299960 ± 0.007829 |
| ETTm2 | 336 | 0.285193 ± 0.005411 | 0.335806 ± 0.004591 |
| ETTm2 | 720 | 0.375472 ± 0.000746 | 0.391677 ± 0.002697 |

ETTh2-H720 的三个测试 MSE 为 0.433716、0.431143 和 0.480684；seed 2023 是主要波动来源。但其验证 MSE 为 0.627805，优于 seed 2021/2022 的 0.638779/0.649174，并且正常完成7轮训练、由第1轮检查点选中。因此不能把该结果作为错误运行删除。

## 证据二：ETTh2 的验证—测试分布迁移

训练段标准化后，ETTh2 验证段与测试段的跨变量平均位置差为0.802个训练标准差，主要来自 MULL（2.119）、OT（1.875）和 HULL（1.454）。ETTm2 对应统计分别为0.800、2.109、1.876和1.447，说明两个 Transformer-2 数据集在不同采样频率下共享近似的宏观状态迁移。

每日周期相关性与季节朴素误差也发生变化：

| 数据集 | 验证日周期相关 | 测试日周期相关 | 验证日周期朴素MSE | 测试日周期朴素MSE |
|---|---:|---:|---:|---:|
| ETTh2 | 0.817 | 0.670 | 0.175 | 0.266 |
| ETTm2 | 0.816 | 0.673 | 0.176 | 0.263 |

ETTh2 每个跨度仅有三个种子，验证 MSE 与测试 MSE 的 Pearson 相关分别为 -0.859、-0.899、-0.685 和 -0.895。这些相关系数不能用于显著性推断，但四个跨度方向一致，足以把“单一验证段能否稳定选出泛化更好的种子/检查点”列为优先风险。

## 证据三：ETTm2 的职责分化有效，但压缩存在信息损失

严格验证集对照显示：

- TimeRole 相对 Recent96 在 ETTm2 四个跨度均提升，三种子平均 MSE 改善约4.995%–6.116%。这说明远期历史修正路径确实利用了额外历史。
- Raw336 相对 TimeRole 在四个跨度均更低，TimeRole 的 MSE 分别落后约4.194%、2.145%、5.729%和5.111%。这说明将前240点压缩为15个均值 token 后，仍损失了直接长输入主干能够利用的细粒度信息。
- 在 ETTm2-H96/H720 的机制对照中，GlobalGate 的验证 MSE 分别比当前通道门控 TimeRole 低约1.132%和1.317%。该结果只覆盖两个跨度，不能支持全面替换门控，但提示当前跨样本共享的通道级系数并非 ETTm2 的稳定最优选择。
- 远期历史时间打乱/反转只使 ETTm2 验证 MSE 上升约1.14%–6.05%，而样本错配造成34.65%–65.48%的退化。模型强依赖“是否来自同一样本”，但对远期内部精确顺序的利用相对有限。

## 结构尺度解释

| 配置 | ETTh2（1小时采样） | ETTm2（15分钟采样） |
|---|---:|---:|
| 最近96点 | 4天 | 1天 |
| 远期240点 | 10天 | 2.5天 |
| 完整336点 | 14天 | 3.5天 |
| patch 4/2 | 4小时/2小时 | 1小时/30分钟 |
| memory pool 16 | 16小时 | 4小时 |

当前模型按“点数”共享超参数，而不是按物理时间共享。ETTm2 的近期主干只观察一个完整日周期，远期压缩单元也从 ETTh2 的16小时缩短为4小时。这能解释为何 DLinear、TimeMixer 等简单或显式多尺度模型在 ETTm2 的部分任务更强，但该解释仍需验证集消融确认。

## 后续验证优先级

1. **先完成公平基线。** 所有七个基线补齐 seed 2022/2023 后，再用统一 mean ± sd 重算任务级排名和差距。
2. **ETTh2：只在训练/验证范围内检验稳健选点。** 比较单一验证段、多个滚动时间块平均验证和邻近最佳轮次权重平均；不得根据测试结果选择方案。
3. **ETTm2：做最小尺度因子实验。** 优先比较按物理时长调整的近期长度、patch跨度与历史压缩率；保持参数量匹配，并与 Recent96、当前TimeRole和Raw336同时对照。
4. **门控只作为次级因素。** 在尺度问题确认后，再比较通道门控、全局门控和样本条件门控，避免把两个因素混在一次实验中。
5. **任何结构改动先做验证集门槛。** 至少覆盖 ETTh2/ETTm2 两个数据集、四个跨度和三个种子；在冻结规则前不访问测试集。

## 统计与推断风险扫描

- Coverage: 11/11 fallacy types checked.
- Simpson's paradox: CAUTION。总体平均会掩盖 ETTh2-H720 和 ETTm2 各跨度方向差异，必须保留任务级结果。
- Ecological fallacy: NOTE。数据集级平均不能外推到每个变量或每个样本。
- Berkson's paradox: NOTE。未发现基于结果筛选种子的做法；必须继续保留全部种子。
- Collider bias: NOTE。当前描述分析未引入后处理控制变量。
- Base-rate neglect: N/A。不是分类任务。
- Regression to the mean: CAUTION。不能因为 seed 2023 极端而只补跑该种子后选择更好结果。
- Survivorship bias: NOTE。现有 TimeRole 60/60 完成；基线队列必须报告全部失败和超时。
- Look-elsewhere effect: CAUTION。20个任务和多项诊断不可只报告有利单元。
- Garden of forking paths: CAUTION。任何尺度或门控候选必须先冻结范围和判定门槛。
- Correlation != causation: CAUTION。数据统计与性能共现只能提出机制假设，不能证明短板原因。
- Reverse causality: NOTE。不适用于时间先后因果，但模型性能差异不能反推出唯一结构原因。

## 可复现来源

- `logs/timerole_table2_multiseed/timerole_mean_std.csv`
- `logs/timerole_table2_multiseed/records/`
- `logs/q2_main_baselines/main_results_long.csv`
- `logs/graphmamba_cmrhm_strict_evidence/group_a_task_mean_sd.csv`
- `logs/graphmamba_cmrhm_strict_evidence/group_b_capacity.csv`
- `logs/graphmamba_cmrhm_strict_evidence/group_c_ablation.csv`
- `logs/cmrhm_interventions/summary.csv`
- `models/TimeRole.py`
- `models/GraphMambaRecent.py`
- `data_provider/data_loader.py`
