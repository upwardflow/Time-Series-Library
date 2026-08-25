# TimeRole 近期预测器简化与替换实验任务书

> 状态：实验前冻结规划。  
> 目标：在保留 TimeRole 远期条件修正核心的前提下，删除近期预测器中缺乏稳定证据的组件，选择更简单、更高效且精度可接受的近期基础预测器。  
> 原则：近期预测器不再承担独立创新任务；以完整 TimeRole 的精度—成本表现作为最终选择依据。

## 1. 决策背景

现有 GraphMamba 近期预测器包含：

- trend/seasonal decomposition；
- coarse/fine 双尺度 patch；
- 双向 Mamba；
- 变量图传播；
- 时间状态与图状态融合；
- 预测头与趋势重组。

现有证据不支持继续增加 SCSD、尺度适配或 Delta calibration：

1. independent_shared 相对 joint 在18个三数据集、双跨度、三种子验证单元中仅8次 MSE 获胜，宏平均 MSE 变化约为 -0.016%；
2. independent_shared 相对 independent_unshared 虽有约0.452%的宏平均 MSE改善，但效应较小，主要支持参数共享作为普通正则化选择；
3. dual-scale 相对每个单元中更好的单尺度仅5/18次获胜，宏平均 MSE约差0.878%；
4. 既有近期主干消融中，删除 decomposition、Graph 或 Mamba 的结果具有明显任务依赖性；
5. TimeRole 的远期条件修正已经具有多随机种子、历史干预和跨主干证据。

因此，本阶段暂停 SCSD 开发，将研究目标改为：

> 寻找能以更少组件和更低成本承载 TimeRole 的近期基础预测器，而不是继续为近期主干构造第二创新。

## 2. 当前证据的组件级判断

| 组件 | 当前证据 | 本阶段处理 |
|---|---|---|
| Decomposition | 删除后经常持平或改善 | 优先删除候选 |
| Dual-scale patch | 双尺度没有稳定超过最佳单尺度 | 优先改为单尺度 |
| Independent state reset | 精度基本中性，推理峰值显存较低 | 不作为创新；按候选结构保留或移除 |
| Parameter sharing | 对 IU 有轻微正向信号 | 仅作为实现选择 |
| Bidirectional Mamba | 价值依赖数据集和跨度 | 比较单向替代 |
| Mamba branch | Recent-only 与 TimeRole 联合结果不同 | 不直接删除，进入组合实验 |
| Graph branch | 平均贡献较小且跨度相关 | 重点测试删除 |
| TimeRole correction | 现有证据最强 | 所有正式候选必须保留 |

## 3. 研究问题

本实验必须回答：

1. 删除 decomposition 后是否可以保持或改善完整 TimeRole？
2. 单一 fine 或 coarse patch 是否足以替代双尺度？
3. 单尺度条件下 Graph 是否仍有必要？
4. 单向 Mamba 是否可以替代双向 Mamba？
5. 纯 Mamba 或纯 Graph 近期预测器是否优于双分支组合？
6. 简化近期预测器后，TimeRole 相对对应 Recent-only 的增益是否仍然存在？
7. 精度轻微变化是否能够由参数、延迟和显存的大幅下降合理补偿？

## 4. 冻结候选结构

### 4.1 主候选

| ID | Decomp | Scale | Mamba | Graph | TimeRole | 目的 |
|---|---|---|---|---|---|---|
| R0 | on | dual | bidirectional | on | on | 当前完整模型参照 |
| R1 | off | fine-only | bidirectional | on | on | 删除分解和粗尺度 |
| R2 | off | coarse-only | bidirectional | on | on | 更低成本粗尺度 |
| R3 | off | fine-only | unidirectional | on | on | 检查单向 Mamba |
| R4 | off | fine-only | bidirectional | off | on | 检查纯 Mamba 近期预测器 |
| R5 | off | fine-only | off | on | on | 检查纯 Graph 近期预测器 |

当前项目的冻结 patch 设置为：

- coarse：patch length/stride = 4/2；
- fine：patch length/stride = 2/1；
- recent length = 96；
- TimeRole 总输入长度 = 336，其中近期96、远期240。

不得在本阶段同时搜索新的 patch 长度、Mamba 深度、Graph top-k 或 TimeRole 压缩率。

### 4.2 对应 Recent-only 控制

第一阶段只要求为 R0、R1、R4、R5 建立对应 Recent-only 控制；进入第二阶段的候选必须全部补齐对应 Recent-only。

对应控制的唯一差异应是 TimeRole 远期修正是否启用。不得为 Recent-only 单独重新调参。

### 4.3 暂不进入第一轮的候选

以下方案不进入第一轮：

- SCSD scale adapter；
- Delta calibration；
- 动态尺度门控；
- 新的 Graph 构造；
- 多专家或路由；
- 新损失函数；
- 新增注意力层；
- 同时改变近期长度或远期长度。

只有现有简化候选全部失败，且失败原因能够明确归因于尺度冲突时，才重新讨论这些机制。

## 5. 实验前审计

在启动新训练前，Codex 必须先完成只读核实：

1. 汇总以下已有结果：
   - logs/graphmamba_backbone_ablation/；
   - logs/cmrhm_backbone_ablation/；
   - logs/cmrhm_no_mamba_cross_domain/；
   - logs/timerole_p0/recent_backbone/final/；
   - logs/graphmamba_scsd_validation/。
2. 检查每组结果的：
   - git SHA 或 source hash；
   - 数据集、预测跨度、seed；
   - validation/test split；
   - test_accessed；
   - 模型配置；
   - 指标口径；
   - 是否存在失败或重试。
3. 标记哪些 R0 结果可以严格复用，哪些必须在当前 clean commit 上重跑。
4. 不允许把不同近期长度、不同扫描模式或不同图配置的结果直接合并。
5. 输出 experiment_results/TimeRole_recent_predictor_existing_evidence_audit.md。

若已有结果无法确认同协议，不得作为正式配对基线。

## 6. Phase A：结构实现与 smoke test

### 6.1 实现要求

优先通过配置开关组合现有代码，不复制整个模型文件。确有必要时再增加清晰的模型别名。

每个候选必须：

- 输出形状与 R0 一致；
- 支持 Recent-only 和 TimeRole；
- 明确记录 active components；
- 只构造启用组件的参数；
- 预测头维度与单尺度 token 数严格匹配；
- 禁用组件后不得保留未使用的大型参数；
- 保持相同归一化、训练协议和数据划分。

### 6.2 Smoke test

只运行：

- dataset = ETTm1；
- horizon = 96；
- seed = 2021；
- split = validation；
- variants = R0–R5。

检查：

- 六个候选均能训练和完整验证；
- 无 NaN/Inf；
- 输出维度一致；
- 参数量随组件删除合理下降；
- 禁用 Graph/Mamba/decomposition 后没有残余前向调用；
- test_accessed=false；
- 日志记录完整配置和 source hashes。

Smoke test 失败时停止正式矩阵，先修复实现问题。

## 7. Phase B：低成本单种子筛选

### 7.1 任务矩阵

- 数据集：ETTm1、ETTh2、Weather；
- 预测长度：96、720；
- seed：2021；
- variants：R0–R5；
- validation only。

总计最多：

6 variants × 3 datasets × 2 horizons = 36 runs。

若 R0 与已有 SCSD/TimeRole 记录协议、代码和配置完全一致，可复用；否则必须重跑。

### 7.2 冻结训练协议

沿用当前 TimeRole 正式协议：

- batch size = 32；
- learning rate = 5e-4；
- max epochs = 100；
- patience = 6；
- validation MSE 选点；
- 相同数据划分和 seed；
- test evaluation disabled；
- 不因中间结果调整单个候选超参数。

### 7.3 记录指标

每次运行至少记录：

- validation MSE、MAE；
- best epoch；
- parameter count；
- training duration；
- inference milliseconds per batch；
- training peak CUDA memory；
- inference peak CUDA memory；
- TimeRole correction MAE/RMS；
- channel gate statistics；
- split 与 test_accessed；
- failure/retry provenance。

### 7.4 第一阶段筛选规则

候选进入 Phase C，至少满足：

1. 相对 R0 的六任务宏平均 MSE 不退化超过0.5%，或直接改善；
2. 宏平均 MAE不退化超过0.5%；
3. 六个任务中至少4个 MSE 不劣于 R0；
4. 参数量下降至少20%，或延迟下降至少15%；
5. 不允许 ETTh2 与 Weather 两个数据域同时系统性退化；
6. TimeRole correction 不是数值坍缩为零。

最多选择两个简化候选进入 Phase C。若没有候选通过，保留 R0，不继续组合搜索。

## 8. Phase C：三随机种子确认

### 8.1 矩阵

比较：

- R0；
- Phase B 最优候选1；
- Phase B 最优候选2。

任务：

- ETTm1、ETTh2、Weather；
- horizons 96、720；
- seeds 2021、2022、2023；
- validation only。

已完成且严格同协议的 seed 2021 可以复用。

### 8.2 最终简化门槛

简化候选可以替换 R0，必须满足：

1. 18个 dataset–horizon–seed 单元中至少11个 MSE 不劣于 R0；
2. 宏平均 MSE退化不超过0.5%，或直接改善；
3. 宏平均 MAE退化不超过0.5%；
4. 参数量降低至少20%；
5. 推理延迟降低至少15%；
6. 推理或训练峰值显存降低至少15%；
7. 每个数据集至少有一个跨度不劣于 R0；
8. TimeRole 相对对应 Recent-only 的宏平均 MSE仍然改善；
9. TimeRole 对 Recent-only 的配对胜场不少于12/18。

如果两个候选均通过，优先选择：

1. 结构更简单者；
2. 参数更少者；
3. 延迟更低者；
4. 最后才比较小于0.2%的误差差异。

## 9. Phase D：机制确认

只对最终胜出候选执行。

### 9.1 Recent-only 对照

比较：

- Simplified Recent；
- Simplified Recent + TimeRole。

验证 TimeRole 的收益没有依赖原 GraphMamba 的复杂组件。

### 9.2 远期历史干预

复用现有干预协议：

- intact；
- batch shuffle；
- temporal shuffle；
- reverse；
- recent mean；
- matched noise。

至少要求：

- batch shuffle 在所有代表任务上导致明显退化；
- recent mean 使 correction 接近零；
- intact 优于主要破坏性干预；
- 结果不能仅来自新增参数或固定偏置。

### 9.3 复杂度核对

报告 R0 与最终简化模型的：

- 总参数量；
- 近期预测器参数量；
- TimeRole 增量参数量；
- 训练时间；
- 推理延迟；
- 峰值显存；
- correction branch 开销。

## 10. Phase E：正式结果

只有 Phase C、D 均通过后：

1. 冻结最终架构和全部超参数；
2. 扩展到正文五数据集；
3. 使用 horizons 96、192、336、720；
4. 使用 seeds 2021、2022、2023；
5. 完整验证结束后一次性访问测试集；
6. 更新主表、核心消融和资源表；
7. 不根据测试结果返回调整架构。

若正式结果显示简化模型明显不稳定，应恢复 R0，而不是继续事后增加组件。

## 11. 产物规范

建议新增：

- scripts/run_timerole_recent_simplification.py；
- scripts/finalize_timerole_recent_simplification.py；
- logs/timerole_recent_simplification/；
- experiment_results/TimeRole_recent_predictor_existing_evidence_audit.md；
- experiment_results/TimeRole_recent_predictor_simplification_result.md。

每条记录至少包含：

- git commit SHA；
- dirty state；
- source-file hashes；
- candidate ID 与 active components；
- dataset、horizon、seed；
- 完整命令与解析配置；
- split、test_accessed；
- MSE、MAE、best epoch；
- parameter count；
- train duration；
- inference latency；
- train/inference peak CUDA memory；
- correction statistics；
- status；
- retry provenance。

调度器必须：

- 串行或显式受控并发；
- 原子写记录；
- completed 自动跳过；
- failed 默认停止；
- 重跑必须显式指定并记录 incident；
- 聚合器只读，不修改原始记录。

## 12. 论文定位

简化成功后，论文不再主张近期预测器是第二创新。

推荐定位：

> TimeRole 使用一个经过精度—成本筛选的轻量近期预测器生成基础预测，并通过远期条件修正分支限制更早历史只表达相对于同一近期状态的边际变化。

近期预测器的作用是：

- 建立可靠基础预测；
- 为远期修正提供近期参照；
- 控制整体计算成本。

不得再主张：

- Graph + Mamba 是核心创新；
- 双尺度是必要贡献；
- decomposition 是必要设计；
- 状态隔离稳定改善精度；
- 所有近期组件都不可替代。

## 13. 停止规则

出现以下任一情况即停止进一步简化：

- 所有候选宏平均 MSE均退化超过0.5%；
- 简化候选在 ETTh2 与 Weather 同时系统性退化；
- TimeRole 相对 Recent-only 的收益明显消失；
- 资源下降不足以补偿误差退化；
- 需要新增复杂门控、专家或损失才能维持性能；
- 正式测试后才发现问题并需要返回调参。

出现以下任一情况即停止 SCSD 路线：

- 简化后的单尺度预测器已经满足精度—成本目标；
- 双尺度继续无法超过最佳单尺度；
- 近期预测器不再承担论文创新职责；
- TimeRole 在简单近期预测器上仍保持稳定收益。

最终目标不是寻找组件最多的模型，而是选择能够最清楚支撑 TimeRole 核心机制的最小充分近期预测器。
