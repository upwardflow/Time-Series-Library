# TimeRole 近期预测器简化：下一阶段执行计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-25
- Verification Status: UNVERIFIED
- Version Label: timerole_recent_simplification_execution_plan_v1
- Authoritative Protocol: `experiment_results/TimeRole_recent_predictor_simplification_plan.md`
- Planning Commit: `7680295`

> 状态：已完成执行规划，尚未实现新开关或启动训练。任务书中的候选、超参数、门槛和停止规则保持不变。

## 1. 目标与实验假设

目标是在保持 TimeRole 远期条件修正收益的前提下，用更少组件、更低参数量、延迟和显存替换当前近期预测器 R0。

- 主假设：至少一个简化候选在验证集宏平均 MSE/MAE 退化不超过 0.5% 的同时，达到任务书规定的资源下降门槛。
- 机制假设：替换近期预测器后，TimeRole 相对严格对应的 Recent-only 仍保持稳定的配对收益。
- 零结果同样有效：若没有候选通过 Phase B/C，则保留 R0 并停止简化，不追加新门控、专家、损失或 SCSD 机制。

### 变量定义

- 自变量：decomposition、patch scale、Mamba 方向、Mamba 分支、Graph 分支、TimeRole 修正开关。
- 主要因变量：validation MSE。
- 次要因变量：validation MAE、参数量、训练耗时、推理延迟、训练/推理峰值显存、correction MAE/RMS、channel gate statistics。
- 固定控制：数据划分、训练 seed、逐 epoch 数据顺序、近期长度 96、总输入长度 336、batch size 32、学习率 `5e-4`、最多 100 epochs、patience 6、validation MSE 选点。

## 2. 冻结候选映射

| ID | 模型 | Decomp | Scale | Mamba | Graph | TimeRole |
|---|---|---:|---|---|---:|---:|
| R0 | TimeRole | 1 | dual | bidirectional | 1 | 1 |
| R1 | TimeRole | 0 | fine | bidirectional | 1 | 1 |
| R2 | TimeRole | 0 | coarse | bidirectional | 1 | 1 |
| R3 | TimeRole | 0 | fine | unidirectional | 1 | 1 |
| R4 | TimeRole | 0 | fine | bidirectional | 0 | 1 |
| R5 | TimeRole | 0 | fine | off | 1 | 1 |

对应 Recent-only 使用 `GraphMambaRecent`，除关闭 TimeRole 远期修正外，其余配置必须与同 ID 的 TimeRole 完全一致。patch/stride 固定为 coarse `4/2`、fine `2/1`；不得搜索 patch、深度、top-k、近期/远期长度或 TimeRole 压缩率。

### 扫描模式冻结

主配对矩阵建议统一使用 `independent_shared`，因为任务书冻结的 coarse/fine patch 几何及 SCSD 证据均对应这一模式。当前正式 TimeRole 在 ETTh2 上可能通过 `auto` 解析为 `periodic_aligned`；该结果只能作为“官方 R0 外部参照”，不得与 `independent_shared` 候选直接组成严格配对。执行前必须在 manifest 中明确写入最终决定：

1. 主矩阵 R0–R5 使用相同的 `independent_shared`；
2. 如需要保留官方 R0，则另列 `R0-official`，不计入简化门槛的配对分母；
3. 禁止把不同扫描模式的历史结果静默复用或合并。

## 3. WP0：执行前保护与公平性修复（硬门槛）

在任何新训练前完成：

1. 从当前 `main` 创建独立实验分支，确认没有无关 tracked 修改。
2. 修复 DataLoader 随机数隔离：训练 loader 使用由实验 seed 初始化的专用 `torch.Generator`；验证/测试 loader 固定 `shuffle=False`。
3. 每轮记录训练采样顺序摘要或 sampler-state hash，证明同 dataset/horizon/seed 的各候选采用相同数据顺序。
4. 单尺度只实例化启用的 patch embedding；禁用 Graph/Mamba/decomposition 后不构造对应大型参数。
5. 在正式训练前提交代码，记录 clean commit、完整 source hashes 和环境指纹：Python、PyTorch、CUDA、mamba_ssm、GPU、驱动版本。
6. 固定资源测量方法：同一 GPU、batch size、验证集、CUDA synchronize、固定 warm-up，保存重复测量原值并以中位数作为门槛判定值。

验收：结构审计、随机数审计、编译检查和 dry-run 全部通过；否则不得进入 smoke。

## 4. WP1：既有证据只读审计

只读检查以下目录：

- `logs/graphmamba_backbone_ablation/`
- `logs/cmrhm_backbone_ablation/`
- `logs/cmrhm_no_mamba_cross_domain/`
- `logs/timerole_p0/recent_backbone/final/`
- `logs/graphmamba_scsd_validation/`

每条候选结果建立 evidence ledger，至少包含：来源路径、Git/source hash、dirty state、dataset/horizon/seed、模型与完整配置、扫描模式、split、`test_accessed`、metric version、状态、失败/重试以及是否满足当前 RNG 公平协议。

复用采用白名单：只有代码、数据划分、近期长度、扫描模式、图配置、训练协议、指标口径和 RNG 公平性均可证明一致时才标记 `REUSABLE`。缺少 provenance、来自 dirty source 且不能重建，或受旧 DataLoader 混杂影响的记录标记 `CONTEXT_ONLY`。

产物：`experiment_results/TimeRole_recent_predictor_existing_evidence_audit.md`。

验收：R0 每个可复用单元均有逐字段证据；无法证明时默认重跑。

## 5. WP2：候选实现与 Phase A smoke

### 静态/结构测试

- R0–R5 输出均为 `[batch, pred_len, channels]`。
- `active_components` 与实际模块/参数对象一致。
- R1/R3/R4/R5 不存在 coarse embedding；R2 不存在 fine embedding。
- R4 不构造或调用 Graph；R5 不构造或调用 Mamba；R1–R5 不调用 decomposition。
- 单/双向 Mamba 的调用方向与配置一致。
- TimeRole 与对应 Recent-only 的近期预测器初始化逐 tensor 对齐；唯一结构差异是修正分支。
- 所有候选参数量变化方向合理，且没有未参与 forward 的大型参数。

### Smoke 矩阵

- ETTm1 × horizon 96 × seed 2021 × R0–R5 × TimeRole，共 6 runs。
- validation only；`test_accessed=false`。

验收：6/6 完成、无 NaN/Inf、checkpoint 可复评、指标和资源字段齐全、无残余禁用分支调用。任一失败即停止正式矩阵。

## 6. WP3：Phase B 单种子筛选

### B1：TimeRole 主矩阵

- 数据集：ETTm1、ETTh2、Weather。
- horizons：96、720。
- seed：2021。
- candidates：R0–R5。
- 总数：36 runs；与 smoke 完全同协议的 R0–R5/ETTm1-96 可复用，因此通常新增 30 runs。

### B2：Recent-only 控制块

任务书同时要求在第一阶段为 R0、R1、R4、R5 建立对应控制。该控制块单独计账：4 × 3 × 2 = 最多 24 runs，不计入“36 个 TimeRole 主任务”的口径。历史结果只有通过 WP1 白名单才可复用。

### Phase B 判定

先按任务书六条规则逐项判定，再做统计描述：

- 每候选相对 R0 的六任务宏平均 MSE/MAE 变化；
- 六任务 MSE 胜场；
- 参数、延迟、显存变化；
- ETTh2/Weather 域级方向；
- correction MAE/RMS 与 gate 分布；
- paired bootstrap 95% CI、效应量、配对检验仅作不确定性说明，不替代冻结门槛。

最多两个候选进入 Phase C。零候选通过时：保留 R0、输出负结果报告并停止。

## 7. WP4：Phase C 三随机种子确认

比较 R0 与最多两个入围候选：

- 3 models × 3 datasets × 2 horizons × 3 seeds = 54 个 TimeRole 记录；
- Phase B 的 18 个 seed-2021 单元可复用，通常新增 36 个 TimeRole runs；
- 每个模型必须有镜像 Recent-only 54-cell 控制矩阵，用于 TimeRole 收益门槛；已严格匹配的 Phase B 控制可复用；
- 取决于入围 ID，Phase C 新增 Recent-only runs 约 36–48 个。

因此 Phase C 预计新增训练量为 72–84 runs，而不是只统计 TimeRole 一侧。

### 统计口径

- 门槛使用未舍入的逐 dataset–horizon–seed 配对指标。
- 主汇总为 18 个单元的等权宏平均相对变化；同时报告原始 MSE/MAE 的宏平均，避免平均方式歧义。
- 胜场定义为候选未舍入 MSE `<= R0`；TimeRole 收益胜场定义为 TimeRole MSE `<` 对应 Recent-only。
- 报告每个 dataset–horizon 的三 seed 均值、样本标准差、CV、paired bootstrap CI、Cohen's dz、paired t-test 与 Wilcoxon；多候选推断使用 Holm 校正。
- 冻结的工程门槛优先于 p 值；统计检验用于量化不确定性，不用于事后改变阈值。

仅当任务书九条 Phase C 门槛全部满足时，简化候选才可替换 R0。两个候选均通过时，依次按结构简单、参数少、延迟低选择；小于 0.2% 的误差差异最后考虑。

## 8. WP5：Phase D 机制确认

只对最终胜出候选执行。

1. 确认其 18-cell TimeRole 与 Recent-only 配对矩阵完整。
2. 对 Phase B 六个代表任务（3 datasets × 2 horizons）的 seed-2021 冻结 checkpoint 执行 intact、batch shuffle、temporal shuffle、reverse、recent mean、matched noise，共 36 次 validation checkpoint evaluation，不重新训练。
3. 检查 batch shuffle 全部明显退化、recent mean 使 correction 接近零、intact 优于主要破坏性干预。
4. 输出 R0 与胜出候选的总参数、近期参数、TimeRole 增量、训练时间、推理延迟、峰值显存和 correction 开销分解。

机制门槛失败时恢复 R0，不进入正式矩阵。

## 9. WP6：Phase E 正式五数据集结果

只有 Phase C/D 均通过后执行：

- 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- horizons：96、192、336、720；
- seeds：2021、2022、2023；
- 最终冻结 TimeRole 架构：60 个 validation runs。

60 个验证任务全部完成并锁定架构、超参数和 checkpoint 清单后，才允许一次性对这些冻结 checkpoint 访问 test。测试结果不得触发返回调参。若正式结果明显不稳定，恢复 R0。

## 10. 调度、监控与产物

### tmux 规划

- session：`timerole_recent_simplification`
- windows：`audit`、`smoke`、`phase_b`、`phase_c`、`mechanism`、`aggregate`
- 同一 GPU 默认只允许一个训练队列；聚合器只读。
- 每个训练任务必须有 process-alive、日志增长和 hard-timeout 监控。

### 目录规划

- runner：`scripts/run_timerole_recent_simplification.py`
- structural audit：`scripts/audit_timerole_recent_simplification.py`
- finalizer：`scripts/finalize_timerole_recent_simplification.py`
- output root：`logs/timerole_recent_simplification/`
- evidence audit：`experiment_results/TimeRole_recent_predictor_existing_evidence_audit.md`
- final report：`experiment_results/TimeRole_recent_predictor_simplification_result.md`

建议按 `manifest/`、`records/`、`raw_logs/`、`checkpoints/`、`incidents/`、`audit/`、`summaries/` 分层保存，并为 Phase A–E 使用独立 stage manifest，禁止后续阶段覆盖前一阶段 manifest。

### 失败与恢复策略

- completed 原子记录自动跳过；聚合器不修改原始记录。
- failed 默认停止当前队列，不自动重试。
- 重跑必须提供显式 `--retry <candidate>`，写入 incident 和 retry provenance。
- checkpoint 不进入 Git；代码、manifest、JSON/CSV、审计报告和必要原始日志可进入 Git。
- 所有 validation 阶段强制 `test_accessed=false`；任何意外 test 访问立即停止整个 session。

## 11. 推荐启动顺序

1. WP0 公平性与 active-parameter 修复。
2. WP1 既有证据审计并冻结 R0/扫描模式。
3. Phase A 结构审计和 6-run smoke。
4. 人工检查 smoke 报告后启动 Phase B1，再补 Phase B2 控制。
5. 自动生成 Phase B gate 报告，但由人确认入围名单。
6. 仅为确认的最多两个候选生成 Phase C 队列。
7. 仅对最终胜者执行 Phase D。
8. Phase C/D 全通过后才生成 Phase E 队列；最终验证锁定后一次性 test。

下一步实施范围应只覆盖 WP0、WP1 和 Phase A。完成并审核这些产物之前，不启动 Phase B。
