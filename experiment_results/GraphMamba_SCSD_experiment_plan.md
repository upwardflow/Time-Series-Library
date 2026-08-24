# GraphMamba 近期预测器第二创新：SCSD 实验任务书

> 状态：实验前设计；本文档不代表机制已经成立。
> 目标：判断“尺度校准、状态隔离的共享状态空间动力学”是否足以成为 TimeRole 的第二核心创新。
> 原则：先验证集、后测试集；先最小机制、后完整矩阵；未通过门槛立即停止。

## 1. 当前证据与研究缺口

当前 GraphMamba 已实现同一近期窗口上的粗、细两组 patch；两组 token 使用同一个 Mamba encoder；independent_shared 分别调用 encoder，因此状态轨迹在尺度间重置；最后将 temporal state 与 variable graph state 融合形成近期基础预测。

已有 joint 对 independent_shared 的验证仅覆盖 ETTh1/ETTh2、预测长度192、seed 2021。独立共享扫描的宏平均 MSE 改善约0.118%，方向不一致，并在同 epoch 的 ETTh1 运行中增加约11.4%耗时。因此，现有证据只能说明状态隔离具有语义合理性，不能证明它是性能贡献。

已有尺度瓶颈诊断表明，粗、细尺度在共享 Mamba 的输入/门控、卷积、输出投影和 FFN 上存在冲突梯度，而状态生成参数 A_log 未表现出跨数据集一致的冲突。这提示一个可检验假设：

> 不同 patch 尺度可以共享潜在状态转移基，但需要尺度相关的输入/输出坐标和物理时间步校准，同时保持独立状态轨迹。

## 2. 候选机制：SCSD

候选名称：

**Scale-Calibrated Shared Dynamics with Scale-Isolated States (SCSD)**  
中文：**尺度校准、状态隔离的共享动力学**

候选机制由三部分组成。

### 2.1 共享潜在动力学

两个尺度共享 Mamba 的主要状态转移核心，至少共享状态生成基：

A(coarse) = A(fine) = A。

该约束表达两个尺度来自同一近期系统，而不是两个无关过程。

### 2.2 独立状态轨迹

两个尺度分别执行扫描，不允许粗尺度末状态作为细尺度首 token 的先验状态。

### 2.3 尺度校准

第一阶段优先实现低风险的尺度适配器：

Z_adapter(s) = Z(s) + U_in(s) V_in(s) Z(s)

H_adapter(s) = H(s) + U_out(s) V_out(s) H(s)

适配器采用低秩或 FiLM 形式，初始化为近似恒等映射，避免把实验变成两套完整 encoder。

第二阶段仅在适配器通过门槛后实现时间步校准：

Delta_t(s) = tau_s × softplus(f_Delta(Z_t(s)) + b_s)

其中 tau_s 由 patch 长度或 stride 的相对物理覆盖确定，b_s 为轻量尺度参数。不得在第一轮同时引入复杂路由、动态图或额外专家。

## 3. 必须建立的对照

### 3.1 扫描与参数共享对照

| ID | 变体 | 目的 |
|---|---|---|
| J | joint | 检验跨尺度伪连续扫描 |
| IS | independent_shared | 当前参数共享、状态隔离基线 |
| IU | independent_unshared | 两套完整 encoder，作为性能上限和参数对照 |
| SA | IS + scale adapters | 检验尺度坐标适配 |
| DC | IS + delta calibration | 检验物理时间步校准；仅第二阶段运行 |
| SCSD | IS + adapters + delta calibration | 完整候选机制；仅前序门槛通过后运行 |

independent_unshared 必须真正包含两个独立初始化、独立更新的 encoder，不能通过复用同一对象或浅复制实现。

### 3.2 双尺度必要性对照

| ID | 变体 | 目的 |
|---|---|---|
| C | coarse only | 粗尺度单独贡献 |
| F | fine only | 细尺度单独贡献 |
| D | dual scale | 双尺度互补性 |

w/o Patch 不能代替本组三方比较。

### 3.3 近期预测器边界对照

仅在 SCSD 通过第一阶段后运行：

- SCSD + Graph；
- SCSD w/o Graph；
- SCSD w/o TimeRole（Recent-only）；
- SCSD + TimeRole。

该组用于判断近期机制是否独立成立，以及它与远期历史修正是否兼容。Graph 只作为辅助变量关系分支，不作为 SCSD 的新颖性来源。

## 4. 冻结实验协议

### 4.1 第一阶段任务

- 数据集：ETTm1、ETTh2、Weather；
- 预测长度：96、720；
- seeds：2021、2022、2023；
- 近期窗口：96；
- 与 TimeRole 联合时总输入：336；
- patch：沿用当前冻结设置，粗尺度4/2，细尺度2/1；
- 选点：仅验证集 MSE；
- 第一阶段严禁访问测试集。

选择 ETTm1、ETTh2、Weather 是为了覆盖分钟级、小时级和气象数据域；96/720 同时覆盖短、长预测跨度。

### 4.2 公平性要求

所有成对变体必须保持：

- 相同数据划分、batch size、学习率、早停规则和最大 epoch；
- 相同 seed 与数据顺序；
- 除被检验机制外完全相同的配置；
- 记录参数量、训练时间、推理延迟和峰值显存；
- 记录最佳 epoch 和完整验证指标；
- 失败不得静默重跑；所有重跑进入 incident log。

IU 参数更多，因此除原始对比外，应补充一个参数匹配控制：减小 IU 的 d_model 或其他容量，使参数量接近 SCSD，并同时保留未缩减 IU 作为性能上界。

## 5. Go/No-Go 门槛

### 5.1 Scale adapter 进入第二阶段的门槛

SA 相对 IS 必须同时满足：

1. 18个 dataset–horizon–seed 单元中至少12个 MSE 改善；
2. 宏平均 MSE 改善不低于1.0%；
3. 宏平均 MAE 不得退化超过0.2%；
4. 三个数据集均不能出现两个跨度同时系统性退化；
5. 参数量明显低于未缩减 IU；
6. 推理延迟相对 IS 增幅原则上不超过15%。

未满足时停止 SCSD 路线，不实现 delta calibration，不访问测试集。

### 5.2 SCSD 升格为第二核心的门槛

完整 SCSD 必须同时满足：

1. 相对 IS 的宏平均 MSE 改善不低于1.0%，且至少12/18单元改善；
2. 相对 IU 的误差接近或更优：宏平均 MSE 差距不超过0.2%，或直接优于 IU；
3. 参数量相对未缩减 IU 至少降低25%；
4. coarse-only、fine-only 的宏平均结果均不优于完整 dual-scale SCSD；
5. Recent-only 与 TimeRole 联合设置下改善方向一致；
6. 不依赖单一数据集或单一预测跨度获得结论。

通过后才能进入完整数据集、四预测长度和测试集评估。未通过时，SCSD 仅作为负结果或补充材料，不修改论文核心贡献。

## 6. Codex 实现顺序

### Phase 0：复现与结构审计

1. 复现现有 J/IS 两个验证任务，确认指标与归档一致；
2. 检查两个尺度的 tensor shape、patch 数量和 encoder 对象身份；
3. 增加单元测试，确认 IS 状态重置、参数共享；
4. 增加单元测试，确认 IU 参数不共享；
5. 输出参数量与前向延迟基线。

### Phase 1：补齐基线

1. 在 models/GraphMamba.py 增加 independent_unshared；
2. 增加 coarse-only、fine-only 配置；
3. 先用单任务单 seed 做 smoke test；
4. 完成 J/IS/IU 与 C/F/D 的第一阶段验证矩阵；
5. 在未读取测试集的情况下生成汇总报告。

### Phase 2：最小尺度适配器

1. 实现低秩 input/output adapter 或 FiLM；
2. 保持 Mamba 状态核心共享；
3. 适配器初始化为近似恒等映射；
4. 先执行结构与梯度检查；
5. 运行 SA 对 IS/IU 的第一阶段验证；
6. 根据第5.1节决定 Go/No-Go。

### Phase 3：时间步校准

仅在 Phase 2 通过后：

1. 定位 Mamba selective step/Delta 的实现位置；
2. 引入由 patch coverage 决定的 tau_s 和轻量偏置；
3. 验证 Delta 数值范围、无 NaN/Inf、反向梯度正常；
4. 分别运行 DC 和完整 SCSD；
5. 按第5.2节决策。

### Phase 4：与 TimeRole 联合验证

仅在完整 SCSD 通过后：

1. Recent-only：IS vs SCSD；
2. TimeRole：IS + TimeRole vs SCSD + TimeRole；
3. w/o Graph 边界对照；
4. 完整数据集与四跨度验证；
5. 最后一次性访问测试集。

## 7. 建议文件与产物

实现时优先复用现有实验基础设施：

- models/GraphMamba.py：扫描模式、单尺度开关、适配器入口；
- layers/GraphMamba_EncDec.py 或实际 Mamba block：尺度条件接口；
- run.py：配置参数；
- scripts/run_graphmamba_scsd_validation.py：严格验证调度；
- scripts/finalize_graphmamba_scsd.py：只读聚合；
- logs/graphmamba_scsd_validation/：原始记录；
- experiment_results/GraphMamba_SCSD_validation_result.md：最终结论。

每条最终 JSON 至少记录：

- git commit SHA；
- dataset、horizon、seed、variant；
- 完整命令和解析后的配置；
- split 与 test_accessed；
- MSE、MAE、最佳 epoch；
- parameter count；
- train duration、inference latency、peak CUDA memory；
- status、failure reason、retry provenance。

## 8. 论文主张边界

在 SCSD 通过全部门槛前，不得写：

- 首个多尺度 Mamba；
- 首个共享状态空间多尺度模型；
- 状态隔离稳定提高预测；
- Graph + Mamba 是本文创新；
- SCSD 是 TimeRole 的第二核心贡献。

通过后可谨慎表述：

> 对覆盖同一近期历史的异分辨率 patch，SCSD 共享潜在状态转移基，同时通过尺度相关的输入/输出校准和物理时间步调节缓解尺度冲突，并保持独立状态轨迹。该设计以显著少于双独立 encoder 的参数取得接近或更好的跨数据域预测结果。

最终论文结构应为：

1. 近期：SCSD 构造尺度校准的高分辨率近期状态；
2. 远期：TimeRole 将压缩远期历史限制为近期状态条件下的边际修正；
3. Graph：近期预测器中的辅助变量依赖建模，而非独立创新。

## 9. 停止规则

出现以下任一情况即停止追加机制：

- SA 未通过第5.1节；
- SCSD 改善主要来自单一数据集；
- 完整机制不优于简单 IS；
- 与 IU 的差距无法由参数效率解释；
- 近期机制加入 TimeRole 后改善方向反转；
- 需要继续叠加动态图、专家路由或复杂损失才能获得微小收益。

停止后应保留 TimeRole 为唯一核心创新，并将 GraphMamba 定位为经过消融验证的近期实验主干。
