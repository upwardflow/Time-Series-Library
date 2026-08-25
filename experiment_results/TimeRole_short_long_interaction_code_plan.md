# TimeRole 短期—长期交互式修正规划

## 1. 目标与当前问题

目标不是再训练一套缩小的近期预测器，也不是让旧历史独立输出修正，而是让旧历史在近期状态的条件下修正真正用于预测的近期潜状态。

当前 `models/TimeRole.py` 的路径为：

1. `GraphMambaRecent` 由最近 96 点产生完整 `base_output`；
2. `recent_context -> memory_decoder` 又学习一条近期到未来的映射；
3. 旧历史只进入旁路的 `with_memory - without_memory`；
4. 最终在输出空间执行 `base_output + correction`。

因此当前差分虽然阻止了显式的“纯旧历史输出”，但旧历史没有修改 GraphMamba 的近期状态，而且旁路重复承担近期预测。

## 2. 冻结的新结构：PCHSR

候选名：Paired Cross-History State Revision（PCHSR，配对式跨历史状态修正）。

张量约定：

- 近期主干状态 `H_r`: `[B, N, D, P]`；
- 压缩旧历史 `X_l`: `[B, N, M]`，当前 `M=240/16=15`；
- 旧历史 tokens `H_l`: `[B, N, M, D]`；
- 近期 query: `[B, N, P, D]`；
- 检索上下文 `C(H_r,H_l)`: `[B, N, P, D]`。

计算路径：

```text
recent 96 -> GraphMamba encoder -> H_r ---------------------> shared head
                                  |                              ^
old 240 -> pooled history tokens -+-> recent-query retrieval -> |ΔH

ΔH = tanh(a_n) * [phi(H_r + C(H_r,H_l)) - phi(H_r)]
H_r' = H_r + ΔH
forecast = GraphMambaHead(H_r') + original trend path
```

约束：

- 只保留 GraphMamba 原有预测头，不再保留 `recent_context` 和 `memory_decoder`；
- 旧历史必须经过由 `H_r` 产生的 query 才能形成上下文；
- 修正采用同一个 `phi` 的成对差分，`H_l=0` 时差分严格为零；
- 每变量幅度 `a_n` 经 `tanh` 有界并初始化为零，使初始模型严格等价于 Recent-only；
- 旧历史 tokenization 使用无 bias 的数值投影和“数值乘时间基”，保证 `recent_mean` 干预归一化为零后不会被位置偏置重新激活；
- 训练时只解码 `H_r'` 一次，避免双预测和两次 dropout；评估时才额外计算 Recent-only 输出用于诊断 correction。

## 3. 文件级代码顺序

### Step 1：行为保持的 GraphMamba 边界重构

修改 `models/GraphMamba.py`，把当前 `forecast` 拆为内部接口：

- `_normalize_input(x)`：返回 normalized、mean、stdev；
- `_encode_forecast_state(normalized)`：返回 `fused_state` 与 `trend_output`；
- `_decode_forecast_state(state, trend, mean, stdev)`：使用现有唯一 head 还原输出；
- `forecast(x)` 仍按原顺序组合这些接口。

本步不得改变模块、参数名、state dict 或默认数值路径。先用相同 state dict 在多种 GraphMamba 配置上比较重构前后 eval 输出，要求 shape 一致、参数键一致，并以 `torch.equal` 为首选门槛；若底层算子只允许近似复现，则记录 max absolute error，门槛 `<=1e-7`。

### Step 2：新增独立候选，不覆盖已发表 TimeRole

新增 `models/TimeRoleInteraction.py`，继承 `GraphMambaRecent`，复用 Step 1 的近期编码/解码接口。保留 `models/TimeRole.py` 作为冻结的 output-correction 对照。

模块仅包含：

- 旧历史 pooling；
- 无 bias 的历史 token/value 投影与时间基；
- 无 bias 的 recent query、history key/value 投影；
- scaled dot-product recent-to-history retrieval；
- 共享 `phi` 和零初始化的变量修正幅度。

首版不加入额外 Graph、专家路由、新损失、双向 cross-attention 或 horizon-specific decoder。

### Step 3：诊断接口

在 eval 模式保存：

- `last_memory_correction`：共享 head 下 `forecast(H_r') - forecast(H_r)`；
- `last_state_revision`：`ΔH`；
- `last_history_attention`：近期 patch 到旧历史段的检索权重；
- `last_interaction_scale`：`tanh(a_n)`。

扩展 `exp/exp_long_term_forecasting.py` 的聚合字段时只读取这些诊断，不改变损失或 checkpoint 选择。

### Step 4：结构审计

新增 `scripts/audit_timerole_interaction.py`，至少验证：

1. 输出为 `[B, pred_len, N]`，支持 R1/R2/R4 的不同 patch 数；
2. `TimeRoleInteraction` 中不存在 `recent_context`、`memory_decoder` 或第二个 forecast head；
3. `a_n=0` 时与同初始化 Recent-only 严格等价；该零门控用例只要求幅度参数本身可获得梯度；
4. `recent_mean` 旧历史干预使 `C`、`ΔH`、correction 为零；
5. 在审计专用的非零 `a_n` 下，intact old history 的 attention 行归一、修正非零且所有新增参数有有限梯度；
6. batch shuffle/reverse 只改变历史交互，不改变近期编码；
7. paired Recent-only/interaction 的共同参数初始化和数据顺序一致；
8. 禁止 test loader，并记录 source/config hashes。

### Step 5：最小 smoke

新增独立 runner `scripts/run_timerole_interaction.py`，不要覆盖近期简化实验记录。先运行：

- ETTm1，horizon 96，seed 2021；
- Recent-only、旧 `TimeRole`、`TimeRoleInteraction`；
- 相同近期结构、训练超参数和 `independent_shared` 扫描；
- validation only，`test_accessed=false`。

只有结构审计全过、三组训练完成、无 NaN/Inf、interaction correction 非零，才进入正式配对门禁。

## 4. 近期主干选择与隔离原则

当前 Phase B 单 seed 结果中 R1、R2、R4 通过简化门槛；R4 是现有报告的综合首选，R1 是精度回退。开始 PCHSR 训练前先完成对应 Recent-only 净收益核对：

- R4 保留 TimeRole 净收益：首轮固定 R4；
- R4 失败但 R1 通过：固定 R1；
- 两者均失败：先用 R0 做结构验证，不把近期简化和交互机制混在同一因果比较中。

同一轮不得同时改变近期 backbone 与历史交互模块。

## 5. 验证门禁与对照

首轮正式矩阵：ETTm1、ETTh2、Weather × horizons 96/720 × seed 2021。每个单元严格配对：

- `C0`: Recent-only；
- `C1`: 当前 `TimeRole` 输出空间修正；
- `C2`: `TimeRoleInteraction` 潜状态交互修正；
- `C3`: 与 C2 参数量相近、但 history context 不使用 recent query 的 no-interaction 控制。

C2 进入三 seed 的最低门槛：

- 相对 C0 宏 MSE 改善，且至少 4/6 单元获胜；
- 相对 C1 宏 MSE 不退化超过 0.3%，同时结构上删除第二近期预测器；
- C2 优于 C3 至少 4/6，证明收益来自条件交互而非增加参数；
- `recent_mean` correction 接近零，破坏旧历史的干预总体恶化；
- 参数、延迟和显存全部记录，不以单次 correction 非零代替预测收益。

未过门槛则保留当前 `TimeRole.py`，归档 PCHSR，不继续加注意力层、门控或超参数搜索。通过后才扩展到 seeds 2022/2023；测试集继续禁止访问。

## 6. 实施提交边界

建议分四个可回退提交：

1. `refactor: expose GraphMamba forecast state boundary`；
2. `feat: add TimeRole latent history interaction candidate`；
3. `test: audit TimeRole interaction invariants`；
4. `experiment: add validation-only interaction gate`。

每个提交先编译和运行对应结构测试。任何一步失败只回退该候选，不修改冻结的 `models/TimeRole.py` 和既有结果记录。
