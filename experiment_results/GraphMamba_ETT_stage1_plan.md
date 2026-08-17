# GraphMamba ETT 第一阶段提升路线

## 阶段目标

在 ETT 四数据集的标准多变量预测协议下，将初代 GraphMamba 的 16 任务平均 MSE 从 **0.3740** 推进到 **0.3582** 或更低，并保证所有参数只通过验证集选择。

| 项目 | 设定 |
|---|---|
| 数据集 | ETTh1、ETTh2、ETTm1、ETTm2 |
| 输入长度 | 96 |
| 预测长度 | 96、192、336、720 |
| 任务 | M→M |
| 主目标 | 16 任务平均 MSE ≤ 0.3582 |
| 辅目标 | 16 任务平均 MAE ≤ 0.3846 |
| 统计 | 最终 seeds 2021/2022/2023，mean ± std |
| 参考 | TimeFilter（ICML 2025）论文报告值 |

## 路线总览

1. **先修实验协议**：训练中不再读取测试集；补齐验证指标、结构化日志、配置哈希与断点续跑。
2. **复验当前基线**：固定 seed=2021 重跑 ETT 16 任务，确认环境和代码结果稳定。
3. **分组粗筛**：在 96 与 720 两类锚点任务上，依次筛优化器、多尺度、图结构和模型容量。
4. **全长度复筛**：每个数据集保留 3 个候选，在四个预测长度上按平均验证 MSE 选出一套共享配置。
5. **冻结后测试**：配置和代码冻结后只测试一次，与 TimeFilter 的 16 个单元格逐项比较。
6. **一次定向细化**：若验证集仍有明显缺口，只针对失败类型做一轮局部搜索；不根据测试结果回调。
7. **最终统计与消融**：3 seeds、mean ± std、效率指标，以及图/Mamba/Patch 组件消融。

## 受控搜索空间

| 模块 | 候选范围 |
|---|---|
| 训练 | lr 1e-4/3e-4/5e-4/1e-3；scheduler type1/type3/cosine；batch 16/32/64；dropout 0/0.1/0.2 |
| 多尺度 | patch/stride 4/2、8/4、16/4、16/8；moving average 13/25/49 |
| 图 | alpha 0/0.25/0.5/0.75/1；top-k 1/2/3/4；node dim 8/16/32 |
| 容量 | d_model 32/64/128；d_state 16/32/64；d_conv 2/4；layers 1/2 |

不做全排列。每轮仅改变一组参数并保留验证分数最好的候选；`d_ff` 固定为 `2*d_model`，第一阶段固定 Mamba-1。

## 结果选优规则

- 搜索阶段固定 seed=2021，主指标为验证集 MSE。
- 多任务排序使用相对当前基线归一化后的平均验证 MSE。
- 差距小于 0.5% 时，优先验证 MAE更低、参数更少、速度更快的配置。
- 每个数据集四个预测长度共享一套配置，不为 16 个任务分别挑参数。
- 正式测试结果出现后不再修改该轮配置；否则结果只能标记为探索性结果。

## 阶段门槛

| 门槛 | 判定 |
|---|---|
| G0 可复现 | seed=2021 基线与历史结果相对误差 ≤1% |
| G1 有效优化 | 16 任务平均 MSE ≤0.3654（距目标 ≤2%） |
| G2 第一阶段达标 | 平均 MSE ≤0.3582，且至少 8/16 个任务达到参考值 |
| G3 论文级确认 | 三种子均值维持 G2 附近，并完成关键消融与效率对比 |

## 停止条件

- 连续两轮搜索的归一化验证 MSE 改善均低于 0.5%，停止扩网格。
- 第一阶段结束后若与目标仍相差超过 2%，转向第二阶段“图条件化 Mamba/层内动态融合”，而不是继续依赖更大范围调参。
- 任一实验若使用测试集参与选择，必须单独标为探索性，不进入论文主结果。

## 第一项执行任务

修改训练与实验记录链路，使搜索阶段完全不可见测试损失，并建立可恢复的验证集搜索脚本。完成该项后再启动基线复验和参数筛选。

## 终端执行与交接

第一轮只运行基线验证。建议在仓库根目录建立独立 tmux 会话：

```bash
cd /home/cwh/Time-Series-Library
tmux new -s graphmamba-ett-s1
.venv/bin/python scripts/run_graphmamba_ett_stage1.py --stage baseline
```

离开 tmux 但保持训练：按 `Ctrl+B`，再按 `D`。重新进入：

```bash
tmux attach -t graphmamba-ett-s1
```

基线完成后不要立即运行下一阶段。通知 Codex 读取以下文件：

```text
logs/graphmamba_ett_stage1_weighted/baseline/runs.csv
logs/graphmamba_ett_stage1_weighted/baseline/candidate_summary.csv
```

Codex 会检查 16 个验证结果，并为四个数据集分别生成：

```text
logs/graphmamba_ett_stage1_weighted/selected/baseline_ETTh1.json
logs/graphmamba_ett_stage1_weighted/selected/baseline_ETTh2.json
logs/graphmamba_ett_stage1_weighted/selected/baseline_ETTm1.json
logs/graphmamba_ett_stage1_weighted/selected/baseline_ETTm2.json
```

之后各阶段命令遵循相同形式：

```bash
.venv/bin/python scripts/run_graphmamba_ett_stage1.py \
  --stage optimizer \
  --datasets ETTh1 \
  --base-config logs/graphmamba_ett_stage1_weighted/selected/baseline_ETTh1.json
```

其余三个数据集替换命令中的数据集名和 JSON 文件名分别运行。每一阶段结束后都先交给 Codex 分析，再为每个数据集分别获得 `optimizer_*.json`、`multiscale_*.json`、`graph_*.json`，并运行下一阶段。不要自行把测试 MSE 最低的配置写入 base config；这些阶段只按验证集选优。

| 阶段 | 默认任务数 | 作用 |
|---|---:|---|
| baseline | 16 | 建立四数据集、四长度的验证基准 |
| optimizer | 80 | 10 个训练候选 × 8 个 96/720 锚点任务 |
| multiscale | 48 | 6 个 Patch/分解候选 × 8 个锚点任务 |
| graph | 96 | 12 个图候选 × 8 个锚点任务 |
| capacity | 56 | 7 个容量候选 × 8 个锚点任务 |

如需先确认环境和命令，不训练：

```bash
.venv/bin/python scripts/run_graphmamba_ett_stage1.py \
  --stage baseline --dry-run --max-runs 1
```

所有搜索运行固定包含 `--test_after_train 0`。单次完成记录会立即写入 JSON，所以终端断开或机器重启后，重新执行原命令即可断点续跑。

第一轮旧目录 `logs/graphmamba_ett_stage1/` 使用 batch 等权验证指标，仅保留作流程审计，不参与选优。正式搜索从 `logs/graphmamba_ett_stage1_weighted/` 开始，验证 MSE/MAE 对全部样本和变量按元素统一加权。

## 后续决策树

### A. Baseline 完成后

1. 检查 16/16 记录完整、无 failed 状态，验证 MSE/MAE 均为有限数。
2. 统计各任务 best epoch。若大量任务在前 5 epoch 达到最佳，optimizer 阶段优先比较 scheduler。
3. Baseline 仅用于建立验证参照，不和旧测试 MSE直接比较数值大小。
4. 为四个数据集分别输出基础配置，后续保持“每数据集一套、四个 horizon 共享”的原则。

### B. Optimizer 阶段

- 每个数据集只运行 96、720 两个锚点长度。
- 比较初始 lr、scheduler、batch size、dropout；按照两个锚点相对 baseline 的平均验证 MSE 排序。
- 只有当候选两个锚点都不明显退化，或总体改善超过 0.5%，才晋级。
- 若不同候选分别擅长 96 与 720，保留前 2 名进入全长度仲裁，而不是直接按单项最低值决定。

### C. Multiscale 阶段

- 从 optimizer 优胜配置出发，搜索 patch/stride 与 moving average。
- 若 720 改善、96 小幅退化不超过 0.5%，候选仍可保留到四长度确认。
- 若大 patch 导致两种锚点均退化，停止向更大 patch 扩展。

### D. Graph 阶段

- 先以 alpha=0 和 alpha=1 判断自适应图、静态图的单独贡献，再比较混合权重。
- top-k 只在 1–4 内搜索；ETT 仅 7 个变量，避免接近全连接后失去稀疏图意义。
- 若 alpha/top-k 改变的验证收益小于 0.5%，保留较简单配置，并把主要提升资源转向结构创新。

### E. Capacity 阶段

- 最后才比较 d_model、d_state、d_conv 和层数，防止用参数量掩盖前面模块设计问题。
- 候选若改善不足 0.5% 且参数量或耗时增加超过 25%，不晋级。
- `d_model=128` 或 2 层只在验证提升稳定时保留。

### F. 全长度确认

- 每个数据集取累计前 3 套配置，运行 96/192/336/720。
- 用四个长度归一化验证 MSE 的平均值选择最终数据集配置。
- 若前两名差距小于 0.5%，选择 MAE 更低者；仍接近则选择参数更少者。
- 输出冻结配置清单，此时才允许正式测试。

### G. 正式测试后的分流

- 达到 G2：进入三种子确认和消融实验。
- 达到 G1 但未达到 G2：只进行一次基于验证集的定向细化。
- 未达到 G1：结束超参数扩搜，进入第二阶段图条件化 Mamba 结构改进。
