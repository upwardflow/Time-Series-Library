# CMRHM阶段结论与GraphMamba后续创新交接

## 1. 阶段目标

本阶段验证的核心问题是：CMRHM是否只是GraphMamba上的局部补丁，还是能够在不同预测
骨干上复用的条件化长历史增强机制。

最终结论：CMRHM在GraphMamba和非图Transformer骨干TimeXer上均取得稳定改善，具备
明确的初步跨骨干证据。本阶段到此冻结，不再继续扩展CMRHM结构；下一阶段研究重心返回
GraphMamba本体创新。

## 2. CMRHM核心思路

CMRHM不直接将全部长历史输入预测主干，也不让旧历史独立预测完整残差。其核心是估计：

> 在给定近期高分辨率上下文后，压缩旧历史还能提供多少新增预测信息。

前向路径为：

```text
近期96点 ─────────────────────────→ 预测主干 → 基础预测
    │
    └──────────────→ 近期上下文表示 zr

旧240点 → 16倍压缩 → 15个记忆token → 历史记忆 m

条件历史增量 = Decoder(zr + m) - Decoder(zr)

最终预测 = 基础预测 + tanh(变量门控) × 条件历史增量
```

对应形式：

```text
Y_hat = Y_recent + G ⊙ [D(Z_recent + M(X_old)) - D(Z_recent)]
```

机制由三部分组成：

1. 多分辨率旧历史压缩：低成本保留远期历史信息。
2. 共享配对差分解码：隔离旧历史相对于近期上下文的边际贡献。
3. 变量级有界门控：根据变量需求控制修正强度，并限制不稳定放大。

## 3. GraphMamba上的冻结结果

协议：GraphMambaRecent和GraphMambaCMRHM都以`seq_len=336`加载相同样本；主干只处理
最后96点，只有CMRHM使用前240点。结构、池化率和隐藏维度在保护性验证前冻结。

### Validation

- ETTm1/ETTm2 × 96/192/336/720共8项。
- MSE与MAE均为8/8胜出。
- 任务宏平均改善：MSE 5.061%，MAE 2.984%。

### 一次性正式Test

- 8/8任务MSE与MAE均胜出。
- 任务宏平均改善：MSE 7.732%，MAE 3.605%。
- 所有checkpoint只由validation选择，test结果没有用于调参。

详细记录：

- `experiment_results/GraphMamba_CMRHM_all_horizons.md`
- `experiment_results/GraphMamba_CMRHM_final_test.md`

## 4. TimeXer跨骨干迁移结果

严格对照使用TimeXerRecent与TimeXerCMRHM：两者都读取`seq_len=336`的相同样本，
TimeXer主干都只使用最后96点，候选唯一新增路径为冻结CMRHM-v1。

### Validation

| Dataset | Horizon | TimeXer MSE | +CMRHM MSE | MSE改善 | MAE改善 |
|---|---:|---:|---:|---:|---:|
| ETTm1 | 96 | 0.386517 | 0.373276 | 3.426% | 1.427% |
| ETTm1 | 720 | 0.946509 | 0.929434 | 1.804% | 1.599% |
| ETTm2 | 96 | 0.124883 | 0.123594 | 1.032% | 0.351% |
| ETTm2 | 720 | 0.291747 | 0.280561 | 3.834% | 2.385% |

- MSE/MAE胜率均为4/4。
- 宏平均改善：MSE 2.524%，MAE 1.441%。
- 最佳checkpoint的记忆门控均明显非零。

### 一次性正式Test

| Dataset | Horizon | TimeXer MSE | +CMRHM MSE | MSE改善 | TimeXer MAE | +CMRHM MAE | MAE改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTm1 | 96 | 0.321663 | 0.297778 | 7.426% | 0.358360 | 0.350331 | 2.240% |
| ETTm1 | 720 | 0.452553 | 0.432111 | 4.517% | 0.441367 | 0.429968 | 2.583% |
| ETTm2 | 96 | 0.171434 | 0.164905 | 3.809% | 0.256019 | 0.253215 | 1.096% |
| ETTm2 | 720 | 0.396538 | 0.369014 | 6.941% | 0.396934 | 0.386716 | 2.574% |

- MSE/MAE胜率均为4/4。
- 宏平均改善：MSE 5.673%，MAE 2.123%。
- 8/8运行均为`is_training=0`，checkpoint只由validation最佳MSE选择。

详细记录：

- `experiment_results/TimeXer_CMRHM_transfer_validation.md`
- `experiment_results/TimeXer_CMRHM_final_test.md`

## 5. 当前可以支持的结论

可以主张：

- 条件化旧历史增量在GraphMamba和TimeXer两个结构不同的骨干上均有效。
- CMRHM并不依赖GraphMamba的图分支，具有初步跨骨干可移植性。
- 旧历史压缩、配对差分与有界注入构成一个完整的长历史利用机制。

暂时不能主张：

- CMRHM是已经充分验证的通用插件。
- 对所有数据域、骨干和随机种子都稳定有效。
- CMRHM本身可以替代GraphMamba主体创新。

限制来自：当前迁移证据仍为单seed、ETTm1/ETTm2两个同族数据集和一个非图骨干。

## 6. 数据使用边界

以下test结果已经消费，不得用于后续结构选择或超参数调整：

- GraphMamba/GraphMambaCMRHM：ETTm1、ETTm2的96/192/336/720。
- TimeXerRecent/TimeXerCMRHM：ETTm1、ETTm2的96/720。

后续GraphMamba创新必须先使用未消费任务的validation进行诊断和选择。已消费test只能作为
冻结历史结果保留，不能反向决定新模块、图结构、门控、损失函数或训练参数。

## 7. 下一阶段研究决策

CMRHM保留为GraphMamba上的有效增强和辅助贡献，但下一阶段不再深化CMRHM，而是回到
GraphMamba本体，寻找属于图—时间建模主干的核心创新。

下一阶段必须遵守：

1. 避免把“图网络 + 双向Mamba”本身作为创新，因为与Neurocomputing 2026 G-Mamba重合。
2. 避免直接将图邻域上下文注入SSM状态选择参数，以免与G-Mamba核心机制再次接近。
3. 不继续堆叠已经验证收益不足的局部融合、图混合比例、普通动态图、Patch尺度和分解校准。
4. 新方向先进行只读诊断或反事实上界；跨至少两个数据集达到1%信号后，才实现端到端候选。
5. 一个阶段只授权一个机制变化，保持与原GraphMamba严格配对并记录参数、显存和耗时。

优先研究假设为“预测跨度条件的变量关系”：不同未来预测区间可能需要不同的跨变量传播
图，而当前GraphMamba对全部预测步共享同一图表示。该假设与G-Mamba的拓扑感知状态更新
问题不同，但必须先通过未消费validation任务诊断后才能进入模型实现。

## 8. 阶段状态

- CMRHM机制：冻结。
- GraphMamba与TimeXer迁移结果：已保存。
- 已消费test：登记完成。
- CMRHM进一步结构搜索：停止。
- 下一活动阶段：GraphMamba本体第二阶段创新诊断。
