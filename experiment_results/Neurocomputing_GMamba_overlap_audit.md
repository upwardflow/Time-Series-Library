# Neurocomputing G-Mamba相似性风险审计与差异化计划

## Material Passport

- Audit target: Chen and Tang, *Graph-enhanced Mamba: Efficient spatiotemporal
  sequence modeling with selective state space and graph neural networks*
- Venue: Neurocomputing 680 (2026), 133280
- DOI: `10.1016/j.neucom.2026.133280`
- Evidence status: publisher metadata and publisher-indexed method excerpts verified;
  full PDF equation/figure-level audit remains pending
- Local candidate: frozen GraphMamba-CMRHM-v1
- Constraint: consumed ETTm1/ETTm2 test results cannot guide further tuning

## 已核验的G-Mamba方法核心

G-Mamba针对具有已知拓扑的结构约束时空预测，将双向Mamba的长程时间建模与图神经
网络结合。其图模块同时使用已知拓扑和可学习动态邻接；关键贡献不是简单后融合，而是
把邻域上下文注入选择性状态空间模块内部，以通道级方式调制状态转移、输入和读出参数，
从而形成拓扑感知的记忆更新。论文还使用时空门控融合以降低朴素加法/拼接的干扰。

## 与当前模型的逐层比较

| 维度 | Neurocomputing G-Mamba | 当前GraphMamba-CMRHM | 风险判断 |
|---|---|---|---|
| 研究问题 | 结构约束场景的拓扑一致性与长程时间建模 | 通用多变量长预测中的有限历史信息瓶颈 | 明显不同 |
| 图语义 | 已知物理拓扑 + 动态邻接 | 训练集变量距离相关图 + 节点嵌入自适应图 | 部分相似，语义不同 |
| 时间主干 | 双向Mamba | 双向Mamba | 高层相似 |
| 图与Mamba关系 | 邻域上下文进入SSM内部参数选择/状态更新 | 时间Mamba和Graph Mixer并行处理Patch token，输出相加 | 机制明显不同 |
| 多尺度来源 | Mamba主干捕获多尺度/非平稳依赖 | 双Patch尺度 + 分解；CMRHM再加入历史分辨率尺度 | 局部相似 |
| 核心新增机制 | 拓扑感知选择性状态更新 + 时空门控融合 | 旧240点压缩为15个token，配对对比解码，变量级有界记忆增量 | 明显不同 |
| 预测任务定位 | 结构化时空/节点预测 | 无预定义拓扑的通用多变量长时序预测 | 明显不同 |
| 效率主张 | 保留Mamba近线性扫描并增强局部可辨识性 | 主干仍只扫描近期96点，以低成本补充旧历史 | 可形成独立效率命题 |

## 风险等级

- 若以“GraphMamba：图神经网络与双向Mamba结合”为论文主贡献：**高风险**。
  同一期刊已有更紧耦合的G-Mamba，审稿人可能认为当前并行相加结构是较弱或增量版本。
- 若以“CMRHM：条件化多分辨率历史记忆”为核心，GraphMamba仅作为承载骨干：
  **中低风险**。两者研究缺口、信息路径和核心机制不同，但仍必须正面引用和比较。
- 若进一步证明CMRHM可迁移到至少一个非图预测主干：**低风险方向**。这能说明贡献
  属于长上下文记忆机制，而不是另一种Graph+Mamba组合。

## 推荐论文定位

避免使用`GraphMamba`或`Graph-enhanced Mamba`作为标题中心。建议的问题陈述为：

> 现有高效时序预测器通常只在固定近期窗口内建模；直接扩大主干上下文会增加计算并
> 引入冗余。如何在保持近期高分辨率主干不变的情况下，提取旧历史对当前预测真正新增
> 的信息？

建议核心贡献层级：

1. 诊断并量化近期固定窗口的历史信息瓶颈，以及无条件旧历史残差映射的跨区间失稳。
2. 提出CMRHM：近期高分辨率主干、旧历史压缩记忆、配对对比增量和变量级有界注入。
3. 证明该机制以低参数/运行开销在多长度、多数据集和不同预测主干上稳定有效。

其中第1、2项构成一个完整方法创新；第3项是使创新摆脱Graph-Mamba同质化的关键证据，
不能仅作为普通补充实验。

## 下一阶段计划

### P0：完整论文核验（必须）

- 获取G-Mamba合法全文，逐项核对Fig. 2、SSM调制公式、数据集、消融和复杂度。
- 建立claim-to-claim矩阵，确认其是否包含历史压缩、差分记忆或变量门控；未核验前不作
  “首次提出”表述。

### P1：冻结定位与命名（立即执行，不训练）

- 冻结CMRHM-v1，不使用已消费的ETTm1/ETTm2 test继续调参。
- 论文方法名以CMRHM或新的记忆导向名称为中心，GraphMamba写作一种backbone实例。
- 在Related Work中单设`Topology-aware Mamba`与`Efficient long-context memory`两条线，
  明确G-Mamba属于前者、本文属于后者。

### P2：决定性差异化实验（最高优先级）

- 将冻结CMRHM思想迁移到至少一个非图主干，优先选择当前仓库已能稳定运行的TimeXer
  或TimeMixer；先做ETTm1/ETTm2的96和720 validation gate。
- 成功门槛：4项中至少3项MSE下降、宏平均改善≥1%，MAE宏平均不恶化。
- 若失败，不修改CMRHM-v1或消费原test；如实将贡献限定为GraphMamba专属增强。

### P3：数据域外推

- 使用未参与结构设计的数据集：ETTh1、ETTh2、Weather、Solar，随后ECL/Traffic。
- 统一运行96/192/336/720；只通过validation选checkpoint，正式test一次性执行。
- 多seed在结构完全冻结后进行，至少3个seed报告均值±标准差。

### P4：机制必要性消融

- Recent336（无旧历史记忆）。
- Raw336主干（直接增加上下文）对比CMRHM的精度、显存、参数和耗时。
- 无条件旧历史支路，验证条件化的必要性。
- 直接`Decoder(Recent+Memory)`对比配对差分增量。
- 共享门控对比变量级有界门控。
- 池化率只在新validation任务选择，不能回看已消费的ETTm test。

### P5：效率与审稿防御

- 报告参数量、FLOPs、峰值显存、训练/推理时间随输入历史长度的增长曲线。
- 将G-Mamba作为最接近相关工作正面讨论；若其代码或完整配置可得，在共同适用的数据集
  上复现，否则只作机制对比，不制造不公平数值表。
- 论文结论限制为“条件化压缩历史记忆”，不宣称首个Graph-Mamba或首个图增强SSM。

## Go / No-Go节点

- **Go（强差异化）**：CMRHM在非图主干和至少两个非ETTm数据集上保持稳定收益。
- **Go（有限定位）**：只在GraphMamba有效，但跨ETT/Weather/Solar稳定；定位为特定
  backbone的长历史扩展，弱化通用插件主张。
- **No-Go/改投定位**：收益只集中于ETTm或无法通过Raw336效率对照；此时Neurocomputing
  同质化风险仍高，应缩小贡献或考虑更偏应用的投稿方向。
