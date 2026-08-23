# GraphMamba–CMRHM：Neurocomputing 相关论文与中文写作地图

检索日期：2026-08-14  
检索范围：Neurocomputing 正式发表或已给出正式 DOI 的时间序列预测论文；关键词覆盖 Mamba、state-space model、graph、multiscale、long-context、patch、periodicity 和 multivariate time-series forecasting。以下信息优先依据 ScienceDirect 出版商页面。

## 一、最需要精读和正面区分的论文

| 优先级 | 论文 | 核心方法 | 与本文的关系 | 中文写作中的处理方式 |
|---|---|---|---|---|
| A1 | **Graph-enhanced Mamba: Efficient spatiotemporal sequence modeling with selective state space and graph neural networks**，Neurocomputing 680 (2026), 133280，DOI: [10.1016/j.neucom.2026.133280](https://www.sciencedirect.com/science/article/pii/S0925231226006776) | 在选择性状态更新中注入图邻域信息，并结合静态拓扑、动态邻接和门控融合 | 与 Graph + Mamba 的重合度最高 | 必须承认图增强 Mamba 已有正式工作；本文新颖性不能建立在 Graph–Mamba 组合本身，而应转向 CMRHM 的远期历史利用机制 |
| A2 | **MPGTimer: A long-context multi-scale patch and graph-enhanced transformer for time series forecasting**，Neurocomputing (2026), 134375，DOI: [10.1016/j.neucom.2026.134375](https://www.sciencedirect.com/science/article/pii/S092523122601773X) | 多尺度 patch 集成、图增强交互注意力和长上下文预测 | 与“长历史 + 多尺度 patch + 图关系”高度接近 | 必须比较：MPGTimer 联合编码长上下文；CMRHM 保持近期主干不变，只压缩旧历史，并以条件边际修正形式注入 |
| A3 | **ms-Mamba: Multi-scale Mamba for time-series forecasting**，Neurocomputing 680 (2026), 133226，DOI: [10.1016/j.neucom.2026.133226](https://www.sciencedirect.com/science/article/abs/pii/S0925231226006235) | 通过具有不同采样率的多个 Mamba 块并行建模多时间尺度 | 与周期多分辨率输入直接相邻 | 不要把“多尺度 Mamba”作为首创；强调本文周期输入由已知采样周期组织近期 token，而 CMRHM 专门处理旧历史 |
| A4 | **Is Mamba effective for time series forecasting?**，Neurocomputing 619 (2025), 129178，DOI: [10.1016/j.neucom.2024.129178](https://www.sciencedirect.com/science/article/abs/pii/S0925231224019490) | S-Mamba 使用双向 Mamba 建模变量间相关性，并以 FFN 建模时间依赖 | Mamba 时序预测的基础直接基线 | 相关工作中用于说明 Mamba 的效率与变量建模能力；实验主表应考虑加入 S-Mamba |
| A5 | **SSMGNN: Spectral temporal graph neural network with state space models for multivariate time-series forecasting**，Neurocomputing 666 (2026), 132295，DOI: [10.1016/j.neucom.2025.132295](https://www.sciencedirect.com/science/article/pii/S0925231225029674) | 将动态谱滤波、Fourier 图算子、状态空间模型和多尺度融合结合 | 与图、SSM、多尺度三部分均相邻 | 用于区分“频谱图滤波”与本文“时域旧历史条件记忆”；建议作为近期强基线或至少放入相关工作 |

## 二、补充支撑论文

| 论文 | 可支撑的论述 | 与本文的关键差异 |
|---|---|---|
| **TF4TF: Multi-semantic modeling within the time–frequency domain for long-term time-series forecasting**，Neurocomputing 617 (2025), 128913，[DOI/页面](https://www.sciencedirect.com/science/article/pii/S0925231224016849) | Patch 后的局部—全局、多尺度及时频建模是长时预测的重要路线 | TF4TF 在统一主干内挖掘多语义；CMRHM 隔离旧历史，并输出相对近期预测状态的边际修正 |
| **Multi-resolution aware pivot-guided dynamic graph neural network for multivariate time series forecasting**，Neurocomputing 683 (2026), 133477，[DOI/页面](https://www.sciencedirect.com/science/article/pii/S092523122600874X) | 多分辨率信息与分辨率相关图结构对多变量预测有价值 | 该方法迭代分解并在各分辨率学习动态图；本文的图主干并非主要创新，重点是远期历史记忆 |
| **ADMS-LSTM: A multi-scale stacked LSTMs long-term prediction method based on an adaptive decomposition framework with DFT-AutoCorrelation**，Neurocomputing 657 (2025), 131362，[DOI/页面](https://www.sciencedirect.com/science/article/pii/S092523122502034X) | 多粒度分解能够帮助提取更远历史中的长期依赖 | 其核心是 DFT/自相关驱动的金字塔分解和多尺度预测融合；CMRHM 不生成多尺度预测，而是压缩旧历史形成受控修正 |
| **Multiple convolutional neural networks for multivariate time series prediction**，Neurocomputing 360 (2019), 107–119，[DOI/页面](https://www.sciencedirect.com/science/article/pii/S092523121930685X) | 周期信息长期以来就是时间序列预测的重要先验 | 该方法以多 CNN 捕获多个周期；本文只将周期用于近期 patch 组织，不能声称首次利用周期信息 |
| **SDVS-Net: A spatial dilated convolution and variable self-attention network for multivariate long-term time series forecasting**，Neurocomputing 619 (2025), 129148，[DOI/页面](https://www.sciencedirect.com/science/article/pii/S0925231224019192) | 长期多变量预测需要同时处理时间依赖和变量依赖 | SDVS-Net 侧重局部/全局变量关系；本文在既有图主干之外讨论如何利用远期历史 |

## 三、建议的中文“相关工作”结构

### 2.1 基于 Mamba 的时间序列预测

写作顺序：

1. Transformer 能够建模长依赖，但注意力计算随序列长度快速增长。
2. S-Mamba 将选择性状态空间模型引入时间序列预测，证明 Mamba 能够兼顾精度与效率。
3. ms-Mamba 从多个状态空间采样率建模多尺度动态；G-Mamba 与 SSMGNN 进一步引入图结构或谱图算子，增强变量间依赖建模。
4. 转折：这些研究主要改进预测主干内部的状态演化、尺度组织或图关系建模，并未直接解决“如何在不显著扩展近期主干的情况下，有选择地利用更早历史”这一问题。

### 2.2 多尺度、周期与长上下文预测

写作顺序：

1. 时间序列同时包含局部变化、周期波动与长期趋势，多尺度表示因此成为常见路线。
2. TF4TF、ADMS-LSTM 和多分辨率动态图模型分别从时频语义、金字塔分解及分辨率感知图学习角度整合多尺度信息。
3. MPGTimer 进一步面向长上下文，将多尺度 patch 与图增强交互结合，是本文必须正面对比的邻近工作。
4. 转折：现有方法大多将不同尺度或完整长上下文共同送入主干并进行融合；这可能增加计算负担，也可能使旧历史中的噪声干扰近期状态。本文据此采用“近期高分辨率主干 + 远期压缩记忆”的非对称处理方式。

### 2.3 本文研究缺口

建议表述：

> 现有研究已经证明了状态空间模型、图结构学习和多尺度表示在多变量时间序列预测中的有效性。然而，多数方法侧重于增强统一预测主干，或者将不同时间尺度的表示直接聚合，较少区分近期观测与远期历史在预测决策中的不同作用。直接扩大回溯窗口不仅增加计算开销，也可能将弱相关或冲突的旧信息引入近期表征。为此，本文关注一个更具体的问题：在保持近期预测主干及其高分辨率表征不变的前提下，如何压缩远期历史，并仅注入其相对于当前预测状态的有效边际信息。

注意：其中“多数”“较少”属于基于当前检索语料的归纳判断，不应写成“首次”或“尚无任何研究”。正式定稿前应通读 A1–A5 全文后再次核对。

## 四、可以直接用于引言的中文问题链

1. 多变量长期预测同时要求建模时间依赖和变量间关系。
2. Transformer 的长序列开销较高，Mamba 提供了近线性的序列建模方案。
3. 图增强 Mamba 和多尺度 Mamba 已分别改善结构关系与多尺度动态建模，但这两点本身已不足以构成新的核心贡献。
4. 延长输入窗口也不等于有效利用历史：越早的观测可能包含互补信息，也可能与当前状态无关甚至冲突。
5. 因此，需要一种不改变近期主干、能够压缩旧历史并按变量控制其边际贡献的机制。
6. CMRHM 据此将旧历史转化为条件修正量，并通过有界变量门控注入预测结果；周期多分辨率输入只负责近期尺度组织。

## 五、论文定位提醒

- 标题和贡献列表以 CMRHM/远期历史有效利用为中心。
- “GraphMamba”作为承载主干，而不是宣称 Graph + Mamba 的首创。
- “周期 patch”作为次级设计，不宣称首次周期建模或普遍协同。
- G-Mamba、MPGTimer、ms-Mamba 是最可能被审稿人用于质疑新颖性的三篇论文，必须在引言或相关工作中明确给出机制差异。
- Neurocomputing 的相关论文普遍重视多数据集主表、模块消融和效率指标；当前稿件仍需在写作期间完成强基线主表的协议核对。

