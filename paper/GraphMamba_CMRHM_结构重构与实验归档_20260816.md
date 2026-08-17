# GraphMamba–CMRHM 结构重构与实验归档（2026-08-16）

## 1. 重构依据

本轮以两篇 Neurocomputing 论文为直接参照：

1. Mo et al., *DiM: Improving multivariate time series forecasting with DI embedding and multi-head graph learning mechanism*, 659, 131777, DOI: 10.1016/j.neucom.2025.131777。
2. Chen et al., *Wrinkles in time: Multi-scale patching and super-resolution for efficient time series forecasting*, 676, 133021, DOI: 10.1016/j.neucom.2026.133021。

正文暂用引用键 `@Mo2026DiM` 和 `@Chen2026Wrinkles`；项目目前没有 BibTeX 文件，正式排版前需建立并核验这两个条目及其他引用键。

DiM 的核心写法是让“两类任务困难—两个对应设计—两组消融证据”严格镜像；Wrinkles 则先确定效率目标，再对分解后的季节项与趋势项分别施加针对性增强，并用精度、参数量、延迟和稳定性共同闭环。两篇论文都没有把所有实现组件提升为同等级创新。

据此，GraphMamba–CMRHM 的主线收束为：

> 长历史不应全部以原始高分辨率进入预测主干；近期窗口负责基础预测，远期历史经压缩后只在近期状态条件下形成受控增量修正。

## 2. 逐章规划

| 章节 | 核心任务 | 保留内容 | 删除或迁移内容 |
|---|---|---|---|
| 摘要 | 问题—方法—证据—边界 | 非对称历史分工、CMRHM、主结果占位 | 主干组件清单、未冻结数字 |
| 1 引言 | 建立“更长不等于更有效”的缺口 | 统一长输入的局限、近期预测/远期修正、3条贡献 | 具体模型罗列、详细结果、周期分支 |
| 2 相关工作 | 只保留与主张直接相邻的两条路线 | 多变量预测；长上下文与多尺度历史利用 | 独立的 Mamba/图综述段落 |
| 3 方法 | 解释统一框架与可复现机制 | 问题定义、总体框架、近期预测器实例、CMRHM、目标与复杂度 | 周期多分辨率独立小节、固定配置清单、重复边界说明 |
| 4 实验 | 让每项实验回答一个主张 | 设置、主结果、核心消融、历史干预/迁移、效率与边界 | 逐格复述、失败的候选结构、探索性周期四格主表 |
| 5 结论 | 总结贡献、证据和限制 | 条件历史利用及适用边界 | 再次罗列网络模块 |

## 3. 术语表

| 规范术语 | 首次定义 | 禁止混用 |
|---|---|---|
| 近期窗口 | recent window, $\mathbf{X}^r$ | 短窗口、局部窗口交替使用 |
| 远期历史 | earlier history, $\mathbf{X}^o$ | 旧历史、长期历史无定义切换 |
| 近期预测器 | recent predictor, $F_\theta$ | 主干、近期路径、GraphMamba 随意互换 |
| 条件多分辨率历史记忆 | Conditioned Multi-Resolution History Memory (CMRHM) | long memory、history module |
| 基础预测 | base forecast, $\widehat{\mathbf{Y}}^b$ | 初始预测、短期预测 |
| 历史修正 | history correction, $\Delta\mathbf{Y}$ | 残差、增量、补偿项无定义切换 |
| 通道级有界系数 | bounded channel-wise coefficient, $g_n$ | 动态门控（当前实现并非样本动态门控） |

## 4. 从主稿删除的内容及保存位置

### 4.1 周期多分辨率近期输入

从方法主线删除。原因不是该方向毫无价值，而是四格实验未证明它与 CMRHM 稳定协同，且统一主实验使用 `independent_shared` 双 patch。完整设计与结果仍保存在：

- `experiment_results/GraphMamba_Q2_factorial_validation.md`
- `experiment_results/GraphMamba_CMRHM_Q2_writing_readiness.md`
- `models/GraphMamba.py`

### 4.2 大型主结果表与逐格复述

从当前中文主稿删除，等待晚间实验完成后按统一协议重建。原始与汇总结果仍保存在：

- `experiment_results/GraphMamba_CMRHM_six_dataset_final_test.md`
- `experiment_results/GraphMamba_CMRHM_all_horizons.md`
- `experiment_results/GraphMamba_CMRHM_final_test.md`
- `logs/graphmamba_cmrhm_six_dataset_final/`

### 4.3 失败或不充分支持的消融

以下结果不进入主文核心证据链，但不得丢失：

- CMRHM 内部 Concat/NoDiff/GlobalGate 未通过联合门槛：`experiment_results/GraphMamba_CMRHM_strict_evidence_result.md`。
- 固定图—Mamba 融合的局部负结果：`experiment_results/GraphMamba_CMRHM_joint_backbone_ablation_result.md`。
- 自适应融合候选未通过验收：`experiment_results/GraphMamba_CMRHM_adaptive_fusion_validation.md`。
- 跨数据域移除 Mamba 的完整审计：`experiment_results/GraphMamba_CMRHM_no_mamba_cross_domain_validation.md`。

这些结果要求正文收缩主张：可以说明 CMRHM 的完整实现，但不能声称条件差分、通道级系数或固定 Graph–Mamba 融合各自在所有任务上不可替代。

## 5. 晚间实验回填顺序

1. 先冻结同协议强基线主表，回填第4.2节和摘要中的第一组数字。
2. 再回填 Recent96—CMRHM 三随机种子配对，作为核心稳定性证据。
3. 回填最小核心消融；若内部机制仍未胜过简化变体，则保留收缩措辞。
4. 回填旧历史干预与 TimeXer 迁移，分别支撑“确实使用样本匹配历史”和“非单一主干技巧”。
5. 最后回填 Recent96/Raw336/CMRHM 的精度—成本表，并据此写适用边界。

## 6. 主张—证据约束

| 主张 | 所需证据 | 当前状态 |
|---|---|---|
| CMRHM 稳定补充近期预测 | ETTm1/ETTm2 四跨度三随机种子配对 | 已有强证据，待最终表格化 |
| 收益来自匹配的远期历史 | 样本错配、时间扰动、均值/噪声替代 | 已有支持，但时间顺序依赖具有数据差异 |
| CMRHM 不绑定 GraphMamba | TimeXer 配对迁移 | 已有支持，范围仅一个额外主干 |
| CMRHM 提供更好成本折中 | Recent96、Raw336、CMRHM 同协议效率比较 | 支持成本折中，不支持全面优于 Raw336 |
| 条件差分和通道级系数各自必要 | NoDiff、Concat、GlobalGate | 当前不支持，不得作为独立实证贡献 |
| Graph 与 Mamba 稳定协同 | 跨域单分支/双分支消融 | 当前只支持任务相关互补，不支持普遍协同 |
