# GraphMamba–CMRHM 第二轮模拟审稿

## Review setup

- **Input scope:** 完整中文稿，包含摘要、引言、相关工作、方法、实验、结论及7张表；同时核对本地实验记录与核心实现。
- **Assessment boundary:** 尚无正式图1、补充表S1、完整参考文献/BibTeX 和目标期刊格式；Solar-Energy 基线矩阵不完整。审稿不假设这些材料已经存在。
- **Shared manuscript claim summary:** 论文提出近期—远期非对称建模，并以 CMRHM 将压缩远期历史限制为近期条件下的预测修正，从而在短输入成本附近利用更长历史。
- **Visible evidence base:** 五数据集20任务统一主比较；ETTm1/ETTm2×四跨度×三种子的 Recent96 对照；四任务内部设计分析；五类历史干预；TimeXer 迁移；Recent96/Raw336/CMRHM 的精度与资源对照。
- **Missing materials affecting confidence:** 主比较只有随机种子2021；TimeMixer 有6个数值发散单元；缺少更近期且与本文最接近的方法的同协议比较；方法总图和完整 MAE 表尚未交付。

## Reviewer 1

- **Overall assessment:** 中心假设清楚，控制实验比一般组合式预测论文更诚实；但主结果的公平性与复现信息尚不足以完全建立性能主张。
- **Who would be interested in the results, and why:** 长期预测、边缘部署与长上下文建模研究者会关注这种把历史范围与主干输入长度解耦的设计，尤其是其明确报告精度—成本边界。
- **Major strengths:** Recent96 是直接且严格的增量对照；历史错配实验验证模型确实使用样本对应的远期信息；Raw336 对照没有回避精度失败；代码核对确认静态图只使用训练区间，未发现图构建数据泄漏。
- **Major concerns:** 方法公式未显式写出解码前的 GELU；核心训练设置、硬件和软件版本缺失；主表只含单随机种子，且一个重要基线在多个任务上数值发散；主文声称有补充 MAE 表，但当前材料未见该表。
- **Technical failings that need to be addressed before the case is established:** 补齐 $D(\mathbf z)=\mathbf W_d\operatorname{GELU}(\mathbf z)$；报告记忆池化率、隐藏维度、patch/stride、优化器、学习率、批量、早停与硬件；解释并修复或重新验证 TimeMixer 发散，不能长期把异常基线直接排除；明确表7是验证集、两个数据集和两个跨度的局部效率结论。
- **Assessment against Nature-style criteria:** 原创性为“可辨认但需更强最近工作区分”；科学重要性目前主要属于时间序列领域内部；跨领域意义来自资源受限长上下文，但证据覆盖仍窄；技术可靠性中等，核心控制较强而主比较仍需加固；中文可读性明显提高，但公式与实验协议仍需补足。
- **Recommendation posture:** 解决实验公平性与复现缺口后可支持进入一般高质量领域期刊评审；当前证据不足以支持更强的广泛影响主张。

## Reviewer 2

- **Overall assessment:** 非对称历史职责是本文最有价值的概念贡献，但必须与“压缩长历史”“残差校正”“多尺度记忆”等邻近思想做更精确的技术区分。
- **Who would be interested in the results, and why:** 研究有效上下文长度、记忆压缩、预测器插件和模型效率的读者会关心该框架是否提供一种可迁移的长历史接口。
- **Major strengths:** 论文没有把 Graph、Mamba、patch 强行拆成多个创新；内部消融失败被转化为设计边界；TimeXer 迁移初步说明 CMRHM 并非只适配图状态空间骨干。
- **Major concerns:** 当前相关工作只用两段概述 Attraos、TimeMixer、DiM 和 Wrinkles，尚不足以证明“近期基础预测+远期条件修正”的新颖边界；跨主干证据只有一个额外主干、两个同族数据集和单种子；内部设计分析显示更简单参数化经常更优，削弱“配对差分+通道门控”作为特定设计的必要性。
- **Technical failings that need to be addressed before the case is established:** 在相关工作中增加逐机制区分，而不是只说职责不同；贡献表述持续锁定为整体条件修正路径；若投稿强调“可迁移”，需在另一数据域或更多种子上验证，否则只能表述为初步可迁移证据；需要一张概念图让读者直观看到统一长输入、简单拼接与条件差分的区别。
- **Assessment against Nature-style criteria:** 原创性潜力来自职责非对称而非单个算子；科学重要性取决于该原则能否跨架构/数据域成立；跨学科意义尚未由现有两个 ETTm 数据集建立；技术上对负结果处理可信，但设计必要性不足；核心概念对非专门读者仍需要图示。
- **Recommendation posture:** 概念有发表价值，但应以“有边界的算法机制”定位；若保持普适或广泛影响措辞，现有证据不够。

## Reviewer 3

- **Overall assessment:** 文章结构已经接近正式论文，章节和表格职责清楚；最大阅读障碍从“结果堆砌”转为“主线之外仍有少量实现与进度语言”。
- **Who would be interested in the results, and why:** 除预测模型研究者外，关注低延迟传感器分析、设备端推理和长记录压缩的读者可能对精度—成本权衡感兴趣。
- **Major strengths:** 摘要按问题—缺口—方法—证据—边界推进；表2沿用参考论文的一问一表逻辑；结论不再逐表复述；负面结果提高了可信度。
- **Major concerns:** 图1说明仍使用“应同时给出”这类作者笔记语气；Solar 缺失在正文中呈现为项目进度，而非稳定的评估范围；“显著降低”容易被理解为统计显著；专业读者还会询问为什么 MSE 在主文、MAE 在未提供的补充材料。
- **Technical failings that need to be addressed before the case is established:** 将图注改成成稿语气；将 Solar 表述改为“当前版本的预设报告范围”，不以待办语气出现；不用“显著”描述未经统计检验的成本差异；说明主文选用 MSE 的理由，并实际生成补充 MAE 表。
- **Assessment against Nature-style criteria:** 可读性在领域论文层面合格；对非专业读者而言，“配对解码差分”的直觉仍依赖图1；科学意义目前是清晰的工程—方法学折中，而非已证明的跨领域结论；技术表述总体克制；原创性叙述比上一稿集中。
- **Recommendation posture:** 文字和版式经过一轮定点修订即可达到完整领域稿水平，但实验公平性问题仍需单独解决。

## Cross-review synthesis

- **Consensus strengths:** 单一主贡献已经稳定；Recent96、历史干预和 Raw336 构成互补证据；对内部变体和长输入失败的报告诚实；结构明显借鉴了 DiM/Wrinkles 的问题—模块—证据镜像关系。
- **Consensus technical risks:** 主比较单种子；TimeMixer 数值发散；强近期基线不足；方法和训练细节不完整；跨主干证据范围有限。
- **Where emphasis differs across reviewers:** Reviewer 1 最关注公平协议与复现；Reviewer 2 最关注新颖性边界与跨架构外推；Reviewer 3 最关注成稿语气、图示和表格入口。
- **Broad-interest / significance readout:** 现有证据足以支撑时间序列预测领域内的成本受控长历史机制，但不足以建立跨领域、普适或远-reaching 的结论。
- **Most important issues to resolve before a strong case is established:** 先修正公式和实验设置；再稳定 TimeMixer 或把该基线从主表降级并给出可复核原因；补齐 MAE 补充表；在投稿前增加至少一个近期强方法的同协议比较或给出严格协议差异说明。

## Risk / unsupported claims

- “显著降低”没有统计检验，需改为定量但非显著性措辞。
- “可迁移”若不带“初步”限定，超出一个额外主干、两个数据集、单种子的证据范围。
- Solar-Energy 尚不能进入六数据集排名；不得在摘要或结论暗示六数据集完整 SOTA。
- 配对差分与通道级门控的独立必要性不受当前消融支持。
- 最近工作覆盖和 BibTeX 尚未完整核验，原创性判断仍受该缺口限制。
