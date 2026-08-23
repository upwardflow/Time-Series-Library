# GraphMamba–CMRHM 引言与相关工作三视角审定报告

## Review setup

- **Input scope：** 激进修订版引言与相关工作，并核对原始初稿中的摘要、方法、实验和结论。
- **Assessment boundary：** 审定论证、新颖性定位、证据一致性和可读性；不重新运行实验，也不对未提供的投稿格式作判断。
- **Shared manuscript claim summary：** CMRHM 将近期基础预测与远期条件修正分离，在成本受控的条件下利用更早历史，但不声称全面优于直接长输入。
- **Visible evidence base：** 五数据集统一主表；ETTm1/ETTm2 三随机种子 Recent96 对照；远期历史干预；TimeXer 跨主干结果；Raw336 精度—资源对照；17 条已核验文献。
- **Missing materials affecting confidence：** 跨主干证据仍为单个额外主干、两个数据集和单随机种子；本报告不据此认定普适性。

## Reviewer 1

- **Overall assessment：** 技术论证已达到可合并状态。引言提出的三类实证判断均能在实验章节找到对应证据，且主动保留数据集和资源折中的边界。
- **Who would be interested, and why：** 长期预测、长上下文建模和高效序列模型研究者会关注该工作，因为它改变的不是单个编码器，而是长历史进入预测的职责。
- **Major strengths：** 近期预测、远期修正和直接长输入三者界限明确；第三条贡献与多随机种子、干预、迁移和效率结果一致；未把内部消融的混合结果包装成普遍机制。
- **Major concerns：** 初版对 MEW 的外推超过原文评估范围，并把异质分支与“减少表征冲突”作了偏强因果连接。
- **Technical failings requiring action：** 将 MEW 限定到其评估的 Transformer 类模型；把“减少冲突”改为“提供设计先例”。两项已在审定版修正。
- **Nature-style criteria：** originality 为有边界的建模职责创新；significance 主要面向时间序列方法社区；interdisciplinary reach 有限但工程意义清楚；technical soundness 在现有文本范围内成立；readability 良好。
- **Recommendation posture：** 支持在上述局部修正后合并。

## Reviewer 2

- **Overall assessment：** 新颖性定位较原稿明显增强。正文主动承认图学习、Mamba/SSM 及其联合建模均有先例，把贡献收缩到远期条件修正，降低了被最近同刊工作直接否定的风险。
- **Who would be interested, and why：** 研究模型模块化、记忆机制及历史窗口有效性的读者会关注“访问范围与主干长度解耦”的思想。
- **Major strengths：** 对 MEW、MPGTimer、Attraos、G-Mamba 和 SSMGNN 的比较基于机制轴而非性能宣传；“保留什么历史”与“历史承担何种职责”的区分清楚。
- **Major concerns：** CMRHM 仍可能被审稿人解释为压缩历史残差分支，因此贡献必须持续落在同一近期状态下的配对解码差值和受限注入，而不能退回“加入长期分支”的宽泛表述。
- **Technical failings requiring action：** 合并时必须原样保留配对解码与边际变化的操作性定义；全文其他章节不得把 Graph+Mamba 重新列为创新。当前方法章节与该要求一致。
- **Nature-style criteria：** originality 可辨识；scientific importance 属于有价值但领域内的算法贡献；interdisciplinary interest 不宜夸大；technical soundness 由多层证据支持；readability 对本领域读者清晰。
- **Recommendation posture：** 支持合并，并建议英文稿继续保持克制的新颖性措辞。

## Reviewer 3

- **Overall assessment：** 结构清晰，但初版引言第二段模型名称过密，会延迟非专业读者理解真正缺口。
- **Who would be interested, and why：** 除预测模型研究者外，关注长序列效率和条件记忆设计的机器学习读者也能理解该问题。
- **Major strengths：** 首段快速建立长窗口的信息—干扰—成本矛盾；第三段明确这是设计假设而非普遍事实；相关工作每段均以 CMRHM 的区别收束。
- **Major concerns：** 过多模型名和未定义的 PIH 缩写影响阅读；“有用但稀疏”可能被误读为已经证明的信息稀疏性。
- **Technical failings requiring action：** 引言第二段改为技术类别概括，将 `MEW/PIH` 改为 `MEW 工作`，删除未经验证的“稀疏”属性。三项已在审定版修正。
- **Nature-style criteria：** originality 与 technical soundness 易于定位；跨学科广度有限；修正后的引言对非专门从事时序预测的机器学习读者更友好。
- **Recommendation posture：** 支持合并修正后的版本。

## Cross-review synthesis

- **Consensus strengths：** 缺口—机制—证据已形成镜像；最近邻文献覆盖充分；新颖性边界诚实；贡献与实验一致。
- **Consensus technical risks：** 跨主干范围有限，配对差值和通道系数的单独优越性不稳定，主表优势并非全面覆盖。
- **Where emphasis differs：** Reviewer 1 更关注主张与实验对应；Reviewer 2 更关注残差分支解释和最近邻定位；Reviewer 3 更关注模型清单对可读性的影响。
- **Broad-interest/significance readout：** 该工作对长上下文预测和高效模型设计具有明确领域价值，但不应写成跨学科基础性突破。
- **Most important issues before merge：** 限定 MEW 证据范围、压缩引言模型清单、删除未证实的因果和稀疏性表述；均已修正。

## Risk / unsupported claims

- 不支持“CMRHM 对任意主干普适”；当前只能写“初步跨主干证据”。
- 不支持“配对差值或通道系数在所有任务独立占优”；只能把它们作为完整 CMRHM 的实现组成。
- 不支持“全面超过直接长输入”或“全面领先所有基线”。
- 不支持将 Graph、Mamba、patch、分解或图—SSM 组合作为本文独立创新。

## 审定结论

完成上述三项局部收紧后，引言与相关工作可以合并进原始初稿。审定状态：**通过，可合并；保留已列出的实验外推边界。**

## 合并后验证

- 第 1、2 节与审定版内容一致。
- 摘要及第 3–5 节未被本轮合并覆盖。
- 全文 17 个 citation keys 与 17 条 BibTeX 一一对应。
- Pandoc `citeproc` 与 Biber 数据模型校验通过。
