# GraphMamba 共享核心多尺度扫描：创新性预审计

## 审计对象

候选机制不是普通“双尺度 patch”，而是：

1. 同一历史生成长、短两套 patch token；
2. 两个尺度不拼成伪时间序列，而是分别重置 SSM state；
3. 两次扫描共享同一个 Mamba 核心参数；
4. 编码后沿 token 维融合，在参数量不增加的情况下形成双分辨率表示。

本审计完成于 2026-08-13，检索来源包括 arXiv、PMLR、OpenReview 和出版社页面，优先
使用论文原文或官方索引页。

## 最接近工作

| 工作 | 已有机制 | 与本候选的重合 | 关键差异 |
|---|---|---|---|
| MTST, AISTATS 2024 | 不同 patch size 的多分辨率 token；各分支独立建模并融合 | “多 patch 分辨率、分支处理后融合”高度重合 | Transformer 分支；未以共享 Mamba state reset 为问题核心 |
| SST, arXiv 2024 | coarse/global 与 fine/local 分开建模；Mamba 和局部 Transformer 专家后路由 | “长短尺度职责分离”重合 | 异构专家，不共享 Mamba 核心 |
| UmambaTSF, arXiv 2024 | U-shaped 多尺度 MLP 与 Mamba 长序列表示结合 | 多尺度 Mamba forecasting 大方向重合 | 尺度来自 U-shaped encoder-decoder，不是双 patch 共享扫描 |
| ms-Mamba, arXiv 2025 / Neurocomputing 2026 | 多个并行 Mamba block，以不同内部 `Delta` 表达多个时间尺度 | “并行多尺度 Mamba”高度重合 | 输入相同、尺度在 SSM 内部；使用多个 block，不是外部 patch 尺度共享权重 |
| 2025–2026 multi-scale Mamba submissions | 多尺度 patcher、频域/层级分支、Mamba 专家与动态融合 | 进一步压缩“多尺度 + Mamba”的宽泛新颖性 | 具体权重共享和 state reset 仍可能不同，但不能支持宽泛首创声明 |

主要来源：

- MTST: https://proceedings.mlr.press/v238/zhang24l.html
- SST: https://arxiv.org/abs/2404.14757
- UmambaTSF: https://arxiv.org/abs/2410.11278
- ms-Mamba: https://arxiv.org/abs/2504.07654；正式 DOI
  `10.1016/j.neucom.2026.133226`

## 可主张与不可主张

不可主张：

- 首个多尺度 Mamba forecasting；
- 首个多 patch time-series encoder；
- 首个长短尺度分支建模；
- 简单“长短 patch 分开进入 Mamba”本身是核心创新。

当前可以谨慎检验的窄主张：

> 对覆盖同一历史的异分辨率 patch，连续拼接会引入不存在的尺度边界状态传递。通过按尺度
> 重置 state 并共享 SSM dynamics，可在零额外核心参数下避免伪连续性，同时保持跨尺度
> 动力学一致性。

这个主张要成为论文贡献，至少需要三类证据：

1. **必要性**：joint scan 在尺度边界产生可量化的状态污染，并且 independent scan 稳定
   改善；
2. **共享价值**：shared-independent 优于或接近 two-encoder independent，同时显著减少
   参数和显存；
3. **机制深化**：共享动力学需要一个尺度可辨识机制，例如 scale-conditioned `Delta`、
   轻量 adapter 或跨尺度状态一致性约束，而不是只调用同一模块两次。

## 创新潜力评级

| 定位 | 当前评级 | 条件 |
|---|---|---|
| 单独作为论文核心创新 | 低 | 与 MTST/ms-Mamba 的方法邻域过近，且当前仅是控制流修正 |
| 作为 GraphMamba 主干的辅助创新 | 中 | 参数中性、跨数据集稳定改善，并有 joint/shared/separate 完整消融 |
| 与 LagGraph 形成统一机制 | 中高潜力、未验证 | 需要让尺度特定 lag/delay 直接控制共享 state dynamics，而非简单模块叠加 |
| 作为工程/语义修正 | 高可信 | 即使精度中性，也可避免无时间含义的跨尺度状态传播，但不能当性能贡献 |

## 实验前 Go/No-Go

第一阶段只做 `joint` 对 `independent_shared`，LagGraph 关闭。

- Go：至少两个任务同向，宏平均 MSE 改善不低于 0.5%，MAE不明显恶化；随后进入多 seed
  和 `independent_separate` 对照。
- Weak Go：宏平均 0–0.5%，但边界诊断清晰且效率不退；只保留为辅助/语义修正。
- No-Go：宏平均退化或结果强烈分裂；恢复 joint 默认，不增加 scale adapter。

所有选择均只基于 validation；本阶段不访问 test。
