# GraphMamba 同相位跨周期扫描验证结果

日期：2026-08-14  
状态：**结构成立，预测门失败，已从活动模型退役；未访问测试集。**

## 候选定义

输入季节项 `[B,N,96]` 按固定周期 `P=24` 重排为
`[B,N,24,4]`。每个相位形成一条跨 4 个完整周期的序列，随后把相位折叠
进 batch，得到 `[B*24,N,D,4]`，与局部 patch4/stride2 分支复用同一个
双向 Mamba。周期状态在 cycle 轴均值汇聚后还原为 24 个相位 token。

候选没有显式正弦相位嵌入，也没有修改 Mamba 的 delta/B/C/A；变化仅是周期
分支的扫描拓扑、标量投影和输出 token 数。

## 结构审计

- 相位 3 的输入严格为 `[3,27,51,75]`，最大索引误差为 `0`；
- 编码器调用形状为局部 `[2,3,16,48]`、周期 `[48,3,16,4]`；
- phase-to-batch 映射最大误差为 `0`，不同相位不共享扫描状态；
- 输出为 `[2,24,3]`，CPU 替身与真实 CUDA Mamba 的前向、反向均有限；
- `periodic_aligned` 未获得任何候选专属参数，重复初始化状态完全一致。

原始审计：`logs/graphmamba_cross_period_phase/structural_audit.json`。

## 预注册验证结果

协议：ETTh1/ETTh2，`96 -> 192`，多变量，seed 2021，validation-only；控制组
为此前冻结的 `periodic_aligned` V1 adapter，训练超参数相同。

| 数据集 | 指标 | V1 控制 | 跨周期候选 | 候选变化 |
|---|---:|---:|---:|---:|
| ETTh1 | MSE | 0.988814 | 1.020100 | +3.164% |
| ETTh1 | MAE | 0.652796 | 0.661112 | +1.274% |
| ETTh2 | MSE | 0.272637 | 0.292189 | +7.172% |
| ETTh2 | MAE | 0.351920 | 0.372431 | +5.828% |

宏平均 MSE 恶化 `5.168%`，宏平均 MAE 恶化 `3.551%`。两项任务、两个指标
均同向变差，因此未达到“两数据集 MSE 均提升、宏平均至少 0.5%”的门槛，
按预注册规则不运行第二 seed、不调 period/pooling/patch、不访问测试集。

## 复杂度与解释

- 完整 ETT-192 参数量：V1 `820,144`，候选 `1,015,280`，增加 `195,136`
  （`23.79%`），主要来自 72-token 预测头替代 56-token 预测头；
- ETTh1 训练记录耗时 `136.44s` vs `33.72s`，ETTh2 为 `137.58s` vs
  `31.04s`，约慢 `4.05x/4.43x`；
- 结构正确只说明物理转移定义成立，并不说明 4 个周期足以让 Mamba 学到优于
  长度 24 周期 patch 的表示。当前证据显示，短 cycle 轴与更宽预测头同时带来
  更高成本和更差泛化。

## 决策与恢复点

`periodic_phase` 从 `models/GraphMamba.py`、`run.py` 和通用实验 runner 的活动
接口移除；默认且受支持的主线仍为 `periodic_aligned`。设计文档、结构审计脚本、
JSON、日志和训练 checkpoint 全部保留，可在新的、预先成立的假设出现后恢复，
但不能把本次候选作为已获支持的创新点。

原始记录：

- `logs/graphmamba_cross_period_phase_validation/validation/pxs_h1_p192_s21.json`
- `logs/graphmamba_cross_period_phase_validation/validation/pxs_h2_p192_s21.json`
- `logs/graphmamba_periodic_v1_validation/validation/pv1a_h1_s21.json`
- `logs/graphmamba_periodic_v1_validation/validation/pv1a_h2_s21.json`

## 先验边界

PhaseFormer 已提出 phase-wise prediction/cross-phase routing；TimesNet 已提出周期
二维重排。因此即使本候选精度通过，也只能声称“局部连续 patch 与同相位跨周期
序列共享 Mamba”的具体组合，不能声称相位建模或周期重排本身新颖。

- PhaseFormer: https://openreview.net/forum?id=Lk9SqMQzhX
- TimesNet: https://openreview.net/forum?id=ju_Uqw384Oq
- ms-Mamba: https://arxiv.org/abs/2504.07654
- TimeMachine: https://arxiv.org/abs/2403.09898

