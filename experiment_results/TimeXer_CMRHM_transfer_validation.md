# TimeXer-CMRHM非图主干迁移验证

## Material Passport

- Experiment type: validation-only paired transfer gate
- Backbone: TimeXer
- Candidate: frozen CMRHM-v1
- Datasets: ETTm1 and ETTm2
- Prediction lengths: 96 and 720
- Seed: 2021
- Metric implementation: `element_weighted_v1`
- Test status: not accessed

## 公平协议

- `TimeXerRecent`与`TimeXerCMRHM`都以`seq_len=336`加载完全相同的样本。
- 两个TimeXer主干都只读取最后96点；候选唯一新增输入是前240点历史。
- CMRHM保持GraphMamba实验中的冻结机制：16倍平均池化得到15个旧历史token、
  32维配对差分解码、7个变量级`tanh`有界门控。
- 35个主干张量在初始化时逐项相同；门控零初始化使两者初始输出最大绝对差为0。
- 使用仓库原始TimeXer ETTm脚本的有效超参数，训练至10 epoch并按validation
  early stopping选checkpoint；`test_after_train=0`。

## Validation结果

| Dataset | Horizon | TimeXer MSE | +CMRHM MSE | MSE改善 | TimeXer MAE | +CMRHM MAE | MAE改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTm1 | 96 | 0.386517 | 0.373276 | 3.426% | 0.408643 | 0.402811 | 1.427% |
| ETTm1 | 720 | 0.946509 | 0.929434 | 1.804% | 0.643710 | 0.633420 | 1.599% |
| ETTm2 | 96 | 0.124883 | 0.123594 | 1.032% | 0.241633 | 0.240784 | 0.351% |
| ETTm2 | 720 | 0.291747 | 0.280561 | 3.834% | 0.370295 | 0.361463 | 2.385% |

- MSE胜率：4/4；MAE胜率：4/4。
- 任务宏平均改善：MSE 2.524%，MAE 1.441%。
- 预注册门槛（至少3/4 MSE胜、宏平均MSE至少1%、宏平均MAE不退化）：通过。

## 模块是否实际启用

最佳checkpoint中变量门控的平均绝对`tanh`值为：

- ETTm1-96: 0.206
- ETTm1-720: 0.228
- ETTm2-96: 0.068
- ETTm2-720: 0.094

门控均明显非零，收益不是因为候选退化成原始TimeXer。

## 判定边界

本实验支持CMRHM从GraphMamba迁移到一个非图Transformer骨干，说明条件化旧历史增量
并非只依赖GraphMamba的图分支。但是当前仍只有单seed、两个同族分钟级数据集和一个
非图骨干，因此结论是“已获得初步跨骨干普适性证据”，而不是“已证明通用插件”。

按照用户确定的研究重心，到此停止扩展CMRHM迁移矩阵，不读取TimeXer test，也不利用
本结果继续修改CMRHM。下一研究阶段回到GraphMamba本体，优先寻找区别于现有G-Mamba
拓扑条件SSM的核心机制。
