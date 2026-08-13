# CMRHM 端到端严格配对验证结果

## Material Passport

- Experiment ID: `graphmamba_cmrhm_validation_v1`
- Type: paired validation-only neural training
- Verification Status: `VERIFIED_SINGLE_SEED_TWO_DATASETS`
- Seed: 2021
- Task: multivariate-to-multivariate forecasting, prediction length 720
- Data access: train for optimization, validation for checkpoint selection, test untouched
- Candidate version: `CMRHM-v1` (pool 16, 15 memory tokens, hidden dim 32)

## 公平协议

CMRHM需要读取336个历史点。为排除训练样本减少和预测时间戳变化的影响，新建
`GraphMambaRecent`严格基线：它与CMRHM都由数据加载器读取336点，但主干只使用
最后96点。两者使用完全相同的训练/验证窗口、seed、batch、学习率、早停策略和
训练上限。

- 每项训练样本：33,505
- 每项验证样本：10,801
- 41个GraphMamba主干张量初始化逐项相同
- 第0步预测最大绝对差：0
- 所有运行均设置`test_after_train=0`

## 端到端结果

| 数据集 | 模型 | Validation MSE | Validation MAE | Best epoch |
|---|---|---:|---:|---:|
| ETTm1-720 | Recent336 | 0.971108 | 0.659672 | 3 |
| ETTm1-720 | CMRHM | **0.940000** | **0.639754** | 3 |
| ETTm2-720 | Recent336 | 0.290188 | 0.369945 | 1 |
| ETTm2-720 | CMRHM | **0.278651** | **0.359677** | 1 |

相对严格基线：

| 数据集 | MSE改善 | MAE改善 |
|---|---:|---:|
| ETTm1-720 | **3.203%** | **3.019%** |
| ETTm2-720 | **3.976%** | **2.775%** |

## 复杂度与机制检查

- 参数量：6,809,318 → 6,835,917，增加26,599（+0.391%）。
- 运行时间：ETTm1 109.73s → 115.04s（约+4.8%）；ETTm2 87.23s → 92.20s（约+5.7%）。
- 最佳checkpoint平均绝对记忆门控：ETTm1 0.203，ETTm2 0.101。
- 记忆编码器和解码器权重范数均非零，模块没有退化为关闭状态。

## 判定

CMRHM在两个预注册长预测任务上均超过1% MSE实质改善门槛，并且MAE同步下降，
因此通过首轮端到端validation gate。它现在可以被称为“验证有效的候选核心创新”，
但尚不能宣称普适有效：当前只有seed 2021、两个数据集、一个预测长度，且尚未
执行保护性任务和正式test。

机器可读结果：`logs/graphmamba_cmrhm_validation/comparison.csv`。
