# GraphMamba 双尺度扫描修正与 LagGraph 首轮验证

## 1. 本轮改动

### 双尺度扫描

原始 `GraphMamba.py` 将覆盖同一段历史的长、短 patch 直接拼接，再交给一次 Mamba
扫描。这会把长尺度末 token 与短尺度首 token 当成时间相邻点。修正后使用同一个编码器
分别扫描长、短尺度，仅在编码后拼接，因此参数共享但状态在尺度边界重置。

### LagGraph-v1

LagGraph 对每个目标变量仅保留静态图的 `top_k` 个候选源变量。对归一化 seasonal
历史执行 FFT，以

`X_target(f) * conj(X_source(f))`

的相位和候选非负延迟的相位基进行匹配；在连续频带内分别得到软延迟分布和有向邻接，
再构造 `source[t-lag]` 的因果消息。消息经共享 patch value projection 后，在长、短
尺度 Mamba 扫描前分别通过有界通道门控注入。

门控默认零初始化；LagGraph 参数在全部基线参数之后初始化，lag context 不额外调用
dropout。因此相同 seed 下，开关 LagGraph 的共享参数初值、零门控训练态前向、dropout
随机流及共享梯度均严格一致。

## 2. 结构与数值检查

- Python compile：通过。
- 双尺度调用：长、短 token 分别调用同一个 encoder，形状和反向传播通过。
- 合成延迟：已知 `lag=3` 的正弦源—目标对恢复 expected lag `3.00001`。
- 零初始化：LagGraph on/off 逐元素等价。
- 严格配对：相同 seed 的共享 state tensor、训练态输出和共享梯度逐项一致。
- CUDA：真实双向 Mamba + LagGraph 完整前向/反向通过，输出 `(2, 24, 7)`。
- 额外参数：5,235，占基线 1,891,575 参数的 0.277%。

## 3. Validation-only 配对结果

统一设置：seed 2021、`seq_len=96`、`pred_len=192`、相同优化器与 early stopping；
所有运行均为 `--test_after_train 0`，没有访问 test。

| 数据集 | 修正扫描 MSE | LagGraph MSE | MSE 相对改善 | 修正扫描 MAE | LagGraph MAE | MAE 相对改善 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 1.002421 | 1.003048 | -0.063% | 0.655899 | 0.656151 | -0.038% |
| ETTh2 | 0.274504 | 0.271807 | +0.982% | 0.352677 | 0.356023 | -0.949% |
| 宏平均 | 0.638463 | 0.637428 | +0.162% | 0.504288 | 0.506087 | -0.357% |

训练耗时：ETTh1 基线/LagGraph 为 39.99/46.87 秒；ETTh2 为 30.14/42.05 秒。
这是完整训练运行时间而非隔离算子 benchmark。

训练后门控绝对值均值为 ETTh1 `0.0270`、ETTh2 `0.0361`；频带权重仍接近均匀，
说明当前任务只弱使用该消息。ETTh2 出现接近 1% 的 MSE 信号，但 MAE 明显退化；ETTh1
基本中性。两任务宏平均没有达到预注册的 1% MSE 门槛。

## 4. 混杂运行登记

初始 LagGraph 实现曾在预测头之前初始化额外参数且零门控时仍执行额外 dropout，导致
共享初始化和随机流不严格配对。该 ETTh1 运行 MSE 为 `1.005159`，已判为有混杂、不可
用于机制选择。修复后重新运行得到上表 `1.003048`。

## 5. 当前判定

- 双尺度独立扫描是语义修正，保留在 `GraphMamba.py` 主线。
- LagGraph-v1 实现保留为默认关闭的可控候选；CLI 必须显式设置
  `--use_lag_graph 1`。
- 当前证据为 **ETTh2 有局部 MSE 信号，但跨数据集宏平均不足且 MAE 不稳**，因此不进入
  test，不扩展频带数、延迟范围、温度或门控搜索。
- 下一步若继续，应先诊断 ETTh2 的收益来自哪些边/延迟，并解释 ETTh1 的弱门控；没有
  新机制证据前，不继续堆叠模块。

## 6. 产物

- 模型：`models/GraphMamba.py`
- CLI：`run.py`
- 配对运行器：`scripts/run_graphmamba_innovation.py`
- 记录：`logs/graphmamba_laggraph_validation/validation/*.json`
- 完整日志：`logs/graphmamba_laggraph_validation/validation/*.log`
