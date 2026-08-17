# TimeXer-CMRHM一次性正式测试

## Material Passport

- Experiment type: one-shot formal test
- Models: TimeXerRecent and TimeXerCMRHM
- Checkpoint selection: validation best MSE only
- Datasets: ETTm1 and ETTm2
- Prediction lengths: 96 and 720
- Seed: 2021
- Training during this phase: none (`is_training=0`)
- Test status: consumed; results must not guide further tuning

## Test结果

| Dataset | Horizon | TimeXer MSE | +CMRHM MSE | MSE改善 | TimeXer MAE | +CMRHM MAE | MAE改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTm1 | 96 | 0.321663 | 0.297778 | 7.426% | 0.358360 | 0.350331 | 2.240% |
| ETTm1 | 720 | 0.452553 | 0.432111 | 4.517% | 0.441367 | 0.429968 | 2.583% |
| ETTm2 | 96 | 0.171434 | 0.164905 | 3.809% | 0.256019 | 0.253215 | 1.096% |
| ETTm2 | 720 | 0.396538 | 0.369014 | 6.941% | 0.396934 | 0.386716 | 2.574% |

- MSE胜率：4/4。
- MAE胜率：4/4。
- 任务宏平均改善：MSE 5.673%，MAE 2.123%。

## 完整性核验

- 8/8推理记录状态为completed。
- 8/8命令均使用`is_training=0`，没有重新训练。
- 8/8 checkpoint均由对应validation最佳MSE选择。
- 测试脚本在开始前验证完整checkpoint矩阵，避免缺失配对造成选择性报告。

## 结论边界

冻结CMRHM-v1在TimeXer上的validation收益成功迁移到正式test，且四项MSE和MAE均改善，
进一步支持其不是GraphMamba特有的局部补丁。由于当前仍为单seed、两个同族分钟级数据集、
一个非图骨干，结论仍限定为“具有明确的初步跨骨干泛化证据”。

ETTm1/ETTm2的96与720测试矩阵已经消费。后续不得利用这些test结果选择池化率、隐藏维度、
门控形式、训练超参数或其他CMRHM变体；新的结构决策只能使用未消费的validation任务。
