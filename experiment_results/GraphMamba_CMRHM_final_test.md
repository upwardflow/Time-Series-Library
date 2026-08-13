# CMRHM 一次性正式测试结果

## Material Passport

- Experiment ID: `graphmamba_cmrhm_final_test_v1`
- Type: one-shot frozen-checkpoint test evaluation
- Verification Status: `VERIFIED_SINGLE_SEED_TWO_DATASETS_FOUR_HORIZONS`
- Seed: 2021
- Tasks: ETTm1 / ETTm2 × {96, 192, 336, 720}
- Selection rule: checkpoint selected only by validation MSE
- Training during test: none (`is_training=0` in all 16 commands)
- Post-test tuning: prohibited

## Test结果

| 数据集 | 长度 | Recent336 MSE | CMRHM MSE | MSE改善 | Recent336 MAE | CMRHM MAE | MAE改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTm1 | 96 | 0.331550 | **0.295054** | **11.008%** | 0.370608 | **0.348728** | **5.904%** |
| ETTm1 | 192 | 0.368139 | **0.331811** | **9.868%** | 0.391468 | **0.374017** | **4.458%** |
| ETTm1 | 336 | 0.399569 | **0.370542** | **7.265%** | 0.415179 | **0.395003** | **4.860%** |
| ETTm1 | 720 | 0.459311 | **0.423542** | **7.788%** | 0.448317 | **0.426063** | **4.964%** |
| ETTm2 | 96 | 0.182228 | **0.171857** | **5.691%** | 0.264699 | **0.261041** | **1.382%** |
| ETTm2 | 192 | 0.242543 | **0.225106** | **7.189%** | 0.301831 | **0.295422** | **2.123%** |
| ETTm2 | 336 | 0.309571 | **0.291423** | **5.862%** | 0.347817 | **0.340736** | **2.036%** |
| ETTm2 | 720 | 0.404274 | **0.375222** | **7.186%** | 0.402370 | **0.389848** | **3.112%** |

汇总：

- MSE胜率：8/8；8/8均超过5%相对改善。
- MAE胜率：8/8。
- 任务级相对改善宏平均：MSE **7.732%**，MAE **3.605%**。
- 16次test推理总耗时约155.74秒。

## 协议审计

- 16/16记录状态为completed。
- 16/16命令均为`is_training=0`，没有重新训练。
- 16/16 checkpoint均由此前validation最佳MSE选出。
- Recent336与CMRHM对每个任务使用相同seq_len、样本范围、seed和训练协议。
- 正式test已经被读取，因此后续不得根据这些结果改变CMRHM-v1结构或超参数。

## 判定与边界

CMRHM-v1在当前一次性正式test中通过两数据集、四预测长度的严格配对检验，且MSE与
MAE方向完全一致。结果支持“压缩旧历史记忆在分钟级ETT长短预测中均提供稳定增益”。
不过它仍是单随机种子证据；论文最终报告前应补充多seed均值±标准差，并验证ETTh及
其他非ETT数据集。后续实验可以检验外推稳定性，但不能再使用本test矩阵调参。

机器可读结果：`logs/graphmamba_cmrhm_final_test/comparison.csv`。
