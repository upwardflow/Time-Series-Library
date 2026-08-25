# TimeRole 近期预测器简化：Phase A Smoke 审核

## 结论

Phase A 技术门禁通过。R0–R5 六个候选均在 ETTm1、预测长度 96、seed 2021 上完成验证集评估；未访问测试集，无失败、重试、超时或训练报错。该阶段只验证实现、配对公平性和初步趋势，不能替代多数据集、多随机种子的正式结论。

## 冻结执行信息

- 分支：`exp/timerole-recent-simplification`
- 训练源码提交：`93866ef`
- 会话：tmux `timerole_recent_simplification:smoke`
- 命令：`.venv/bin/python -u scripts/run_timerole_recent_simplification.py --stage smoke --role timerole --gpu 0 --timeout-seconds 3600`
- 数据：ETTm1，输入长度 336，预测长度 96，seed 2021
- 选择规则：验证集 MSE；训练后不运行测试集

## 结果

| 变体 | Val MSE | Val MAE | 最佳 epoch | 参数量 | ms/batch | 训练峰值显存 (MiB) | 校正 RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.364151 | 0.398597 | 6 | 1,004,013 | 4.944 | 520.9 | 0.334687 |
| R1 | 0.364156 | 0.400884 | 6 | 699,533 | 3.202 | 340.0 | 0.302320 |
| R2 | 0.369595 | 0.403179 | 6 | 404,749 | 3.072 | 186.8 | 0.309030 |
| R3 | 0.366523 | 0.403014 | 9 | 661,005 | 2.726 | 231.3 | 0.316723 |
| R4 | **0.362137** | 0.399346 | 9 | 691,143 | 3.124 | 315.6 | 0.312189 |
| R5 | 0.373479 | 0.402794 | 9 | 605,517 | **2.177** | **78.3** | 0.339154 |

相对 R0，R4 的参数量减少约 31.2%，单批延迟减少约 36.8%，同时本次验证 MSE 改善约 0.55%。R1 的 MSE 与 R0 几乎相同（相对差约 0.002%），参数量减少约 30.3%，单批延迟减少约 35.2%。R5 的速度和显存优势最大，但 MSE 最差，提示完全移除 Mamba 存在精度风险。

## 门禁检查

- 6/6 运行状态为 `completed`，指标均为有限值。
- R0–R5 使用相同 seed 的独立 DataLoader generator，前两个 epoch 的训练顺序哈希一致。
- 验证集使用顺序采样；全部记录 `test_accessed=false`。
- 简化变体参数量均低于 R0，TimeRole 校正量均非零。
- 训练记录的源码状态干净，六个配置均对应提交 `93866ef`。

## 已知异常

六份日志各包含一次 `RESOURCE_ALERT`。原因是监控器在子进程刚创建时读取到约 8 KiB RSS，并误将其作为稳定基线；没有 OOM、Traceback、超时或停滞。该问题不影响训练结果。smoke 结束后已为后续阶段加入至少 30 秒的 RSS 基线预热，原始 smoke 日志和记录保持不变。

## 下一门禁

Phase B 尚未启动。根据冻结任务书，应在确认本报告后运行 36 个 TimeRole 筛选任务（3 个数据集 × 2 个预测长度 × 6 个变体，seed 2021），并继续保持验证集专用、配对数据顺序和失败不自动重试策略。

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Mode：run
- Verification Status：`UNVERIFIED`（尚未进行独立验证审计）
- Evidence：原始日志、原子 JSON 记录、阶段 manifest、CSV/JSON 汇总及 smoke gate JSON
