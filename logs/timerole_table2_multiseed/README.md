# TimeRole 表2三随机种子实验

- tmux 会话：`timerole_table2_3seed`
- 随机种子：2021、2022、2023
- 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather
- 预测长度：96、192、336、720
- 扫描协议：`independent_shared`（严格匹配表2的2021冻结结果）

## 查看运行状态

```bash
tmux attach -t timerole_table2_3seed
```

从 tmux 脱离但不中止实验：按 `Ctrl-b`，再按 `d`。

无需进入 tmux 也可查看：

```bash
sed -n '1,160p' logs/timerole_table2_multiseed/status.json
tail -f logs/timerole_table2_multiseed/logs/*.log
```

## 输出

- 单任务记录：`records/*.json`
- 单任务日志：`logs/*.log`
- 全部种子长表：`timerole_results_long.csv`
- 各任务均值与样本标准差：`timerole_mean_std.csv`
- 检查点：`checkpoints/`

运行器具有断点续跑能力；如会话意外停止，可重新执行：

```bash
.venv/bin/python -u scripts/run_timerole_table2_multiseed.py --seeds 2021 2022 2023 --gpu 0
```

已有冻结测试结果只做来源核验和复制，不会再次读取测试集；仅缺失组合会重新训练并在验证选点后执行一次测试。
