# GraphMambaCMRHM-v1 严格实验结构审查

日期：2026-08-14

- Full、Concat、NoDiff、GlobalGate 在零门控时均与 `GraphMambaRecent` 逐元素完全相等（最大绝对误差 0）。
- Full、NoDiff、GlobalGate 的 state dict 键、参数值与初始化完全相同，只改变指定计算机制。
- 四个旧历史分支激活后均对旧 240 点扰动产生非零响应，门控与 memory context 梯度非零。
- 预测长度 96 时参数量：Recent 997,382；Full/NoDiff/GlobalGate 1,004,013；Concat 1,007,085。
- 52 个新任务名称唯一，分为 A=32、B=8、C=12；16 个 seed-2021 复用记录完整。
- 所有新任务固定输入 336 且 `test_after_train=0`，实验期间不访问测试集。

因此实现与命令级审查通过，可以进入预注册的长时间验证实验；这不预示结果一定通过。
