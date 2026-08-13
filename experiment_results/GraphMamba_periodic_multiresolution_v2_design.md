# GraphMamba 周期多分辨率 V2：置信度调制适配器

V1 的输出路由几乎停留在等权，而尺度适配器贡献了主要增益。V2 不再在编码后整体缩放分支，而在完整周期分支进入共享 Mamba 前调制适配残差：

`period_tokens = tokens + confidence_24 * adapter(tokens, scale_descriptor)`

其中 `confidence_24` 联合输入窗口的 24 步附近频谱能量和 lag-24 自相关，形状为样本×变量；它只读取输入，不读取预测标签。局部尺度适配保持 V1 形式。

严格对照为 V1 adapter-only 与 V2 confidence-adapter；alignment/router 均关闭。通过条件仍为两个任务 MSE 同方向且宏平均至少改善 0.5%。
