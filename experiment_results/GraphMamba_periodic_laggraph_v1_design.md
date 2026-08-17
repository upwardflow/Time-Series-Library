# GraphMamba 周期主干 × LagGraph：集成设计

已通过门控的周期主干具有局部 `(4,2)` 与完整周期 `(24,12)` 两个独立共享-Mamba 扫描。LagGraph 的因果时滞消息不再使用旧 `4/2` embedding，而分别投影到这两个真实尺度：

`seasonal → cross-spectral causal lag message → local/period patch projection → zero-gated residual → scale adapter → shared independent Mamba`

两个尺度共享 lag-context projection，但使用独立通道门；门从零初始化，确保启用 LagGraph 时初始预测严格等价于周期 adapter-only 主干。旧 joint/independent LagGraph 路径继续保留。

验证隔离：只比较相同 periodic V1 主干上的 LagGraph off/on，ETTh1-192 与 ETTh2-192，seed 2021；不访问测试集。
