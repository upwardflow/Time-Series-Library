# CMRHM + w/o Mamba 跨数据域验证记录

## 结论摘要

六组新增跨域消融全部成功完成。新增任务中，`w/o Mamba` 仅在 ETTh2-720 上降低 MSE（1/6），六任务宏平均 MSE/MAE 分别退化 1.625%/1.065%。合并此前 ETTm1/ETTm2 的四组任务后，`w/o Mamba` 在十个任务中获得 4 个 MSE 胜格，十任务宏平均 MSE/MAE 分别退化 1.006%/0.520%。因此未达到预先冻结的“至少 7/10 个 MSE 胜格且宏平均 MSE 更低”条件，不能从论文主模型中直接删除 Mamba。

## Material Passport

| 项目 | 内容 |
|---|---|
| Origin Skill | academic-research-suite / experiment-agent |
| Run | CMRHM + w/o Mamba cross-domain validation |
| Date | 2026-08-16 |
| Verification Status | ANALYZED |
| 新增数据 | `dataset/ETT-small/ETTh1.csv`、`dataset/ETT-small/ETTh2.csv`、`dataset/weather/weather.csv` |
| 新增任务 | ETTh1、ETTh2、Weather × 预测长度 96/720，共6组 |
| 候选模型 | `GraphMambaCMRHM`，`use_time_mamba=0`、`use_graph=1`、`cmrhm_old_intervention=intact` |
| 历史划分 | 总输入336点；CMRHM使用远期240点，近期图主干使用最后96点 |
| Patch/扫描 | patch length 4、stride 2、`independent_shared` |
| 随机种子 | 2021 |
| 选点与评价 | 验证集早停；最佳检查点在完整验证集上按元素数计算 MSE/MAE |
| 新实验测试集 | 未访问，六条最终记录均为 `split=val`、`test_accessed=false` |
| 停止规则 | 单任务硬上限1800秒；首个失败停止；不自动重试 |
| 总控脚本 | `scripts/run_cmrhm_no_mamba_cross_domain.py` |
| 启动命令 | `.venv/bin/python -u scripts/run_cmrhm_no_mamba_cross_domain.py --gpu 0 --seed 2021 --datasets ETTh1 ETTh2 Weather --horizons 96 720 --epochs 100 --patience 6 --timeout-seconds 1800` |
| 输出目录 | `logs/cmrhm_no_mamba_cross_domain/` |

## 新增六任务结果

| 数据集 | 预测长度 | 完整 CMRHM MSE/MAE | CMRHM + w/o Mamba MSE/MAE | MSE变化 | MSE胜负 |
|---|---:|---:|---:|---:|:---:|
| ETTh1 | 96 | **0.6797**/0.5511 | 0.6810/**0.5510** | +0.192% | 负 |
| ETTh1 | 720 | **1.4783**/**0.8320** | 1.5314/0.8501 | +3.589% | 负 |
| ETTh2 | 96 | **0.2235**/**0.3242** | 0.2241/0.3287 | +0.233% | 负 |
| ETTh2 | 720 | 0.6388/0.5625 | **0.6350**/**0.5621** | -0.589% | 胜 |
| Weather | 96 | **0.3837**/**0.2692** | 0.3911/0.2754 | +1.937% | 负 |
| Weather | 720 | **0.6445**/**0.4407** | 0.6517/0.4442 | +1.119% | 负 |
| 宏平均 | — | **0.6747**/**0.4966** | 0.6857/0.5019 | +1.625% | 1/6 |

正变化表示去除 Mamba 后误差上升。完整 CMRHM 数值取自既有 seed-2021 验证过程；Weather 的旧 `auto` 参数在 `run.py` 中实际解析为 `independent_shared`。旧完整模型运行在验证选点后曾继续执行其原定测试流程，但本表仅使用训练期间的完整验证集指标；本轮六个删减模型没有访问测试集。

## 十任务联合判断

将新增六任务与既有 ETTm1/ETTm2 × 96/720 四任务合并：

| 指标 | 完整 CMRHM | CMRHM + w/o Mamba | 相对变化 | 去除 Mamba 胜格 |
|---|---:|---:|---:|---:|
| 宏平均 MSE | **0.5758** | 0.5816 | +1.006% | 4/10 |
| 宏平均 MAE | **0.4623** | 0.4647 | +0.520% | 5/10 |

预先冻结的删除条件为：`w/o Mamba` 至少取得 7/10 个 MSE 胜格，并且十任务宏平均 MSE 低于完整模型。实际为 4/10 个胜格且宏平均 MSE 上升 1.006%，故判定为 **FAIL：保留 Mamba**。

## 论文解释边界

- ETTm 上去除 Mamba 的局部收益不能推广到 ETTh 和 Weather；此前“可以删除 Mamba”的判断缺少跨域证据，本实验已将其否定。
- Mamba 的边际作用具有数据域和预测跨度依赖性：它在 ETTm2 和 ETTh2 的长跨度任务上可能冗余，但在 ETTh1 与 Weather 上总体有正向作用。
- 结果支持保留“图结构 + Mamba + CMRHM”的完整模型作为论文主方法，但不支持宣称固定图—Mamba相加在每个任务上都最优。
- 自适应门控候选此前未超过仅图分支，因此不应为追求统一结论继续追加复杂融合。论文中应把图—Mamba主干表述为互补的实验宿主，把 CMRHM 作为核心创新，并用本跨域消融说明两类近期依赖编码器具有任务相关互补性。

## 完整性核对

- 6个训练 JSON、6个最终验证 JSON、6个最佳检查点均存在。
- 六条最终记录的 `(dataset, horizon, seed)` 唯一，状态均为 `completed`。
- 六条命令均包含 `use_time_mamba=0`、`use_graph=1` 和 `cmrhm_old_intervention=intact`。
- 六条最终记录均为 `split=val`、`test_accessed=false`。
- 本轮无失败、超时或重试。

