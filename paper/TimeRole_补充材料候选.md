# TimeRole 补充材料候选内容

## A. 与 SimpleTM 公开结果的非同协议定位

为初步判断模型的跨数据表现，将 TimeRole 的单次测试结果与 SimpleTM 的公开报告值进行了对照。在六个数据集、四个预测长度共24个任务上，TimeRole 获得15/24个 MSE 胜场和11/24个 MAE 胜场：其在 ETTm1 和 Weather 上表现稳定，在 ETTm2 上多数 MSE 较优，但在 Solar-Energy 上的 MSE 高出19.43%–25.08%，ETTh1/ETTh2 亦呈现混合结果。全部任务的宏平均 MSE 和 MAE 分别高1.056%和1.003%。

需要强调的是，本方法读取336点，而所引用的 SimpleTM 结果使用96点输入，二者训练实现、输入预算与结果来源并不完全一致。因此，该比较只用于补充呈现外部定位和失败边界，不作为公平的主要基线比较，也不支持“优于现有先进方法”的结论。旧实验原始数据仍保存在 `logs/graphmamba_cmrhm_six_dataset_final/comparison_simpletm.csv`；该路径属于重命名前的历史溯源标识。
