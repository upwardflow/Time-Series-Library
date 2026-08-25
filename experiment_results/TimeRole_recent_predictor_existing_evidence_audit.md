# TimeRole 近期预测器既有证据审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / preflight audit
- Origin Date: 2026-08-25T16:05:18.029700+08:00
- Verification Status: ANALYZED
- Version Label: timerole_existing_evidence_audit_v1

## 结论

共读取 141 条候选 JSON 记录；严格可复用 0 条，仅作背景证据 141 条，无效 0 条。

当前记录均不能证明在新协议下具有架构无关的逐 epoch 数据顺序，因此不作为 R0/R1–R5 正式配对基线。新 smoke 与后续矩阵从当前冻结代码重跑。

## 来源计数

| 来源 | 记录数 |
|---|---:|
| `logs/cmrhm_backbone_ablation` | 20 |
| `logs/cmrhm_no_mamba_cross_domain` | 6 |
| `logs/graphmamba_backbone_ablation` | 20 |
| `logs/graphmamba_scsd_validation` | 94 |
| `logs/timerole_p0/recent_backbone/final` | 1 |

## 复用判定规则

- 必须是 completed validation 记录且 `test_accessed=false`；
- 必须包含 clean Git SHA、source hashes 和可重建配置；
- 必须满足 `seq_len=336`、`recent_len=96` 和冻结扫描/图配置；
- 必须证明相同 seed 下各架构逐 epoch 数据顺序一致；
- 任一字段缺失即降级为 `CONTEXT_ONLY`。

完整逐记录 ledger：`logs/timerole_recent_simplification/audit/existing_evidence_ledger.json`。
