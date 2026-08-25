#!/usr/bin/env python3
"""Build a read-only reuse ledger for TimeRole recent-predictor evidence."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "timerole_recent_simplification" / "audit"
REPORT = ROOT / "experiment_results" / "TimeRole_recent_predictor_existing_evidence_audit.md"
SOURCES = (
    ROOT / "logs" / "graphmamba_backbone_ablation",
    ROOT / "logs" / "cmrhm_backbone_ablation",
    ROOT / "logs" / "cmrhm_no_mamba_cross_domain",
    ROOT / "logs" / "timerole_p0" / "recent_backbone" / "final",
    ROOT / "logs" / "graphmamba_scsd_validation",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_command(command: object) -> dict[str, str]:
    if not isinstance(command, list):
        return {}
    result: dict[str, str] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if isinstance(item, str) and item.startswith("--") and index + 1 < len(command):
            result[item[2:]] = str(command[index + 1])
            index += 2
        else:
            index += 1
    return result


def candidate_json(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    if path.name in {"status.json", "manifest.json", "summary.json"}:
        return False
    return any(part in {"records", "validation", "final"} for part in path.parts)


def audit_record(source: Path, path: Path) -> dict[str, object]:
    reasons: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "source_root": str(source.relative_to(ROOT)),
            "record": str(path.relative_to(ROOT)),
            "reuse_status": "INVALID",
            "reasons": [f"unreadable_json:{type(error).__name__}"],
        }
    if not isinstance(payload, dict):
        reasons.append("payload_not_object")
        payload = {}
    config = payload.get("resolved_config")
    if not isinstance(config, dict):
        config = parse_command(payload.get("command"))

    status = payload.get("status")
    split = payload.get("split")
    test_accessed = payload.get("test_accessed")
    git_sha = payload.get("git_commit_sha")
    source_hashes = payload.get("source_files_sha256")
    dirty = payload.get("git_dirty")
    scan_mode = payload.get("scan_mode") or config.get("dual_scale_scan_mode")
    seq_len = config.get("seq_len")
    recent_len = config.get("timerole_recent_len") or config.get("cmrhm_recent_len")

    if status != "completed":
        reasons.append("status_not_completed")
    if split not in {"val", "validation"}:
        reasons.append("not_validation_split")
    if test_accessed is not False:
        reasons.append("test_access_not_explicitly_false")
    if not git_sha:
        reasons.append("missing_git_sha")
    if not isinstance(source_hashes, dict) or not source_hashes:
        reasons.append("missing_source_hashes")
    if dirty is not False:
        reasons.append("source_not_proven_clean")
    if seq_len != "336":
        reasons.append("seq_len_not_336")
    if recent_len not in {"96", None}:
        reasons.append("recent_len_not_96")
    if scan_mode != "independent_shared":
        reasons.append("scan_mode_not_independent_shared")
    # Old records predate the isolated DataLoader generator and cannot prove
    # equal epoch-wise sample order across architecture variants.
    reasons.append("paired_data_order_not_proven")

    return {
        "source_root": str(source.relative_to(ROOT)),
        "record": str(path.relative_to(ROOT)),
        "reuse_status": "CONTEXT_ONLY" if reasons else "REUSABLE",
        "reasons": reasons,
        "dataset": payload.get("dataset") or config.get("data"),
        "horizon": payload.get("horizon") or payload.get("pred_len") or config.get("pred_len"),
        "seed": payload.get("seed") or config.get("seed"),
        "variant": payload.get("variant"),
        "model": payload.get("model") or config.get("model"),
        "status": status,
        "split": split,
        "test_accessed": test_accessed,
        "git_commit_sha": git_sha,
        "git_dirty": dirty,
        "scan_mode": scan_mode,
        "seq_len": seq_len,
        "recent_len": recent_len,
        "metric_version": payload.get("metric_version"),
    }


def main() -> int:
    rows = []
    source_counts: Counter[str] = Counter()
    for source in SOURCES:
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*.json")):
            if not candidate_json(path):
                continue
            row = audit_record(source, path)
            rows.append(row)
            source_counts[str(source.relative_to(ROOT))] += 1

    reuse_counts = Counter(str(row["reuse_status"]) for row in rows)
    ledger = {
        "created_at": now(),
        "protocol": "TimeRole recent-predictor simplification existing-evidence audit",
        "policy": "default non-reuse; exact protocol and paired data order must be proven",
        "source_counts": dict(sorted(source_counts.items())),
        "reuse_counts": dict(sorted(reuse_counts.items())),
        "records": rows,
    }
    atomic_write(
        OUTPUT / "existing_evidence_ledger.json",
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    lines = [
        "# TimeRole 近期预测器既有证据审计",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run / preflight audit",
        f"- Origin Date: {now()}",
        "- Verification Status: ANALYZED",
        "- Version Label: timerole_existing_evidence_audit_v1",
        "",
        "## 结论",
        "",
        f"共读取 {len(rows)} 条候选 JSON 记录；严格可复用 {reuse_counts.get('REUSABLE', 0)} 条，"
        f"仅作背景证据 {reuse_counts.get('CONTEXT_ONLY', 0)} 条，"
        f"无效 {reuse_counts.get('INVALID', 0)} 条。",
        "",
        "当前记录均不能证明在新协议下具有架构无关的逐 epoch 数据顺序，因此不作为 R0/R1–R5 正式配对基线。新 smoke 与后续矩阵从当前冻结代码重跑。",
        "",
        "## 来源计数",
        "",
        "| 来源 | 记录数 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{source}` | {count} |" for source, count in sorted(source_counts.items()))
    lines.extend([
        "",
        "## 复用判定规则",
        "",
        "- 必须是 completed validation 记录且 `test_accessed=false`；",
        "- 必须包含 clean Git SHA、source hashes 和可重建配置；",
        "- 必须满足 `seq_len=336`、`recent_len=96` 和冻结扫描/图配置；",
        "- 必须证明相同 seed 下各架构逐 epoch 数据顺序一致；",
        "- 任一字段缺失即降级为 `CONTEXT_ONLY`。",
        "",
        "完整逐记录 ledger：`logs/timerole_recent_simplification/audit/existing_evidence_ledger.json`。",
        "",
    ])
    atomic_write(REPORT, "\n".join(lines))
    print(json.dumps({
        "records": len(rows), "source_counts": dict(source_counts),
        "reuse_counts": dict(reuse_counts), "report": str(REPORT),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
