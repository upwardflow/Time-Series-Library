#!/usr/bin/env python3
"""Aggregate the frozen TimeRole-v1 validation audit and apply its fixed gates."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "logs" / "timerole_strict_evidence" / "validation"
OLD = ROOT / "logs" / "timerole_all_horizons" / "validation"
OUT = NEW.parent
REPORT = ROOT / "experiment_results" / "TimeRole_strict_evidence_result.md"
DATASETS = ("ETTm1", "ETTm2")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
PARAMS = {
    96: (997382, 3232262, 1004013),
    192: (1891526, 6361286, 1901229),
    336: (3232742, 11054822, 3247053),
    720: (6809318, 23570918, 6835917),
}


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return row if row.get("status") == "completed" and "best_mse" in row else None


def old(dataset: str, horizon: int, variant: str) -> dict | None:
    return load(OLD / f"{dataset.lower()}_{horizon}_{variant}_s2021.json")


def new(group: str, dataset: str, horizon: int, variant: str, seed: int) -> dict | None:
    return load(NEW / f"{group}_{dataset.lower()}_p{horizon}_{variant}_s{seed}.json")


def improve(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def avg(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def group_a() -> tuple[list[dict], list[dict]]:
    pairs, summaries = [], []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            task = []
            for seed in SEEDS:
                recent = old(dataset, horizon, "recent336") if seed == 2021 else new("a", dataset, horizon, "r", seed)
                full = old(dataset, horizon, "timerole") if seed == 2021 else new("a", dataset, horizon, "c", seed)
                if recent and full:
                    row = {
                        "dataset": dataset, "horizon": horizon, "seed": seed,
                        "recent_mse": recent["best_mse"], "timerole_mse": full["best_mse"],
                        "mse_improvement_pct": improve(recent["best_mse"], full["best_mse"]),
                        "recent_mae": recent["best_mae"], "timerole_mae": full["best_mae"],
                        "mae_improvement_pct": improve(recent["best_mae"], full["best_mae"]),
                    }
                    pairs.append(row); task.append(row)
            if len(task) == 3:
                summaries.append({
                    "dataset": dataset, "horizon": horizon,
                    "recent_mse_mean": avg(x["recent_mse"] for x in task),
                    "recent_mse_sd": statistics.stdev(x["recent_mse"] for x in task),
                    "timerole_mse_mean": avg(x["timerole_mse"] for x in task),
                    "timerole_mse_sd": statistics.stdev(x["timerole_mse"] for x in task),
                    "timerole_mean_mse_improvement_pct": improve(
                        avg(x["recent_mse"] for x in task), avg(x["timerole_mse"] for x in task)),
                    "recent_mae_mean": avg(x["recent_mae"] for x in task),
                    "recent_mae_sd": statistics.stdev(x["recent_mae"] for x in task),
                    "timerole_mae_mean": avg(x["timerole_mae"] for x in task),
                    "timerole_mae_sd": statistics.stdev(x["timerole_mae"] for x in task),
                })
    return pairs, summaries


def group_b() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            recent, raw, full = old(dataset, horizon, "recent336"), new("b", dataset, horizon, "raw", 2021), old(dataset, horizon, "timerole")
            if recent and raw and full:
                rows.append({
                    "dataset": dataset, "horizon": horizon,
                    "recent_mse": recent["best_mse"], "raw_mse": raw["best_mse"], "timerole_mse": full["best_mse"],
                    "timerole_vs_raw_mse_pct": improve(raw["best_mse"], full["best_mse"]),
                    "recent_mae": recent["best_mae"], "raw_mae": raw["best_mae"], "timerole_mae": full["best_mae"],
                    "timerole_vs_raw_mae_pct": improve(raw["best_mae"], full["best_mae"]),
                    "recent_params": PARAMS[horizon][0], "raw_params": PARAMS[horizon][1], "timerole_params": PARAMS[horizon][2],
                    "raw_runtime_s": raw["duration_seconds"], "timerole_runtime_s": full["duration_seconds"],
                })
    return rows


def group_c() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for horizon in (96, 720):
            full = old(dataset, horizon, "timerole")
            for code, label in (("cat", "Concat"), ("nd", "NoDiff"), ("gg", "GlobalGate")):
                ablation = new("c", dataset, horizon, code, 2021)
                if full and ablation:
                    rows.append({
                        "dataset": dataset, "horizon": horizon, "ablation": label,
                        "ablation_mse": ablation["best_mse"], "full_mse": full["best_mse"],
                        "full_mse_improvement_pct": improve(ablation["best_mse"], full["best_mse"]),
                        "ablation_mae": ablation["best_mae"], "full_mae": full["best_mae"],
                        "full_mae_improvement_pct": improve(ablation["best_mae"], full["best_mae"]),
                    })
    return rows


def state(ok: bool, complete: bool) -> str:
    return ("PASS" if ok else "FAIL") if complete else "PENDING"


def main() -> int:
    a, a_summary = group_a(); b, c = group_b(), group_c()
    write_csv("group_a_pairs.csv", a); write_csv("group_a_task_mean_sd.csv", a_summary)
    write_csv("group_b_capacity.csv", b); write_csv("group_c_ablation.csv", c)

    a_mse_wins = sum(x["timerole_mse"] < x["recent_mse"] for x in a)
    a_mae_wins = sum(x["timerole_mae"] < x["recent_mae"] for x in a)
    a_mse = avg(x["mse_improvement_pct"] for x in a); a_mae = avg(x["mae_improvement_pct"] for x in a)
    ds_seed = []
    for dataset in DATASETS:
        for seed in SEEDS:
            values = [x["mse_improvement_pct"] for x in a if x["dataset"] == dataset and x["seed"] == seed]
            if len(values) == 4: ds_seed.append(avg(values))
    a_ok = a_mse_wins >= 20 and a_mse >= 1 and len(ds_seed) == 6 and min(ds_seed) >= 0 and a_mae_wins >= 18 and a_mae >= 0

    b_wins = sum(x["timerole_mse"] < x["raw_mse"] for x in b)
    b_mse = avg(x["timerole_vs_raw_mse_pct"] for x in b); b_mae = avg(x["timerole_vs_raw_mae_pct"] for x in b)
    b_ok = b_wins >= 6 and b_mse >= 1 and b_mae >= 0

    parts = []
    for label in ("Concat", "NoDiff", "GlobalGate"):
        rows = [x for x in c if x["ablation"] == label]
        wins = sum(x["full_mse"] < x["ablation_mse"] for x in rows)
        mse = avg(x["full_mse_improvement_pct"] for x in rows); mae = avg(x["full_mae_improvement_pct"] for x in rows)
        parts.append((label, len(rows), wins, mse, mae, len(rows) == 4 and wins >= 3 and mse >= .5 and mae >= 0))
    c_ok = all(x[-1] for x in parts)
    status = json.loads((OUT / "status.json").read_text()) if (OUT / "status.json").is_file() else {}

    lines = ["# TimeRole-v1 严格有效性审计", "", f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
             "实验只能支持或反驳主张，不能预设“证明有效”。本审计全部为验证集，不再次访问测试集。", "", "## 进度", "",
             f"- 新任务：{status.get('completed_new_jobs', 0)}/52；失败：{len(status.get('failed_jobs', []))}。",
             f"- 完整配对：A {len(a)}/24，B {len(b)}/8，C {len(c)}/12。", "", "## A：三随机种子配对", "",
             f"- {state(a_ok, len(a)==24)}；MSE 胜场 {a_mse_wins}/{len(a)}，配对宏平均改善 {a_mse:.3f}%。",
             f"- MAE 胜场 {a_mae_wins}/{len(a)}，配对宏平均改善 {a_mae:.3f}%。",
             f"- 数据集×种子四跨度宏平均最小值：{min(ds_seed) if ds_seed else float('nan'):.3f}%。", "", "## B：Raw336 容量控制", "",
             f"- {state(b_ok, len(b)==8)}；MSE 胜场 {b_wins}/{len(b)}，MSE/MAE 宏平均改善 {b_mse:.3f}%/{b_mae:.3f}%。",
             f"- 平均训练时间：Raw336 {avg(x['raw_runtime_s'] for x in b):.1f}s，TimeRole {avg(x['timerole_runtime_s'] for x in b):.1f}s。", "",
             "| horizon | Recent 参数 | Raw336 参数 | TimeRole 参数 |", "|---:|---:|---:|---:|"]
    lines += [f"| {h} | {p[0]:,} | {p[1]:,} | {p[2]:,} |" for h, p in PARAMS.items()]
    lines += ["", "## C：机制消融", ""]
    for label, count, wins, mse, mae, ok in parts:
        lines.append(f"- {label}: {state(ok, count==4)}；MSE 胜场 {wins}/{count}，MSE/MAE 宏平均改善 {mse:.3f}%/{mae:.3f}%。")
    lines += ["", f"- 三项机制联合门槛：{state(c_ok, len(c)==12)}。", "", "## 结论边界", "",
              "- 只有完整组达到预注册门槛才支持对应主张；失败项必须收缩或删除，不能结果后改门槛。",
              "- A 报告样本标准差；B 显式报告容量与运行时间；Concat 是高参数控制，另外两项为等参数控制。",
              "- 明细 CSV、原始 JSON 和逐任务日志均位于 `logs/timerole_strict_evidence/`。", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8"); temporary.replace(REPORT)
    print(REPORT); print(f"A={state(a_ok,len(a)==24)} B={state(b_ok,len(b)==8)} C={state(c_ok,len(c)==12)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
