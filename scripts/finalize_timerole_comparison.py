#!/usr/bin/env python3
"""Finalize the six-dataset TimeRole report with literature comparisons."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "timerole_six_dataset_final"
RESULTS = OUTPUT / "results.csv"
COMPARISON = OUTPUT / "comparison_simpletm.csv"
REPORT = ROOT / "experiment_results" / "TimeRole_six_dataset_final_test.md"

# ICLR 2025 SimpleTM, Table 6. All entries use lookback 96 and the same four
# standard horizons. Values are copied from the primary paper, not a leaderboard.
SIMPLETM = {
    "ETTh1": {96: (.366, .392), 192: (.422, .421), 336: (.440, .438), 720: (.463, .462)},
    "ETTh2": {96: (.281, .338), 192: (.355, .387), 336: (.365, .401), 720: (.413, .436)},
    "ETTm1": {96: (.321, .361), 192: (.360, .380), 336: (.390, .404), 720: (.454, .438)},
    "ETTm2": {96: (.173, .257), 192: (.238, .299), 336: (.296, .338), 720: (.393, .395)},
    "weather": {96: (.162, .207), 192: (.208, .248), 336: (.263, .290), 720: (.340, .341)},
    "solar": {96: (.163, .232), 192: (.182, .247), 336: (.193, .257), 720: (.199, .252)},
}

# TimeCDS forecasting averages from its ICLR 2026 withdrawn submission.
# These are context only: input length 672 and recursive 96-step generation.
TIMECDS_AVG = {
    "ETTh1": (.355, .424), "ETTh2": (.269, .301),
    "ETTm1": (.381, .401), "ETTm2": (.357, .395),
    "weather": (.356, .420), "solar": (.207, .233),
}


def load_rows() -> list[dict[str, object]]:
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 completed rows, found {len(rows)}")
    normalized = []
    for row in rows:
        normalized.append({
            **row,
            "pred_len": int(row["pred_len"]),
            "test_mse": float(row["test_mse"]),
            "test_mae": float(row["test_mae"]),
            "periodic_backbone_active": row["periodic_backbone_active"].lower() == "true",
        })
    return normalized


def main() -> int:
    rows = load_rows()
    compared = []
    for row in rows:
        dataset, horizon = row["dataset"], row["pred_len"]
        baseline_mse, baseline_mae = SIMPLETM[dataset][horizon]
        mse_delta_pct = 100 * (row["test_mse"] - baseline_mse) / baseline_mse
        mae_delta_pct = 100 * (row["test_mae"] - baseline_mae) / baseline_mae
        compared.append({
            **row,
            "simpletm_mse": baseline_mse,
            "simpletm_mae": baseline_mae,
            "mse_delta_vs_simpletm_pct": mse_delta_pct,
            "mae_delta_vs_simpletm_pct": mae_delta_pct,
            "mse_win": row["test_mse"] < baseline_mse,
            "mae_win": row["test_mae"] < baseline_mae,
        })

    with COMPARISON.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compared[0]))
        writer.writeheader()
        writer.writerows(compared)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in compared:
        grouped[row["dataset"]].append(row)
    timecds_rows = []
    for dataset in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "solar"):
        ours_mse = sum(row["test_mse"] for row in grouped[dataset]) / 4
        ours_mae = sum(row["test_mae"] for row in grouped[dataset]) / 4
        other_mse, other_mae = TIMECDS_AVG[dataset]
        timecds_rows.append((dataset, ours_mse, ours_mae, other_mse, other_mae))

    mse_wins = sum(row["mse_win"] for row in compared)
    mae_wins = sum(row["mae_win"] for row in compared)
    macro_mse_delta = sum(row["mse_delta_vs_simpletm_pct"] for row in compared) / 24
    macro_mae_delta = sum(row["mae_delta_vs_simpletm_pct"] for row in compared) / 24

    lines = [
        "# TimeRole 六数据集正式测试与公开模型对比",
        "",
        "## 最终测试结果（MSE、MAE 前置）",
        "",
        "| Dataset | Horizon | **TimeRole MSE** | **TimeRole MAE** | SimpleTM MSE | SimpleTM MAE | ΔMSE | ΔMAE | 主干 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in compared:
        backbone = (
            "TimeRole（周期近期主干）"
            if row["periodic_backbone_active"]
            else "TimeRole（双 Patch 近期主干）"
        )
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | **{row['test_mse']:.6f}** | "
            f"**{row['test_mae']:.6f}** | {row['simpletm_mse']:.3f} | "
            f"{row['simpletm_mae']:.3f} | {row['mse_delta_vs_simpletm_pct']:+.2f}% | "
            f"{row['mae_delta_vs_simpletm_pct']:+.2f}% | {backbone} |"
        )
    lines.extend([
        "",
        "负的 Δ 表示 TimeRole 误差更低。",
        "",
        "## 与最近正式发表、可逐跨度核对的模型比较",
        "",
        f"- SimpleTM（ICLR 2025）逐跨度比较：MSE 胜出 **{mse_wins}/24**，MAE 胜出 **{mae_wins}/24**。",
        f"- 24 项任务级相对差值宏平均：MSE **{macro_mse_delta:+.2f}%**，MAE **{macro_mae_delta:+.2f}%**。",
        "- SimpleTM 使用固定输入长度 96；TimeRole 读取 336，其中图增强状态空间近期预测器只使用最近 96，额外 240 点仅供压缩历史修正分支。因此这是公开结果定位，不是完全等预算比较。",
        "- SimpleTM 原始论文及完整 Table 6：https://proceedings.iclr.cc/paper_files/paper/2025/file/27c546ab1e4f1d7d638e6a8dfbad9a07-Paper-Conference.pdf",
        "",
        "## 更新但不作为主公平基线：TimeCDS",
        "",
        "| Dataset | TimeRole Avg MSE | TimeRole Avg MAE | TimeCDS Avg MSE | TimeCDS Avg MAE |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset, ours_mse, ours_mae, other_mse, other_mae in timecds_rows:
        lines.append(
            f"| {dataset} | {ours_mse:.6f} | {ours_mae:.6f} | {other_mse:.3f} | {other_mae:.3f} |"
        )
    lines.extend([
        "",
        "TimeCDS 是 2026 年公开但已撤回的 ICLR submission；它使用输入长度 672，并以滚动 96 步方式生成 192/336/720，只公开这里所用的四跨度平均值。因此不能把该表当作逐跨度公平 SOTA 对比。",
        "OpenReview 状态：https://openreview.net/forum?id=0x1a6fSSeL",
        "",
        "## 实验协议和边界",
        "",
        "- 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Solar；不含 Traffic/ECL。",
        "- Horizon：96、192、336、720；seed 2021；M→M。",
        "- checkpoint 只由验证集最佳 MSE 选择，然后测试集一次性评估。",
        "- ETTm1/ETTm2 复用此前冻结的一次性 test 记录，没有再次读取测试集。",
        "- 只有 ETTh1/ETTh2 启用周期 24 主干；其他四个数据集使用独立双 Patch 主干。",
        "- 所有结果均为单 seed；不得根据本测试表修改 TimeRole-v1 结构或超参数。",
        "",
        "机器可读文件：",
        "",
        "- `logs/timerole_six_dataset_final/results.csv`",
        "- `logs/timerole_six_dataset_final/comparison_simpletm.csv`",
        "- `logs/timerole_six_dataset_final/records/`",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": "completed_with_literature_comparison",
        "tasks": 24,
        "simpletm_mse_wins": mse_wins,
        "simpletm_mae_wins": mae_wins,
        "macro_mse_delta_vs_simpletm_pct": macro_mse_delta,
        "macro_mae_delta_vs_simpletm_pct": macro_mae_delta,
        "comparison_warning": "TimeRole input=336 vs SimpleTM input=96; TimeCDS is withdrawn and input=672 rolling forecast.",
    }
    (OUTPUT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
