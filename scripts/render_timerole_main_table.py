#!/usr/bin/env python3
"""Render TimeRole Table 2 and reviewer-facing summary from audited records."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs" / "q2_main_baselines" / "main_results_long.csv"
TABLE = ROOT / "paper" / "tables" / "Table2_TimeRole_MSE_MAE.md"
SUMMARY = ROOT / "logs" / "q2_main_baselines" / "timerole_main_summary.json"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")
HORIZONS = (96, 192, 336, 720)
MODELS = (
    "DLinear", "PatchTST", "iTransformer", "TimeMixer", "TimesNet",
    "SMamba", "MSGNet", "TimeRole",
)
DISPLAY = {"SMamba": "S-Mamba", "weather": "Weather"}


def ranked(value: float, ordered: list[float]) -> str:
    shown = f"{value:.4f}"
    unique = sorted(set(ordered))
    if value == unique[0]:
        return f"**{shown}**"
    if len(unique) > 1 and value == unique[1]:
        return f"<u>{shown}</u>"
    return shown


def main() -> int:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        (row["dataset"], int(row["horizon"]), row["model"]): row
        for row in rows
    }
    expected = len(DATASETS) * len(HORIZONS) * len(MODELS)
    if len(lookup) != expected:
        raise RuntimeError(f"expected {expected} audited rows, found {len(lookup)}")

    lines = [
        "# 表2：五数据集统一协议下的长期预测结果", "",
        "单元格依次为 MSE / MAE；两项指标均越低越好。最佳结果以粗体表示，次优结果以下划线表示；排名依据未舍入数值计算，显示值保留四位小数。", "",
        "| 数据集 | $H$ | " + " | ".join(DISPLAY.get(m, m) for m in MODELS) + " |",
        "|---|---:|" + "---:|" * len(MODELS),
    ]
    metric_stats = {}
    for metric in ("test_mse", "test_mae"):
        wins = seconds = 0
        relative_improvements = []
        nonwins = []
        baselines = MODELS[:-1]
        for dataset in DATASETS:
            for horizon in HORIZONS:
                values = {
                    model: float(lookup[(dataset, horizon, model)][metric])
                    for model in MODELS
                }
                ours = values["TimeRole"]
                baseline_best_model = min(baselines, key=lambda model: values[model])
                baseline_best = values[baseline_best_model]
                ordered = sorted(set(values.values()))
                wins += ours == ordered[0]
                seconds += len(ordered) > 1 and ours == ordered[1]
                relative_improvements.append((baseline_best - ours) / baseline_best * 100)
                if ours >= baseline_best:
                    nonwins.append({
                        "dataset": DISPLAY.get(dataset, dataset),
                        "horizon": horizon,
                        "timerole": ours,
                        "best_baseline": baseline_best,
                        "best_baseline_model": DISPLAY.get(baseline_best_model, baseline_best_model),
                    })
        metric_stats[metric] = {
            "best_count": wins,
            "second_count": seconds,
            "macro_relative_improvement_vs_taskwise_best_baseline_percent": (
                sum(relative_improvements) / len(relative_improvements)
            ),
            "nonwinning_tasks": nonwins,
        }

    for dataset in DATASETS:
        for row_index, horizon in enumerate(HORIZONS):
            mse_values = [
                float(lookup[(dataset, horizon, model)]["test_mse"])
                for model in MODELS
            ]
            mae_values = [
                float(lookup[(dataset, horizon, model)]["test_mae"])
                for model in MODELS
            ]
            cells = []
            for model, mse, mae in zip(MODELS, mse_values, mae_values):
                cells.append(f"{ranked(mse, mse_values)} / {ranked(mae, mae_values)}")
            dataset_label = DISPLAY.get(dataset, dataset) if row_index == 0 else ""
            lines.append(f"| {dataset_label} | {horizon} | " + " | ".join(cells) + " |")

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY.write_text(
        json.dumps(metric_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(TABLE)
    print(SUMMARY)
    print(json.dumps(metric_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
