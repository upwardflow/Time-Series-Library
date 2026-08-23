#!/usr/bin/env python3
"""Aggregate the five-dataset main-paper baseline records into CSV and Markdown."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "q2_main_baselines"
TIMEMIXER_STABILITY = ROOT / "logs" / "q2_timemixer_stability" / "summary.csv"
MODELS = (
    "DLinear", "PatchTST", "iTransformer", "TimeMixer", "TimesNet",
    "SMamba", "MSGNet", "TimeRole",
)
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")
HORIZONS = (96, 192, 336, 720)
SEED = 2021
NUMERICAL_DIVERGENCE_MSE = 10.0


def slug(model: str, dataset: str, horizon: int) -> str:
    return f"{model.lower()}_{dataset.lower()}_sl336_pl{horizon}_s{SEED}"


def record_path(model: str, dataset: str, horizon: int) -> Path:
    # TimeRole was frozen before the manuscript rename; retain provenance while
    # presenting the final model name in aggregate outputs.
    stored_model = "GraphMambaCMRHM" if model == "TimeRole" else model
    return OUTPUT / "records" / f"{slug(stored_model, dataset, horizon)}.json"


def load_timemixer_stability() -> dict[tuple[str, int], dict[str, str]]:
    if not TIMEMIXER_STABILITY.is_file():
        return {}
    with TIMEMIXER_STABILITY.open(newline="", encoding="utf-8") as handle:
        return {
            (row["dataset"], int(row["horizon"])): row
            for row in csv.DictReader(handle)
        }


def main() -> int:
    rows, missing, failed, unstable = [], [], [], []
    timemixer_stability = load_timemixer_stability()
    for model in MODELS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                path = record_path(model, dataset, horizon)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "completed" or "test_mse" not in payload:
                    failed.append(str(path))
                    continue
                stable = timemixer_stability.get((dataset, horizon)) if model == "TimeMixer" else None
                mse = float(stable["test_mse"]) if stable else float(payload["test_mse"])
                mae = float(stable["test_mae"]) if stable else float(payload["test_mae"])
                quality = (
                    "numerically_unstable"
                    if not math.isfinite(mse)
                    or not math.isfinite(mae)
                    or mse >= NUMERICAL_DIVERGENCE_MSE
                    else "valid"
                )
                if quality != "valid":
                    unstable.append(str(path))
                rows.append({
                    "model": model, "dataset": dataset, "horizon": horizon,
                    "seed": SEED, "seq_len": 336,
                    "validation_mse": (float(stable["validation_mse"])
                                       if stable else payload.get("validation_best_mse")),
                    "validation_mae": (float(stable["validation_mae"])
                                       if stable else payload.get("validation_best_mae")),
                    "test_mse": mse, "test_mae": mae, "quality": quality,
                    "duration_seconds": payload.get("duration_seconds"),
                    "result_source": ("q2_timemixer_stability_summary"
                                      if stable else payload.get("result_source")),
                })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "main_results_long.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    status = {
        "expected": len(MODELS) * len(DATASETS) * len(HORIZONS),
        "completed": len(rows), "missing": len(missing), "failed": len(failed),
        "numerically_unstable": len(unstable),
        "missing_records": missing, "failed_records": failed,
        "numerically_unstable_records": unstable,
    }
    (OUTPUT / "aggregate_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = OUTPUT / "main_table.md"
    lines = [
        "# 五数据集统一协议主实验表", "",
        "- 输入长度：336；预测长度：96/192/336/720；seed：2021；M→M。",
        "- 所有新运行均由验证集最佳 MSE 选择检查点，再进行一次性测试。",
        f"- 完成度：{len(rows)}/{status['expected']}；缺失 {len(missing)}；失败 {len(failed)}；数值发散 {len(unstable)}。",
        "- 数值发散单元保留原始记录，但不参与表中平均值与模型排名。", "",
    ]
    lookup = {(r["model"], r["dataset"], r["horizon"]): r for r in rows}
    for dataset in DATASETS:
        lines.extend([
            f"## {dataset}", "",
            "| Model | 96 MSE/MAE | 192 MSE/MAE | 336 MSE/MAE | 720 MSE/MAE | Avg MSE | Avg MAE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for model in MODELS:
            cells, mses, maes = [], [], []
            for horizon in HORIZONS:
                row = lookup.get((model, dataset, horizon))
                if row:
                    if row["quality"] != "valid":
                        cells.append("数值发散")
                        continue
                    mse, mae = float(row["test_mse"]), float(row["test_mae"])
                    cells.append(f"{mse:.4f}/{mae:.4f}")
                    mses.append(mse); maes.append(mae)
                else:
                    cells.append("—")
            avg_mse = f"{sum(mses)/len(mses):.4f}" if mses else "—"
            avg_mae = f"{sum(maes)/len(maes):.4f}" if maes else "—"
            lines.append(f"| {model} | {' | '.join(cells)} | {avg_mse} | {avg_mae} |")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"Table: {report}")
    return 0 if not missing and not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
