#!/usr/bin/env python3
"""Run and summarize the frozen DARC universality matrix on ETT datasets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUNNER = REPO_ROOT / "scripts" / "run_graphmamba_innovation.py"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "graphmamba_darc_universality"
DEFAULT_REPORT = REPO_ROOT / "experiment_results" / "GraphMamba_DARC_universality.md"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
PRED_LENS = (96, 192, 336, 720)
MODELS = (("baseline", "GraphMamba"), ("darc", "GraphMambaAF"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--pred-lens", nargs="+", type=int, choices=PRED_LENS, default=list(PRED_LENS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    args.datasets = list(dict.fromkeys(args.datasets))
    args.pred_lens = list(dict.fromkeys(args.pred_lens))
    args.seeds = list(dict.fromkeys(args.seeds))
    args.output_dir = args.output_dir.resolve()
    args.report = args.report.resolve()
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("max-runs must be positive")
    return args


def candidate_name(kind: str, dataset: str, pred_len: int, seed: int) -> str:
    return f"{dataset.lower()}_p{pred_len}_s{seed}_{kind}_frozen"


def record_path(output_dir: Path, candidate: str) -> Path:
    return output_dir / "final" / f"{candidate}.json"


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = {"best_mse", "best_mae", "test_mse", "test_mae"}
    return payload.get("status") == "completed" and required.issubset(payload)


def load_records(output_dir: Path) -> list[dict]:
    records = []
    for path in sorted((output_dir / "final").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "completed" and row.get("final_test"):
            records.append(row)
    return records


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(args: argparse.Namespace) -> None:
    records = load_records(args.output_dir)
    run_fields = [
        "dataset", "pred_len", "seed", "model", "candidate", "best_mse", "best_mae",
        "test_mse", "test_mae", "best_epoch", "epochs_ran", "duration_seconds", "status",
    ]
    write_csv(args.output_dir / "runs.csv", records, run_fields)

    grouped = {}
    for row in records:
        key = (row["dataset"], int(row["pred_len"]), int(row["seed"]))
        kind = "darc" if row["model"] == "GraphMambaAF" else "baseline"
        grouped.setdefault(key, {})[kind] = row

    summary_rows = []
    for dataset in args.datasets:
        for pred_len in args.pred_lens:
            pairs = [
                grouped[(dataset, pred_len, seed)]
                for seed in args.seeds
                if (dataset, pred_len, seed) in grouped
                and {"baseline", "darc"}.issubset(grouped[(dataset, pred_len, seed)])
            ]
            if not pairs:
                continue
            row = {"dataset": dataset, "pred_len": pred_len, "paired_seeds": len(pairs)}
            for split, metric in (("val", "best_mse"), ("val_mae", "best_mae"),
                                  ("test", "test_mse"), ("test_mae", "test_mae")):
                b_mean, b_std = mean_std([pair["baseline"][metric] for pair in pairs])
                d_mean, d_std = mean_std([pair["darc"][metric] for pair in pairs])
                row[f"baseline_{split}_mean"] = b_mean
                row[f"baseline_{split}_std"] = b_std
                row[f"darc_{split}_mean"] = d_mean
                row[f"darc_{split}_std"] = d_std
                row[f"{split}_improvement_pct"] = 100.0 * (b_mean - d_mean) / b_mean
            row["test_mse_seed_wins"] = sum(
                pair["darc"]["test_mse"] < pair["baseline"]["test_mse"] for pair in pairs
            )
            row["test_mae_seed_wins"] = sum(
                pair["darc"]["test_mae"] < pair["baseline"]["test_mae"] for pair in pairs
            )
            summary_rows.append(row)

    if summary_rows:
        write_csv(args.output_dir / "summary.csv", summary_rows, list(summary_rows[0]))

    expected_pairs = len(args.datasets) * len(args.pred_lens)
    complete = len(summary_rows) == expected_pairs and all(
        row["paired_seeds"] == len(args.seeds) for row in summary_rows
    )
    mse_task_wins = sum(row["test_improvement_pct"] > 0 for row in summary_rows)
    mae_task_wins = sum(row["test_mae_improvement_pct"] > 0 for row in summary_rows)
    avg_mse_gain = statistics.mean([row["test_improvement_pct"] for row in summary_rows]) if summary_rows else float("nan")
    avg_mae_gain = statistics.mean([row["test_mae_improvement_pct"] for row in summary_rows]) if summary_rows else float("nan")

    lines = [
        "# GraphMamba DARC 普适性实验报告",
        "",
        "## Material Passport",
        "",
        "- Experiment: frozen DARC universality matrix",
        f"- Status: {'COMPLETE' if complete else 'IN PROGRESS'}",
        "- Protocol: ETT, M→M, input=96, prediction=96/192/336/720",
        f"- Seeds: {', '.join(map(str, args.seeds))}",
        "- Selection: element-weighted validation MSE; test uses the same best-validation checkpoint",
        "- No architecture or hyperparameter changes after test inspection",
        "",
        "## 汇总结论",
        "",
        f"- 完成配对任务：{len(summary_rows)}/{expected_pairs}；每项计划 {len(args.seeds)} 个随机种子。",
        f"- Test MSE 均值获胜：{mse_task_wins}/{len(summary_rows) if summary_rows else 0}。",
        f"- Test MAE 均值获胜：{mae_task_wins}/{len(summary_rows) if summary_rows else 0}。",
        f"- 任务级平均相对变化：MSE {avg_mse_gain:+.3f}%，MAE {avg_mae_gain:+.3f}%（正值表示DARC改善）。",
        "",
        "## 三随机种子测试结果（mean ± std）",
        "",
        "| Dataset | Pred | Baseline MSE | DARC MSE | ΔMSE | Baseline MAE | DARC MAE | ΔMAE | seed wins MSE/MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | "
            f"{row['baseline_test_mean']:.6f} ± {row['baseline_test_std']:.6f} | "
            f"{row['darc_test_mean']:.6f} ± {row['darc_test_std']:.6f} | "
            f"{row['test_improvement_pct']:+.3f}% | "
            f"{row['baseline_test_mae_mean']:.6f} ± {row['baseline_test_mae_std']:.6f} | "
            f"{row['darc_test_mae_mean']:.6f} ± {row['darc_test_mae_std']:.6f} | "
            f"{row['test_mae_improvement_pct']:+.3f}% | "
            f"{row['test_mse_seed_wins']}/{row['test_mae_seed_wins']} |"
        )
    lines += [
        "",
        "## 审计说明",
        "",
        "- Baseline 与 DARC 对每个 dataset/prediction/seed 严格配对。",
        "- DARC 固定使用 `variable_scale_residual`，未进行数据集或预测长度专属调参。",
        "- 原始逐次记录位于 `logs/graphmamba_darc_universality/final/`。",
        "- `runs.csv` 保存每次运行；`summary.csv` 保存配对统计。",
        "- 正值 Δ 表示误差下降，负值如实表示退化。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        (dataset, pred_len, seed, kind, model)
        for dataset in args.datasets
        for pred_len in args.pred_lens
        for seed in args.seeds
        for kind, model in MODELS
    ]
    if args.max_runs is not None:
        planned = planned[:args.max_runs]
    print(f"Planned runs: {len(planned)}")
    started = time.monotonic()
    for index, (dataset, pred_len, seed, kind, model) in enumerate(planned, 1):
        candidate = candidate_name(kind, dataset, pred_len, seed)
        path = record_path(args.output_dir, candidate)
        if completed(path):
            print(f"[{index}/{len(planned)}] skip completed {candidate}")
            continue
        command = [
            sys.executable, "-u", str(SINGLE_RUNNER),
            "--dataset", dataset,
            "--pred-len", str(pred_len),
            "--model", model,
            "--candidate", candidate,
            "--seed", str(seed),
            "--gpu", str(args.gpu),
            "--af-mode", "variable_scale_residual",
            "--final-test",
            "--output-dir", str(args.output_dir),
        ]
        print(f"[{index}/{len(planned)}] run {candidate}")
        if args.dry_run:
            print(" ".join(command))
            continue
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        summarize(args)
        if result.returncode != 0:
            print(f"Stopped after failed run: {candidate}", file=sys.stderr)
            return result.returncode
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"[{index}/{len(planned)}] done {candidate}: "
            f"test MSE={payload['test_mse']:.6f}, MAE={payload['test_mae']:.6f}"
        )
    summarize(args)
    elapsed = (time.monotonic() - started) / 60.0
    print(f"Universality matrix finished in {elapsed:.2f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
