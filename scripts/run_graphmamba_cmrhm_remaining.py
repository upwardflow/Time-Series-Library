#!/usr/bin/env python3
"""Run strict paired CMRHM validation on the remaining ETTm horizons."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graphmamba_innovation.py"
OUTPUT = ROOT / "logs" / "graphmamba_cmrhm_all_horizons"
DATASETS = ("ETTm1", "ETTm2")
PRED_LENS = (96, 192, 336, 720)
REMAINING_PRED_LENS = (96, 192, 336)
MODELS = (("GraphMambaRecent", "recent336"), ("GraphMambaCMRHM", "cmrhm"))
SEED = 2021


def candidate_name(dataset: str, pred_len: int, label: str) -> str:
    return f"{dataset.lower()}_{pred_len}_{label}_s{SEED}"


def completed_record(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def import_720_records() -> None:
    """Reuse the already completed, protocol-identical 720 validation records."""
    source = ROOT / "logs" / "graphmamba_cmrhm_validation" / "validation"
    destination = OUTPUT / "validation"
    destination.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for _, label in MODELS:
            name = candidate_name(dataset, 720, label)
            source_path = source / f"{name}.json"
            destination_path = destination / f"{name}.json"
            if not destination_path.exists() and source_path.exists():
                destination_path.write_text(
                    source_path.read_text(encoding="utf-8"), encoding="utf-8"
                )


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            records = {}
            for _, label in MODELS:
                name = candidate_name(dataset, pred_len, label)
                path = OUTPUT / "validation" / f"{name}.json"
                if not completed_record(path):
                    continue
                records[label] = json.loads(path.read_text(encoding="utf-8"))
            if len(records) != len(MODELS):
                continue
            baseline, cmrhm = records["recent336"], records["cmrhm"]
            rows.append({
                "dataset": dataset,
                "pred_len": pred_len,
                "seed": SEED,
                "baseline_mse": baseline["best_mse"],
                "cmrhm_mse": cmrhm["best_mse"],
                "mse_improvement_pct": 100 * (baseline["best_mse"] - cmrhm["best_mse"]) / baseline["best_mse"],
                "baseline_mae": baseline["best_mae"],
                "cmrhm_mae": cmrhm["best_mae"],
                "mae_improvement_pct": 100 * (baseline["best_mae"] - cmrhm["best_mae"]) / baseline["best_mae"],
                "baseline_best_epoch": baseline["best_epoch"],
                "cmrhm_best_epoch": cmrhm["best_epoch"],
                "baseline_duration_seconds": baseline["duration_seconds"],
                "cmrhm_duration_seconds": cmrhm["duration_seconds"],
            })
    return rows


def write_comparison(rows: list[dict[str, object]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    import_720_records()
    for dataset in DATASETS:
        for pred_len in REMAINING_PRED_LENS:
            for model, label in MODELS:
                name = candidate_name(dataset, pred_len, label)
                record = OUTPUT / "validation" / f"{name}.json"
                if completed_record(record):
                    continue
                command = [
                    sys.executable,
                    "-u",
                    str(RUNNER),
                    "--dataset",
                    dataset,
                    "--pred-len",
                    str(pred_len),
                    "--seq-len",
                    "336",
                    "--model",
                    model,
                    "--candidate",
                    name,
                    "--seed",
                    str(SEED),
                    "--output-dir",
                    str(OUTPUT),
                ]
                print(f"\n=== {dataset}-{pred_len} {label} ===", flush=True)
                if subprocess.run(command, cwd=ROOT).returncode:
                    write_comparison(collect_rows())
                    return 1
                write_comparison(collect_rows())

    rows = collect_rows()
    write_comparison(rows)
    for row in rows:
        print(
            f"{row['dataset']}-{row['pred_len']}: "
            f"MSE {row['mse_improvement_pct']:+.3f}%, "
            f"MAE {row['mae_improvement_pct']:+.3f}%"
        )
    return 0 if len(rows) == len(DATASETS) * len(PRED_LENS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
