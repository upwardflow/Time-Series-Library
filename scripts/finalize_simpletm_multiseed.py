#!/usr/bin/env python3
"""Aggregate SimpleTM five-dataset, three-seed results."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ETTH1_RECORDS = ROOT / "logs/simpletm_etth1_sl336/records"
REMAINING_RECORDS = ROOT / "logs/simpletm_remaining_sl336/records"
OUTPUT = ROOT / "logs/simpletm_multiseed"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)


def record_path(dataset: str, horizon: int, seed: int) -> Path:
    directory = ETTH1_RECORDS if dataset == "ETTh1" else REMAINING_RECORDS
    return directory / f"simpletm_{dataset.lower()}_sl336_pl{horizon}_s{seed}.json"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    long_rows: list[dict[str, object]] = []
    missing, failed = [], []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                path = record_path(dataset, horizon, seed)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "completed" or "test_mse" not in payload:
                    failed.append(str(path))
                    continue
                long_rows.append({
                    "model": "SimpleTM", "dataset": dataset,
                    "horizon": horizon, "seq_len": 336, "seed": seed,
                    "validation_mse": payload.get("validation_best_mse"),
                    "validation_mae": payload.get("validation_best_mae"),
                    "best_epoch": payload.get("validation_best_epoch"),
                    "test_mse": payload["test_mse"],
                    "test_mae": payload["test_mae"],
                    "duration_seconds": payload.get("duration_seconds"),
                    "record_path": str(path),
                })

    expected = len(DATASETS) * len(HORIZONS) * len(SEEDS)
    status = {
        "status": "completed" if len(long_rows) == expected and not missing and not failed else "incomplete",
        "expected": expected, "completed": len(long_rows),
        "missing": len(missing), "failed": len(failed),
        "missing_records": missing, "failed_records": failed,
        "std_definition": "sample standard deviation (n-1; statistics.stdev)",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if status["status"] != "completed":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2

    write_csv(OUTPUT / "results_long.csv", long_rows)
    lookup: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in long_rows:
        lookup.setdefault((str(row["dataset"]), int(row["horizon"])), []).append(row)

    summary_rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            rows = sorted(lookup[(dataset, horizon)], key=lambda row: int(row["seed"]))
            mses = [float(row["test_mse"]) for row in rows]
            maes = [float(row["test_mae"]) for row in rows]
            summary_rows.append({
                "model": "SimpleTM", "dataset": dataset, "horizon": horizon,
                "seq_len": 336, "n_seeds": len(rows),
                "seeds": ";".join(str(row["seed"]) for row in rows),
                "test_mse_mean": statistics.mean(mses),
                "test_mse_std": statistics.stdev(mses),
                "test_mae_mean": statistics.mean(maes),
                "test_mae_std": statistics.stdev(maes),
            })
    write_csv(OUTPUT / "mean_std.csv", summary_rows)

    lines = [
        "# SimpleTM three-seed results", "",
        "- Protocol: seq_len=336, seeds=2021/2022/2023, validation-selected checkpoint.",
        "- Dispersion: sample standard deviation (n-1).", "",
        "| Dataset | Horizon | MSE (mean ± std) | MAE (mean ± std) |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | "
            f"{row['test_mse_mean']:.4f} ± {row['test_mse_std']:.4f} | "
            f"{row['test_mae_mean']:.4f} ± {row['test_mae_std']:.4f} |"
        )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"LONG_CSV {OUTPUT / 'results_long.csv'}")
    print(f"MEAN_STD_CSV {OUTPUT / 'mean_std.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
