#!/usr/bin/env python3
"""Frozen six-task validation-only gate for GraphMamba TIRGE."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graphmamba_innovation.py"
OUTPUT = ROOT / "logs" / "graphmamba_rg_validation"
UNIVERSALITY = ROOT / "logs" / "graphmamba_darc_universality" / "final"
TASKS = (
    ("ETTh1", 192, "development"),
    ("ETTm1", 720, "development"),
    ("ETTm2", 720, "development"),
    ("ETTh1", 96, "protection"),
    ("ETTh2", 96, "protection"),
    ("ETTm2", 336, "protection"),
)


def main() -> int:
    for dataset, pred_len, role in TASKS:
        candidate = f"rg_{dataset.lower()}_{pred_len}_{role[:4]}"
        record = OUTPUT / "validation" / f"{candidate}.json"
        if record.exists() and json.loads(record.read_text()).get("status") == "completed":
            continue
        command = [
            sys.executable, "-u", str(RUNNER),
            "--dataset", dataset, "--pred-len", str(pred_len),
            "--model", "GraphMambaRG", "--candidate", candidate,
            "--seed", "2021", "--output-dir", str(OUTPUT),
        ]
        if subprocess.run(command, cwd=ROOT).returncode:
            return 1

    references = {}
    for path in UNIVERSALITY.glob("*.json"):
        row = json.loads(path.read_text())
        if int(row["seed"]) == 2021:
            references[(row["dataset"], int(row["pred_len"]), row["model"])] = row

    rows = []
    for dataset, pred_len, role in TASKS:
        candidate = f"rg_{dataset.lower()}_{pred_len}_{role[:4]}"
        rg = json.loads((OUTPUT / "validation" / f"{candidate}.json").read_text())
        baseline = references[(dataset, pred_len, "GraphMamba")]
        darc = references[(dataset, pred_len, "GraphMambaAF")]
        rows.append({
            "dataset": dataset,
            "pred_len": pred_len,
            "role": role,
            "baseline_val_mse": baseline["best_mse"],
            "darc_val_mse": darc["best_mse"],
            "rg_val_mse": rg["best_mse"],
            "rg_vs_baseline_mse_pct": 100 * (baseline["best_mse"] - rg["best_mse"]) / baseline["best_mse"],
            "rg_vs_darc_mse_pct": 100 * (darc["best_mse"] - rg["best_mse"]) / darc["best_mse"],
            "baseline_val_mae": baseline["best_mae"],
            "darc_val_mae": darc["best_mae"],
            "rg_val_mae": rg["best_mae"],
            "rg_vs_baseline_mae_pct": 100 * (baseline["best_mae"] - rg["best_mae"]) / baseline["best_mae"],
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['dataset']}-{row['pred_len']} {row['role']}: "
            f"MSE vs baseline {row['rg_vs_baseline_mse_pct']:+.3f}%, "
            f"vs DARC {row['rg_vs_darc_mse_pct']:+.3f}%; "
            f"MAE vs baseline {row['rg_vs_baseline_mae_pct']:+.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
