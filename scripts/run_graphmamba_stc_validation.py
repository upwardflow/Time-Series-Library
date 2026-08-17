#!/usr/bin/env python3
"""Four-task validation-only gate for GraphMambaSTC."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graphmamba_innovation.py"
OUTPUT = ROOT / "logs" / "graphmamba_stc_validation"
UNIVERSALITY = ROOT / "logs" / "graphmamba_darc_universality" / "final"
TASKS = (("ETTm1", 96), ("ETTm2", 96), ("ETTm1", 720), ("ETTm2", 720))


def main() -> int:
    for dataset, pred_len in TASKS:
        candidate = f"stc_{dataset.lower()}_{pred_len}_gate"
        record = OUTPUT / "validation" / f"{candidate}.json"
        if record.exists() and json.loads(record.read_text()).get("status") == "completed":
            continue
        command = [
            sys.executable, "-u", str(RUNNER), "--dataset", dataset,
            "--pred-len", str(pred_len), "--model", "GraphMambaSTC",
            "--candidate", candidate, "--seed", "2021", "--output-dir", str(OUTPUT),
        ]
        if subprocess.run(command, cwd=ROOT).returncode:
            return 1

    references = {}
    for path in UNIVERSALITY.glob("*.json"):
        row = json.loads(path.read_text())
        if int(row["seed"]) == 2021 and row["model"] == "GraphMamba":
            references[(row["dataset"], int(row["pred_len"]))] = row
    rows = []
    for dataset, pred_len in TASKS:
        candidate = f"stc_{dataset.lower()}_{pred_len}_gate"
        result = json.loads((OUTPUT / "validation" / f"{candidate}.json").read_text())
        baseline = references[(dataset, pred_len)]
        rows.append({
            "dataset": dataset, "pred_len": pred_len,
            "baseline_val_mse": baseline["best_mse"], "stc_val_mse": result["best_mse"],
            "mse_improvement_pct": 100*(baseline["best_mse"]-result["best_mse"])/baseline["best_mse"],
            "baseline_val_mae": baseline["best_mae"], "stc_val_mae": result["best_mae"],
            "mae_improvement_pct": 100*(baseline["best_mae"]-result["best_mae"])/baseline["best_mae"],
            "best_epoch": result["best_epoch"], "epochs_ran": result["epochs_ran"],
        })
    with (OUTPUT/"comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:
        print(f"{row['dataset']}-{row['pred_len']}: MSE {row['mse_improvement_pct']:+.3f}%, MAE {row['mae_improvement_pct']:+.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
