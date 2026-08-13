#!/usr/bin/env python3
"""Strict paired validation gate for CMRHM on ETTm1/ETTm2-720."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graphmamba_innovation.py"
OUTPUT = ROOT / "logs" / "graphmamba_cmrhm_validation"
TASKS = (("ETTm1", 720), ("ETTm2", 720))
MODELS = (("GraphMambaRecent", "recent336"), ("GraphMambaCMRHM", "cmrhm"))


def main() -> int:
    for dataset, pred_len in TASKS:
        for model, label in MODELS:
            candidate = f"{dataset.lower()}_{pred_len}_{label}_s2021"
            record = OUTPUT / "validation" / f"{candidate}.json"
            if record.exists() and json.loads(record.read_text()).get("status") == "completed":
                continue
            command = [
                sys.executable, "-u", str(RUNNER), "--dataset", dataset,
                "--pred-len", str(pred_len), "--seq-len", "336",
                "--model", model, "--candidate", candidate, "--seed", "2021",
                "--output-dir", str(OUTPUT),
            ]
            if subprocess.run(command, cwd=ROOT).returncode:
                return 1

    rows = []
    for dataset, pred_len in TASKS:
        records = {}
        for _, label in MODELS:
            candidate = f"{dataset.lower()}_{pred_len}_{label}_s2021"
            records[label] = json.loads(
                (OUTPUT / "validation" / f"{candidate}.json").read_text()
            )
        baseline, candidate = records["recent336"], records["cmrhm"]
        rows.append({
            "dataset": dataset, "pred_len": pred_len, "seed": 2021,
            "baseline_mse": baseline["best_mse"], "cmrhm_mse": candidate["best_mse"],
            "mse_improvement_pct": 100 * (baseline["best_mse"] - candidate["best_mse"]) / baseline["best_mse"],
            "baseline_mae": baseline["best_mae"], "cmrhm_mae": candidate["best_mae"],
            "mae_improvement_pct": 100 * (baseline["best_mae"] - candidate["best_mae"]) / baseline["best_mae"],
            "baseline_best_epoch": baseline["best_epoch"], "cmrhm_best_epoch": candidate["best_epoch"],
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['dataset']}-720: MSE {row['mse_improvement_pct']:+.3f}%, MAE {row['mae_improvement_pct']:+.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
