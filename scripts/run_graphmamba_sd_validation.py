#!/usr/bin/env python3
"""Validation-only gate for GraphMamba scale-disagreement calibration."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graphmamba_innovation.py"
OUTPUT = ROOT / "logs" / "graphmamba_sd_validation"
UNIVERSALITY = ROOT / "logs" / "graphmamba_darc_universality" / "final"
TASKS = (
    ("ETTh1", 192, "development"), ("ETTm1", 720, "development"),
    ("ETTm2", 720, "development"), ("ETTh1", 96, "protection"),
    ("ETTh2", 96, "protection"), ("ETTm2", 336, "protection"),
)


def main() -> int:
    for dataset, pred_len, role in TASKS:
        candidate = f"sd_{dataset.lower()}_{pred_len}_{role[:4]}"
        record = OUTPUT / "validation" / f"{candidate}.json"
        if record.exists() and json.loads(record.read_text()).get("status") == "completed":
            continue
        cmd = [sys.executable, "-u", str(RUNNER), "--dataset", dataset,
               "--pred-len", str(pred_len), "--model", "GraphMambaSD",
               "--candidate", candidate, "--seed", "2021",
               "--output-dir", str(OUTPUT)]
        if subprocess.run(cmd, cwd=ROOT).returncode:
            return 1

    references = {}
    for path in UNIVERSALITY.glob("*.json"):
        row = json.loads(path.read_text())
        if int(row["seed"]) == 2021:
            references[(row["dataset"], int(row["pred_len"]), row["model"])] = row
    rows = []
    for dataset, pred_len, role in TASKS:
        candidate = f"sd_{dataset.lower()}_{pred_len}_{role[:4]}"
        sd = json.loads((OUTPUT / "validation" / f"{candidate}.json").read_text())
        base = references[(dataset, pred_len, "GraphMamba")]
        darc = references[(dataset, pred_len, "GraphMambaAF")]
        rows.append({
            "dataset": dataset, "pred_len": pred_len, "role": role,
            "baseline_val_mse": base["best_mse"], "darc_val_mse": darc["best_mse"],
            "sd_val_mse": sd["best_mse"],
            "sd_vs_baseline_mse_pct": 100*(base["best_mse"]-sd["best_mse"])/base["best_mse"],
            "sd_vs_darc_mse_pct": 100*(darc["best_mse"]-sd["best_mse"])/darc["best_mse"],
            "baseline_val_mae": base["best_mae"], "darc_val_mae": darc["best_mae"],
            "sd_val_mae": sd["best_mae"],
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for r in rows:
        print(f"{r['dataset']}-{r['pred_len']} {r['role']}: vs baseline {r['sd_vs_baseline_mse_pct']:+.3f}%, vs DARC {r['sd_vs_darc_mse_pct']:+.3f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
