#!/usr/bin/env python3
"""Validation-only gate for the single rank-16 DARC refinement."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graphmamba_innovation.py"
OUTPUT = ROOT / "logs" / "graphmamba_darc_lowrank_validation"
TASKS = (
    ("ETTh1", 192, "development"),
    ("ETTm1", 720, "development"),
    ("ETTm2", 720, "development"),
    ("ETTh1", 96, "protection"),
    ("ETTh2", 96, "protection"),
    ("ETTm2", 336, "protection"),
)


def candidate_name(dataset: str, pred_len: int, role: str) -> str:
    role_code = "dev" if role == "development" else "prot"
    return f"lr16_{dataset.lower()}_{pred_len}_{role_code}"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset, pred_len, role in TASKS:
        candidate = candidate_name(dataset, pred_len, role)
        record = OUTPUT / "validation" / f"{candidate}.json"
        if record.exists():
            payload = json.loads(record.read_text(encoding="utf-8"))
            if payload.get("status") == "completed":
                print(f"skip {candidate}")
                continue
        command = [
            sys.executable, "-u", str(RUNNER),
            "--dataset", dataset, "--pred-len", str(pred_len),
            "--model", "GraphMambaAF", "--candidate", candidate,
            "--seed", "2021", "--af-mode", "variable_scale_lowrank",
            "--af-rank", "16", "--output-dir", str(OUTPUT),
        ]
        print(f"run {candidate}")
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode

    darc_records = {
        (r["dataset"], int(r["pred_len"]), int(r["seed"])): r
        for path in (ROOT / "logs" / "graphmamba_darc_universality" / "final").glob("*.json")
        if (r := json.loads(path.read_text(encoding="utf-8")))["model"] == "GraphMambaAF"
    }
    rows = []
    for dataset, pred_len, role in TASKS:
        candidate = candidate_name(dataset, pred_len, role)
        lowrank = json.loads(
            (OUTPUT / "validation" / f"{candidate}.json").read_text(encoding="utf-8")
        )
        full = darc_records[(dataset, pred_len, 2021)]
        rows.append({
            "dataset": dataset, "pred_len": pred_len, "role": role,
            "full_darc_val_mse": full["best_mse"],
            "lowrank_val_mse": lowrank["best_mse"],
            "mse_improvement_pct": 100 * (full["best_mse"] - lowrank["best_mse"]) / full["best_mse"],
            "full_darc_val_mae": full["best_mae"],
            "lowrank_val_mae": lowrank["best_mae"],
            "mae_improvement_pct": 100 * (full["best_mae"] - lowrank["best_mae"]) / full["best_mae"],
        })
    with (OUTPUT / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(
            f"{row['dataset']}-{row['pred_len']} {row['role']}: "
            f"MSE {row['mse_improvement_pct']:+.3f}%, "
            f"MAE {row['mae_improvement_pct']:+.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
