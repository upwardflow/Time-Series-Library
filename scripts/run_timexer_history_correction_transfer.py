#!/usr/bin/env python3
"""Validation-only transfer gate for frozen TimeRole-v1 on TimeXer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
OUTPUT = ROOT / "logs" / "timexer_timerole_transfer"
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
TASKS = (("ETTm1", 96), ("ETTm1", 720), ("ETTm2", 96), ("ETTm2", 720))
MODELS = (("TimeXerRecent", "recent336"), ("TimeXerHistoryCorrection", "timerole"))

# Exact effective values from the repository's original TimeXer ETTm scripts;
# omitted options use run.py defaults and are made explicit here.
PRESETS = {
    ("ETTm1", 96): {"d_model": 256, "d_ff": 2048, "batch_size": 4},
    ("ETTm1", 720): {"d_model": 256, "d_ff": 512, "batch_size": 4},
    ("ETTm2", 96): {"d_model": 256, "d_ff": 2048, "batch_size": 32},
    ("ETTm2", 720): {"d_model": 512, "d_ff": 2048, "batch_size": 32},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args, dataset: str, pred_len: int, model: str, candidate: str):
    preset = PRESETS[(dataset, pred_len)]
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(ROOT / "dataset" / "ETT-small"),
        "--data_path", f"{dataset}.csv", "--data", dataset,
        "--model_id", f"{dataset}_336_{pred_len}_{candidate}",
        "--model", model, "--seed", str(args.seed), "--features", "M",
        "--seq_len", "336", "--label_len", "48", "--pred_len", str(pred_len),
        "--e_layers", "1", "--factor", "3", "--enc_in", "7",
        "--dec_in", "7", "--c_out", "7", "--d_model", str(preset["d_model"]),
        "--d_ff", str(preset["d_ff"]), "--batch_size", str(preset["batch_size"]),
        "--learning_rate", "0.0001", "--train_epochs", str(args.epochs),
        "--patience", str(args.patience), "--num_workers", str(args.num_workers),
        "--gpu", "0", "--des", candidate, "--itr", "1",
        "--test_after_train", "0",
    ]


def run_one(command, log_path: Path, env):
    validation = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            match = VALIDATION_PATTERN.match(line.strip())
            if match:
                validation = json.loads(match.group(1))
        return_code = process.wait()
    return return_code, validation, round(time.monotonic() - started, 3)


def write_summary() -> list[dict]:
    rows = []
    for dataset, pred_len in TASKS:
        records = {}
        for _, label in MODELS:
            path = OUTPUT / "validation" / f"{dataset.lower()}_{pred_len}_{label}_s2021.json"
            records[label] = json.loads(path.read_text())
        baseline, timerole = records["recent336"], records["timerole"]
        rows.append({
            "dataset": dataset, "pred_len": pred_len, "seed": 2021,
            "baseline_mse": baseline["best_mse"], "timerole_mse": timerole["best_mse"],
            "mse_improvement_pct": 100 * (baseline["best_mse"] - timerole["best_mse"]) / baseline["best_mse"],
            "baseline_mae": baseline["best_mae"], "timerole_mae": timerole["best_mae"],
            "mae_improvement_pct": 100 * (baseline["best_mae"] - timerole["best_mae"]) / baseline["best_mae"],
            "baseline_best_epoch": baseline["best_epoch"],
            "timerole_best_epoch": timerole["best_epoch"],
        })
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    args = parse_args()
    validation_dir = OUTPUT / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for dataset, pred_len in TASKS:
        for model, label in MODELS:
            candidate = f"{dataset.lower()}_{pred_len}_{label}_s{args.seed}"
            record_path = validation_dir / f"{candidate}.json"
            if record_path.exists() and not args.force:
                record = json.loads(record_path.read_text())
                if record.get("status") == "completed":
                    continue
            command = build_command(args, dataset, pred_len, model, candidate)
            log_path = validation_dir / f"{candidate}.log"
            print("Command:", shlex.join(command))
            if args.dry_run:
                continue
            return_code, validation, duration = run_one(command, log_path, env)
            payload = {
                "status": "completed" if return_code == 0 and validation else "failed",
                "model": model, "candidate": candidate, "dataset": dataset,
                "pred_len": pred_len, "seed": args.seed, "final_test": False,
                "protocol": f"{dataset}_recent96_from_seq336_{pred_len}_M",
                "return_code": return_code, "duration_seconds": duration,
                "recorded_at": datetime.now().astimezone().isoformat(),
                "command": command, "log_path": str(log_path),
            }
            if validation:
                payload.update(validation)
            temporary = record_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(record_path)
            if payload["status"] != "completed":
                print(f"Run failed; stopping without retry: {candidate}", file=sys.stderr)
                return return_code or 1

    if args.dry_run:
        return 0
    rows = write_summary()
    mean_mse = sum(row["mse_improvement_pct"] for row in rows) / len(rows)
    mean_mae = sum(row["mae_improvement_pct"] for row in rows) / len(rows)
    wins = sum(row["mse_improvement_pct"] > 0 for row in rows)
    for row in rows:
        print(f"{row['dataset']}-{row['pred_len']}: MSE {row['mse_improvement_pct']:+.3f}%, MAE {row['mae_improvement_pct']:+.3f}%")
    print(f"Gate: {wins}/4 MSE wins; macro MSE {mean_mse:+.3f}%, MAE {mean_mae:+.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
