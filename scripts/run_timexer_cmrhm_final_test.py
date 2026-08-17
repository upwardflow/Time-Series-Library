#!/usr/bin/env python3
"""One-shot test of validation-selected TimeXer/CMRHM checkpoints."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "logs" / "timexer_cmrhm_transfer" / "validation"
OUTPUT = ROOT / "logs" / "timexer_cmrhm_final_test"
TASKS = (("ETTm1", 96), ("ETTm1", 720), ("ETTm2", 96), ("ETTm2", 720))
LABELS = ("recent336", "cmrhm")
SEED = 2021
TEST_PATTERN = re.compile(r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:")


def name(dataset: str, pred_len: int, label: str) -> str:
    return f"{dataset.lower()}_{pred_len}_{label}_s{SEED}"


def load_validation(dataset: str, pred_len: int, label: str) -> dict:
    path = VALIDATION / f"{name(dataset, pred_len, label)}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "completed" or record.get("final_test") is not False:
        raise RuntimeError(f"Invalid validation source: {path}")
    return record


def test_command(record: dict) -> list[str]:
    command = list(record["command"])
    command[command.index("--is_training") + 1] = "0"
    command[command.index("--test_after_train") + 1] = "0"
    return command


def setting_from_command(command: list[str]) -> str:
    def value(option: str) -> str:
        return command[command.index(option) + 1]
    return (
        f"{value('--task_name')}_{value('--model_id')}_{value('--model')}_{value('--data')}"
        f"_ft{value('--features')}_sl{value('--seq_len')}_ll{value('--label_len')}"
        f"_pl{value('--pred_len')}_dm{value('--d_model')}_nh8_el{value('--e_layers')}"
        f"_dl1_df{value('--d_ff')}_expand2_dc4_fc{value('--factor')}"
        f"_ebtimeF_dtTrue_{value('--des')}_0"
    )


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "completed" and "test_mse" in record


def run_one(dataset: str, pred_len: int, label: str) -> int:
    candidate = name(dataset, pred_len, label)
    destination = OUTPUT / "records" / f"{candidate}.json"
    log_path = OUTPUT / "logs" / f"{candidate}.log"
    if completed(destination):
        return 0
    validation = load_validation(dataset, pred_len, label)
    command = test_command(validation)
    checkpoint = ROOT / "checkpoints" / setting_from_command(command) / "checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen validation checkpoint not found: {checkpoint}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    metrics = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
            match = TEST_PATTERN.match(line.strip())
            if match:
                metrics = {
                    "test_mse": float(match.group(1)),
                    "test_mae": float(match.group(2)),
                }
        return_code = process.wait()

    payload = {
        "status": "completed" if return_code == 0 and metrics else "failed",
        "dataset": dataset, "pred_len": pred_len, "model": validation["model"],
        "label": label, "candidate": candidate, "seed": SEED,
        "checkpoint_selected_by": "validation_best_mse",
        "validation_best_epoch": validation["best_epoch"],
        "validation_best_mse": validation["best_mse"],
        "validation_best_mae": validation["best_mae"],
        "checkpoint_path": str(checkpoint), "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "source_validation_record": str(VALIDATION / f"{candidate}.json"),
        "log_path": str(log_path), "command": command,
    }
    if metrics:
        payload.update(metrics)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return 0 if payload["status"] == "completed" else 1


def collect() -> list[dict]:
    rows = []
    for dataset, pred_len in TASKS:
        pair = {}
        for label in LABELS:
            path = OUTPUT / "records" / f"{name(dataset, pred_len, label)}.json"
            if completed(path):
                pair[label] = json.loads(path.read_text(encoding="utf-8"))
        if len(pair) != 2:
            continue
        baseline, cmrhm = pair["recent336"], pair["cmrhm"]
        rows.append({
            "dataset": dataset, "pred_len": pred_len, "seed": SEED,
            "baseline_test_mse": baseline["test_mse"],
            "cmrhm_test_mse": cmrhm["test_mse"],
            "mse_improvement_pct": 100 * (baseline["test_mse"] - cmrhm["test_mse"]) / baseline["test_mse"],
            "baseline_test_mae": baseline["test_mae"],
            "cmrhm_test_mae": cmrhm["test_mae"],
            "mae_improvement_pct": 100 * (baseline["test_mae"] - cmrhm["test_mae"]) / baseline["test_mae"],
        })
    return rows


def write_comparison(rows: list[dict]) -> None:
    if not rows:
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    # Fail before any test access if the complete frozen checkpoint matrix is absent.
    for dataset, pred_len in TASKS:
        for label in LABELS:
            record = load_validation(dataset, pred_len, label)
            checkpoint = ROOT / "checkpoints" / setting_from_command(test_command(record)) / "checkpoint.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

    for dataset, pred_len in TASKS:
        for label in LABELS:
            print(f"\n=== ONE-SHOT TEST {dataset}-{pred_len} {label} ===", flush=True)
            if run_one(dataset, pred_len, label):
                write_comparison(collect())
                return 1
            write_comparison(collect())
    rows = collect()
    for row in rows:
        print(
            f"{row['dataset']}-{row['pred_len']}: "
            f"MSE {row['mse_improvement_pct']:+.3f}%, "
            f"MAE {row['mae_improvement_pct']:+.3f}%"
        )
    return 0 if len(rows) == len(TASKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
