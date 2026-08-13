#!/usr/bin/env python3
"""Evaluate frozen Recent336/CMRHM validation checkpoints on the test split once."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "logs" / "graphmamba_cmrhm_all_horizons" / "validation"
OUTPUT = ROOT / "logs" / "graphmamba_cmrhm_final_test"
DATASETS = ("ETTm1", "ETTm2")
PRED_LENS = (96, 192, 336, 720)
LABELS = ("recent336", "cmrhm")
SEED = 2021
TEST_PATTERN = re.compile(r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:")


def name(dataset: str, pred_len: int, label: str) -> str:
    return f"{dataset.lower()}_{pred_len}_{label}_s{SEED}"


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and "test_mse" in payload


def test_command(validation_record: dict[str, object]) -> list[str]:
    command = list(validation_record["command"])
    index = command.index("--is_training")
    command[index + 1] = "0"
    test_index = command.index("--test_after_train")
    command[test_index + 1] = "0"
    return command


def run_one(dataset: str, pred_len: int, label: str) -> int:
    candidate = name(dataset, pred_len, label)
    source = VALIDATION / f"{candidate}.json"
    destination = OUTPUT / "records" / f"{candidate}.json"
    log_path = OUTPUT / "logs" / f"{candidate}.log"
    if completed(destination):
        return 0
    validation_record = json.loads(source.read_text(encoding="utf-8"))
    if validation_record.get("status") != "completed":
        raise RuntimeError(f"Validation record is not completed: {source}")

    command = test_command(validation_record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    metrics = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
            match = TEST_PATTERN.match(line.strip())
            if match:
                metrics = {"test_mse": float(match.group(1)), "test_mae": float(match.group(2))}
        return_code = process.wait()

    payload = {
        "status": "completed" if return_code == 0 and metrics else "failed",
        "dataset": dataset,
        "pred_len": pred_len,
        "model": validation_record["model"],
        "label": label,
        "candidate": candidate,
        "seed": SEED,
        "checkpoint_selected_by": "validation_best_mse",
        "validation_best_epoch": validation_record["best_epoch"],
        "validation_best_mse": validation_record["best_mse"],
        "validation_best_mae": validation_record["best_mae"],
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "source_validation_record": str(source),
        "log_path": str(log_path),
        "command": command,
    }
    if metrics:
        payload.update(metrics)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return 0 if payload["status"] == "completed" else 1


def collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            pair = {}
            for label in LABELS:
                path = OUTPUT / "records" / f"{name(dataset, pred_len, label)}.json"
                if completed(path):
                    pair[label] = json.loads(path.read_text(encoding="utf-8"))
            if len(pair) != 2:
                continue
            baseline, cmrhm = pair["recent336"], pair["cmrhm"]
            rows.append({
                "dataset": dataset,
                "pred_len": pred_len,
                "seed": SEED,
                "baseline_test_mse": baseline["test_mse"],
                "cmrhm_test_mse": cmrhm["test_mse"],
                "mse_improvement_pct": 100 * (baseline["test_mse"] - cmrhm["test_mse"]) / baseline["test_mse"],
                "baseline_test_mae": baseline["test_mae"],
                "cmrhm_test_mae": cmrhm["test_mae"],
                "mae_improvement_pct": 100 * (baseline["test_mae"] - cmrhm["test_mae"]) / baseline["test_mae"],
            })
    return rows


def write_comparison(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            for label in LABELS:
                print(f"\n=== FINAL TEST {dataset}-{pred_len} {label} ===", flush=True)
                if run_one(dataset, pred_len, label):
                    write_comparison(collect())
                    return 1
                write_comparison(collect())
    rows = collect()
    for row in rows:
        print(
            f"{row['dataset']}-{row['pred_len']}: MSE {row['mse_improvement_pct']:+.3f}%, "
            f"MAE {row['mae_improvement_pct']:+.3f}%"
        )
    return 0 if len(rows) == len(DATASETS) * len(PRED_LENS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
