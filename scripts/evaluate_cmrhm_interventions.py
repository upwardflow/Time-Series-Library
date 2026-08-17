#!/usr/bin/env python3
"""Validation-only CMRHM old-history interventions and efficiency audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "cmrhm_interventions"
DATASETS = ("ETTm1", "ETTm2")
HORIZONS = (96, 720)
INTERVENTIONS = (
    "intact", "batch_shuffle", "temporal_shuffle", "reverse",
    "recent_mean", "noise",
)
RESULT_PATTERN = re.compile(r"^EVALUATION_RESULT\s+(\{.*\})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument(
        "--interventions", nargs="+", choices=INTERVENTIONS,
        default=list(INTERVENTIONS),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_record(dataset: str, horizon: int) -> Path:
    return (
        ROOT / "logs" / "graphmamba_cmrhm_all_horizons" / "validation"
        / f"{dataset.lower()}_{horizon}_cmrhm_s2021.json"
    )


def record_path(dataset: str, horizon: int, intervention: str) -> Path:
    return (
        OUTPUT / "records"
        / f"{dataset.lower()}_p{horizon}_{intervention}_s2021.json"
    )


def replace_option(command: list[str], option: str, value: str) -> None:
    if option in command:
        command[command.index(option) + 1] = value
    else:
        command.extend((option, value))


def completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and "mse" in payload


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_one(
    dataset: str, horizon: int, intervention: str, args: argparse.Namespace
) -> int:
    destination = record_path(dataset, horizon, intervention)
    if completed(destination):
        return 0
    source = source_record(dataset, horizon)
    validation = json.loads(source.read_text(encoding="utf-8"))
    command = list(validation["command"])
    replace_option(command, "--is_training", "0")
    replace_option(command, "--test_after_train", "0")
    replace_option(command, "--evaluation_split", "val")
    replace_option(command, "--cmrhm_old_intervention", intervention)
    replace_option(command, "--cmrhm_noise_std", "1.0")
    replace_option(command, "--gpu", str(args.gpu))
    if args.dry_run:
        print(" ".join(command))
        return 0
    log_path = OUTPUT / "logs" / destination.with_suffix(".log").name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                handle.write(line)
                handle.flush()
                match = RESULT_PATTERN.match(line.strip())
                if match:
                    result = json.loads(match.group(1))
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = 124
    payload = {
        "status": "completed" if return_code == 0 and result else "failed",
        "dataset": dataset, "pred_len": horizon, "seed": 2021,
        "split": "val", "intervention": intervention,
        "scan_mode": "independent_shared", "source_record": str(source),
        "command": command, "log_path": str(log_path),
        "duration_seconds": round(time.monotonic() - started, 3),
        "return_code": return_code,
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if result:
        payload.update(result)
    write(destination, payload)
    return 0 if payload["status"] == "completed" else 1


def summarize() -> int:
    rows = []
    gates = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            intact_path = record_path(dataset, horizon, "intact")
            intact = json.loads(intact_path.read_text(encoding="utf-8")) if completed(intact_path) else None
            for intervention in INTERVENTIONS:
                path = record_path(dataset, horizon, intervention)
                if not completed(path):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "dataset": dataset, "horizon": horizon,
                    "intervention": intervention,
                    "mse": payload["mse"], "mae": payload["mae"],
                    "mse_change_pct": (
                        100 * (payload["mse"] - intact["mse"]) / intact["mse"]
                        if intact else None
                    ),
                    "mae_change_pct": (
                        100 * (payload["mae"] - intact["mae"]) / intact["mae"]
                        if intact else None
                    ),
                    "memory_correction_mae": payload.get("memory_correction_mae"),
                    "memory_correction_rms": payload.get("memory_correction_rms"),
                    "milliseconds_per_batch": payload.get("milliseconds_per_batch"),
                    "peak_cuda_memory_bytes": payload.get("peak_cuda_memory_bytes"),
                    "parameter_count": payload.get("parameter_count"),
                })
                if intervention == "intact":
                    for variable, value in enumerate(payload.get("gate_values", [])):
                        gates.append({
                            "dataset": dataset, "horizon": horizon,
                            "variable_index": variable, "gate": value,
                        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, data in (("summary.csv", rows), ("gate_values.csv", gates)):
        if data:
            with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    status = {
        "expected": len(DATASETS) * len(HORIZONS) * len(INTERVENTIONS),
        "completed": len(rows), "split": "validation", "test_accessed": False,
        "traffic_excluded": True,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    write(OUTPUT / "status.json", status)
    return len(rows)


def main() -> int:
    args = parse_args()
    failed = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            for intervention in args.interventions:
                print(f"=== {dataset}-{horizon} {intervention} ===", flush=True)
                if run_one(dataset, horizon, intervention, args):
                    failed.append(f"{dataset}-{horizon}-{intervention}")
                    break
            if failed:
                break
        if failed:
            break
    count = summarize()
    print(json.dumps({"completed": count, "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
