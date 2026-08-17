#!/usr/bin/env python3
"""Validation-only efficiency benchmark for Recent96, Raw336, and CMRHM."""

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
OUTPUT = ROOT / "logs" / "cmrhm_efficiency"
DATASETS = ("ETTm1", "ETTm2")
HORIZONS = (96, 720)
VARIANTS = ("Recent96", "Raw336", "CMRHM")
PATTERN = re.compile(r"^EVALUATION_RESULT\s+(\{.*\})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source(dataset: str, horizon: int, variant: str) -> Path:
    if variant == "Raw336":
        return (
            ROOT / "logs" / "graphmamba_cmrhm_strict_evidence" / "validation"
            / f"b_{dataset.lower()}_p{horizon}_raw_s2021.json"
        )
    label = "recent336" if variant == "Recent96" else "cmrhm"
    return (
        ROOT / "logs" / "graphmamba_cmrhm_all_horizons" / "validation"
        / f"{dataset.lower()}_{horizon}_{label}_s2021.json"
    )


def destination(dataset: str, horizon: int, variant: str) -> Path:
    return OUTPUT / "records" / f"{dataset.lower()}_p{horizon}_{variant.lower()}.json"


def replace(command: list[str], option: str, value: str) -> None:
    if option in command:
        command[command.index(option) + 1] = value
    else:
        command.extend((option, value))


def complete(path: Path) -> bool:
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_one(dataset: str, horizon: int, variant: str, args: argparse.Namespace) -> int:
    target = destination(dataset, horizon, variant)
    if complete(target):
        return 0
    origin = source(dataset, horizon, variant)
    record = json.loads(origin.read_text(encoding="utf-8"))
    command = list(record["command"])
    replace(command, "--is_training", "0")
    replace(command, "--test_after_train", "0")
    replace(command, "--evaluation_split", "val")
    replace(command, "--gpu", str(args.gpu))
    if variant == "CMRHM":
        replace(command, "--cmrhm_old_intervention", "intact")
    if args.dry_run:
        print(" ".join(command))
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log_path = OUTPUT / "logs" / target.with_suffix(".log").name
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
                match = PATTERN.match(line.strip())
                if match:
                    result = json.loads(match.group(1))
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = 124
    payload = {
        "status": "completed" if return_code == 0 and result else "failed",
        "dataset": dataset, "horizon": horizon, "variant": variant,
        "split": "val", "seed": 2021, "source_record": str(origin),
        "command": command, "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if result:
        payload.update(result)
    write(target, payload)
    return 0 if payload["status"] == "completed" else 1


def summarize() -> int:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for variant in VARIANTS:
                path = destination(dataset, horizon, variant)
                if not complete(path):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "dataset": dataset, "horizon": horizon, "variant": variant,
                    "validation_mse": payload["mse"],
                    "validation_mae": payload["mae"],
                    "parameter_count": payload["parameter_count"],
                    "milliseconds_per_batch": payload["milliseconds_per_batch"],
                    "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
                    "memory_correction_mae": payload.get("memory_correction_mae"),
                })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(OUTPUT / "status.json", {
        "expected": len(DATASETS) * len(HORIZONS) * len(VARIANTS),
        "completed": len(rows), "split": "validation", "test_accessed": False,
        "traffic_excluded": True,
        "updated_at": datetime.now().astimezone().isoformat(),
    })
    return len(rows)


def main() -> int:
    args = parse_args()
    failures = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for variant in VARIANTS:
                print(f"=== {dataset}-{horizon} {variant} ===", flush=True)
                if run_one(dataset, horizon, variant, args):
                    failures.append(f"{dataset}-{horizon}-{variant}")
                    break
            if failures:
                break
        if failures:
            break
    count = summarize()
    print(json.dumps({"completed": count, "failed": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
