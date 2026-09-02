#!/usr/bin/env python3
"""Build the unified non-periodic TimeRole main-result set without Traffic.

ETTh1/ETTh2 192/720 reuse validation-selected independent-shared checkpoints
and read the test split once. ETTh1/ETTh2 96/336 are the only training jobs.
ETTm1/ETTm2, Weather, and Solar reuse completed independent-shared records.
"""

from __future__ import annotations

import argparse
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
OUTPUT = ROOT / "logs" / "timerole_unified_main"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "solar")
HORIZONS = (96, 192, 336, 720)
TEST_PATTERN = re.compile(r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def destination(dataset: str, horizon: int) -> Path:
    return OUTPUT / "records" / f"timerole_{dataset.lower()}_p{horizon}_s2021.json"


def completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("scan_mode") == "independent_shared"
        and "test_mse" in payload
        and "test_mae" in payload
    )


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def replace_option(command: list[str], option: str, value: str) -> None:
    if option in command:
        command[command.index(option) + 1] = value
    else:
        command.extend((option, value))


def normalize_paper_model(command: list[str]) -> None:
    if "--model" not in command:
        return
    index = command.index("--model") + 1
    command[index] = {
        "GraphMambaCMRHM": "TimeRole",
        "CMRHM": "TimeRole",
        "GraphMambaRecent": "TimeRoleRecent",
        "GraphMamba": "TimeRoleFullHistory",
    }.get(command[index], command[index])


def reuse_source(dataset: str, horizon: int) -> Path | None:
    if dataset in {"ETTm1", "ETTm2"}:
        return (
            ROOT / "logs" / "timerole_final_test" / "records"
            / f"{dataset.lower()}_{horizon}_timerole_s2021.json"
        )
    if dataset in {"weather", "solar"}:
        return (
            ROOT / "logs" / "timerole_six_dataset_final" / "records"
            / f"sixds_{dataset}_p{horizon}_s2021.json"
        )
    return None


def reuse_existing(dataset: str, horizon: int) -> bool:
    source = reuse_source(dataset, horizon)
    if source is None or not source.is_file():
        return False
    payload = json.loads(source.read_text(encoding="utf-8"))
    periodic = bool(payload.get("periodic_backbone_active", False))
    command = payload.get("command", [])
    if isinstance(command, list) and "--dual_scale_scan_mode" in command:
        mode = command[command.index("--dual_scale_scan_mode") + 1]
        periodic = periodic or mode == "periodic_aligned"
    if periodic or payload.get("status") != "completed" or "test_mse" not in payload:
        return False
    record = {
        "status": "completed",
        "model": "TimeRole",
        "host_backbone": "graph_enhanced_ssm",
        "dataset": dataset,
        "pred_len": horizon,
        "seq_len": 336,
        "recent_len": 96,
        "seed": 2021,
        "scan_mode": "independent_shared",
        "checkpoint_selected_by": "validation_best_mse",
        "test_mse": payload["test_mse"],
        "test_mae": payload["test_mae"],
        "validation_best_mse": payload.get("validation_best_mse"),
        "validation_best_mae": payload.get("validation_best_mae"),
        "duration_seconds": payload.get("duration_seconds"),
        "result_source": "reused_completed_independent_shared_record",
        "source_record": str(source),
        "test_access": "reused_record_no_new_test_read",
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    atomic_write(destination(dataset, horizon), record)
    return True


def factorial_source(dataset: str, horizon: int) -> Path:
    return (
        ROOT / "logs" / "graphmamba_q2_factorial" / "validation"
        / f"q2f_{dataset.lower()}_p{horizon}_c_s2021.json"
    )


def test_existing_checkpoint(
    dataset: str, horizon: int, args: argparse.Namespace
) -> int:
    source = factorial_source(dataset, horizon)
    validation = json.loads(source.read_text(encoding="utf-8"))
    command = list(validation["command"])
    normalize_paper_model(command)
    replace_option(command, "--is_training", "0")
    replace_option(command, "--test_after_train", "0")
    replace_option(command, "--evaluation_split", "test")
    replace_option(command, "--gpu", str(args.gpu))
    log_path = OUTPUT / "logs" / f"timerole_{dataset.lower()}_p{horizon}_s2021.log"
    if args.dry_run:
        print("TEST_REUSE", " ".join(command))
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    metrics = None
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
                match = TEST_PATTERN.match(line.strip())
                if match:
                    metrics = {
                        "test_mse": float(match.group(1)),
                        "test_mae": float(match.group(2)),
                    }
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = 124
    payload = {
        "status": "completed" if return_code == 0 and metrics else "failed",
        "model": "TimeRole",
        "host_backbone": "graph_enhanced_ssm",
        "dataset": dataset,
        "pred_len": horizon,
        "seq_len": 336,
        "recent_len": 96,
        "seed": 2021,
        "scan_mode": "independent_shared",
        "checkpoint_selected_by": "validation_best_mse",
        "validation_best_mse": validation["best_mse"],
        "validation_best_mae": validation["best_mae"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "result_source": "reused_independent_checkpoint_one_shot_test",
        "source_validation_record": str(source),
        "test_access": "one_shot_after_frozen_validation_selection",
        "return_code": return_code,
        "command": command,
        "log_path": str(log_path),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if metrics:
        payload.update(metrics)
    atomic_write(destination(dataset, horizon), payload)
    return 0 if payload["status"] == "completed" else 1


def train_missing(dataset: str, horizon: int, args: argparse.Namespace) -> int:
    candidate = f"timerole_unified_{dataset.lower()}_p{horizon}_s2021"
    command = [
        sys.executable, "-u", str(ROOT / "scripts" / "run_timerole_experiment.py"),
        "--model", "TimeRole", "--candidate", candidate,
        "--dataset", dataset, "--pred-len", str(horizon), "--seq-len", "336",
        "--dual-scale-scan-mode", "independent_shared", "--seed", "2021",
        "--gpu", str(args.gpu), "--epochs", "100", "--patience", "6",
        "--final-test", "--output-dir", str(OUTPUT / "training"),
    ]
    if args.dry_run:
        print("TRAIN_MISSING", " ".join(command))
        return 0
    return_code = subprocess.run(
        command, cwd=ROOT, timeout=args.timeout_seconds
    ).returncode
    source = OUTPUT / "training" / "final" / f"{candidate}.json"
    if return_code or not source.is_file():
        return return_code or 1
    trained = json.loads(source.read_text(encoding="utf-8"))
    payload = {
        **trained,
        "model": "TimeRole",
        "implementation_model": "TimeRole",
        "host_backbone": "graph_enhanced_ssm",
        "seq_len": 336,
        "recent_len": 96,
        "scan_mode": "independent_shared",
        "checkpoint_selected_by": "validation_best_mse",
        "result_source": "new_unified_independent_shared_run",
        "test_access": "one_shot_after_validation_selection",
        "source_record": str(source),
    }
    atomic_write(destination(dataset, horizon), payload)
    return 0 if completed(destination(dataset, horizon)) else 1


def write_manifest_and_table() -> tuple[int, int]:
    rows = []
    missing = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = destination(dataset, horizon)
            if not completed(path):
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "dataset": dataset, "horizon": horizon, "seed": 2021,
                "seq_len": 336, "recent_len": 96,
                "scan_mode": payload["scan_mode"],
                "test_mse": payload["test_mse"], "test_mae": payload["test_mae"],
                "result_source": payload.get("result_source"),
                "source_record": payload.get("source_record")
                or payload.get("source_validation_record"),
            })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "protocol": "TimeRole unified independent_shared main results",
        "datasets": DATASETS,
        "horizons": HORIZONS,
        "expected": len(DATASETS) * len(HORIZONS),
        "completed": len(rows),
        "missing": missing,
        "traffic_excluded": True,
        "periodic_primary_records": 0,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    atomic_write(OUTPUT / "manifest.json", manifest)
    return len(rows), len(missing)


def main() -> int:
    args = parse_args()
    failures = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            target = destination(dataset, horizon)
            if completed(target):
                continue
            if reuse_existing(dataset, horizon):
                print(f"REUSED {dataset}-{horizon}", flush=True)
                continue
            if dataset in {"ETTh1", "ETTh2"} and horizon in {192, 720}:
                code = test_existing_checkpoint(dataset, horizon, args)
            elif dataset in {"ETTh1", "ETTh2"} and horizon in {96, 336}:
                code = train_missing(dataset, horizon, args)
            else:
                code = 1
            if code:
                failures.append(f"{dataset}-{horizon}")
                break
        if failures:
            break
    completed_count, missing_count = write_manifest_and_table()
    print(json.dumps({
        "completed": completed_count, "missing": missing_count,
        "failures": failures, "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
