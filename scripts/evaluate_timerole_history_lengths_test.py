#!/usr/bin/env python3
"""Evaluate preregistered TimeRole sensitivity checkpoints on the test split.

This secondary complete-grid analysis follows the validation-only sensitivity
study. It reuses frozen validation-best checkpoints, never retrains, and writes
test results to a separate directory so the validation records remain immutable.
The default remains the 72-job total-history grid; ``--factors recent pool``
selects the remaining 48 boundary/compression jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_graphmamba_backbone_ablation as base
import run_timerole_p0_sensitivity as sensitivity


SOURCE = ROOT / "logs" / "timerole_p0" / "sensitivity"
HISTORY_OUTPUT = ROOT / "logs" / "timerole_p0" / "history_length_test"
CONFIG_OUTPUT = ROOT / "logs" / "timerole_p0" / "boundary_pool_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--factors", nargs="+", choices=("history", "recent", "pool"),
        default=("history",),
    )
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds < 1 or args.max_jobs < 0:
        parser.error("timeout must be positive; max-jobs cannot be negative")
    args.source_dir = args.source_dir.resolve()
    if args.output_dir is None:
        args.output_dir = (
            HISTORY_OUTPUT if set(args.factors) == {"history"} else CONFIG_OUTPUT
        )
    args.output_dir = args.output_dir.resolve()
    return args


def tasks(args: argparse.Namespace) -> list[sensitivity.Task]:
    selected = [task for task in sensitivity.matrix() if task.factor in args.factors]
    assert selected and len({task.name for task in selected}) == len(selected)
    return selected


def source_record(task: sensitivity.Task, args: argparse.Namespace) -> Path:
    return args.source_dir / "records" / f"{task.name}.json"


def output_record(task: sensitivity.Task, args: argparse.Namespace) -> Path:
    return args.output_dir / "records" / f"{task.name}.json"


def evaluation_command(task: sensitivity.Task, args: argparse.Namespace) -> list[str]:
    source_args = argparse.Namespace(
        output_dir=args.source_dir,
        epochs=100,
        patience=6,
        gpu=args.gpu,
    )
    command = sensitivity.command(task, source_args)
    base.replace(command, "--is_training", 0)
    base.replace(command, "--evaluation_split", "test")
    return command


def preflight(task: sensitivity.Task, args: argparse.Namespace) -> tuple[dict, Path]:
    validation_path = source_record(task, args)
    if not base.completed(validation_path):
        raise RuntimeError(f"missing completed validation record: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("split") != "val" or validation.get("test_accessed") is not False:
        raise RuntimeError(f"invalid source split metadata: {validation_path}")
    command = evaluation_command(task, args)
    setting = validation.get("setting")
    if not setting:
        # The exact setting is recoverable from the checkpoint path saved by run.py.
        candidates = list((args.source_dir / "checkpoints").glob(f"*_{task.name}_0/checkpoint.pth"))
    else:
        candidates = [args.source_dir / "checkpoints" / str(setting) / "checkpoint.pth"]
    if len(candidates) != 1 or not candidates[0].is_file():
        raise RuntimeError(f"expected exactly one checkpoint for {task.name}, found {candidates}")
    return validation, candidates[0]


def run_one(task: sensitivity.Task, args: argparse.Namespace) -> int:
    destination = output_record(task, args)
    if base.completed(destination):
        print(f"SKIP completed: {task.name}", flush=True)
        return 0
    if destination.exists():
        print(f"REFUSE retry of failed/incomplete test record: {destination}", flush=True)
        return 1
    validation, checkpoint = preflight(task, args)
    command = evaluation_command(task, args)
    print("TEST", shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    log_path = args.output_dir / "logs" / f"{task.name}.test.log"
    code, result, elapsed = base.execute(
        command, log_path, base.EVALUATION_PATTERN, args.gpu,
        args.timeout_seconds,
    )
    success = (
        code == 0 and isinstance(result, dict)
        and result.get("split") == "test"
        and result.get("test_accessed") is True
    )
    payload = {
        "status": "completed" if success else "failed",
        "stage": "secondary_complete_sensitivity_grid_test",
        "analysis_scope": "post_validation_complete_grid_no_configuration_selection",
        "model": "TimeRole",
        **task.__dict__,
        "split": "test" if success else None,
        "test_accessed": True if success else None,
        "return_code": code,
        "eval_duration_seconds": round(elapsed, 3),
        "validation_mse": validation.get("mse"),
        "validation_mae": validation.get("mae"),
        "source_validation_record": str(source_record(task, args)),
        "source_checkpoint": str(checkpoint),
        "command": command,
        "log_path": str(log_path),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if result:
        payload.update(result)
    base.atomic_write(destination, payload)
    return 0 if success else 1


def summarize(selected: list[sensitivity.Task], args: argparse.Namespace, status: str) -> None:
    rows = []
    failed = 0
    for task in selected:
        path = output_record(task, args)
        if base.completed(path):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "factor": task.factor,
                "dataset": task.dataset,
                "horizon": task.horizon,
                "seq_len": task.seq_len,
                "recent_len": task.recent_len,
                "pool": task.pool,
                "seed": task.seed,
                "test_mse": payload["mse"],
                "test_mae": payload["mae"],
                "validation_mse": payload.get("validation_mse"),
                "validation_mae": payload.get("validation_mae"),
                "parameter_count": payload.get("parameter_count"),
                "record": str(path),
            })
        elif path.exists():
            failed += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "factor", "dataset", "horizon", "seq_len", "recent_len", "pool",
        "seed", "test_mse", "test_mae", "validation_mse", "validation_mae",
        "parameter_count", "record",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    base.atomic_write(args.output_dir / "status.json", {
        "status": status,
        "analysis_scope": "post_validation_complete_grid_no_configuration_selection",
        "factors": list(args.factors),
        "expected": len(selected),
        "completed": len(rows),
        "failed": failed,
        "split": "test",
        "test_accessed": bool(rows),
        "updated_at": datetime.now().astimezone().isoformat(),
    })


def main() -> int:
    args = parse_args()
    selected = tasks(args)
    if args.max_jobs:
        selected = selected[:args.max_jobs]
    for task in selected:
        preflight(task, args)
    if args.dry_run:
        for task in selected:
            run_one(task, args)
        print(json.dumps({
            "jobs": len(selected), "factors": list(args.factors),
            "split": "test", "retrain": False,
        }))
        return 0
    for index, task in enumerate(selected, 1):
        print(f"=== [{index}/{len(selected)}] {task.name} ===", flush=True)
        if run_one(task, args):
            summarize(selected, args, "failed")
            return 1
        summarize(selected, args, "running")
    summarize(selected, args, "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
