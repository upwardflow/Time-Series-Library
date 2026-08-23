#!/usr/bin/env python3
"""Run the preregistered 120-job TimeRole boundary sensitivity matrix.

All jobs are validation-only.  The matrix changes exactly one of total history,
recent history, or memory-pool width at a time and stops at the first failure.
Completed atomic records are resumable but failed jobs are never retried here.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_graphmamba_backbone_ablation as base


OUTPUT = ROOT / "logs" / "timerole_p0" / "sensitivity"
SEEDS = (2021, 2022, 2023)
HORIZONS = (96, 720)


@dataclass(frozen=True)
class Dataset:
    root: Path
    file: str
    data_type: str
    channels: int
    target: str


DATASETS = {
    "ETTm1": Dataset(ROOT / "dataset" / "ETT-small", "ETTm1.csv", "ETTm1", 7, "OT"),
    "ETTm2": Dataset(ROOT / "dataset" / "ETT-small", "ETTm2.csv", "ETTm2", 7, "OT"),
    "weather": Dataset(ROOT / "dataset" / "weather", "weather.csv", "custom", 21, "CO2 (ppm)"),
}


@dataclass(frozen=True)
class Task:
    factor: str
    dataset: str
    horizon: int
    seed: int
    seq_len: int
    recent_len: int
    pool: int

    @property
    def name(self) -> str:
        return (
            f"sensitivity_{self.factor}_{self.dataset.lower()}_h{self.horizon}"
            f"_l{self.seq_len}_r{self.recent_len}_p{self.pool}_s{self.seed}"
        )


def matrix() -> list[Task]:
    tasks: list[Task] = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for seq_len in (192, 336, 720, 960):
                    tasks.append(Task("history", dataset, horizon, seed, seq_len, 96, 16))
        for dataset in ("ETTm1", "ETTm2"):
            for horizon in HORIZONS:
                for recent_len in (48, 192):
                    tasks.append(Task("recent", dataset, horizon, seed, 336, recent_len, 16))
                for pool in (8, 24):
                    tasks.append(Task("pool", dataset, horizon, seed, 336, 96, pool))
    assert len(tasks) == 120 and len({task.name for task in tasks}) == 120
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--factors", nargs="+", choices=("history", "recent", "pool"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.timeout_seconds) < 1 or args.max_jobs < 0:
        parser.error("epochs, patience, timeout must be positive; max-jobs cannot be negative")
    args.output_dir = args.output_dir.resolve()
    return args


def command(task: Task, args: argparse.Namespace) -> list[str]:
    data = DATASETS[task.dataset]
    return [
        sys.executable, "-u", str(ROOT / "run.py"),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(data.root), "--data_path", data.file,
        "--model_id", task.name, "--model", "TimeRole", "--seed", str(task.seed),
        "--data", data.data_type, "--features", "M", "--target", data.target,
        "--seq_len", str(task.seq_len), "--label_len", "48", "--pred_len", str(task.horizon),
        "--enc_in", str(data.channels), "--dec_in", str(data.channels), "--c_out", str(data.channels),
        "--timerole_recent_len", str(task.recent_len),
        "--timerole_memory_pool", str(task.pool),
        "--timerole_old_intervention", "intact",
        "--patch_len", "4", "--stride", "2", "--d_model", "64", "--d_ff", "128",
        "--d_state", "32", "--d_conv", "2", "--e_layers", "1", "--expand", "2",
        "--mamba_version", "1", "--mamba_bidirectional", "1",
        "--use_graph", "1", "--use_time_mamba", "1", "--use_patch", "1",
        "--use_decomp", "1", "--moving_avg", "25",
        "--dual_scale_scan_mode", "independent_shared",
        "--periodic_period", "24", "--periodic_local_patch", "4",
        "--periodic_local_stride", "2", "--periodic_period_stride", "12",
        "--periodic_use_adapter", "1", "--graph_alpha", "0.5", "--graph_top_k", "2",
        "--graph_sample_size", "2000", "--graph_sample_method", "uniform",
        "--static_graph_mode", "weighted", "--graph_cache", "1",
        "--gc_graph_dim", "16", "--gc_temperature", "1.0", "--gc_residual_init", "0.5",
        "--gc_dynamic_graph", "1", "--gc_symmetric_graph", "1",
        "--gc_input_modulation", "1", "--gc_direction_fusion", "1",
        "--gc_parallel_residual", "1", "--dropout", "0.1", "--batch_size", "32",
        "--learning_rate", "0.0005", "--lradj", "type1",
        "--train_epochs", str(args.epochs), "--patience", str(args.patience),
        "--num_workers", "0", "--gpu", str(args.gpu),
        "--checkpoints", str(args.output_dir / "checkpoints"),
        "--des", task.name, "--itr", "1", "--test_after_train", "0",
    ]


def record_path(task: Task, args: argparse.Namespace) -> Path:
    return args.output_dir / "records" / f"{task.name}.json"


def run_one(task: Task, args: argparse.Namespace) -> int:
    destination = record_path(task, args)
    if base.completed(destination):
        print(f"SKIP completed: {task.name}", flush=True)
        return 0
    train_command = command(task, args)
    print("TRAIN", shlex.join(train_command), flush=True)
    if args.dry_run:
        return 0
    train_log = args.output_dir / "logs" / f"{task.name}.train.log"
    code, validation, train_seconds = base.execute(
        train_command, train_log, base.VALIDATION_PATTERN, args.gpu, args.timeout_seconds
    )
    if code != 0 or not validation:
        base.atomic_write(destination, {
            "status": "failed", "stage": "training", "return_code": code,
            "task": task.__dict__, "split": "validation", "test_accessed": False,
            "duration_seconds": round(train_seconds, 3), "command": train_command,
            "log_path": str(train_log), "recorded_at": datetime.now().astimezone().isoformat(),
        })
        print(f"FAILED {task.name}; no automatic retry", flush=True)
        return 1
    eval_command = list(train_command)
    base.replace(eval_command, "--is_training", 0)
    base.replace(eval_command, "--evaluation_split", "val")
    eval_log = args.output_dir / "logs" / f"{task.name}.val.log"
    code, result, eval_seconds = base.execute(
        eval_command, eval_log, base.EVALUATION_PATTERN, args.gpu, args.timeout_seconds
    )
    success = (
        code == 0 and isinstance(result, dict)
        and result.get("split") == "val" and result.get("test_accessed") is False
    )
    payload = {
        "status": "completed" if success else "failed",
        "stage": "checkpoint_validation", "model": "TimeRole", **task.__dict__,
        "split": "validation", "test_accessed": False if success else None,
        "return_code": code, "train_duration_seconds": round(train_seconds, 3),
        "eval_duration_seconds": round(eval_seconds, 3), "command": eval_command,
        "train_log": str(train_log), "log_path": str(eval_log),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if result:
        payload.update(result)
    base.atomic_write(destination, payload)
    if not success:
        print(f"FAILED {task.name}; no automatic retry", flush=True)
        return 1
    return 0


def selected(args: argparse.Namespace) -> list[Task]:
    tasks = [
        task for task in matrix()
        if (not args.factors or task.factor in args.factors)
        and (not args.datasets or task.dataset in args.datasets)
        and (not args.horizons or task.horizon in args.horizons)
        and (not args.seeds or task.seed in args.seeds)
    ]
    return tasks[: args.max_jobs] if args.max_jobs else tasks


def summarize(tasks: list[Task], args: argparse.Namespace, status: str) -> None:
    rows = []
    for task in tasks:
        path = record_path(task, args)
        if not base.completed(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            **task.__dict__, "mse": payload["mse"], "mae": payload["mae"],
            "parameter_count": payload.get("parameter_count"), "record": str(path),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "factor", "dataset", "horizon", "seed", "seq_len", "recent_len", "pool",
        "mse", "mae", "parameter_count", "record",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    base.atomic_write(args.output_dir / "status.json", {
        "status": status, "expected": len(tasks), "completed": len(rows),
        "failed": sum(1 for task in tasks if record_path(task, args).is_file() and not base.completed(record_path(task, args))),
        "split": "validation", "test_accessed": False,
        "updated_at": datetime.now().astimezone().isoformat(),
    })


def main() -> int:
    args = parse_args()
    tasks = selected(args)
    if args.dry_run:
        for task in tasks:
            run_one(task, args)
        print(json.dumps({"jobs": len(tasks), "test_accessed": False}))
        return 0
    for index, task in enumerate(tasks, 1):
        print(f"=== [{index}/{len(tasks)}] {task.name} ===", flush=True)
        if run_one(task, args):
            summarize(tasks, args, "failed")
            return 1
        summarize(tasks, args, "running")
    summarize(tasks, args, "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
