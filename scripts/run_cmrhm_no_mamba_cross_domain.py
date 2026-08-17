#!/usr/bin/env python3
"""Run validation-only cross-domain CMRHM ablations without Mamba.

The matrix is intentionally restricted to ETTh1, ETTh2, and Weather at
prediction lengths 96 and 720.  It never evaluates the test split, stops on
the first failure, and resumes only records that completed successfully.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import select
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_graphmamba_backbone_ablation import (
    EVALUATION_PATTERN,
    ROOT,
    RUN_PY,
    VALIDATION_PATTERN,
    atomic_write,
    completed,
    replace,
)


DATASET_SPECS = {
    "ETTh1": {
        "root_path": ROOT / "dataset" / "ETT-small",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "target": "OT",
        "channels": 7,
    },
    "ETTh2": {
        "root_path": ROOT / "dataset" / "ETT-small",
        "data_path": "ETTh2.csv",
        "data": "ETTh2",
        "target": "OT",
        "channels": 7,
    },
    "Weather": {
        "root_path": ROOT / "dataset" / "weather",
        "data_path": "weather.csv",
        "data": "custom",
        "target": "CO2 (ppm)",
        "channels": 21,
    },
}
HORIZONS = (96, 720)


def execute(
    command: list[str], log_path: Path, pattern: re.Pattern[str],
    gpu: int, timeout_seconds: int,
) -> tuple[int, dict[str, object] | None, float]:
    """Execute one job with a true wall-clock deadline."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = None
    started = time.monotonic()
    deadline = started + timeout_seconds
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return 124, result, time.monotonic() - started
            readable, _, _ = select.select(
                [process.stdout], [], [], min(1.0, remaining)
            )
            if not readable:
                continue
            line = process.stdout.readline()
            if not line:
                continue
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            match = pattern.match(line.strip())
            if match:
                result = json.loads(match.group(1))
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            match = pattern.match(line.strip())
            if match:
                result = json.loads(match.group(1))
        return process.returncode, result, time.monotonic() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs" / "cmrhm_no_mamba_cross_domain",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, choices=HORIZONS,
        default=list(HORIZONS),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("timeout, epochs, and patience must be positive")
    args.output_dir = args.output_dir.resolve()
    return args


def candidate(dataset: str, horizon: int, seed: int) -> str:
    return f"crossdomain_no_mamba_{dataset.lower()}_p{horizon}_s{seed}"


def build_train_command(
    dataset: str, horizon: int, args: argparse.Namespace
) -> list[str]:
    spec = DATASET_SPECS[dataset]
    name = candidate(dataset, horizon, args.seed)
    channels = str(spec["channels"])
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(spec["root_path"]),
        "--data_path", str(spec["data_path"]),
        "--model_id", f"{dataset}_96_{horizon}_{name}",
        "--model", "GraphMambaCMRHM", "--seed", str(args.seed),
        "--data", str(spec["data"]), "--features", "M",
        "--target", str(spec["target"]),
        "--seq_len", "336", "--label_len", "48",
        "--pred_len", str(horizon),
        "--enc_in", channels, "--dec_in", channels, "--c_out", channels,
        "--patch_len", "4", "--stride", "2",
        "--d_model", "64", "--d_ff", "128", "--d_state", "32",
        "--d_conv", "2", "--e_layers", "1", "--expand", "2",
        "--mamba_version", "1", "--mamba_bidirectional", "1",
        "--use_graph", "1", "--use_time_mamba", "0",
        "--use_patch", "1", "--use_decomp", "1", "--moving_avg", "25",
        "--dual_scale_scan_mode", "independent_shared",
        "--periodic_period", "24", "--periodic_local_patch", "0",
        "--periodic_local_stride", "0", "--periodic_period_stride", "12",
        "--periodic_use_adapter", "1",
        "--graph_alpha", "0.5", "--graph_top_k", "2",
        "--graph_sample_size", "2000", "--graph_sample_method", "uniform",
        "--static_graph_mode", "weighted", "--graph_cache", "0",
        "--gc_graph_dim", "16", "--gc_temperature", "1.0",
        "--gc_residual_init", "0.5", "--gc_dynamic_graph", "1",
        "--gc_symmetric_graph", "1", "--gc_input_modulation", "1",
        "--gc_direction_fusion", "1", "--gc_parallel_residual", "1",
        "--af_hidden_dim", "32", "--af_rank", "16",
        "--af_mode", "variable_scale_residual",
        "--dropout", "0.1", "--batch_size", "32",
        "--learning_rate", "0.0005", "--lradj", "type1",
        "--train_epochs", str(args.epochs), "--patience", str(args.patience),
        "--num_workers", "0", "--gpu", str(args.gpu),
        "--checkpoints", str(args.output_dir / "checkpoints"),
        "--des", name, "--itr", "1", "--test_after_train", "0",
        "--cmrhm_old_intervention", "intact",
    ]


def record_path(args: argparse.Namespace, dataset: str, horizon: int) -> Path:
    name = candidate(dataset, horizon, args.seed)
    return args.output_dir / "records" / f"{name}.json"


def training_path(args: argparse.Namespace, dataset: str, horizon: int) -> Path:
    name = candidate(dataset, horizon, args.seed)
    return args.output_dir / "training" / f"{name}.json"


def run_one(dataset: str, horizon: int, args: argparse.Namespace) -> int:
    final_path = record_path(args, dataset, horizon)
    if completed(final_path):
        print(f"SKIP completed: {final_path}", flush=True)
        return 0

    name = candidate(dataset, horizon, args.seed)
    train_path = training_path(args, dataset, horizon)
    command = build_train_command(dataset, horizon, args)
    if args.dry_run:
        print("TRAIN", shlex.join(command))
        evaluation = list(command)
        replace(evaluation, "--is_training", 0)
        replace(evaluation, "--evaluation_split", "val")
        print("EVAL ", shlex.join(evaluation))
        return 0

    if not completed(train_path, "best_mse"):
        log_path = args.output_dir / "logs" / f"{name}.train.log"
        code, metrics, duration = execute(
            command, log_path, VALIDATION_PATTERN, args.gpu,
            args.timeout_seconds,
        )
        payload: dict[str, object] = {
            "status": "completed" if code == 0 and metrics else "failed",
            "stage": "training", "model": "GraphMambaCMRHM",
            "dataset": dataset, "horizon": horizon,
            "variant": "no_mamba", "variant_label": "CMRHM + w/o Mamba",
            "seed": args.seed, "split": "validation", "test_accessed": False,
            "return_code": code, "duration_seconds": round(duration, 3),
            "recorded_at": datetime.now().astimezone().isoformat(),
            "command": command, "log_path": str(log_path),
        }
        if metrics:
            payload.update(metrics)
        atomic_write(train_path, payload)
        if payload["status"] != "completed":
            print(f"FAILED training: {name}; no automatic retry", flush=True)
            return 1
    else:
        payload = json.loads(train_path.read_text(encoding="utf-8"))
        command = list(payload["command"])

    evaluation = list(command)
    replace(evaluation, "--is_training", 0)
    replace(evaluation, "--test_after_train", 0)
    replace(evaluation, "--evaluation_split", "val")
    eval_log = args.output_dir / "logs" / f"{name}.val.log"
    code, metrics, duration = execute(
        evaluation, eval_log, EVALUATION_PATTERN, args.gpu,
        args.timeout_seconds,
    )
    payload = {
        "status": "completed" if code == 0 and metrics else "failed",
        "stage": "checkpoint_validation", "model": "GraphMambaCMRHM",
        "dataset": dataset, "horizon": horizon,
        "variant": "no_mamba", "variant_label": "CMRHM + w/o Mamba",
        "seed": args.seed, "split": "val", "test_accessed": False,
        "training_record": str(train_path), "return_code": code,
        "duration_seconds": round(duration, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "command": evaluation, "log_path": str(eval_log),
    }
    if metrics:
        payload.update(metrics)
    atomic_write(final_path, payload)
    if payload["status"] != "completed":
        print(f"FAILED evaluation: {name}; no automatic retry", flush=True)
        return 1
    return 0


def summarize(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            path = record_path(args, dataset, horizon)
            if not completed(path):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "model": "GraphMambaCMRHM", "variant": "no_mamba",
                "variant_label": "CMRHM + w/o Mamba",
                "dataset": dataset, "horizon": horizon, "seed": args.seed,
                "validation_mse": payload["mse"],
                "validation_mae": payload["mae"],
                "parameter_count": payload["parameter_count"],
                "record": str(path),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    expected = len(args.datasets) * len(args.horizons)
    atomic_write(args.output_dir / "status.json", {
        "expected": expected, "completed": len(rows),
        "failed": expected - len(rows), "split": "validation",
        "test_accessed": False, "seed": args.seed,
        "model": "GraphMambaCMRHM", "variant": "no_mamba",
        "updated_at": datetime.now().astimezone().isoformat(),
    })
    return len(rows)


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            print(f"=== CMRHM + w/o Mamba | {dataset}-{horizon} ===", flush=True)
            if run_one(dataset, horizon, args):
                failures.append(f"{dataset}-{horizon}")
                break
        if failures:
            break
    count = summarize(args) if not args.dry_run else 0
    print(json.dumps({"completed": count, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
