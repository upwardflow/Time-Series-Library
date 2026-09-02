#!/usr/bin/env python3
"""Run the paper TimeRole configuration on its five benchmark datasets."""

from __future__ import annotations

import argparse
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
DATASETS = {
    "ETTh1": {
        "root": ROOT / "dataset/ETT-small",
        "path": "ETTh1.csv",
        "data": "ETTh1",
        "target": "OT",
        "channels": 7,
    },
    "ETTh2": {
        "root": ROOT / "dataset/ETT-small",
        "path": "ETTh2.csv",
        "data": "ETTh2",
        "target": "OT",
        "channels": 7,
    },
    "ETTm1": {
        "root": ROOT / "dataset/ETT-small",
        "path": "ETTm1.csv",
        "data": "ETTm1",
        "target": "OT",
        "channels": 7,
    },
    "ETTm2": {
        "root": ROOT / "dataset/ETT-small",
        "path": "ETTm2.csv",
        "data": "ETTm2",
        "target": "OT",
        "channels": 7,
    },
    "Weather": {
        "root": ROOT / "dataset/weather",
        "path": "weather.csv",
        "data": "custom",
        "target": "CO2 (ppm)",
        "channels": 21,
    },
}
HORIZONS = (96, 192, 336, 720)
DEFAULT_SEEDS = (2021, 2022, 2023)
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$", re.MULTILINE)
TEST_PATTERN = re.compile(
    r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:", re.MULTILINE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--test-after-train", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "logs/timerole_datasets"
    )
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.timeout_seconds) < 1:
        parser.error("epochs, patience, batch size, and timeout must be positive")
    if args.num_workers < 0 or args.max_jobs < 0:
        parser.error("num-workers and max-jobs cannot be negative")
    args.output_dir = args.output_dir.resolve()
    return args


def now() -> str:
    return datetime.now().astimezone().isoformat()


def candidate(dataset: str, horizon: int, seed: int) -> str:
    return f"timerole_{dataset.lower()}_sl336_pl{horizon}_s{seed}"


def record_path(args: argparse.Namespace, dataset: str, horizon: int, seed: int) -> Path:
    return args.output_dir / "records" / f"{candidate(dataset, horizon, seed)}.json"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def completed(path: Path, require_test: bool) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "completed" or payload.get("validation_best_mse") is None:
        return False
    return not require_test or payload.get("test_mse") is not None


def build_command(
    args: argparse.Namespace, dataset: str, horizon: int, seed: int
) -> list[str]:
    spec = DATASETS[dataset]
    name = candidate(dataset, horizon, seed)
    channels = str(spec["channels"])
    command = [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", name,
        "--model", "TimeRole",
        "--seed", str(seed),
        "--data", str(spec["data"]),
        "--root_path", str(spec["root"]),
        "--data_path", str(spec["path"]),
        "--features", "M",
        "--target", str(spec["target"]),
        "--seq_len", "336",
        "--label_len", "48",
        "--pred_len", str(horizon),
        "--enc_in", channels,
        "--dec_in", channels,
        "--c_out", channels,
        "--timerole_recent_len", "96",
        "--timerole_memory_pool", "16",
        "--timerole_hidden_dim", "32",
        "--timerole_old_intervention", "intact",
        "--patch_len", "4",
        "--stride", "2",
        "--d_model", "64",
        "--d_ff", "128",
        "--d_state", "32",
        "--d_conv", "2",
        "--e_layers", "1",
        "--expand", "2",
        "--mamba_version", "1",
        "--mamba_bidirectional", "1",
        "--use_graph", "1",
        "--use_time_mamba", "1",
        "--use_patch", "1",
        "--use_decomp", "1",
        "--moving_avg", "25",
        "--dual_scale_scan_mode", "independent_shared",
        "--dual_scale_selection", "dual",
        "--timerole_branch_fusion", "fixed_sum",
        "--graph_alpha", "0.5",
        "--graph_top_k", "2",
        "--graph_sample_size", "2000",
        "--graph_sample_method", "uniform",
        "--static_graph_mode", "weighted",
        "--static_graph_only", "0",
        "--graph_cache", "0",
        "--dropout", "0.1",
        "--batch_size", str(args.batch_size),
        "--learning_rate", "0.0005",
        "--lradj", "type1",
        "--train_epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--num_workers", str(args.num_workers),
        "--gpu", str(args.gpu),
        "--checkpoints", str(args.output_dir / "checkpoints"),
        "--des", name,
        "--itr", "1",
        "--test_after_train", "1" if args.test_after_train else "0",
    ]
    if args.cpu:
        command.append("--no_use_gpu")
    return command


def run_job(
    args: argparse.Namespace, dataset: str, horizon: int, seed: int
) -> bool:
    destination = record_path(args, dataset, horizon, seed)
    if args.resume and completed(destination, args.test_after_train):
        print(f"SKIP {dataset} H={horizon} seed={seed}", flush=True)
        return True

    command = build_command(args, dataset, horizon, seed)
    print("RUN " + shlex.join(command), flush=True)
    if args.dry_run:
        return True

    name = candidate(dataset, horizon, seed)
    log_path = args.output_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if not args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
        )
        output = process.stdout
        return_code = process.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return_code = 124
    print(output, end="")
    log_path.write_text(output, encoding="utf-8")

    validations = VALIDATION_PATTERN.findall(output)
    validation = json.loads(validations[-1]) if validations else None
    tests = TEST_PATTERN.findall(output)
    test = tests[-1] if tests else None
    success = return_code == 0 and validation is not None
    if args.test_after_train:
        success = success and test is not None
    payload: dict[str, object] = {
        "status": "completed" if success else "failed",
        "model": "TimeRole",
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "seq_len": 336,
        "recent_len": 96,
        "memory_pool": 16,
        "checkpoint_selected_by": "validation_mse",
        "test_after_train": args.test_after_train,
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "command": command,
        "log_path": str(log_path),
        "recorded_at": now(),
    }
    if validation:
        payload["validation_best_epoch"] = validation.get("best_epoch")
        payload["validation_best_mse"] = validation.get("best_mse")
        payload["validation_best_mae"] = validation.get("best_mae")
    if test:
        payload["test_mse"] = float(test[0])
        payload["test_mae"] = float(test[1])
    atomic_json(destination, payload)
    return success


def main() -> int:
    args = parse_args()
    missing = [
        str(DATASETS[name]["root"] / str(DATASETS[name]["path"]))
        for name in args.datasets
        if not (DATASETS[name]["root"] / str(DATASETS[name]["path"])).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing dataset files: " + ", ".join(missing))
    jobs = [
        (dataset, horizon, seed)
        for dataset in args.datasets
        for horizon in args.horizons
        for seed in args.seeds
    ]
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    if args.dry_run:
        print(f"DRY RUN: {len(jobs)} jobs", flush=True)

    failed: list[str] = []
    for index, (dataset, horizon, seed) in enumerate(jobs, start=1):
        name = candidate(dataset, horizon, seed)
        print(f"[{index}/{len(jobs)}] {name}", flush=True)
        if not run_job(args, dataset, horizon, seed):
            failed.append(name)
        if not args.dry_run:
            status = "failed" if failed else (
                "completed" if index == len(jobs) else "running"
            )
            atomic_json(
                args.output_dir / "status.json",
                {
                    "status": status,
                    "total": len(jobs),
                    "processed": index,
                    "failed": failed,
                    "updated_at": now(),
                },
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
