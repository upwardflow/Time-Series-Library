#!/usr/bin/env python3
"""Run frozen Attraos/DiM comparison jobs with one-shot checkpoint testing.

Compatibility pilots must pass before this runner is used. Training selects the
checkpoint exclusively on validation loss; the selected checkpoint is then
evaluated on test once. The serial runner resumes completed records and stops
on the first failure without retrying it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_graphmamba_backbone_ablation as base
import run_timerole_attraos_pilot as attraos_pilot
import run_timerole_dim_pilot as dim_pilot


OUTPUT = ROOT / "logs" / "timerole_p0" / "closest" / "formal"
MODELS = ("Attraos", "DiM")
SEEDS = (2021, 2022, 2023)
HORIZONS = (96, 720)


@dataclass(frozen=True)
class Dataset:
    root: Path
    file: str
    data_type: str
    channels: int
    target: str
    freq: str
    batch_size: int


DATASETS = {
    "ETTm1": Dataset(ROOT / "dataset" / "ETT-small", "ETTm1.csv", "ETTm1", 7, "OT", "t", 32),
    "ETTm2": Dataset(ROOT / "dataset" / "ETT-small", "ETTm2.csv", "ETTm2", 7, "OT", "t", 32),
    "electricity": Dataset(ROOT / "dataset" / "electricity", "electricity.csv", "custom", 321, "0", "h", 8),
    "solar": Dataset(ROOT / "dataset" / "solar", "solar.csv", "custom", 137, "channel_99", "h", 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=["ETTm1", "ETTm2"])
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.timeout_seconds) < 1 or args.max_jobs < 0:
        parser.error("epochs, patience, timeout must be positive; max-jobs cannot be negative")
    args.output_dir = args.output_dir.resolve()
    return args


def set_flag(command: list[str], flag: str, value: object) -> None:
    base.replace(command, flag, value)


def name(model: str, dataset: str, horizon: int, seed: int) -> str:
    return f"closest_{model.lower()}_{dataset.lower()}_l336_h{horizon}_s{seed}"


def build_command(model: str, dataset: str, horizon: int, seed: int, args: argparse.Namespace) -> list[str]:
    data = DATASETS[dataset]
    task_name = name(model, dataset, horizon, seed)
    module = attraos_pilot if model == "Attraos" else dim_pilot
    namespace = SimpleNamespace(
        gpu=args.gpu, seed=seed, epochs=args.epochs, patience=args.patience,
        timeout_seconds=args.timeout_seconds,
        output_dir=args.output_dir / model.lower(), dry_run=args.dry_run,
    )
    command = module.build_command(namespace)
    for flag, value in (
        ("--model_id", task_name), ("--des", task_name),
        ("--data", data.data_type), ("--root_path", data.root),
        ("--data_path", data.file), ("--target", data.target), ("--freq", data.freq),
        ("--pred_len", horizon), ("--enc_in", data.channels),
        ("--dec_in", data.channels), ("--c_out", data.channels),
        ("--batch_size", data.batch_size),
        ("--checkpoints", args.output_dir / model.lower() / "checkpoints"),
        ("--test_after_train", 1), ("--evaluation_split", "test"),
    ):
        set_flag(command, flag, value)
    return command


def record_path(model: str, dataset: str, horizon: int, seed: int, args: argparse.Namespace) -> Path:
    return args.output_dir / "records" / f"{name(model, dataset, horizon, seed)}.json"


def completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("split") == "test"
        and payload.get("test_accessed") is True
        and all(payload.get(metric) is not None for metric in ("mse", "mae"))
    )


def run_one(model: str, dataset: str, horizon: int, seed: int, args: argparse.Namespace) -> int:
    destination = record_path(model, dataset, horizon, seed, args)
    if completed(destination):
        print(f"SKIP completed: {destination.name}", flush=True)
        return 0
    command = build_command(model, dataset, horizon, seed, args)
    print("COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    repository = attraos_pilot.ATTRAOS if model == "Attraos" else dim_pilot.DIM
    log_path = args.output_dir / "logs" / f"{name(model, dataset, horizon, seed)}.log"
    code, result, duration = base.execute(
        command, log_path, base.EVALUATION_PATTERN, args.gpu, args.timeout_seconds,
        cwd=repository,
    )
    success = (
        code == 0 and isinstance(result, dict)
        and result.get("split") == "test" and result.get("test_accessed") is True
    )
    payload = {
        "status": "completed" if success else "failed", "model": model,
        "dataset": dataset, "horizon": horizon, "seq_len": 336, "seed": seed,
        "checkpoint_selected_by": "validation_loss_early_stopping",
        "test_access": "one_shot_after_validation_selection" if success else None,
        "return_code": code, "duration_seconds": round(duration, 3),
        "command": command, "cwd": str(repository), "log_path": str(log_path),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if result:
        payload.update(result)
    base.atomic_write(destination, payload)
    if not success:
        print(f"FAILED {name(model, dataset, horizon, seed)}; no automatic retry", flush=True)
        return 1
    return 0


def jobs(args: argparse.Namespace) -> list[tuple[str, str, int, int]]:
    result = [
        (model, dataset, horizon, seed)
        for model in args.models for seed in args.seeds
        for dataset in args.datasets for horizon in args.horizons
    ]
    return result[: args.max_jobs] if args.max_jobs else result


def summarize(matrix: list[tuple[str, str, int, int]], args: argparse.Namespace, status: str) -> None:
    rows = []
    for model, dataset, horizon, seed in matrix:
        path = record_path(model, dataset, horizon, seed, args)
        if not completed(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": model, "dataset": dataset, "horizon": horizon, "seed": seed,
            "mse": payload["mse"], "mae": payload["mae"],
            "parameter_count": payload.get("parameter_count"), "record": str(path),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model", "dataset", "horizon", "seed", "mse", "mae", "parameter_count", "record"]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    base.atomic_write(args.output_dir / "status.json", {
        "status": status, "expected": len(matrix), "completed": len(rows),
        "failed": sum(1 for job in matrix if record_path(*job, args).is_file() and not completed(record_path(*job, args))),
        "updated_at": datetime.now().astimezone().isoformat(),
    })


def main() -> int:
    args = parse_args()
    matrix = jobs(args)
    if args.dry_run:
        for job in matrix:
            run_one(*job, args)
        print(json.dumps({"jobs": len(matrix), "models": args.models, "datasets": args.datasets}))
        return 0
    for index, job in enumerate(matrix, 1):
        print(f"=== [{index}/{len(matrix)}] {name(*job)} ===", flush=True)
        if run_one(*job, args):
            summarize(matrix, args, "failed")
            return 1
        summarize(matrix, args, "running")
    summarize(matrix, args, "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
