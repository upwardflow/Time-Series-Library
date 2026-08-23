#!/usr/bin/env python3
"""Resumable same-budget PatchTST comparison for Recent96, IRPA, and TimeRole."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
OUTPUT = ROOT / "logs" / "patchtst_irpa_timerole_2x2x3"
MODELS = ("PatchTSTRecent", "PatchTSTIRPA", "PatchTSTHistoryCorrection")
DATASET_ORDER = ("ETTm1", "weather")
HORIZONS = (96, 720)
SEEDS = (2021, 2022, 2023)
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$", re.MULTILINE)
TEST_PATTERN = re.compile(
    r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:", re.MULTILINE
)
SETTING_PATTERN = re.compile(r"^>+start training : (.*?)>+$", re.MULTILINE)


@dataclass(frozen=True)
class DatasetConfig:
    root_path: Path
    data_path: str
    data_type: str
    channels: int
    target: str
    freq: str
    e_layers: int
    n_heads: int


DATASETS = {
    "ETTm1": DatasetConfig(
        ROOT / "dataset" / "ETT-small", "ETTm1.csv", "ETTm1", 7,
        "OT", "t", 3, 8,
    ),
    "weather": DatasetConfig(
        ROOT / "dataset" / "weather", "weather.csv", "custom", 21,
        "CO2 (ppm)", "t", 2, 4,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER)
    )
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS,
                        default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slug(model: str, dataset: str, horizon: int, seed: int) -> str:
    return f"{model.lower()}_{dataset.lower()}_sl960_pl{horizon}_s{seed}"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_path(model: str, dataset: str, horizon: int, seed: int) -> Path:
    return OUTPUT / "records" / f"{slug(model, dataset, horizon, seed)}.json"


def is_completed(path: Path, validation_only: bool) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = "validation_best_mse" if validation_only else "test_mse"
    return record.get("status") == "completed" and required in record


def validate_data() -> None:
    failures = []
    for name, config in DATASETS.items():
        path = config.root_path / config.data_path
        if not path.is_file():
            failures.append(f"missing dataset: {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle), [])
        values = [column for column in columns if column != "date"]
        if len(values) != config.channels:
            failures.append(f"{name}: expected {config.channels} variables, got {len(values)}")
        if config.target not in values:
            failures.append(f"{name}: target {config.target!r} missing")
    if failures:
        raise RuntimeError("\n".join(failures))


def build_command(
    model: str, dataset: str, horizon: int, seed: int, args: argparse.Namespace
) -> list[str]:
    config = DATASETS[dataset]
    name = slug(model, dataset, horizon, seed)
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(config.root_path), "--data_path", config.data_path,
        "--model_id", name, "--model", model, "--seed", str(seed),
        "--data", config.data_type, "--features", "M", "--target", config.target,
        "--freq", config.freq, "--seq_len", "960", "--label_len", "48",
        "--pred_len", str(horizon), "--enc_in", str(config.channels),
        "--dec_in", str(config.channels), "--c_out", str(config.channels),
        "--d_model", "512", "--d_ff", "2048", "--n_heads", str(config.n_heads),
        "--e_layers", str(config.e_layers), "--d_layers", "1", "--factor", "3",
        "--dropout", "0.1", "--batch_size", "32", "--learning_rate", "0.0001",
        "--train_epochs", str(args.epochs), "--patience", str(args.patience),
        "--lradj", "type1", "--num_workers", "0", "--gpu", str(args.gpu),
        "--des", name, "--itr", "1", "--checkpoints", str(OUTPUT / "checkpoints"),
        "--test_after_train", "0" if args.validation_only else "1",
        "--moving_avg", "25", "--irpa_revise_len", "96", "--irpa_topk", "3",
        "--timerole_hidden_dim", "32", "--timerole_memory_pool", "16",
    ]


def parse_log(text: str):
    validation_matches = list(VALIDATION_PATTERN.finditer(text))
    test_matches = list(TEST_PATTERN.finditer(text))
    setting_matches = list(SETTING_PATTERN.finditer(text))
    validation = json.loads(validation_matches[-1].group(1)) if validation_matches else None
    test = None
    if test_matches:
        match = test_matches[-1]
        test = {"test_mse": float(match.group(1)), "test_mae": float(match.group(2))}
    setting = setting_matches[-1].group(1) if setting_matches else None
    return validation, test, setting


def run_one(model: str, dataset: str, horizon: int, seed: int,
            args: argparse.Namespace) -> int:
    destination = record_path(model, dataset, horizon, seed)
    if args.resume and is_completed(destination, args.validation_only):
        print(f"SKIP completed: {destination.name}", flush=True)
        return 0
    command = build_command(model, dataset, horizon, seed, args)
    print("COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    log_path = OUTPUT / "logs" / f"{slug(model, dataset, horizon, seed)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                text=True, timeout=args.timeout_seconds, check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
            )
        return_code = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = 124

    text = log_path.read_text(encoding="utf-8", errors="replace")
    validation, test, setting = parse_log(text)
    success = return_code == 0 and validation is not None
    if not args.validation_only:
        success = success and test is not None
    payload: dict[str, object] = {
        "status": "completed" if success else ("timeout" if timed_out else "failed"),
        "model": model, "dataset": dataset, "pred_len": horizon, "seq_len": 960,
        "recent_or_revise_len": 96, "seed": seed, "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "none" if args.validation_only else "one_shot_after_validation_selection",
        "protocol": "same_budget_sl960_patchtst_recent96",
        "command": command, "setting": setting, "log_path": str(log_path),
    }
    if validation:
        payload.update({f"validation_{key}": value for key, value in validation.items()})
    if test:
        payload.update(test)
    atomic_json(destination, payload)
    print(f"FINISH {payload['status']}: {destination.name} ({payload['duration_seconds']}s)",
          flush=True)
    return 0 if success else return_code or 1


def write_summary() -> None:
    rows = []
    for dataset in DATASET_ORDER:
        for horizon in HORIZONS:
            for seed in SEEDS:
                for model in MODELS:
                    path = record_path(model, dataset, horizon, seed)
                    if not path.is_file():
                        continue
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if "test_mse" not in record:
                        continue
                    rows.append({
                        "dataset": dataset, "horizon": horizon, "seed": seed,
                        "model": model, "test_mse": record["test_mse"],
                        "test_mae": record["test_mae"],
                        "duration_seconds": record["duration_seconds"],
                    })
    if not rows:
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "all_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    aggregates = []
    for dataset in DATASET_ORDER:
        for horizon in HORIZONS:
            for model in MODELS:
                selected = [r for r in rows if r["dataset"] == dataset
                            and r["horizon"] == horizon and r["model"] == model]
                if len(selected) != len(SEEDS):
                    continue
                mse = [float(r["test_mse"]) for r in selected]
                mae = [float(r["test_mae"]) for r in selected]
                aggregates.append({
                    "dataset": dataset, "horizon": horizon, "model": model,
                    "mse_mean": statistics.mean(mse), "mse_sample_std": statistics.stdev(mse),
                    "mae_mean": statistics.mean(mae), "mae_sample_std": statistics.stdev(mae),
                })
    if aggregates:
        with (OUTPUT / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader(); writer.writerows(aggregates)


def main() -> int:
    args = parse_args()
    validate_data()
    jobs = [
        (model, dataset, horizon, seed)
        for dataset in args.datasets for horizon in args.horizons
        for seed in args.seeds for model in args.models
    ]
    if args.max_jobs > 0:
        jobs = jobs[:args.max_jobs]
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {slug(*job)}", flush=True)
        code = run_one(*job, args)
        if code != 0:
            print(f"STOP after failure: {job}", file=sys.stderr, flush=True)
            return code
    if not args.validation_only and not args.dry_run:
        write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
