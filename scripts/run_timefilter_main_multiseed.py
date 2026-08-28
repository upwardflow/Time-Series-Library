#!/usr/bin/env python3
"""Run TimeFilter on the five-dataset, four-horizon, three-seed main protocol.

The data/evaluation protocol matches the TimeRole main table (M->M, seq_len
336, validation-selected checkpoint, one test pass). TimeFilter architecture
hyperparameters follow the official per-dataset/per-horizon scripts, with only
the lookback changed from the official 96 to the paper-wide 336 comparison.

The runner is serial, resumable, atomic, and stops at the first failed job.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
OUTPUT = ROOT / "logs" / "timefilter_main_multiseed"
DATASET_ORDER = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")
HORIZONS = (96, 192, 336, 720)
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


DATASETS = {
    "ETTh1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh1.csv", "ETTh1", 7, "OT", "h"),
    "ETTh2": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh2.csv", "ETTh2", 7, "OT", "h"),
    "ETTm1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm1.csv", "ETTm1", 7, "OT", "t"),
    "ETTm2": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm2.csv", "ETTm2", 7, "OT", "t"),
    "weather": DatasetConfig(
        ROOT / "dataset/weather", "weather.csv", "custom", 21, "CO2 (ppm)", "t"
    ),
}


OFFICIAL_SCRIPTS = {
    "ETTh1": "https://github.com/TROUBADOUR000/TimeFilter/blob/main/scripts/ETTh1.sh",
    "ETTh2": "https://github.com/TROUBADOUR000/TimeFilter/blob/main/scripts/ETTh2.sh",
    "ETTm1": "https://github.com/TROUBADOUR000/TimeFilter/blob/main/scripts/ETTm1.sh",
    "ETTm2": "https://github.com/TROUBADOUR000/TimeFilter/blob/main/scripts/ETTm2.sh",
    "weather": "https://github.com/TROUBADOUR000/TimeFilter/blob/main/scripts/Weather.sh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()
    if args.timeout_seconds < 1 or args.max_jobs < 0:
        parser.error("timeout must be positive and max-jobs cannot be negative")
    return args


def preset(dataset: str, horizon: int) -> dict[str, object]:
    """Official TimeFilter settings; seq_len is intentionally controlled elsewhere."""
    if dataset == "ETTh1":
        return dict(
            e_layers=2, d_model=128, d_ff=128 if horizon == 720 else 256,
            dropout=0.8, patch_len=2, pos=0, alpha=0.1, top_p=0.5,
            learning_rate=1e-4, batch_size=32,
        )
    if dataset == "ETTh2":
        table = {
            96: dict(e_layers=1, d_model=128, d_ff=256, dropout=0.8,
                     patch_len=4, alpha=0.1, top_p=0.0),
            192: dict(e_layers=1, d_model=128, d_ff=256, dropout=0.6,
                      patch_len=4, alpha=0.8, top_p=0.5),
            336: dict(e_layers=2, d_model=256, d_ff=256, dropout=0.7,
                      patch_len=8, alpha=0.4, top_p=0.0),
            720: dict(e_layers=2, d_model=256, d_ff=256, dropout=0.3,
                      patch_len=8, alpha=0.9, top_p=0.0),
        }
        return dict(**table[horizon], pos=1, learning_rate=1e-4, batch_size=32)
    if dataset == "ETTm1":
        return dict(
            e_layers=2, d_model=256, d_ff=256,
            dropout={96: 0.3, 192: 0.5, 336: 0.5, 720: 0.7}[horizon],
            patch_len=16 if horizon == 720 else 8, pos=1, alpha=0.1, top_p=0.0,
            learning_rate=1e-4, batch_size=32,
        )
    if dataset == "ETTm2":
        return dict(
            e_layers=2, d_model=128, d_ff=128,
            dropout=0.8 if horizon == 720 else 0.6,
            patch_len=16, pos=1, alpha=0.1, top_p=0.0,
            learning_rate=1e-4, batch_size=32,
        )
    if dataset == "weather":
        return dict(
            e_layers=2, d_model=128, d_ff=256, dropout=0.3,
            patch_len=48, pos=1, alpha=0.1, top_p=0.5,
            learning_rate=5e-4, batch_size=32,
        )
    raise ValueError(dataset)


def slug(dataset: str, horizon: int, seed: int) -> str:
    return f"timefilter_{dataset.lower()}_sl336_pl{horizon}_s{seed}"


def record_path(dataset: str, horizon: int, seed: int) -> Path:
    return OUTPUT / "records" / f"{slug(dataset, horizon, seed)}.json"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("test_mse") is not None
        and payload.get("test_mae") is not None
    )


def validate_data() -> None:
    failures: list[str] = []
    for dataset, config in DATASETS.items():
        path = config.root_path / config.data_path
        if not path.is_file():
            failures.append(f"missing dataset: {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle), [])
        values = [column for column in columns if column != "date"]
        if len(values) != config.channels:
            failures.append(f"{path}: expected {config.channels} variables, found {len(values)}")
        if config.target not in values:
            failures.append(f"{path}: target {config.target!r} not found")
    if failures:
        raise RuntimeError("\n".join(failures))


def build_command(dataset: str, horizon: int, seed: int, gpu: int) -> list[str]:
    data = DATASETS[dataset]
    config = preset(dataset, horizon)
    name = slug(dataset, horizon, seed)
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(data.root_path), "--data_path", data.data_path,
        "--model_id", name, "--model", "TimeFilter", "--seed", str(seed),
        "--data", data.data_type, "--features", "M", "--target", data.target,
        "--freq", data.freq, "--seq_len", "336", "--label_len", "48",
        "--pred_len", str(horizon), "--enc_in", str(data.channels),
        "--dec_in", str(data.channels), "--c_out", str(data.channels),
        "--d_model", str(config["d_model"]), "--d_ff", str(config["d_ff"]),
        "--n_heads", "8", "--e_layers", str(config["e_layers"]),
        "--d_layers", "1", "--factor", "3", "--dropout", str(config["dropout"]),
        "--patch_len", str(config["patch_len"]), "--stride", str(config["patch_len"]),
        "--pos", str(config["pos"]), "--alpha", str(config["alpha"]),
        "--top_p", str(config["top_p"]), "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]), "--train_epochs", "10",
        "--patience", "3", "--lradj", "type1", "--num_workers", "0",
        "--gpu", str(gpu), "--des", name, "--itr", "1",
        "--checkpoints", str(OUTPUT / "checkpoints"), "--test_after_train", "1",
    ]


def parse_log(text: str) -> tuple[dict[str, object] | None, dict[str, float] | None, str | None]:
    validation_matches = list(VALIDATION_PATTERN.finditer(text))
    test_matches = list(TEST_PATTERN.finditer(text))
    setting_matches = list(SETTING_PATTERN.finditer(text))
    validation = json.loads(validation_matches[-1].group(1)) if validation_matches else None
    metrics = None
    if test_matches:
        match = test_matches[-1]
        metrics = {"test_mse": float(match.group(1)), "test_mae": float(match.group(2))}
    setting = setting_matches[-1].group(1) if setting_matches else None
    return validation, metrics, setting


def run_one(dataset: str, horizon: int, seed: int, args: argparse.Namespace) -> int:
    destination = record_path(dataset, horizon, seed)
    if args.resume and completed(destination):
        print(f"SKIP completed: {destination.name}", flush=True)
        return 0
    command = build_command(dataset, horizon, seed, args.gpu)
    print("COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    log_path = OUTPUT / "logs" / f"{slug(dataset, horizon, seed)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    return_code, timed_out = 1, False
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            result = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                text=True, timeout=args.timeout_seconds, check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
            )
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, return_code = True, 124

    text = log_path.read_text(encoding="utf-8", errors="replace")
    validation, metrics, setting = parse_log(text)
    status = "completed" if return_code == 0 and validation and metrics else (
        "timeout" if timed_out else "failed"
    )
    payload: dict[str, object] = {
        "status": status, "model": "TimeFilter", "dataset": dataset,
        "horizon": horizon, "seq_len": 336, "seed": seed,
        "return_code": return_code, "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "one_shot_after_validation_selection",
        "result_source": "official_timefilter_hparams_common_timerole_protocol",
        "official_script": OFFICIAL_SCRIPTS[dataset],
        "protocol_adaptation": "seq_len changed from official 96 to common 336",
        "command": command, "setting": setting, "log_path": str(log_path),
        "preset": preset(dataset, horizon),
    }
    if validation:
        payload.update({f"validation_{key}": value for key, value in validation.items()})
    if metrics:
        payload.update(metrics)
    atomic_json(destination, payload)
    print(f"FINISH {status}: {dataset} H={horizon} seed={seed} ({payload['duration_seconds']}s)", flush=True)
    return 0 if status == "completed" else return_code or 1


def write_summaries(active: str | None, state: str) -> None:
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for horizon in HORIZONS:
            for seed in SEEDS:
                path = record_path(dataset, horizon, seed)
                if not completed(path):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "model": "TimeFilter", "dataset": dataset, "horizon": horizon,
                    "seed": seed, "seq_len": 336,
                    "validation_mse": payload.get("validation_best_mse"),
                    "validation_mae": payload.get("validation_best_mae"),
                    "test_mse": payload["test_mse"], "test_mae": payload["test_mae"],
                    "duration_seconds": payload.get("duration_seconds"),
                })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    long_path = OUTPUT / "results_long.csv"
    if rows:
        with long_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    grouped: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for horizon in HORIZONS:
            selected = [r for r in rows if r["dataset"] == dataset and r["horizon"] == horizon]
            if not selected:
                continue
            mses = [float(r["test_mse"]) for r in selected]
            maes = [float(r["test_mae"]) for r in selected]
            grouped.append({
                "model": "TimeFilter", "dataset": dataset, "horizon": horizon,
                "n_seeds": len(selected),
                "seeds": ";".join(str(r["seed"]) for r in selected),
                "test_mse_mean": statistics.fmean(mses),
                "test_mse_std": statistics.stdev(mses) if len(mses) > 1 else math.nan,
                "test_mae_mean": statistics.fmean(maes),
                "test_mae_std": statistics.stdev(maes) if len(maes) > 1 else math.nan,
            })
    if grouped:
        with (OUTPUT / "mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
            writer.writeheader()
            writer.writerows(grouped)
    atomic_json(OUTPUT / "status.json", {
        "status": state, "active_or_last": active,
        "completed": len(rows), "expected": len(DATASET_ORDER) * len(HORIZONS) * len(SEEDS),
        "updated_at": datetime.now().astimezone().isoformat(),
        "results_long": str(long_path), "mean_std": str(OUTPUT / "mean_std.csv"),
    })


def main() -> int:
    args = parse_args()
    validate_data()
    jobs = [
        (dataset, horizon, seed)
        for dataset in args.datasets for horizon in args.horizons for seed in args.seeds
    ]
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    done = 0
    for index, (dataset, horizon, seed) in enumerate(jobs, start=1):
        active = slug(dataset, horizon, seed)
        write_summaries(active, "dry_run" if args.dry_run else "running")
        print(f"[{index}/{len(jobs)}] {active}", flush=True)
        code = run_one(dataset, horizon, seed, args)
        if code != 0:
            write_summaries(active, "failed")
            print(f"STOP after failure: inspect {record_path(dataset, horizon, seed)}", flush=True)
            return code
        done += 1
        write_summaries(active, "dry_run" if args.dry_run else "running")
    write_summaries(None, "dry_run" if args.dry_run else "completed")
    print(f"Completed this invocation: {done}/{len(jobs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
