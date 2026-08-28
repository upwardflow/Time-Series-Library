#!/usr/bin/env python3
"""Run remaining SimpleTM datasets under the TimeRole main protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "simpletm_remaining_sl336"
HORIZONS = (96, 192, 336, 720)
DATASET_ORDER = ("ETTh2", "ETTm1", "ETTm2", "weather")
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$", re.MULTILINE)
TEST_PATTERN = re.compile(
    r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:", re.MULTILINE
)


@dataclass(frozen=True)
class DatasetConfig:
    root_path: Path
    data_path: str
    data_type: str
    channels: int
    target: str
    freq: str


DATASETS = {
    "ETTh2": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh2.csv", "ETTh2", 7, "OT", "h"),
    "ETTm1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm1.csv", "ETTm1", 7, "OT", "t"),
    "ETTm2": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm2.csv", "ETTm2", 7, "OT", "t"),
    "weather": DatasetConfig(
        ROOT / "dataset/weather", "weather.csv", "custom", 21, "CO2 (ppm)", "t"
    ),
}


# Official SimpleTM horizon-specific presets. The common lookback and experiment
# seed are intentionally supplied by this runner instead of these presets.
PRESETS = {
    "ETTh2": {
        96: dict(e_layers=1, d_model=32, d_ff=32, lr=0.006, batch=256,
                 wavelet="bior3.1", levels=1, alpha=0.1, l1=5e-4),
        192: dict(e_layers=1, d_model=32, d_ff=32, lr=0.006, batch=256,
                  wavelet="db1", levels=1, alpha=0.1, l1=5e-3),
        336: dict(e_layers=1, d_model=32, d_ff=32, lr=0.003, batch=256,
                  wavelet="db1", levels=1, alpha=0.9, l1=0.0),
        720: dict(e_layers=1, d_model=32, d_ff=32, lr=0.003, batch=256,
                  wavelet="db1", levels=1, alpha=1.0, l1=5e-5),
    },
    "ETTm1": {
        96: dict(e_layers=1, d_model=32, d_ff=32, lr=0.02, batch=256,
                 wavelet="db1", levels=3, alpha=0.1, l1=5e-3),
        192: dict(e_layers=1, d_model=32, d_ff=32, lr=0.02, batch=256,
                  wavelet="db1", levels=3, alpha=0.1, l1=5e-3),
        336: dict(e_layers=1, d_model=32, d_ff=32, lr=0.02, batch=256,
                  wavelet="db1", levels=1, alpha=0.1, l1=5e-3),
        720: dict(e_layers=1, d_model=32, d_ff=32, lr=0.02, batch=256,
                  wavelet="db1", levels=3, alpha=0.1, l1=5e-3),
    },
    "ETTm2": {
        96: dict(e_layers=1, d_model=32, d_ff=32, lr=0.006, batch=256,
                 wavelet="bior3.1", levels=3, alpha=0.3, l1=5e-4),
        192: dict(e_layers=1, d_model=32, d_ff=32, lr=0.006, batch=256,
                  wavelet="bior3.1", levels=1, alpha=0.0, l1=5e-3),
        336: dict(e_layers=1, d_model=64, d_ff=64, lr=0.006, batch=128,
                  wavelet="bior3.3", levels=1, alpha=0.6, l1=5e-5),
        720: dict(e_layers=1, d_model=96, d_ff=96, lr=0.003, batch=256,
                  wavelet="db1", levels=3, alpha=1.0, l1=0.0),
    },
    "weather": {
        96: dict(e_layers=4, d_model=32, d_ff=32, lr=0.01, batch=256,
                 wavelet="db4", levels=1, alpha=0.3, l1=5e-5),
        192: dict(e_layers=4, d_model=32, d_ff=32, lr=0.009, batch=256,
                  wavelet="db4", levels=1, alpha=0.3, l1=0.0),
        336: dict(e_layers=1, d_model=32, d_ff=32, lr=0.009, batch=256,
                  wavelet="db4", levels=3, alpha=1.0, l1=5e-5),
        720: dict(e_layers=1, d_model=32, d_ff=32, lr=0.02, batch=256,
                  wavelet="db4", levels=1, alpha=0.9, l1=5e-3),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER,
                        default=list(DATASET_ORDER))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS,
                        default=list(HORIZONS))
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def job_paths(dataset, horizon, seed):
    slug = f"simpletm_{dataset.lower()}_sl336_pl{horizon}_s{seed}"
    return slug, OUTPUT / "logs" / f"{slug}.log", OUTPUT / "records" / f"{slug}.json"


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def is_completed(path):
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and "test_mse" in payload


def validate_data(datasets):
    for dataset in datasets:
        config = DATASETS[dataset]
        path = config.root_path / config.data_path
        if not path.is_file():
            raise FileNotFoundError(path)


def command_for(dataset, horizon, seed, gpu):
    config = DATASETS[dataset]
    preset = PRESETS[dataset][horizon]
    slug, _, _ = job_paths(dataset, horizon, seed)
    return [
        str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "run.py"),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(config.root_path), "--data_path", config.data_path,
        "--data", config.data_type, "--model_id", slug, "--model", "SimpleTM",
        "--features", "M", "--target", config.target, "--freq", config.freq,
        "--seq_len", "336", "--label_len", "0", "--pred_len", str(horizon),
        "--enc_in", str(config.channels), "--dec_in", str(config.channels),
        "--c_out", str(config.channels), "--e_layers", str(preset["e_layers"]),
        "--d_model", str(preset["d_model"]), "--d_ff", str(preset["d_ff"]),
        "--n_heads", "8", "--factor", "1", "--dropout", "0.1",
        "--simpletm_geom_dropout", "0.5",
        "--simpletm_wavelet", str(preset["wavelet"]),
        "--simpletm_levels", str(preset["levels"]),
        "--simpletm_alpha", str(preset["alpha"]),
        "--simpletm_l1_weight", str(preset["l1"]),
        "--simpletm_pct_start", "0.2", "--use_norm", "1",
        "--learning_rate", str(preset["lr"]), "--batch_size", str(preset["batch"]),
        "--train_epochs", "10", "--patience", "3", "--num_workers", "0",
        "--seed", str(seed), "--gpu", str(gpu), "--des", slug, "--itr", "1",
        "--test_after_train", "1", "--checkpoints", str(OUTPUT / "checkpoints"),
    ]


def run_one(dataset, horizon, args):
    slug, log_path, record_path = job_paths(dataset, horizon, args.seed)
    if args.resume and is_completed(record_path):
        print(f"SKIP completed: {slug}", flush=True)
        return 0
    command = command_for(dataset, horizon, args.seed, args.gpu)
    print("COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            result = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                text=True, timeout=args.timeout_seconds, check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
            )
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            return_code, timed_out = 124, True

    text = log_path.read_text(encoding="utf-8", errors="replace")
    validation_matches = list(VALIDATION_PATTERN.finditer(text))
    test_matches = list(TEST_PATTERN.finditer(text))
    validation = json.loads(validation_matches[-1].group(1)) if validation_matches else None
    metrics = None
    if test_matches:
        metrics = {
            "test_mse": float(test_matches[-1].group(1)),
            "test_mae": float(test_matches[-1].group(2)),
        }
    status = "completed" if return_code == 0 and validation and metrics else (
        "timeout" if timed_out else "failed"
    )
    payload = {
        "status": status, "model": "SimpleTM", "dataset": dataset,
        "seq_len": 336, "pred_len": horizon, "seed": args.seed,
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "one_shot_after_validation_selection",
        "hyperparameter_source": "official_dataset_horizon_preset_with_common_lookback_and_seed",
        "command": command, "log_path": str(log_path),
    }
    if validation:
        payload.update({f"validation_{key}": value for key, value in validation.items()})
    if metrics:
        payload.update(metrics)
    atomic_json(record_path, payload)
    print(f"FINISH {status}: {slug} ({payload['duration_seconds']}s)", flush=True)
    return 0 if status == "completed" else return_code or 1


def main():
    args = parse_args()
    validate_data(args.datasets)
    jobs = [(dataset, horizon) for dataset in args.datasets for horizon in args.horizons]
    completed = 0
    for dataset, horizon in jobs:
        active = f"{dataset}-{horizon}"
        atomic_json(OUTPUT / "status.json", {
            "status": "running", "active": active, "completed": completed,
            "total": len(jobs), "updated_at": datetime.now().astimezone().isoformat(),
        })
        code = run_one(dataset, horizon, args)
        if code:
            atomic_json(OUTPUT / "status.json", {
                "status": "failed", "active": active, "completed": completed,
                "total": len(jobs), "updated_at": datetime.now().astimezone().isoformat(),
            })
            return code
        completed += 1
    atomic_json(OUTPUT / "status.json", {
        "status": "dry_run" if args.dry_run else "completed", "active": None,
        "completed": completed, "total": len(jobs),
        "updated_at": datetime.now().astimezone().isoformat(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
