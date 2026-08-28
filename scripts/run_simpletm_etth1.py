#!/usr/bin/env python3
"""Run resumable SimpleTM ETTh1 experiments under the TimeRole protocol."""

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
OUTPUT = ROOT / "logs" / "simpletm_etth1_sl336"
HORIZONS = (96, 192, 336, 720)
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$", re.MULTILINE)
TEST_PATTERN = re.compile(
    r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:", re.MULTILINE
)

# Official ETTh1 horizon-specific settings. Only the common lookback and seed
# are changed to match the TimeRole main comparison protocol.
PRESETS = {
    96: dict(e_layers=1, d_model=32, d_ff=32, learning_rate=0.02,
             levels=3, alpha=0.3, l1_weight=5e-4),
    192: dict(e_layers=1, d_model=32, d_ff=32, learning_rate=0.02,
              levels=3, alpha=1.0, l1_weight=5e-5),
    336: dict(e_layers=4, d_model=64, d_ff=64, learning_rate=0.002,
              levels=3, alpha=0.0, l1_weight=0.0),
    720: dict(e_layers=1, d_model=32, d_ff=32, learning_rate=0.009,
              levels=1, alpha=0.9, l1_weight=5e-4),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS,
                        default=list(HORIZONS))
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def paths(horizon, seed):
    slug = f"simpletm_etth1_sl336_pl{horizon}_s{seed}"
    return (
        slug,
        OUTPUT / "logs" / f"{slug}.log",
        OUTPUT / "records" / f"{slug}.json",
    )


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
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def command_for(horizon, seed, gpu):
    preset = PRESETS[horizon]
    slug, _, _ = paths(horizon, seed)
    command = [
        str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "run.py"),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(ROOT / "dataset/ETT-small"),
        "--data_path", "ETTh1.csv", "--data", "ETTh1",
        "--model_id", slug, "--model", "SimpleTM", "--features", "M",
        "--target", "OT", "--freq", "h", "--seq_len", "336",
        "--label_len", "0", "--pred_len", str(horizon),
        "--enc_in", "7", "--dec_in", "7", "--c_out", "7",
        "--e_layers", str(preset["e_layers"]),
        "--d_model", str(preset["d_model"]), "--d_ff", str(preset["d_ff"]),
        "--n_heads", "8", "--factor", "1", "--dropout", "0.1",
        "--simpletm_geom_dropout", "0.5", "--simpletm_wavelet", "db1",
        "--simpletm_levels", str(preset["levels"]),
        "--simpletm_alpha", str(preset["alpha"]),
        "--simpletm_l1_weight", str(preset["l1_weight"]),
        "--simpletm_pct_start", "0.2", "--use_norm", "1",
        "--learning_rate", str(preset["learning_rate"]),
        "--batch_size", "256", "--train_epochs", "10", "--patience", "3",
        "--num_workers", "0", "--seed", str(seed), "--gpu", str(gpu),
        "--des", slug, "--itr", "1", "--test_after_train", "1",
        "--checkpoints", str(OUTPUT / "checkpoints"),
    ]
    return command


def run_one(horizon, args):
    slug, log_path, record_path = paths(horizon, args.seed)
    if args.resume and is_completed(record_path):
        print(f"SKIP completed: {slug}", flush=True)
        return 0
    command = command_for(horizon, args.seed, args.gpu)
    print("COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
            )
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
            timed_out = True

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
        "status": status,
        "model": "SimpleTM",
        "dataset": "ETTh1",
        "seq_len": 336,
        "pred_len": horizon,
        "seed": args.seed,
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "one_shot_after_validation_selection",
        "hyperparameter_source": "official_ETTh1_horizon_preset_with_common_lookback_and_seed",
        "command": command,
        "log_path": str(log_path),
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
    dataset = ROOT / "dataset/ETT-small/ETTh1.csv"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    completed = 0
    for horizon in args.horizons:
        atomic_json(OUTPUT / "status.json", {
            "status": "running", "active": f"ETTh1-{horizon}",
            "completed": completed, "total": len(args.horizons),
            "updated_at": datetime.now().astimezone().isoformat(),
        })
        code = run_one(horizon, args)
        if code:
            atomic_json(OUTPUT / "status.json", {
                "status": "failed", "active": f"ETTh1-{horizon}",
                "completed": completed, "total": len(args.horizons),
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            return code
        completed += 1
    atomic_json(OUTPUT / "status.json", {
        "status": "dry_run" if args.dry_run else "completed", "active": None,
        "completed": completed, "total": len(args.horizons),
        "updated_at": datetime.now().astimezone().isoformat(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
