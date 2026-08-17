#!/usr/bin/env python3
"""Run the preregistered validation-only B0/B1 stage-1 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
OUTPUT = ROOT / "logs/graphmamba_period_normalized_v2/stage1"
SEED = 2021
HORIZONS = (192, 720)
DATASET_ORDER = ("ETTh1", "ETTm1", "weather", "solar")
VARIANTS = ("b0_recent96", "b1_period_norm")
VALIDATION_RE = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
SETTING_RE = re.compile(r"^>+start training : (.*?)>+$")
TEST_RE = re.compile(r"^mse:[-+0-9.eE]+,\s*mae:[-+0-9.eE]+,\s*dtw:")


@dataclass(frozen=True)
class DatasetConfig:
    root_path: Path
    data_path: str
    data_type: str
    channels: int
    target: str
    freq: str
    batch_size: int
    samples_per_hour: int


DATASETS = {
    "ETTh1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh1.csv", "ETTh1", 7, "OT", "h", 32, 1),
    "ETTm1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm1.csv", "ETTm1", 7, "OT", "t", 32, 4),
    "weather": DatasetConfig(ROOT / "dataset/weather", "weather.csv", "custom", 21, "CO2 (ppm)", "t", 32, 6),
    "solar": DatasetConfig(ROOT / "dataset/solar", "solar.csv", "custom", 137, "channel_99", "t", 16, 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_data(names: list[str]) -> None:
    errors: list[str] = []
    for name in names:
        cfg = DATASETS[name]
        path = cfg.root_path / cfg.data_path
        if not path.is_file():
            errors.append(f"missing dataset: {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle), [])
        values = [column for column in columns if column != "date"]
        if "date" not in columns:
            errors.append(f"{path}: missing date column")
        if len(values) != cfg.channels:
            errors.append(f"{path}: expected {cfg.channels} variables, found {len(values)}")
        if cfg.target not in values:
            errors.append(f"{path}: missing target {cfg.target!r}")
    if errors:
        raise RuntimeError("\n".join(errors))


def job_id(variant: str, dataset: str, horizon: int) -> str:
    return f"{variant}_{dataset.lower()}_sl336_pl{horizon}_s{SEED}"


def build_command(variant: str, dataset: str, horizon: int, gpu: int) -> list[str]:
    cfg = DATASETS[dataset]
    model = "GraphMambaRecent" if variant == "b0_recent96" else "GraphMambaPeriodNorm"
    name = job_id(variant, dataset, horizon)
    # GraphMambaPeriodNorm has an intentionally descriptive checkpoint suffix;
    # keep its user-controlled fields compact to stay below NAME_MAX (255).
    run_name = name if variant == "b0_recent96" else f"s1pn_{dataset.lower()}_p{horizon}"
    description = name if variant == "b0_recent96" else "s1pn"
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(cfg.root_path), "--data_path", cfg.data_path,
        "--model_id", run_name, "--model", model, "--seed", str(SEED),
        "--data", cfg.data_type, "--features", "M", "--target", cfg.target,
        "--freq", cfg.freq, "--seq_len", "336", "--label_len", "48",
        "--pred_len", str(horizon), "--enc_in", str(cfg.channels),
        "--dec_in", str(cfg.channels), "--c_out", str(cfg.channels),
        "--patch_len", "4", "--stride", "2", "--d_model", "64",
        "--d_ff", "128", "--d_state", "32", "--d_conv", "2",
        "--e_layers", "1", "--expand", "2", "--mamba_version", "1",
        "--mamba_bidirectional", "1", "--use_graph", "1",
        "--use_time_mamba", "1", "--use_patch", "1", "--use_decomp", "1",
        "--moving_avg", "25", "--dual_scale_scan_mode", "independent_shared",
        "--periodic_period", "24", "--periodic_local_patch", "4",
        "--periodic_local_stride", "2", "--periodic_period_stride", "12",
        "--periodic_use_adapter", "1", "--period_norm_factor", str(cfg.samples_per_hour),
        "--period_norm_recent_len", "96", "--graph_alpha", "0.5",
        "--graph_top_k", "2", "--graph_sample_size", "2000",
        "--graph_sample_method", "uniform", "--static_graph_mode", "weighted",
        "--graph_cache", "1", "--dropout", "0.1",
        "--batch_size", str(cfg.batch_size), "--learning_rate", "0.0005",
        "--lradj", "type1", "--train_epochs", "100", "--patience", "6",
        "--num_workers", "0", "--gpu", str(gpu), "--des", description, "--itr", "1",
        "--checkpoints", str(OUTPUT / "checkpoints"), "--test_after_train", "0",
    ]


def fingerprint(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command).encode("utf-8")).hexdigest()


def completed_record(path: Path, command: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("command_fingerprint") == fingerprint(command)
        and "best_mse" in payload
        and "best_mae" in payload
        and not payload.get("test_metric_seen", False)
    )


def run_job(variant: str, dataset: str, horizon: int, gpu: int) -> dict[str, object]:
    name = job_id(variant, dataset, horizon)
    command = build_command(variant, dataset, horizon, gpu)
    record_path = OUTPUT / "records" / f"{name}.json"
    log_path = OUTPUT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = datetime.now().astimezone().isoformat()
    started = time.monotonic()
    atomic_json(record_path, {
        "status": "running", "job_id": name, "variant": variant,
        "dataset": dataset, "pred_len": horizon, "seed": SEED,
        "started_at": started_wall, "command": command,
        "command_fingerprint": fingerprint(command), "log_path": str(log_path),
        "test_after_train": 0,
    })
    print(f"\nSTAGE1_JOB_START {name}", flush=True)
    print("Command:", shlex.join(command), flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    validation = None
    setting = None
    test_metric_seen = False
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            match = VALIDATION_RE.match(line.strip())
            if match:
                validation = json.loads(match.group(1))
            match = SETTING_RE.match(line.strip())
            if match:
                setting = match.group(1)
            if TEST_RE.match(line.strip()):
                test_metric_seen = True
        return_code = process.wait()
    payload: dict[str, object] = {
        "status": "completed" if return_code == 0 and validation and not test_metric_seen else "failed",
        "job_id": name, "variant": variant,
        "model": "GraphMambaRecent" if variant == "b0_recent96" else "GraphMambaPeriodNorm",
        "dataset": dataset, "pred_len": horizon, "seq_len": 336,
        "seed": SEED, "samples_per_hour": DATASETS[dataset].samples_per_hour,
        "return_code": return_code, "duration_seconds": round(time.monotonic() - started, 3),
        "started_at": started_wall, "finished_at": datetime.now().astimezone().isoformat(),
        "setting": setting, "command": command,
        "command_fingerprint": fingerprint(command), "log_path": str(log_path),
        "test_after_train": 0, "test_metric_seen": test_metric_seen,
    }
    if validation:
        payload.update(validation)
    atomic_json(record_path, payload)
    print(f"STAGE1_JOB_END {name} status={payload['status']}", flush=True)
    if payload["status"] != "completed":
        raise RuntimeError(f"stage-1 job failed without retry: {name}; see {log_path}")
    return payload


def write_summary() -> None:
    rows = []
    for path in sorted((OUTPUT / "records").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            rows.append({key: payload.get(key) for key in (
                "job_id", "variant", "dataset", "pred_len", "seed", "best_mse",
                "best_mae", "best_epoch", "epochs_ran", "duration_seconds",
                "test_after_train", "test_metric_seen", "command_fingerprint",
            )})
    atomic_json(OUTPUT / "summary.json", rows)


def main() -> int:
    args = parse_args()
    validate_data(args.datasets)
    jobs = [
        (variant, dataset, horizon)
        for dataset in args.datasets
        for horizon in args.horizons
        for variant in args.variants
    ]
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    manifest = []
    for variant, dataset, horizon in jobs:
        command = build_command(variant, dataset, horizon, args.gpu)
        manifest.append({
            "job_id": job_id(variant, dataset, horizon), "variant": variant,
            "dataset": dataset, "pred_len": horizon, "seed": SEED,
            "command": command, "command_fingerprint": fingerprint(command),
            "test_after_train": 0,
        })
    atomic_json(OUTPUT / "frozen_manifest.json", manifest)
    print(f"Frozen {len(manifest)} commands in {OUTPUT / 'frozen_manifest.json'}")
    if args.dry_run:
        for item in manifest:
            print(item["job_id"], shlex.join(item["command"]))
        return 0
    for variant, dataset, horizon in jobs:
        command = build_command(variant, dataset, horizon, args.gpu)
        path = OUTPUT / "records" / f"{job_id(variant, dataset, horizon)}.json"
        if not args.no_resume and completed_record(path, command):
            print(f"STAGE1_JOB_SKIP completed {job_id(variant, dataset, horizon)}")
            continue
        run_job(variant, dataset, horizon, args.gpu)
        write_summary()
    write_summary()
    print("STAGE1_MATRIX_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
