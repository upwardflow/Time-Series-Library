#!/usr/bin/env python3
"""Run a resumable six-dataset, fixed-lookback Q2 baseline matrix.

The runner does not tune models. It preserves dataset/model presets from the
repository's official scripts while fixing seq_len=336, seed=2021, the data
split, checkpoint selection, and metric implementation. Frozen one-shot CMRHM
records are reused instead of reading already-consumed test splits again.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
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
OUTPUT = ROOT / "logs" / "q2_main_baselines"
MODELS = (
    "DLinear",
    "PatchTST",
    "iTransformer",
    "TimeMixer",
    "TimesNet",
    "GraphMambaCMRHM",
)
DATASET_ORDER = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "solar")
HORIZONS = (96, 192, 336, 720)
SEED = 2021
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
    batch_size: int


DATASETS = {
    "ETTh1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh1.csv", "ETTh1", 7, "OT", "h", 32),
    "ETTh2": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh2.csv", "ETTh2", 7, "OT", "h", 32),
    "ETTm1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm1.csv", "ETTm1", 7, "OT", "t", 32),
    "ETTm2": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm2.csv", "ETTm2", 7, "OT", "t", 32),
    "weather": DatasetConfig(ROOT / "dataset/weather", "weather.csv", "custom", 21, "CO2 (ppm)", "t", 32),
    "solar": DatasetConfig(ROOT / "dataset/solar", "solar.csv", "custom", 137, "channel_99", "h", 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    return parser.parse_args()


def slug(model: str, dataset: str, horizon: int) -> str:
    return f"{model.lower()}_{dataset.lower()}_sl336_pl{horizon}_s{SEED}"


def record_path(model: str, dataset: str, horizon: int) -> Path:
    return OUTPUT / "records" / f"{slug(model, dataset, horizon)}.json"


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
    return payload.get("status") == "completed" and "test_mse" in payload


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
        if "date" not in columns:
            failures.append(f"{path}: missing date column")
        if len(values) != config.channels:
            failures.append(f"{path}: expected {config.channels} variables, found {len(values)}")
        if config.target not in values:
            failures.append(f"{path}: target {config.target!r} not found")
    if failures:
        raise RuntimeError("\n".join(failures))


def model_preset(model: str, dataset: str) -> dict[str, object]:
    large = False
    if model == "DLinear":
        return dict(label_len=48, e_layers=2, d_model=32, d_ff=64, n_heads=4,
                    batch_size=DATASETS[dataset].batch_size, learning_rate=1e-4,
                    epochs=10, patience=3)
    if model == "PatchTST":
        e_layers = 1 if dataset == "ETTh1" else (3 if dataset.startswith("ETT") else 2)
        n_heads = 4 if dataset in {"ETTh2", "weather"} else 8
        return dict(label_len=48, e_layers=e_layers, d_model=512,
                    d_ff=2048, n_heads=n_heads,
                    batch_size=DATASETS[dataset].batch_size,
                    learning_rate=1e-4, epochs=10, patience=3,
                    patch_len=16, stride=8)
    if model == "iTransformer":
        if dataset.startswith("ETT"):
            e_layers, d_model, d_ff, lr = 2, 128, 128, 1e-4
        else:
            e_layers = 3
            d_model, d_ff = 512, 512
            lr = 1e-4
        return dict(label_len=48, e_layers=e_layers, d_model=d_model, d_ff=d_ff,
                    n_heads=8, batch_size=16 if large else DATASETS[dataset].batch_size,
                    learning_rate=lr, epochs=10, patience=3)
    if model == "TimeMixer":
        return dict(label_len=0, e_layers=2 if dataset.startswith("ETT") else 3,
                    d_model=16, d_ff=32,
                    n_heads=4, batch_size=DATASETS[dataset].batch_size,
                    learning_rate=1e-2, epochs=20 if dataset in {"weather", "solar"} else 10,
                    patience=10, down_sampling_layers=3,
                    down_sampling_window=2, down_sampling_method="avg")
    if model == "TimesNet":
        d_model, d_ff, batch_size = 32, 32, DATASETS[dataset].batch_size
        return dict(label_len=48, e_layers=2, d_model=d_model, d_ff=d_ff,
                    n_heads=4, batch_size=batch_size, learning_rate=1e-4,
                    epochs=10, patience=3, top_k=5)
    if model == "GraphMambaCMRHM":
        return dict(label_len=48, e_layers=1, d_model=64, d_ff=128,
                    n_heads=8, batch_size=DATASETS[dataset].batch_size,
                    learning_rate=5e-4, epochs=100, patience=6,
                    patch_len=4, stride=2)
    raise ValueError(model)


def build_command(model: str, dataset: str, horizon: int, gpu: int) -> list[str]:
    config = DATASETS[dataset]
    preset = model_preset(model, dataset)
    name = slug(model, dataset, horizon)
    command = [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(config.root_path), "--data_path", config.data_path,
        "--model_id", name, "--model", model, "--seed", str(SEED),
        "--data", config.data_type, "--features", "M", "--target", config.target,
        "--freq", config.freq, "--seq_len", "336",
        "--label_len", str(preset["label_len"]), "--pred_len", str(horizon),
        "--enc_in", str(config.channels), "--dec_in", str(config.channels),
        "--c_out", str(config.channels), "--d_model", str(preset["d_model"]),
        "--d_ff", str(preset["d_ff"]), "--n_heads", str(preset["n_heads"]),
        "--e_layers", str(preset["e_layers"]), "--d_layers", "1", "--factor", "3",
        "--dropout", "0.1", "--batch_size", str(preset["batch_size"]),
        "--learning_rate", str(preset["learning_rate"]), "--train_epochs", str(preset["epochs"]),
        "--patience", str(preset["patience"]), "--lradj", "type1",
        "--num_workers", "0", "--gpu", str(gpu), "--des", name, "--itr", "1",
        "--checkpoints", str(OUTPUT / "checkpoints"), "--test_after_train", "1",
    ]
    for key in ("patch_len", "stride", "top_k", "down_sampling_layers", "down_sampling_window"):
        if key in preset:
            command.extend(["--" + key, str(preset[key])])
    if "down_sampling_method" in preset:
        command.extend(["--down_sampling_method", str(preset["down_sampling_method"])])
    if model == "GraphMambaCMRHM":
        command.extend([
            "--d_state", "32", "--d_conv", "2", "--expand", "2",
            "--mamba_version", "1", "--mamba_bidirectional", "1",
            "--use_graph", "1", "--use_time_mamba", "1", "--use_patch", "1",
            "--use_decomp", "1", "--moving_avg", "25",
            "--dual_scale_scan_mode", "auto", "--periodic_period", "24",
            "--periodic_local_patch", "4", "--periodic_local_stride", "2",
            "--periodic_period_stride", "12", "--periodic_use_adapter", "1",
            "--graph_alpha", "0.5", "--graph_top_k", "2",
            "--graph_sample_size", "2000", "--graph_sample_method", "uniform",
            "--static_graph_mode", "weighted", "--graph_cache", "1",
        ])
    return command


def frozen_cmrhm_source(dataset: str, horizon: int) -> Path:
    """Return the protocol-unified, independent-shared CMRHM record."""
    return ROOT / "logs/cmrhm_unified_main/records" / (
        f"cmrhm_{dataset.lower()}_p{horizon}_s{SEED}.json"
    )


def reuse_frozen_cmrhm(dataset: str, horizon: int, destination: Path) -> bool:
    source = frozen_cmrhm_source(dataset, horizon)
    if not source.is_file():
        return False
    payload = json.loads(source.read_text(encoding="utf-8"))
    validation_mse = payload.get("validation_best_mse", payload.get("best_mse"))
    validation_mae = payload.get("validation_best_mae", payload.get("best_mae"))
    if (
        payload.get("status") != "completed"
        or payload.get("scan_mode") != "independent_shared"
        or validation_mse is None
        or validation_mae is None
        or "test_mse" not in payload
        or "test_mae" not in payload
    ):
        raise RuntimeError(f"invalid frozen CMRHM record: {source}")
    normalized = {
        "status": "completed", "model": "GraphMambaCMRHM", "dataset": dataset,
        "pred_len": horizon, "seq_len": 336, "seed": SEED,
        "scan_mode": "independent_shared",
        "validation_best_mse": validation_mse,
        "validation_best_mae": validation_mae,
        "test_mse": payload["test_mse"], "test_mae": payload["test_mae"],
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "reused_frozen_one_shot_record_no_new_test_read",
        "result_source": "reused_frozen_cmrhm",
        "source_record": str(source), "recorded_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(destination, normalized)
    print(f"REUSED frozen CMRHM: {dataset} {horizon}", flush=True)
    return True


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


def run_one(model: str, dataset: str, horizon: int, args: argparse.Namespace) -> int:
    destination = record_path(model, dataset, horizon)
    if model == "GraphMambaCMRHM" and not args.dry_run:
        if reuse_frozen_cmrhm(dataset, horizon, destination):
            return 0
    if args.resume and completed(destination):
        print(f"SKIP completed: {destination.name}", flush=True)
        return 0

    command = build_command(model, dataset, horizon, args.gpu)
    print("COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    log_path = OUTPUT / "logs" / f"{slug(model, dataset, horizon)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT / "heavy_dataset.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    return_code = 1
    timed_out = False
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if dataset == "solar":
            print(f"WAIT heavy-dataset lock: {model} {dataset} {horizon}", flush=True)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                try:
                    result = subprocess.run(
                        command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                        text=True, timeout=args.timeout_seconds, check=False,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
                    )
                    return_code = result.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    return_code = 124
        finally:
            if dataset == "solar":
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    validation, metrics, setting = parse_log(text)
    status = "completed" if return_code == 0 and validation and metrics else ("timeout" if timed_out else "failed")
    payload: dict[str, object] = {
        "status": status, "model": model, "dataset": dataset, "pred_len": horizon,
        "seq_len": 336, "seed": SEED, "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "one_shot_after_validation_selection",
        "result_source": "q2_same_protocol_local_run", "command": command,
        "setting": setting, "log_path": str(log_path),
    }
    if validation:
        payload.update({f"validation_{key}": value for key, value in validation.items()})
    if metrics:
        payload.update(metrics)
    atomic_json(destination, payload)
    print(f"FINISH {status}: {model} {dataset} {horizon} ({payload['duration_seconds']}s)", flush=True)
    return 0 if status == "completed" else return_code or 1


def write_worker_status(args: argparse.Namespace, completed_jobs: int, active: str | None, status: str) -> None:
    worker = "_".join(model.lower() for model in args.models)
    atomic_json(OUTPUT / f"status_{worker}.json", {
        "status": status, "models": args.models, "datasets": args.datasets,
        "horizons": args.horizons, "completed_jobs_this_invocation": completed_jobs,
        "active_or_last": active, "updated_at": datetime.now().astimezone().isoformat(),
    })


def main() -> int:
    args = parse_args()
    validate_data()
    jobs = [(m, d, h) for m in args.models for d in args.datasets for h in args.horizons]
    if args.max_jobs > 0:
        jobs = jobs[:args.max_jobs]
    done = 0
    for index, (model, dataset, horizon) in enumerate(jobs, start=1):
        active = slug(model, dataset, horizon)
        write_worker_status(args, done, active, "running")
        print(f"[{index}/{len(jobs)}] {active}", flush=True)
        code = run_one(model, dataset, horizon, args)
        if code != 0:
            write_worker_status(args, done, active, "failed")
            print(f"STOP after failure: inspect {record_path(model, dataset, horizon)}", flush=True)
            return code
        done += 1
    write_worker_status(args, done, None, "dry_run" if args.dry_run else "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
