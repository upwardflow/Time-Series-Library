#!/usr/bin/env python3
"""Resumable, validation-only SCSD Phase-0 and Phase-1 experiment scheduler."""

from __future__ import annotations

import argparse
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
OUTPUT = ROOT / "logs" / "graphmamba_scsd_validation"
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
EVALUATION_PATTERN = re.compile(r"^EVALUATION_RESULT\s+(\{.*\})\s*$")
SEEDS = (2021, 2022, 2023)
HORIZONS = (96, 720)


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    data_path: str
    data: str
    channels: int
    target: str
    freq: str


DATASETS = {
    "ETTh1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh1.csv", "ETTh1", 7, "OT", "h"),
    "ETTm1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm1.csv", "ETTm1", 7, "OT", "t"),
    "ETTh2": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh2.csv", "ETTh2", 7, "OT", "h"),
    "weather": DatasetConfig(ROOT / "dataset/weather", "weather.csv", "custom", 21, "CO2 (ppm)", "t"),
}
PHASE1_DATASETS = ("ETTm1", "ETTh2", "weather")

VARIANTS = {
    "J": ("joint", "dual"),
    "IS": ("independent_shared", "dual"),
    "IU": ("independent_unshared", "dual"),
    "C": ("independent_shared", "coarse"),
    "F": ("independent_shared", "fine"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("phase0", "baseline", "all"), default="all")
    parser.add_argument("--datasets", nargs="+", choices=PHASE1_DATASETS, default=list(PHASE1_DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_incident(payload: dict[str, object]) -> None:
    path = OUTPUT / "incidents.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def source_state() -> dict[str, object]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    source_paths = (
        ROOT / "models/GraphMamba.py",
        ROOT / "run.py",
        ROOT / "exp/exp_long_term_forecasting.py",
        Path(__file__).resolve(),
    )
    return {
        "git_commit_sha": sha,
        "git_dirty": bool(status.strip()),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_files_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
    }


def slug(stage: str, dataset: str, horizon: int, seed: int, variant: str) -> str:
    return f"scsd_{stage}_{dataset.lower()}_p{horizon}_s{seed}_{variant.lower()}"


def paths(name: str) -> tuple[Path, Path]:
    return OUTPUT / "records" / f"{name}.json", OUTPUT / "raw_logs" / f"{name}.log"


def load_record(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parsed_config(command: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    index = 3
    while index < len(command):
        item = command[index]
        if item.startswith("--") and index + 1 < len(command):
            result[item[2:]] = command[index + 1]
            index += 2
        else:
            index += 1
    return result


def build_command(dataset: str, horizon: int, seed: int, variant: str, args: argparse.Namespace, name: str) -> list[str]:
    cfg = DATASETS[dataset]
    scan_mode, selection = VARIANTS[variant]
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(cfg.root), "--data_path", cfg.data_path,
        "--model_id", name, "--model", "GraphMamba", "--seed", str(seed),
        "--data", cfg.data, "--features", "M", "--target", cfg.target,
        "--freq", cfg.freq, "--seq_len", "96", "--label_len", "48",
        "--pred_len", str(horizon), "--enc_in", str(cfg.channels),
        "--dec_in", str(cfg.channels), "--c_out", str(cfg.channels),
        "--patch_len", "4", "--stride", "2", "--d_model", "64",
        "--d_ff", "128", "--d_state", "32", "--d_conv", "2",
        "--e_layers", "1", "--expand", "2", "--mamba_version", "1",
        "--mamba_bidirectional", "1", "--use_graph", "1",
        "--use_time_mamba", "1", "--use_patch", "1", "--use_decomp", "1",
        "--moving_avg", "25", "--dual_scale_scan_mode", scan_mode,
        "--dual_scale_selection", selection, "--graph_alpha", "0.5",
        "--graph_top_k", "2", "--graph_sample_size", "2000",
        "--graph_sample_method", "uniform", "--static_graph_mode", "weighted",
        "--graph_cache", "0", "--dropout", "0.1", "--batch_size", "32",
        "--learning_rate", "0.0005", "--lradj", "type1",
        "--train_epochs", str(args.epochs), "--patience", str(args.patience),
        "--num_workers", "0", "--gpu", "0", "--checkpoints", str(OUTPUT / "checkpoints"),
        "--des", name, "--itr", "1", "--test_after_train", "0",
    ]


def task_list(args: argparse.Namespace) -> list[tuple[str, str, int, int, str]]:
    tasks: list[tuple[str, str, int, int, str]] = []
    if args.stage in {"phase0", "all"}:
        for dataset in ("ETTh1", "ETTh2"):
            for variant in ("J", "IS"):
                tasks.append(("phase0", dataset, 192, 2021, variant))
    if args.stage in {"baseline", "all"}:
        for dataset in args.datasets:
            for horizon in args.horizons:
                for seed in args.seeds:
                    for variant in args.variants:
                        tasks.append(("baseline", dataset, horizon, seed, variant))
    return tasks


def run_one(task: tuple[str, str, int, int, str], args: argparse.Namespace) -> int:
    stage, dataset, horizon, seed, variant = task
    name = slug(stage, dataset, horizon, seed, variant)
    record_path, log_path = paths(name)
    previous = load_record(record_path)
    if previous and previous.get("status") == "completed":
        print(f"already completed: {name}", flush=True)
        return 0
    if previous and not args.retry_failed:
        print(f"failed/incomplete record requires --retry-failed: {record_path}", flush=True)
        return 1

    command = build_command(dataset, horizon, seed, variant, args, name)
    print("Command:", shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    record_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 1 + int(previous.get("attempt", 0) if previous else 0)
    if previous:
        append_incident({
            "recorded_at": now(), "candidate": name, "event": "explicit_retry",
            "attempt": attempt, "previous_status": previous.get("status"),
            "previous_return_code": previous.get("return_code"),
        })

    validation = None
    evaluation = None
    started = time.monotonic()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with log_path.open("a" if previous else "w", encoding="utf-8") as handle:
        if previous:
            handle.write(f"\n===== RETRY attempt={attempt} at={now()} =====\n")
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            match = VALIDATION_PATTERN.match(line.strip())
            if match:
                validation = json.loads(match.group(1))
            match = EVALUATION_PATTERN.match(line.strip())
            if match:
                candidate_evaluation = json.loads(match.group(1))
                if candidate_evaluation.get("split") == "val":
                    evaluation = candidate_evaluation
        return_code = process.wait()

    status = "completed" if return_code == 0 and validation and evaluation else "failed"
    payload: dict[str, object] = {
        "status": status, "attempt": attempt, "stage": stage,
        "candidate": name, "dataset": dataset, "horizon": horizon,
        "seed": seed, "variant": variant, "scan_mode": VARIANTS[variant][0],
        "scale_selection": VARIANTS[variant][1], "split": "val",
        "test_accessed": False, "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": now(), "command": command,
        "resolved_config": parsed_config(command), "log_path": str(log_path),
        "retry_provenance": [] if not previous else [str(record_path)],
        **source_state(),
    }
    if validation:
        payload.update(validation)
        payload["train_duration_seconds"] = validation.get("train_duration_seconds")
        payload["train_peak_cuda_memory_bytes"] = validation.get("train_peak_cuda_memory_bytes")
    if evaluation:
        payload.update({
            "validation_evaluation": evaluation,
            "parameter_count": evaluation.get("parameter_count"),
            "inference_elapsed_seconds": evaluation.get("elapsed_seconds"),
            "inference_milliseconds_per_batch": evaluation.get("milliseconds_per_batch"),
            "peak_cuda_memory_bytes": evaluation.get("peak_cuda_memory_bytes"),
        })
    atomic_json(record_path, payload)
    if status != "completed":
        append_incident({
            "recorded_at": now(), "candidate": name, "event": "run_failed",
            "attempt": attempt, "return_code": return_code,
            "validation_parsed": validation is not None,
            "evaluation_parsed": evaluation is not None,
            "log_path": str(log_path),
        })
        return return_code or 1
    return 0


def write_manifest(args: argparse.Namespace, tasks: list[tuple[str, str, int, int, str]]) -> None:
    payload = {
        "created_at": now(), "validation_only": True, "test_accessed": False,
        "output_root": str(OUTPUT), "arguments": vars(args),
        "tasks": [dict(zip(("stage", "dataset", "horizon", "seed", "variant"), task)) for task in tasks],
        **source_state(),
    }
    atomic_json(OUTPUT / "manifest.json", payload)


def main() -> int:
    args = parse_args()
    tasks = task_list(args)
    if args.max_jobs > 0:
        tasks = tasks[:args.max_jobs]
    write_manifest(args, tasks)
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task}", flush=True)
        if run_one(task, args):
            print("Stopping after recorded failure; no silent retry.", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
