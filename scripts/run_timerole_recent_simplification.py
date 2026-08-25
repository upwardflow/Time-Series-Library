#!/usr/bin/env python3
"""Run gated, validation-only TimeRole recent-predictor simplification jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
OUTPUT = ROOT / "logs" / "timerole_recent_simplification"
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


@dataclass(frozen=True)
class CandidateConfig:
    use_decomp: int
    scale: str
    bidirectional: int
    use_graph: int
    use_mamba: int


DATASETS = {
    "ETTm1": DatasetConfig(ROOT / "dataset/ETT-small", "ETTm1.csv", "ETTm1", 7, "OT", "t"),
    "ETTh2": DatasetConfig(ROOT / "dataset/ETT-small", "ETTh2.csv", "ETTh2", 7, "OT", "h"),
    "weather": DatasetConfig(ROOT / "dataset/weather", "weather.csv", "custom", 21, "CO2 (ppm)", "t"),
}
CANDIDATES = {
    "R0": CandidateConfig(1, "dual", 1, 1, 1),
    "R1": CandidateConfig(0, "fine", 1, 1, 1),
    "R2": CandidateConfig(0, "coarse", 1, 1, 1),
    "R3": CandidateConfig(0, "fine", 0, 1, 1),
    "R4": CandidateConfig(0, "fine", 1, 0, 1),
    "R5": CandidateConfig(0, "fine", 1, 1, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "phase_b"), default="smoke")
    parser.add_argument("--role", choices=("timerole", "recent"), default="timerole")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    parser.add_argument("--variants", nargs="+", choices=tuple(CANDIDATES))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--check-interval-seconds", type=int, default=30)
    parser.add_argument("--stall-seconds", type=int, default=90)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.timeout_seconds, args.check_interval_seconds) < 1:
        parser.error("epochs, patience, timeout, and check interval must be positive")
    if args.stall_seconds < args.check_interval_seconds:
        parser.error("stall-seconds must be at least check-interval-seconds")
    return args


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_incident(payload: dict[str, object]) -> None:
    path = OUTPUT / "incidents" / "incidents.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_state() -> dict[str, object]:
    source_paths = (
        ROOT / "data_provider/data_factory.py",
        ROOT / "models/GraphMamba.py",
        ROOT / "models/GraphMambaRecent.py",
        ROOT / "models/TimeRole.py",
        ROOT / "exp/exp_long_term_forecasting.py",
        ROOT / "run.py",
        Path(__file__).resolve(),
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    )
    full_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    return {
        "git_commit_sha": sha,
        "git_dirty": bool(full_status.strip()),
        "source_dirty": bool(tracked_status.strip()),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_files_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths
        },
    }


def environment_fingerprint() -> dict[str, object]:
    import torch
    try:
        import mamba_ssm
        mamba_version = getattr(mamba_ssm, "__version__", "unknown")
    except Exception as error:  # pragma: no cover - diagnostic only
        mamba_version = f"unavailable:{type(error).__name__}"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
    except Exception as error:  # pragma: no cover - diagnostic only
        driver = [f"unavailable:{type(error).__name__}"]
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mamba_ssm": mamba_version,
        "gpu_driver": driver,
    }


def active_components(variant: str, role: str) -> dict[str, object]:
    cfg = CANDIDATES[variant]
    return {
        "decomposition": bool(cfg.use_decomp),
        "scale": cfg.scale,
        "mamba": bool(cfg.use_mamba),
        "mamba_bidirectional": bool(cfg.bidirectional and cfg.use_mamba),
        "graph": bool(cfg.use_graph),
        "timerole": role == "timerole",
    }


def slug(stage: str, role: str, dataset: str, horizon: int, seed: int, variant: str) -> str:
    return f"trps_{stage}_{role}_{dataset.lower()}_p{horizon}_s{seed}_{variant.lower()}"


def parsed_config(command: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if item.startswith("--") and index + 1 < len(command):
            result[item[2:]] = command[index + 1]
            index += 2
        else:
            index += 1
    return result


def build_command(
    dataset: str, horizon: int, seed: int, variant: str,
    role: str, args: argparse.Namespace, name: str,
) -> list[str]:
    data = DATASETS[dataset]
    candidate = CANDIDATES[variant]
    model = "TimeRole" if role == "timerole" else "GraphMambaRecent"
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(data.root), "--data_path", data.data_path,
        "--model_id", name, "--model", model, "--seed", str(seed),
        "--data", data.data, "--features", "M", "--target", data.target,
        "--freq", data.freq, "--seq_len", "336", "--label_len", "48",
        "--pred_len", str(horizon), "--enc_in", str(data.channels),
        "--dec_in", str(data.channels), "--c_out", str(data.channels),
        "--timerole_recent_len", "96", "--timerole_hidden_dim", "32",
        "--timerole_memory_pool", "16", "--timerole_old_intervention", "intact",
        "--patch_len", "4", "--stride", "2", "--d_model", "64",
        "--d_ff", "128", "--d_state", "32", "--d_conv", "2",
        "--e_layers", "1", "--expand", "2", "--mamba_version", "1",
        "--mamba_bidirectional", str(candidate.bidirectional),
        "--use_graph", str(candidate.use_graph),
        "--use_time_mamba", str(candidate.use_mamba),
        "--use_patch", "1", "--use_decomp", str(candidate.use_decomp),
        "--moving_avg", "25", "--dual_scale_scan_mode", "independent_shared",
        "--dual_scale_selection", candidate.scale,
        "--graph_mamba_fusion", "fixed_sum", "--graph_alpha", "0.5",
        "--graph_top_k", "2", "--graph_sample_size", "2000",
        "--graph_sample_method", "uniform", "--static_graph_mode", "weighted",
        "--graph_cache", "0", "--gc_graph_dim", "16",
        "--gc_temperature", "1.0", "--gc_residual_init", "0.5",
        "--gc_dynamic_graph", "1", "--gc_symmetric_graph", "1",
        "--gc_input_modulation", "1", "--gc_direction_fusion", "1",
        "--gc_parallel_residual", "1", "--dropout", "0.1",
        "--batch_size", "32", "--learning_rate", "0.0005",
        "--lradj", "type1", "--train_epochs", str(args.epochs),
        "--patience", str(args.patience), "--num_workers", "0",
        "--augmentation_ratio", "0", "--gpu", "0",
        "--checkpoints", str(OUTPUT / "checkpoints"),
        "--des", name, "--itr", "1", "--test_after_train", "0",
    ]


def tasks(args: argparse.Namespace) -> list[tuple[str, str, str, int, int, str]]:
    if args.stage == "smoke":
        datasets = args.datasets or ["ETTm1"]
        horizons = args.horizons or [96]
        seeds = args.seeds or [2021]
        variants = args.variants or list(CANDIDATES)
    else:
        datasets = args.datasets or list(DATASETS)
        horizons = args.horizons or list(HORIZONS)
        seeds = args.seeds or [2021]
        variants = args.variants or list(CANDIDATES)
    result = [
        (args.stage, args.role, dataset, horizon, seed, variant)
        for dataset in datasets for horizon in horizons
        for seed in seeds for variant in variants
    ]
    if args.max_jobs > 0:
        result = result[:args.max_jobs]
    return result


def load_record(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def execute(
    command: list[str], log_path: Path, args: argparse.Namespace,
) -> tuple[int, dict[str, object] | None, dict[str, object] | None, dict[str, object]]:
    validation = None
    evaluation = None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_output = started
    stall_warned = False
    resource_warned = False
    timeout_hit = False
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        initial_rss = rss_bytes(process.pid)
        resource_baseline_rss = initial_rss
        resource_warmup_seconds = max(30, args.check_interval_seconds)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                events = selector.select(timeout=args.check_interval_seconds)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                    last_output = time.monotonic()
                    match = VALIDATION_PATTERN.match(line.strip())
                    if match:
                        validation = json.loads(match.group(1))
                    match = EVALUATION_PATTERN.match(line.strip())
                    if match:
                        candidate_eval = json.loads(match.group(1))
                        if candidate_eval.get("split") == "val":
                            evaluation = candidate_eval

                elapsed = time.monotonic() - started
                current_rss = rss_bytes(process.pid)
                if current_rss and elapsed <= resource_warmup_seconds:
                    resource_baseline_rss = max(resource_baseline_rss or 0, current_rss)
                if (
                    elapsed > resource_warmup_seconds
                    and resource_baseline_rss and current_rss
                    and current_rss > 3 * resource_baseline_rss
                    and not resource_warned
                ):
                    message = (
                        "[MONITOR RESOURCE_ALERT] "
                        f"rss={current_rss} baseline={resource_baseline_rss}\n"
                    )
                    print(message, end="", flush=True)
                    handle.write(message)
                    handle.flush()
                    resource_warned = True
                if time.monotonic() - last_output >= args.stall_seconds and not stall_warned:
                    message = f"[MONITOR OUTPUT_STALL advisory] no output for {args.stall_seconds}s\n"
                    print(message, end="", flush=True)
                    handle.write(message)
                    handle.flush()
                    stall_warned = True
                if time.monotonic() - last_output < args.stall_seconds:
                    stall_warned = False
                if elapsed >= args.timeout_seconds:
                    message = f"[MONITOR HARD_TIMEOUT] elapsed={elapsed:.1f}s\n"
                    print(message, end="", flush=True)
                    handle.write(message)
                    handle.flush()
                    timeout_hit = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                if process.poll() is not None:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        handle.write(line)
                        match = VALIDATION_PATTERN.match(line.strip())
                        if match:
                            validation = json.loads(match.group(1))
                        match = EVALUATION_PATTERN.match(line.strip())
                        if match:
                            candidate_eval = json.loads(match.group(1))
                            if candidate_eval.get("split") == "val":
                                evaluation = candidate_eval
                    break
        finally:
            selector.close()
    return_code = 124 if timeout_hit else int(process.returncode or 0)
    monitor = {
        "pid": process.pid,
        "initial_rss_bytes": initial_rss,
        "resource_baseline_rss_bytes": resource_baseline_rss,
        "resource_warmup_seconds": resource_warmup_seconds,
        "duration_seconds": round(time.monotonic() - started, 3),
        "output_stall_advisory": stall_warned,
        "resource_alert": resource_warned,
        "hard_timeout": timeout_hit,
        "timeout_seconds": args.timeout_seconds,
        "check_interval_seconds": args.check_interval_seconds,
    }
    return return_code, validation, evaluation, monitor


def run_one(task: tuple[str, str, str, int, int, str], args: argparse.Namespace) -> int:
    stage, role, dataset, horizon, seed, variant = task
    name = slug(stage, role, dataset, horizon, seed, variant)
    record_path = OUTPUT / "records" / stage / f"{name}.json"
    log_path = OUTPUT / "raw_logs" / stage / f"{name}.log"
    previous = load_record(record_path)
    if previous and previous.get("status") == "completed":
        print(f"already completed: {name}", flush=True)
        return 0
    if previous and not args.retry_failed:
        print(f"failed/incomplete record requires --retry-failed: {record_path}", flush=True)
        return 1

    command = build_command(dataset, horizon, seed, variant, role, args, name)
    print("Command:", shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    attempt = 1 + int(previous.get("attempt", 0) if previous else 0)
    if previous:
        append_incident({
            "recorded_at": now(), "candidate": name, "event": "explicit_retry",
            "attempt": attempt, "previous_status": previous.get("status"),
            "previous_return_code": previous.get("return_code"),
        })

    return_code, validation, evaluation, monitor = execute(command, log_path, args)
    integrity_ok = bool(
        validation and evaluation
        and evaluation.get("split") == "val"
        and evaluation.get("test_accessed") is False
    )
    status = "completed" if return_code == 0 and integrity_ok else (
        "timeout" if monitor["hard_timeout"] else "failed"
    )
    payload: dict[str, object] = {
        "status": status, "attempt": attempt, "stage": stage,
        "candidate": name, "role": role, "dataset": dataset,
        "horizon": horizon, "seed": seed, "variant": variant,
        "active_components": active_components(variant, role),
        "scan_mode": "independent_shared", "split": "val",
        "test_accessed": False, "return_code": return_code,
        "recorded_at": now(), "command": command,
        "resolved_config": parsed_config(command), "log_path": str(log_path),
        "monitor": monitor, "data_order_seed": seed,
        "validation_shuffle": False,
        "retry_provenance": [] if not previous else [str(record_path)],
        **source_state(),
    }
    if validation:
        payload.update(validation)
    if evaluation:
        payload["validation_evaluation"] = evaluation
        for key in (
            "mse", "mae", "parameter_count", "milliseconds_per_batch",
            "peak_cuda_memory_bytes", "memory_correction_mae",
            "memory_correction_rms", "gate_mean", "gate_abs_mean",
            "gate_min", "gate_max", "gate_values", "fusion_gate_mean",
            "fusion_gate_std", "fusion_gate_min", "fusion_gate_max",
        ):
            if key in evaluation:
                payload[key] = evaluation[key]
    atomic_json(record_path, payload)
    if status != "completed":
        append_incident({
            "recorded_at": now(), "candidate": name, "event": "run_failed",
            "attempt": attempt, "return_code": return_code,
            "validation_parsed": validation is not None,
            "evaluation_parsed": evaluation is not None,
            "monitor": monitor, "log_path": str(log_path),
        })
        return return_code or 1
    return 0


def write_manifest(args: argparse.Namespace, task_list: list[tuple[str, str, str, int, int, str]]) -> None:
    payload = {
        "created_at": now(), "protocol": "TimeRole recent-predictor simplification",
        "stage": args.stage, "validation_only": True, "test_accessed": False,
        "scan_mode": "independent_shared", "data_order_isolated": True,
        "validation_shuffle": False, "output_root": str(OUTPUT),
        "arguments": vars(args), "environment": environment_fingerprint(),
        "tasks": [
            dict(zip(("stage", "role", "dataset", "horizon", "seed", "variant"), task))
            for task in task_list
        ],
        **source_state(),
    }
    atomic_json(OUTPUT / "manifests" / f"{args.stage}_{args.role}.json", payload)


def main() -> int:
    args = parse_args()
    task_list = tasks(args)
    write_manifest(args, task_list)
    for index, task in enumerate(task_list, 1):
        print(f"[{index}/{len(task_list)}] {task}", flush=True)
        if run_one(task, args):
            print("Stopping after recorded failure; no automatic retry.", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
