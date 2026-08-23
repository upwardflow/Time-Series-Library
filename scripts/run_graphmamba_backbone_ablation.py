#!/usr/bin/env python3
"""Train and validate one-factor recent-backbone ablations.

The script is deliberately validation-only.  Each job trains from a fresh random
initialization, preserves the 336-point data window / recent-96 backbone input,
and evaluates the selected checkpoint over the complete validation split. It
supports both the Recent-only control and the TimeRole-active complete model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import selectors
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
DEFAULT_OUTPUTS = {
    "GraphMambaRecent": ROOT / "logs" / "graphmamba_backbone_ablation",
    "TimeRole": ROOT / "logs" / "timerole_backbone_ablation",
    "TimeRoleAGF": ROOT / "logs" / "timerole_adaptive_fusion",
}
DATASETS = ("ETTm1", "ETTm2")
HORIZONS = (96, 720)
VARIANT_ORDER = ("no_decomp", "no_patch", "uni_mamba", "no_mamba", "no_graph")
VARIANTS = {
    "full": {"label": "Adaptive Graph-Mamba Fusion"},
    "no_decomp": {"label": "w/o Decomp", "use_decomp": 0},
    "no_patch": {"label": "w/o Patch", "use_patch": 0},
    "uni_mamba": {"label": "Uni-Mamba", "mamba_bidirectional": 0},
    "no_mamba": {"label": "w/o Mamba", "use_time_mamba": 0},
    "no_graph": {"label": "w/o Graph", "use_graph": 0},
}
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
EVALUATION_PATTERN = re.compile(r"^EVALUATION_RESULT\s+(\{.*\})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--model",
        choices=("GraphMambaRecent", "TimeRole", "TimeRoleAGF"),
        default="GraphMambaRecent",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANT_ORDER))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("timeout, epochs, and patience must be positive")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUTS[args.model]
    args.output_dir = args.output_dir.resolve()
    return args


def replace(command: list[str], option: str, value: object) -> None:
    text = str(value)
    if option in command:
        command[command.index(option) + 1] = text
    else:
        command.extend((option, text))


def candidate(
    dataset: str, horizon: int, variant: str, seed: int, model: str
) -> str:
    prefix = {
        "GraphMambaRecent": "table2",
        "TimeRole": "timerole_backbone",
        "TimeRoleAGF": "timerole_agf",
    }[model]
    return f"{prefix}_{dataset.lower()}_p{horizon}_{variant}_s{seed}"


def build_train_command(
    dataset: str, horizon: int, variant: str, args: argparse.Namespace
) -> list[str]:
    name = candidate(dataset, horizon, variant, args.seed, args.model)
    flags = {
        "mamba_bidirectional": 1,
        "use_graph": 1,
        "use_time_mamba": 1,
        "use_patch": 1,
        "use_decomp": 1,
    }
    flags.update({key: value for key, value in VARIANTS[variant].items() if key != "label"})
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(ROOT / "dataset" / "ETT-small"),
        "--data_path", f"{dataset}.csv",
        "--model_id", f"{dataset}_96_{horizon}_{name}",
        "--model", args.model, "--seed", str(args.seed),
        "--data", dataset, "--features", "M", "--target", "OT",
        "--seq_len", "336", "--label_len", "48", "--pred_len", str(horizon),
        "--enc_in", "7", "--dec_in", "7", "--c_out", "7",
        "--patch_len", "4", "--stride", "2",
        "--d_model", "64", "--d_ff", "128", "--d_state", "32",
        "--d_conv", "2", "--e_layers", "1", "--expand", "2",
        "--mamba_version", "1",
        "--mamba_bidirectional", str(flags["mamba_bidirectional"]),
        "--use_graph", str(flags["use_graph"]),
        "--use_time_mamba", str(flags["use_time_mamba"]),
        "--use_patch", str(flags["use_patch"]),
        "--use_decomp", str(flags["use_decomp"]), "--moving_avg", "25",
        "--dual_scale_scan_mode", "independent_shared",
        "--periodic_period", "24", "--periodic_local_patch", "0",
        "--periodic_local_stride", "0", "--periodic_period_stride", "12",
        "--periodic_use_adapter", "1",
        "--graph_alpha", "0.5", "--graph_top_k", "2",
        "--graph_sample_size", "2000", "--graph_sample_method", "uniform",
        "--static_graph_mode", "weighted", "--graph_cache", "0",
        "--gc_graph_dim", "16", "--gc_temperature", "1.0",
        "--gc_residual_init", "0.5", "--gc_dynamic_graph", "1",
        "--gc_symmetric_graph", "1", "--gc_input_modulation", "1",
        "--gc_direction_fusion", "1", "--gc_parallel_residual", "1",
        "--af_hidden_dim", "32", "--af_rank", "16",
        "--af_mode", "variable_scale_residual",
        "--dropout", "0.1", "--batch_size", "32",
        "--learning_rate", "0.0005", "--lradj", "type1",
        "--train_epochs", str(args.epochs), "--patience", str(args.patience),
        "--num_workers", "0", "--gpu", str(args.gpu),
        "--checkpoints", str(args.output_dir / "checkpoints"),
        "--des", name, "--itr", "1", "--test_after_train", "0",
        "--timerole_old_intervention", "intact",
    ]


def record_path(args: argparse.Namespace, dataset: str, horizon: int, variant: str) -> Path:
    name = candidate(dataset, horizon, variant, args.seed, args.model)
    return args.output_dir / "records" / f"{name}.json"


def train_record_path(args: argparse.Namespace, dataset: str, horizon: int, variant: str) -> Path:
    name = candidate(dataset, horizon, variant, args.seed, args.model)
    return args.output_dir / "training" / f"{name}.json"


def completed(path: Path, metric: str = "mse") -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and metric in payload


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute(
    command: list[str], log_path: Path, pattern: re.Pattern[str],
    gpu: int, timeout_seconds: int,
) -> tuple[int, dict[str, object] | None, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = started + timeout_seconds
        while True:
            for key, _ in selector.select(timeout=1.0):
                line = key.fileobj.readline()
                if line:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                    match = pattern.match(line.strip())
                    if match:
                        result = json.loads(match.group(1))
            if process.poll() is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    match = pattern.match(line.strip())
                    if match:
                        result = json.loads(match.group(1))
                return_code = process.returncode
                break
            if time.monotonic() >= deadline:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return_code = 124
                break
        selector.close()
    return return_code, result, time.monotonic() - started


def run_one(dataset: str, horizon: int, variant: str, args: argparse.Namespace) -> int:
    final_path = record_path(args, dataset, horizon, variant)
    if completed(final_path):
        print(f"SKIP completed: {final_path}", flush=True)
        return 0

    name = candidate(dataset, horizon, variant, args.seed, args.model)
    train_path = train_record_path(args, dataset, horizon, variant)
    train_command = build_train_command(dataset, horizon, variant, args)
    if args.dry_run:
        print("TRAIN", shlex.join(train_command))
        eval_command = list(train_command)
        replace(eval_command, "--is_training", 0)
        replace(eval_command, "--evaluation_split", "val")
        print("EVAL ", shlex.join(eval_command))
        return 0

    if not completed(train_path, "best_mse"):
        log_path = args.output_dir / "logs" / f"{name}.train.log"
        code, validation, duration = execute(
            train_command, log_path, VALIDATION_PATTERN, args.gpu, args.timeout_seconds
        )
        payload: dict[str, object] = {
            "status": "completed" if code == 0 and validation else "failed",
            "stage": "training", "model": args.model,
            "dataset": dataset, "horizon": horizon,
            "variant": variant, "variant_label": VARIANTS[variant]["label"],
            "seed": args.seed, "split": "validation", "test_accessed": False,
            "return_code": code, "duration_seconds": round(duration, 3),
            "recorded_at": datetime.now().astimezone().isoformat(),
            "command": train_command, "log_path": str(log_path),
        }
        if validation:
            payload.update(validation)
        atomic_write(train_path, payload)
        if payload["status"] != "completed":
            print(f"FAILED training: {name}; no automatic retry", flush=True)
            return 1
    else:
        payload = json.loads(train_path.read_text(encoding="utf-8"))
        train_command = list(payload["command"])

    eval_command = list(train_command)
    replace(eval_command, "--is_training", 0)
    replace(eval_command, "--test_after_train", 0)
    replace(eval_command, "--evaluation_split", "val")
    eval_log = args.output_dir / "logs" / f"{name}.val.log"
    code, evaluation, duration = execute(
        eval_command, eval_log, EVALUATION_PATTERN, args.gpu, args.timeout_seconds
    )
    final_payload: dict[str, object] = {
        "status": "completed" if code == 0 and evaluation else "failed",
        "stage": "checkpoint_validation", "model": args.model,
        "dataset": dataset, "horizon": horizon,
        "variant": variant, "variant_label": VARIANTS[variant]["label"],
        "seed": args.seed, "split": "val", "test_accessed": False,
        "training_record": str(train_path), "return_code": code,
        "duration_seconds": round(duration, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "command": eval_command, "log_path": str(eval_log),
    }
    if evaluation:
        final_payload.update(evaluation)
    atomic_write(final_path, final_payload)
    if final_payload["status"] != "completed":
        print(f"FAILED evaluation: {name}; no automatic retry", flush=True)
        return 1
    return 0


def summarize(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    for variant in args.variants:
        for dataset in args.datasets:
            for horizon in args.horizons:
                path = record_path(args, dataset, horizon, variant)
                if not completed(path):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "model": args.model, "variant": variant,
                    "variant_label": VARIANTS[variant]["label"],
                    "dataset": dataset, "horizon": horizon, "seed": args.seed,
                    "validation_mse": payload["mse"], "validation_mae": payload["mae"],
                    "parameter_count": payload["parameter_count"],
                    "record": str(path),
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    expected = len(args.datasets) * len(args.horizons) * len(args.variants)
    atomic_write(args.output_dir / "status.json", {
        "expected": expected, "completed": len(rows),
        "failed": expected - len(rows), "split": "validation",
        "test_accessed": False, "seed": args.seed, "model": args.model,
        "updated_at": datetime.now().astimezone().isoformat(),
    })
    return len(rows)


def main() -> int:
    args = parse_args()
    failures = []
    for variant in args.variants:
        for dataset in args.datasets:
            for horizon in args.horizons:
                print(
                    f"=== {VARIANTS[variant]['label']} | {dataset}-{horizon} ===",
                    flush=True,
                )
                if run_one(dataset, horizon, variant, args):
                    failures.append(f"{variant}:{dataset}-{horizon}")
                    break
            if failures:
                break
        if failures:
            break
    completed_count = summarize(args) if not args.dry_run else 0
    print(json.dumps({"completed": completed_count, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
