#!/usr/bin/env python3
"""Compare native TimeXer-336 with TimeXer+DHC under a frozen protocol.

The validation stage trains every requested pair with test access disabled and
selects checkpoints by validation MSE.  The test stage refuses to start until
the complete requested validation/checkpoint matrix exists.
"""

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
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
DEFAULT_OUTPUT = ROOT / "logs" / "timexer336_dhc_multiseed"
ALL_DATASETS = ("ETTm1", "ETTm2")
ALL_HORIZONS = (96, 720)
ALL_SEEDS = (2021, 2022, 2023)
VARIANTS = (
    ("TimeXer", "timexer336"),
    ("TimeXerHistoryCorrection", "timexer_dhc"),
)
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
TEST_PATTERN = re.compile(r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:")

# Frozen from the repository's original TimeXer ETTm commands.  Only seq_len
# changes for native TimeXer; TimeXer+DHC internally retains a 96-point backbone.
PRESETS = {
    ("ETTm1", 96): {"d_model": 256, "d_ff": 2048, "batch_size": 4},
    ("ETTm1", 720): {"d_model": 256, "d_ff": 512, "batch_size": 4},
    ("ETTm2", 96): {"d_model": 256, "d_ff": 2048, "batch_size": 32},
    ("ETTm2", 720): {"d_model": 512, "d_ff": 2048, "batch_size": 32},
}


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated integers: {value}") from exc
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def parse_str_list(value: str) -> tuple[str, ...]:
    values = tuple(item for item in value.split(",") if item)
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "test", "both"), default="validation")
    parser.add_argument("--datasets", type=parse_str_list, default=ALL_DATASETS)
    parser.add_argument("--horizons", type=parse_int_list, default=ALL_HORIZONS)
    parser.add_argument("--seeds", type=parse_int_list, default=ALL_SEEDS)
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU exposed to the child process")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    unknown_datasets = set(args.datasets) - set(ALL_DATASETS)
    unknown_horizons = set(args.horizons) - set(ALL_HORIZONS)
    if unknown_datasets:
        parser.error(f"unsupported datasets: {sorted(unknown_datasets)}")
    if unknown_horizons:
        parser.error(f"unsupported horizons: {sorted(unknown_horizons)}")
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    args.output_dir = args.output_dir.resolve()
    return args


def candidate_name(dataset: str, horizon: int, label: str, seed: int) -> str:
    return f"{dataset.lower()}_p{horizon}_{label}_s{seed}"


def requested_tasks(args: argparse.Namespace):
    for dataset in args.datasets:
        for horizon in args.horizons:
            for seed in args.seeds:
                for model, label in VARIANTS:
                    yield dataset, horizon, seed, model, label


def build_validation_command(
    args: argparse.Namespace,
    dataset: str,
    horizon: int,
    seed: int,
    model: str,
    label: str,
) -> list[str]:
    preset = PRESETS[(dataset, horizon)]
    candidate = candidate_name(dataset, horizon, label, seed)
    return [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(ROOT / "dataset" / "ETT-small"),
        "--data_path",
        f"{dataset}.csv",
        "--data",
        dataset,
        "--model_id",
        f"{dataset}_336_{horizon}_{candidate}",
        "--model",
        model,
        "--seed",
        str(seed),
        "--features",
        "M",
        "--seq_len",
        "336",
        "--label_len",
        "48",
        "--pred_len",
        str(horizon),
        "--patch_len",
        "16",
        "--timerole_recent_len",
        "96",
        "--timerole_memory_pool",
        "16",
        "--timerole_hidden_dim",
        "32",
        "--e_layers",
        "1",
        "--factor",
        "3",
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
        "--d_model",
        str(preset["d_model"]),
        "--d_ff",
        str(preset["d_ff"]),
        "--batch_size",
        str(preset["batch_size"]),
        "--learning_rate",
        "0.0001",
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        "0",
        "--des",
        candidate,
        "--itr",
        "1",
        "--test_after_train",
        "0",
        "--checkpoints",
        str(args.output_dir / "checkpoints"),
    ]


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def setting_from_command(command: list[str]) -> str:
    return (
        f"{option(command, '--task_name')}_{option(command, '--model_id')}_"
        f"{option(command, '--model')}_{option(command, '--data')}_"
        f"ft{option(command, '--features')}_sl{option(command, '--seq_len')}_"
        f"ll{option(command, '--label_len')}_pl{option(command, '--pred_len')}_"
        f"dm{option(command, '--d_model')}_nh8_el{option(command, '--e_layers')}_"
        f"dl1_df{option(command, '--d_ff')}_expand2_dc4_fc{option(command, '--factor')}_"
        f"ebtimeF_dtTrue_{option(command, '--des')}_0"
    )


def checkpoint_path(command: list[str]) -> Path:
    return Path(option(command, "--checkpoints")) / setting_from_command(command) / "checkpoint.pth"


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def validation_record_path(args, dataset, horizon, label, seed) -> Path:
    return args.output_dir / "validation" / f"{candidate_name(dataset, horizon, label, seed)}.json"


def test_record_path(args, dataset, horizon, label, seed) -> Path:
    return args.output_dir / "test" / f"{candidate_name(dataset, horizon, label, seed)}.json"


def validation_complete(path: Path) -> bool:
    record = load_json(path)
    return bool(
        record
        and record.get("status") == "completed"
        and record.get("split") == "validation"
        and checkpoint_path(record["command"]).is_file()
    )


def test_complete(path: Path) -> bool:
    record = load_json(path)
    return bool(record and record.get("status") == "completed" and "test_mse" in record)


def run_process(command: list[str], log_path: Path, env: dict, pattern: re.Pattern):
    matched = None
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            match = pattern.match(line.strip())
            if match:
                matched = match.groups()
        return_code = process.wait()
    return return_code, matched, round(time.monotonic() - started, 3)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_validation(args: argparse.Namespace, env: dict) -> int:
    for dataset, horizon, seed, model, label in requested_tasks(args):
        data_path = ROOT / "dataset" / "ETT-small" / f"{dataset}.csv"
        if not data_path.is_file():
            raise FileNotFoundError(f"missing dataset: {data_path}")
        record_path = validation_record_path(args, dataset, horizon, label, seed)
        if validation_complete(record_path) and not args.force:
            print(f"SKIP completed validation: {record_path.name}")
            continue
        command = build_validation_command(args, dataset, horizon, seed, model, label)
        print("Command:", shlex.join(command), flush=True)
        if args.dry_run:
            continue
        candidate = candidate_name(dataset, horizon, label, seed)
        log_path = args.output_dir / "validation" / f"{candidate}.log"
        return_code, matched, duration = run_process(command, log_path, env, VALIDATION_PATTERN)
        validation = json.loads(matched[0]) if matched else None
        payload = {
            "status": "completed" if return_code == 0 and validation else "failed",
            "split": "validation",
            "dataset": dataset,
            "pred_len": horizon,
            "seed": seed,
            "model": model,
            "label": label,
            "candidate": candidate,
            "protocol": "timexer336_vs_timexer96_dhc240",
            "checkpoint_selected_by": "validation_best_mse",
            "checkpoint_path": str(checkpoint_path(command)),
            "return_code": return_code,
            "duration_seconds": duration,
            "recorded_at": datetime.now().astimezone().isoformat(),
            "command": command,
            "log_path": str(log_path),
        }
        if validation:
            payload.update(validation)
        atomic_write_json(record_path, payload)
        if payload["status"] != "completed":
            print(f"Validation failed: {candidate}", file=sys.stderr)
            return return_code or 1
    return 0


def ensure_complete_validation_matrix(args: argparse.Namespace) -> None:
    missing = []
    for dataset, horizon, seed, _, label in requested_tasks(args):
        path = validation_record_path(args, dataset, horizon, label, seed)
        if not validation_complete(path):
            missing.append(str(path))
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:12])
        raise RuntimeError(f"test stage blocked; incomplete validation matrix:\n{preview}")


def run_test(args: argparse.Namespace, env: dict) -> int:
    ensure_complete_validation_matrix(args)
    for dataset, horizon, seed, _, label in requested_tasks(args):
        destination = test_record_path(args, dataset, horizon, label, seed)
        if test_complete(destination) and not args.force:
            print(f"SKIP completed test: {destination.name}")
            continue
        source_path = validation_record_path(args, dataset, horizon, label, seed)
        source = load_json(source_path)
        assert source is not None
        command = list(source["command"])
        command[command.index("--is_training") + 1] = "0"
        command[command.index("--test_after_train") + 1] = "0"
        print("Command:", shlex.join(command), flush=True)
        if args.dry_run:
            continue
        candidate = candidate_name(dataset, horizon, label, seed)
        log_path = args.output_dir / "test" / f"{candidate}.log"
        return_code, matched, duration = run_process(command, log_path, env, TEST_PATTERN)
        metrics = None
        if matched:
            metrics = {"test_mse": float(matched[0]), "test_mae": float(matched[1])}
        payload = {
            "status": "completed" if return_code == 0 and metrics else "failed",
            "split": "test",
            "dataset": dataset,
            "pred_len": horizon,
            "seed": seed,
            "model": source["model"],
            "label": label,
            "candidate": candidate,
            "protocol": source["protocol"],
            "checkpoint_selected_by": "validation_best_mse",
            "validation_best_epoch": source["best_epoch"],
            "validation_best_mse": source["best_mse"],
            "validation_best_mae": source["best_mae"],
            "checkpoint_path": source["checkpoint_path"],
            "source_validation_record": str(source_path),
            "return_code": return_code,
            "duration_seconds": duration,
            "recorded_at": datetime.now().astimezone().isoformat(),
            "command": command,
            "log_path": str(log_path),
        }
        if metrics:
            payload.update(metrics)
        atomic_write_json(destination, payload)
        write_summaries(args)
        if payload["status"] != "completed":
            print(f"Test failed: {candidate}", file=sys.stderr)
            return return_code or 1
    return 0


def paired_rows(args: argparse.Namespace) -> list[dict]:
    rows = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            for seed in args.seeds:
                pair = {}
                for _, label in VARIANTS:
                    record = load_json(test_record_path(args, dataset, horizon, label, seed))
                    if record and record.get("status") == "completed":
                        pair[label] = record
                if len(pair) != len(VARIANTS):
                    continue
                baseline = pair["timexer336"]
                dhc = pair["timexer_dhc"]
                rows.append({
                    "dataset": dataset,
                    "pred_len": horizon,
                    "seed": seed,
                    "timexer336_mse": baseline["test_mse"],
                    "timexer_dhc_mse": dhc["test_mse"],
                    "mse_improvement_pct": 100.0 * (baseline["test_mse"] - dhc["test_mse"]) / baseline["test_mse"],
                    "timexer336_mae": baseline["test_mae"],
                    "timexer_dhc_mae": dhc["test_mae"],
                    "mae_improvement_pct": 100.0 * (baseline["test_mae"] - dhc["test_mae"]) / baseline["test_mae"],
                })
    return rows


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(args: argparse.Namespace) -> None:
    rows = paired_rows(args)
    write_csv(args.output_dir / "paired_test_results.csv", rows)
    aggregate = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            group = [row for row in rows if row["dataset"] == dataset and row["pred_len"] == horizon]
            if not group:
                continue
            entry = {"dataset": dataset, "pred_len": horizon, "n_seeds": len(group)}
            for field in (
                "timexer336_mse",
                "timexer_dhc_mse",
                "mse_improvement_pct",
                "timexer336_mae",
                "timexer_dhc_mae",
                "mae_improvement_pct",
            ):
                values = [float(row[field]) for row in group]
                entry[f"{field}_mean"] = statistics.fmean(values)
                entry[f"{field}_sd"] = sample_sd(values)
            aggregate.append(entry)
    write_csv(args.output_dir / "mean_std.csv", aggregate)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.stage in ("validation", "both"):
        result = run_validation(args, env)
        if result:
            return result
    if args.stage in ("test", "both"):
        result = run_test(args, env)
        if result:
            return result
    if not args.dry_run:
        write_summaries(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
