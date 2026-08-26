#!/usr/bin/env python3
"""Run the frozen R4 all-ETT validation matrix and one-shot final tests.

The test firewall is deliberate: ``--stage test`` first requires all 48
validation-selected checkpoints.  A test intent record is written before each
test process starts, so an interrupted test is never silently repeated.
"""

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
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
OUTPUT = ROOT / "logs" / "graphmamba_recent_r4_ett"
SOURCE = ROOT / "logs" / "timerole_recent_simplification"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
REUSE_DATASETS = {"ETTh2", "ETTm1"}
REUSE_HORIZONS = {96, 720}
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
EVALUATION_PATTERN = re.compile(r"^EVALUATION_RESULT\s+(\{.*\})\s*$")

FROZEN_OPTIONS = {
    "--timerole_recent_len": "96",
    "--use_decomp": "0",
    "--use_patch": "1",
    "--dual_scale_scan_mode": "independent_shared",
    "--dual_scale_selection": "fine",
    "--use_time_mamba": "1",
    "--mamba_bidirectional": "1",
    "--use_graph": "0",
    "--graph_mamba_fusion": "fixed_sum",
    "--patch_len": "4",
    "--stride": "2",
    "--d_model": "64",
    "--d_ff": "128",
    "--d_state": "32",
    "--d_conv": "2",
    "--e_layers": "1",
    "--expand": "2",
    "--dropout": "0.1",
    "--batch_size": "32",
    "--learning_rate": "0.0005",
    "--lradj": "type1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("validation", "test", "full"), default="full"
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.datasets = list(dict.fromkeys(args.datasets))
    args.horizons = list(dict.fromkeys(args.horizons))
    args.seeds = list(dict.fromkeys(args.seeds))
    if min(args.epochs, args.patience) < 1 or args.max_jobs < 0:
        parser.error("epochs/patience must be positive and max-jobs non-negative")
    if args.stage in {"test", "full"} and (
        tuple(args.datasets) != DATASETS
        or tuple(args.horizons) != HORIZONS
        or tuple(args.seeds) != SEEDS
    ):
        parser.error("test/full is frozen to the complete 4 x 4 x 3 matrix")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate(dataset: str, horizon: int, seed: int) -> str:
    return f"r4_ett_{dataset.lower()}_p{horizon}_s{seed}"


def tasks(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    result = [
        (dataset, horizon, seed)
        for dataset in args.datasets
        for horizon in args.horizons
        for seed in args.seeds
    ]
    return result[: args.max_jobs or None]


def option(command: list[str], name: str) -> str:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing command option {name}") from error


def replace_option(command: list[str], name: str, value: object) -> None:
    command[command.index(name) + 1] = str(value)


def completed(path: Path, split: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("split") == split
        and isinstance(payload.get("mse"), (int, float))
        and isinstance(payload.get("mae"), (int, float))
    )


def validation_record_path(dataset: str, horizon: int, seed: int) -> Path:
    return OUTPUT / "validation_records" / f"{candidate(dataset, horizon, seed)}.json"


def test_record_path(dataset: str, horizon: int, seed: int) -> Path:
    return OUTPUT / "test_records" / f"{candidate(dataset, horizon, seed)}.json"


def is_reusable(dataset: str, horizon: int) -> bool:
    return dataset in REUSE_DATASETS and horizon in REUSE_HORIZONS


def source_record_path(dataset: str, horizon: int, seed: int) -> Path:
    stage = "phase_b" if seed == 2021 else "phase_c"
    name = f"trps_{stage}_recent_{dataset.lower()}_p{horizon}_s{seed}_r4"
    return SOURCE / "records" / stage / f"{name}.json"


def resolve_checkpoint(command: list[str]) -> Path:
    checkpoint_root = Path(option(command, "--checkpoints"))
    model_id = option(command, "--model_id")
    model = option(command, "--model")
    description = option(command, "--des")
    pattern = f"long_term_forecast_{model_id}_{model}_*_{description}_0/checkpoint.pth"
    matches = list(checkpoint_root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one checkpoint for {model_id}, found {len(matches)}"
        )
    return matches[0].resolve()


def assert_r4_command(command: list[str], dataset: str, horizon: int, seed: int) -> None:
    for name, expected in FROZEN_OPTIONS.items():
        actual = option(command, name)
        if actual != expected:
            raise RuntimeError(f"R4 mismatch {name}: {actual!r} != {expected!r}")
    expected_core = {
        "--data": dataset,
        "--data_path": f"{dataset}.csv",
        "--pred_len": str(horizon),
        "--seed": str(seed),
        "--seq_len": "336",
        "--features": "M",
        "--target": "OT",
        "--test_after_train": "0",
    }
    for name, expected in expected_core.items():
        if option(command, name) != expected:
            raise RuntimeError(f"config mismatch {name} for {dataset}-{horizon}-s{seed}")


def import_reused_validation(dataset: str, horizon: int, seed: int) -> None:
    destination = validation_record_path(dataset, horizon, seed)
    if completed(destination, "val"):
        return
    source_path = source_record_path(dataset, horizon, seed)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not (
        source.get("status") == "completed"
        and source.get("role") == "recent"
        and source.get("variant") == "R4"
        and source.get("split") == "val"
        and source.get("test_accessed") is False
    ):
        raise RuntimeError(f"invalid reusable validation record: {source_path}")
    command = list(source["command"])
    assert_r4_command(command, dataset, horizon, seed)
    checkpoint = resolve_checkpoint(command)
    payload = {
        "status": "completed",
        "split": "val",
        "test_accessed": False,
        "model": "GraphMambaRecent",
        "variant": "R4",
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "reused": True,
        "equivalent_explicit_model": "GraphMambaRecentR4",
        "source_validation_record": str(source_path.resolve()),
        "source_validation_record_sha256": sha256(source_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_selected_by": "validation_mse",
        "command": command,
        "best_epoch": source.get("best_epoch"),
        "mse": source["mse"],
        "mae": source["mae"],
        "parameter_count": source.get("parameter_count"),
        "recorded_at": now(),
    }
    atomic_json(destination, payload)


def build_train_command(
    dataset: str, horizon: int, seed: int, args: argparse.Namespace
) -> list[str]:
    name = candidate(dataset, horizon, seed)
    freq = "h" if dataset.startswith("ETTh") else "t"
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(ROOT / "dataset" / "ETT-small"),
        "--data_path", f"{dataset}.csv", "--model_id", name,
        "--model", "GraphMambaRecentR4", "--seed", str(seed),
        "--data", dataset, "--features", "M", "--target", "OT",
        "--freq", freq, "--seq_len", "336", "--label_len", "48",
        "--pred_len", str(horizon), "--enc_in", "7", "--dec_in", "7",
        "--c_out", "7", "--timerole_recent_len", "96",
        "--patch_len", "4", "--stride", "2", "--d_model", "64",
        "--d_ff", "128", "--d_state", "32", "--d_conv", "2",
        "--e_layers", "1", "--expand", "2", "--mamba_version", "1",
        "--mamba_bidirectional", "1", "--use_graph", "0",
        "--use_time_mamba", "1", "--use_patch", "1", "--use_decomp", "0",
        "--moving_avg", "25", "--dual_scale_scan_mode", "independent_shared",
        "--dual_scale_selection", "fine", "--graph_mamba_fusion", "fixed_sum",
        "--dropout", "0.1", "--batch_size", "32",
        "--learning_rate", "0.0005", "--lradj", "type1",
        "--train_epochs", str(args.epochs), "--patience", str(args.patience),
        "--num_workers", "0", "--augmentation_ratio", "0", "--gpu", "0",
        "--checkpoints", str(OUTPUT / "checkpoints"), "--des", name,
        "--itr", "1", "--test_after_train", "0",
    ]


def execute(
    command: list[str], log_path: Path, gpu: int
) -> tuple[int, dict[str, object] | None, dict[str, object] | None, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    validation = None
    evaluation = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
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
                evaluation = json.loads(match.group(1))
        return_code = process.wait()
    return return_code, validation, evaluation, time.monotonic() - started


def run_validation(
    dataset: str, horizon: int, seed: int, args: argparse.Namespace
) -> int:
    destination = validation_record_path(dataset, horizon, seed)
    if completed(destination, "val"):
        print(f"SKIP validation: {destination.name}", flush=True)
        return 0
    if destination.exists():
        raise RuntimeError(f"incomplete validation record requires manual audit: {destination}")
    if is_reusable(dataset, horizon):
        import_reused_validation(dataset, horizon, seed)
        print(f"REUSE validation: {destination.name}", flush=True)
        return 0

    command = build_train_command(dataset, horizon, seed, args)
    assert_r4_command(command, dataset, horizon, seed)
    if args.dry_run:
        print("TRAIN", shlex.join(command), flush=True)
        return 0
    log_path = OUTPUT / "raw_logs" / "validation" / f"{destination.stem}.log"
    code, validation, evaluation, duration = execute(command, log_path, args.gpu)
    checkpoint = resolve_checkpoint(command) if code == 0 else None
    integrity = bool(
        code == 0 and validation and evaluation
        and evaluation.get("split") == "val"
        and evaluation.get("test_accessed") is False
        and checkpoint and checkpoint.is_file()
    )
    payload: dict[str, object] = {
        "status": "completed" if integrity else "failed",
        "split": "val", "test_accessed": False,
        "model": "GraphMambaRecentR4", "variant": "R4",
        "dataset": dataset, "horizon": horizon, "seed": seed,
        "reused": False, "checkpoint_selected_by": "validation_mse",
        "return_code": code, "duration_seconds": round(duration, 3),
        "recorded_at": now(), "command": command, "log_path": str(log_path),
    }
    if validation:
        payload.update(validation)
    if evaluation:
        payload["validation_evaluation"] = evaluation
        payload.update({key: evaluation[key] for key in (
            "mse", "mae", "parameter_count", "milliseconds_per_batch",
            "peak_cuda_memory_bytes",
        ) if key in evaluation})
    if checkpoint:
        payload["checkpoint_path"] = str(checkpoint)
        payload["checkpoint_sha256"] = sha256(checkpoint)
    atomic_json(destination, payload)
    return 0 if integrity else 1


def load_complete_matrix() -> list[dict[str, object]]:
    records = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                path = validation_record_path(dataset, horizon, seed)
                if not completed(path, "val"):
                    raise RuntimeError(f"test firewall: missing validation record {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = Path(str(record["checkpoint_path"]))
                if not checkpoint.is_file() or sha256(checkpoint) != record["checkpoint_sha256"]:
                    raise RuntimeError(f"test firewall: checkpoint mismatch {checkpoint}")
                records.append(record)
    if len(records) != 48:
        raise RuntimeError(f"test firewall: expected 48 records, found {len(records)}")
    return records


def run_test(record: dict[str, object], args: argparse.Namespace) -> int:
    dataset = str(record["dataset"])
    horizon = int(record["horizon"])
    seed = int(record["seed"])
    destination = test_record_path(dataset, horizon, seed)
    if completed(destination, "test"):
        print(f"SKIP test: {destination.name}", flush=True)
        return 0
    if destination.exists():
        raise RuntimeError(f"one-shot firewall: test was already attempted: {destination}")
    command = list(record["command"])
    replace_option(command, "--is_training", 0)
    replace_option(command, "--test_after_train", 0)
    if "--evaluation_split" in command:
        replace_option(command, "--evaluation_split", "test")
    else:
        command.extend(["--evaluation_split", "test"])
    if args.dry_run:
        print("TEST ", shlex.join(command), flush=True)
        return 0

    checkpoint = Path(str(record["checkpoint_path"]))
    intent: dict[str, object] = {
        "status": "running", "split": "test", "test_accessed": True,
        "model": record["model"], "variant": "R4", "dataset": dataset,
        "horizon": horizon, "seed": seed, "one_shot": True,
        "checkpoint_selected_by": "validation_mse",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "source_validation_record": str(validation_record_path(dataset, horizon, seed)),
        "command": command, "started_at": now(),
    }
    atomic_json(destination, intent)
    log_path = OUTPUT / "raw_logs" / "test" / f"{destination.stem}.log"
    code, _, evaluation, duration = execute(command, log_path, args.gpu)
    integrity = bool(
        code == 0 and evaluation and evaluation.get("split") == "test"
        and evaluation.get("test_accessed") is True
    )
    intent.update({
        "status": "completed" if integrity else "failed",
        "return_code": code, "duration_seconds": round(duration, 3),
        "recorded_at": now(), "log_path": str(log_path),
    })
    if evaluation:
        intent["evaluation"] = evaluation
        intent.update({key: evaluation[key] for key in (
            "mse", "mae", "parameter_count", "milliseconds_per_batch",
            "peak_cuda_memory_bytes", "origin_metric_version", "origin_count",
            "origin_metrics_path",
        ) if key in evaluation})
    atomic_json(destination, intent)
    return 0 if integrity else 1


def write_status() -> None:
    validation = []
    tests = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                vp = validation_record_path(dataset, horizon, seed)
                tp = test_record_path(dataset, horizon, seed)
                if completed(vp, "val"):
                    validation.append(json.loads(vp.read_text(encoding="utf-8")))
                if completed(tp, "test"):
                    tests.append(json.loads(tp.read_text(encoding="utf-8")))
    atomic_json(OUTPUT / "status.json", {
        "variant": "R4", "expected_validation": 48,
        "completed_validation": len(validation), "reused_validation": sum(
            bool(row.get("reused")) for row in validation
        ), "expected_test": 48, "completed_test": len(tests),
        "updated_at": now(),
    })
    if tests:
        rows = [{
            "dataset": row["dataset"], "horizon": row["horizon"],
            "seed": row["seed"], "test_mse": row["mse"],
            "test_mae": row["mae"], "model": row["model"],
        } for row in tests]
        path = OUTPUT / "test_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_manifest(args: argparse.Namespace) -> None:
    atomic_json(OUTPUT / "manifests" / f"{args.stage}.json", {
        "protocol": "frozen GraphMambaRecent R4 all-ETT",
        "stage": args.stage, "datasets": args.datasets,
        "horizons": args.horizons, "seeds": args.seeds,
        "validation_checkpoint_selection": "best validation MSE",
        "test_policy": "one attempt per frozen checkpoint",
        "arguments": vars(args), "created_at": now(),
        "source_files_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                ROOT / "models" / "GraphMamba.py",
                ROOT / "models" / "GraphMambaRecent.py",
                ROOT / "models" / "GraphMambaRecentR4.py",
                ROOT / "exp" / "exp_long_term_forecasting.py",
                ROOT / "run.py", Path(__file__).resolve(),
            )
        },
    })


def main() -> int:
    args = parse_args()
    write_manifest(args)
    if args.stage in {"validation", "full"}:
        selected = tasks(args)
        for index, (dataset, horizon, seed) in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] VALIDATION {dataset}-{horizon}-s{seed}", flush=True)
            if run_validation(dataset, horizon, seed, args):
                write_status()
                return 1
            write_status()
    if args.stage in {"test", "full"}:
        matrix = load_complete_matrix()
        for index, record in enumerate(matrix, 1):
            print(
                f"[{index}/48] ONE-SHOT TEST {record['dataset']}-{record['horizon']}-s{record['seed']}",
                flush=True,
            )
            if run_test(record, args):
                write_status()
                return 1
            write_status()
    write_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
