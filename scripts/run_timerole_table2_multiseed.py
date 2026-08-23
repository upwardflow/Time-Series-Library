#!/usr/bin/env python3
"""Complete the three-seed TimeRole block for Table 2 without test reuse.

The 2021 Table-2 commands are the protocol source of truth. Existing frozen
test records are copied with provenance; only missing seed/dataset/horizon
combinations are trained and evaluated. The runner is serial, resumable, and
safe to keep inside a tmux session.
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
OUTPUT = ROOT / "logs" / "timerole_table2_multiseed"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)

VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$", re.MULTILINE)
TEST_PATTERN = re.compile(
    r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:", re.MULTILINE
)
SETTING_PATTERN = re.compile(r"^>+start training : (.*?)>+$", re.MULTILINE)

EXPECTED_FLAGS = {
    "--seq_len": "336",
    "--d_model": "64",
    "--d_ff": "128",
    "--e_layers": "1",
    "--d_state": "32",
    "--d_conv": "2",
    "--expand": "2",
    "--mamba_version": "1",
    "--mamba_bidirectional": "1",
    "--use_graph": "1",
    "--use_time_mamba": "1",
    "--use_patch": "1",
    "--use_decomp": "1",
    "--moving_avg": "25",
    "--patch_len": "4",
    "--stride": "2",
    "--dual_scale_scan_mode": "independent_shared",
    "--graph_alpha": "0.5",
    "--graph_top_k": "2",
    "--graph_sample_size": "2000",
    "--graph_sample_method": "uniform",
    "--static_graph_mode": "weighted",
    "--learning_rate": "0.0005",
    "--train_epochs": "100",
    "--patience": "6",
    "--batch_size": "32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    return parser.parse_args()


def slug(dataset: str, horizon: int, seed: int) -> str:
    return f"timerole_{dataset.lower()}_sl336_pl{horizon}_s{seed}"


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
        isinstance(payload, dict)
        and payload.get("status") == "completed"
        and payload.get("test_mse") is not None
        and payload.get("test_mae") is not None
    )


def flag_value(command: list[str], flag: str) -> str | None:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def set_flag(command: list[str], flag: str, value: object) -> None:
    text = str(value)
    try:
        index = command.index(flag)
    except ValueError:
        command.extend([flag, text])
    else:
        if index + 1 >= len(command):
            raise RuntimeError(f"flag without value in template: {flag}")
        command[index + 1] = text


def canonical_2021_record(dataset: str, horizon: int) -> Path:
    return ROOT / "logs" / "cmrhm_unified_main" / "records" / (
        f"cmrhm_{dataset.lower()}_p{horizon}_s2021.json"
    )


def historical_multiseed_record(dataset: str, horizon: int, seed: int) -> Path | None:
    if dataset not in {"ETTm1", "ETTm2"} or seed not in {2022, 2023}:
        return None
    return ROOT / "logs" / "graphmamba_cmrhm_table34_final_test" / "records" / (
        f"a_{dataset.lower()}_p{horizon}_c_s{seed}.json"
    )


def load_record(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"record is not a JSON object: {path}")
    return payload


def resolve_auditable_command(path: Path) -> tuple[list[str], Path]:
    """Follow provenance links until the originating training command is found."""
    current = path
    visited: set[Path] = set()
    for _ in range(8):
        current = current.resolve()
        if current in visited:
            raise RuntimeError(f"cycle in record provenance: {current}")
        visited.add(current)
        payload = load_record(current)
        command = payload.get("command")
        if (
            isinstance(command, list)
            and all(isinstance(item, str) for item in command)
            and flag_value(command, "--is_training") == "1"
        ):
            return list(command), current
        source = payload.get("source_validation_record") or payload.get("source_record")
        if not isinstance(source, str):
            break
        candidate = Path(source)
        if not candidate.is_file():
            break
        current = candidate
    raise RuntimeError(f"no auditable command in provenance chain: {path}")


def effective_flag_value(command: list[str], flag: str, dataset: str) -> str | None:
    value = flag_value(command, flag)
    if flag == "--dual_scale_scan_mode" and value in {None, "auto"}:
        return "periodic_aligned" if dataset in {"ETTh1", "ETTh2"} else "independent_shared"
    if value is not None:
        return value
    return None


def audit_command(command: list[str], dataset: str, horizon: int, seed: int) -> None:
    expected = dict(EXPECTED_FLAGS)
    expected.update({"--pred_len": str(horizon), "--seed": str(seed)})
    failures = [
        f"{flag}: expected {value}, found {effective_flag_value(command, flag, dataset)}"
        for flag, value in expected.items()
        if effective_flag_value(command, flag, dataset) != value
    ]
    expected_data = dataset if dataset != "weather" else "custom"
    if flag_value(command, "--data") != expected_data:
        failures.append(
            f"--data: expected {expected_data}, found {flag_value(command, '--data')}"
        )
    if failures:
        raise RuntimeError("protocol mismatch:\n" + "\n".join(failures))


def command_template(dataset: str, horizon: int) -> list[str]:
    source = canonical_2021_record(dataset, horizon)
    command, _ = resolve_auditable_command(source)
    audit_command(command, dataset, horizon, 2021)
    for flag, value in EXPECTED_FLAGS.items():
        set_flag(command, flag, value)
    return list(command)


def build_command(dataset: str, horizon: int, seed: int, gpu: int) -> list[str]:
    command = command_template(dataset, horizon)
    name = slug(dataset, horizon, seed)
    command[0] = sys.executable
    if len(command) < 3:
        raise RuntimeError("invalid command template")
    command[2] = str(RUN_PY)
    set_flag(command, "--model", "TimeRole")
    set_flag(command, "--seed", seed)
    set_flag(command, "--model_id", name)
    set_flag(command, "--des", name)
    set_flag(command, "--gpu", gpu)
    set_flag(command, "--checkpoints", OUTPUT / "checkpoints")
    set_flag(command, "--test_after_train", 1)
    audit_command(command, dataset, horizon, seed)
    return command


def reusable_source(dataset: str, horizon: int, seed: int) -> Path | None:
    if seed == 2021:
        return canonical_2021_record(dataset, horizon)
    return historical_multiseed_record(dataset, horizon, seed)


def reuse_frozen(dataset: str, horizon: int, seed: int, destination: Path) -> bool:
    source = reusable_source(dataset, horizon, seed)
    if source is None or not source.is_file():
        return False
    payload = load_record(source)
    command, command_source = resolve_auditable_command(source)
    audit_command(command, dataset, horizon, seed)
    if (
        payload.get("status") != "completed"
        or payload.get("test_mse") is None
        or payload.get("test_mae") is None
    ):
        raise RuntimeError(f"incomplete frozen record: {source}")
    validation_mse = payload.get("validation_best_mse", payload.get("best_mse"))
    validation_mae = payload.get("validation_best_mae", payload.get("best_mae"))
    normalized = {
        "status": "completed",
        "model": "TimeRole",
        "dataset": dataset,
        "horizon": horizon,
        "pred_len": horizon,
        "seq_len": 336,
        "seed": seed,
        "validation_best_mse": validation_mse,
        "validation_best_mae": validation_mae,
        "test_mse": payload["test_mse"],
        "test_mae": payload["test_mae"],
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "reused_frozen_record_no_new_test_read",
        "result_source": "reused_protocol_matched_timerole_record",
        "source_record": str(source),
        "protocol_command_source": str(command_source),
        "source_implementation_model": payload.get("model"),
        "protocol_scan_mode": "independent_shared",
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(destination, normalized)
    print(f"REUSED {dataset} H={horizon} seed={seed}", flush=True)
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


def run_one(dataset: str, horizon: int, seed: int, args: argparse.Namespace) -> int:
    destination = record_path(dataset, horizon, seed)
    if args.resume and completed(destination):
        print(f"SKIP completed: {destination.name}", flush=True)
        return 0
    if not args.dry_run and reuse_frozen(dataset, horizon, seed, destination):
        return 0

    command = build_command(dataset, horizon, seed, args.gpu)
    print("COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    log_path = OUTPUT / "logs" / f"{slug(dataset, horizon, seed)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    return_code = 1
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
            timed_out = True
            return_code = 124

    text = log_path.read_text(encoding="utf-8", errors="replace")
    validation, metrics, setting = parse_log(text)
    status = (
        "completed"
        if return_code == 0 and validation is not None and metrics is not None
        else ("timeout" if timed_out else "failed")
    )
    payload: dict[str, object] = {
        "status": status,
        "model": "TimeRole",
        "dataset": dataset,
        "horizon": horizon,
        "pred_len": horizon,
        "seq_len": 336,
        "seed": seed,
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "one_shot_after_validation_selection",
        "result_source": "new_table2_protocol_matched_run",
        "protocol_scan_mode": "independent_shared",
        "command": command,
        "setting": setting,
        "log_path": str(log_path),
    }
    if validation:
        payload.update({f"validation_{key}": value for key, value in validation.items()})
    if metrics:
        payload.update(metrics)
    atomic_json(destination, payload)
    print(
        f"FINISH {status}: {dataset} H={horizon} seed={seed} "
        f"({payload['duration_seconds']}s)",
        flush=True,
    )
    return 0 if status == "completed" else return_code or 1


def write_summaries() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted((OUTPUT / "records").glob("*.json")):
        if not completed(path):
            continue
        payload = load_record(path)
        rows.append(
            {
                "model": "TimeRole",
                "dataset": payload["dataset"],
                "horizon": int(payload["horizon"]),
                "seed": int(payload["seed"]),
                "seq_len": 336,
                "validation_mse": payload.get("validation_best_mse"),
                "validation_mae": payload.get("validation_best_mae"),
                "test_mse": payload["test_mse"],
                "test_mae": payload["test_mae"],
                "result_source": payload.get("result_source"),
            }
        )
    rows.sort(key=lambda row: (DATASETS.index(str(row["dataset"])), int(row["horizon"]), int(row["seed"])))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    long_path = OUTPUT / "timerole_results_long.csv"
    fields = list(rows[0]) if rows else [
        "model", "dataset", "horizon", "seed", "seq_len",
        "validation_mse", "validation_mae", "test_mse", "test_mae", "result_source",
    ]
    with long_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), int(row["horizon"])), []).append(row)
    summary_path = OUTPUT / "timerole_mean_std.csv"
    summary_fields = [
        "model", "dataset", "horizon", "n_seeds", "seeds",
        "test_mse_mean", "test_mse_std", "test_mae_mean", "test_mae_std",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for (dataset, horizon), group in sorted(
            grouped.items(), key=lambda item: (DATASETS.index(item[0][0]), item[0][1])
        ):
            mse = [float(row["test_mse"]) for row in group]
            mae = [float(row["test_mae"]) for row in group]
            seeds = sorted(int(row["seed"]) for row in group)
            writer.writerow(
                {
                    "model": "TimeRole",
                    "dataset": dataset,
                    "horizon": horizon,
                    "n_seeds": len(group),
                    "seeds": ";".join(map(str, seeds)),
                    "test_mse_mean": statistics.mean(mse),
                    "test_mse_std": statistics.stdev(mse) if len(mse) > 1 else "",
                    "test_mae_mean": statistics.mean(mae),
                    "test_mae_std": statistics.stdev(mae) if len(mae) > 1 else "",
                }
            )


def write_status(args: argparse.Namespace, done: int, active: str | None, status: str) -> None:
    atomic_json(
        OUTPUT / "status.json",
        {
            "status": status,
            "seeds": args.seeds,
            "datasets": args.datasets,
            "horizons": args.horizons,
            "completed_jobs_this_invocation": done,
            "active_or_last": active,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def main() -> int:
    args = parse_args()
    jobs = [
        (dataset, horizon, seed)
        for seed in args.seeds
        for dataset in args.datasets
        for horizon in args.horizons
    ]
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    done = 0
    for index, (dataset, horizon, seed) in enumerate(jobs, start=1):
        active = slug(dataset, horizon, seed)
        write_status(args, done, active, "running")
        print(f"[{index}/{len(jobs)}] {active}", flush=True)
        code = run_one(dataset, horizon, seed, args)
        write_summaries()
        if code != 0:
            write_status(args, done, active, "failed")
            print(f"STOP after failure: inspect {record_path(dataset, horizon, seed)}", flush=True)
            return code
        done += 1
    write_status(args, done, None, "dry_run" if args.dry_run else "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
