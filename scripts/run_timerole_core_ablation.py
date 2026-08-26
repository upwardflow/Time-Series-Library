#!/usr/bin/env python3
"""Run the publication-facing TimeRole core ablation matrix.

The runner is serial, resumable, and test-set disciplined.  Checkpoints are
selected by validation MSE during training; the test split is evaluated once
after selection.  Existing frozen test records are reused only when they match
the requested protocol and contain explicit test MSE/MAE values.
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
OUTPUT = ROOT / "logs" / "timerole_core_ablation"
DATASETS = ("ETTm1", "ETTm2", "Weather")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
VARIANT_ORDER = ("full", "no_dhc", "no_decomp", "no_patch", "no_mamba", "no_graph")
VARIANTS = {
    "full": {"label": "TimeRole (Full)", "model": "TimeRole"},
    "no_dhc": {"label": "w/o DHC", "model": "GraphMambaRecent"},
    "no_decomp": {"label": "w/o Decomposition", "model": "TimeRole", "use_decomp": 0},
    "no_patch": {"label": "w/o Patch", "model": "TimeRole", "use_patch": 0},
    "no_mamba": {"label": "w/o Mamba", "model": "TimeRole", "use_time_mamba": 0},
    "no_graph": {"label": "w/o Graph", "model": "TimeRole", "use_graph": 0},
}
DATASET_SPECS = {
    "ETTm1": {"root": "dataset/ETT-small", "path": "ETTm1.csv", "data": "ETTm1", "target": "OT", "channels": 7},
    "ETTm2": {"root": "dataset/ETT-small", "path": "ETTm2.csv", "data": "ETTm2", "target": "OT", "channels": 7},
    "Weather": {"root": "dataset/weather", "path": "weather.csv", "data": "custom", "target": "CO2 (ppm)", "channels": 21},
}

VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$", re.MULTILINE)
TEST_PATTERN = re.compile(r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER, default=list(VARIANT_ORDER))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-after-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    if args.max_jobs < 0:
        parser.error("--max-jobs cannot be negative")
    if not args.test_after_selection and not args.dry_run:
        parser.error("publication runs require --test-after-selection")
    return args


def now() -> str:
    return datetime.now().astimezone().isoformat()


def candidate(dataset: str, horizon: int, variant: str, seed: int) -> str:
    return f"coreabl_{dataset.lower()}_p{horizon}_{variant}_s{seed}"


def record_path(dataset: str, horizon: int, variant: str, seed: int) -> Path:
    return OUTPUT / "records" / f"{candidate(dataset, horizon, variant, seed)}.json"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"record is not a JSON object: {path}")
    return payload


def completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False
    return payload.get("status") == "completed" and payload.get("test_mse") is not None and payload.get("test_mae") is not None


def frozen_source(dataset: str, horizon: int, variant: str, seed: int) -> Path | None:
    slug = dataset.lower()
    if variant == "full":
        return ROOT / "logs" / "timerole_table2_multiseed" / "records" / f"timerole_{slug}_sl336_pl{horizon}_s{seed}.json"
    if variant == "no_dhc" and dataset in {"ETTm1", "ETTm2"}:
        if seed == 2021:
            return ROOT / "logs" / "graphmamba_cmrhm_final_test" / "records" / f"{slug}_{horizon}_recent336_s2021.json"
        return ROOT / "logs" / "graphmamba_cmrhm_table34_final_test" / "records" / f"a_{slug}_p{horizon}_r_s{seed}.json"
    return None


def validate_frozen(payload: dict[str, object], source: Path, dataset: str, horizon: int, variant: str, seed: int) -> None:
    failures: list[str] = []
    if payload.get("status") != "completed":
        failures.append("status is not completed")
    if payload.get("test_mse") is None or payload.get("test_mae") is None:
        failures.append("test MSE/MAE missing")
    recorded_dataset = str(payload.get("dataset", "")).lower()
    if recorded_dataset != dataset.lower():
        failures.append(f"dataset={payload.get('dataset')!r}")
    recorded_horizon = payload.get("horizon", payload.get("pred_len"))
    if int(recorded_horizon or -1) != horizon:
        failures.append(f"horizon={recorded_horizon!r}")
    if int(payload.get("seed", -1)) != seed:
        failures.append(f"seed={payload.get('seed')!r}")
    expected_model = VARIANTS[variant]["model"]
    recorded_model = payload.get("model")
    if variant == "full" and recorded_model not in {"TimeRole", "CMRHM", "GraphMambaCMRHM"}:
        failures.append(f"model={recorded_model!r}")
    if variant == "no_dhc" and recorded_model != expected_model:
        failures.append(f"model={recorded_model!r}")
    if failures:
        raise RuntimeError(f"frozen record protocol mismatch: {source}: " + "; ".join(failures))


def reuse_frozen(dataset: str, horizon: int, variant: str, seed: int, destination: Path, dry_run: bool) -> bool:
    source = frozen_source(dataset, horizon, variant, seed)
    if source is None or not source.is_file():
        return False
    payload = load_json(source)
    validate_frozen(payload, source, dataset, horizon, variant, seed)
    if dry_run:
        print(f"REUSE {dataset} H={horizon} {variant} seed={seed} <- {source.relative_to(ROOT)}", flush=True)
        return True
    normalized = {
        "status": "completed",
        "dataset": dataset,
        "horizon": horizon,
        "pred_len": horizon,
        "seq_len": 336,
        "seed": seed,
        "variant": variant,
        "variant_label": VARIANTS[variant]["label"],
        "model": VARIANTS[variant]["model"],
        "validation_best_epoch": payload.get("validation_best_epoch", payload.get("best_epoch")),
        "validation_best_mse": payload.get("validation_best_mse", payload.get("best_mse")),
        "validation_best_mae": payload.get("validation_best_mae", payload.get("best_mae")),
        "test_mse": float(payload["test_mse"]),
        "test_mae": float(payload["test_mae"]),
        "checkpoint_selected_by": "validation_best_mse",
        "test_access": "reused_frozen_test_record",
        "result_source": "protocol_matched_frozen_record",
        "source_record": str(source),
        "recorded_at": now(),
    }
    atomic_json(destination, normalized)
    print(f"REUSED {dataset} H={horizon} {variant} seed={seed}", flush=True)
    return True


def build_command(dataset: str, horizon: int, variant: str, seed: int, gpu: int) -> list[str]:
    spec = DATASET_SPECS[dataset]
    definition = VARIANTS[variant]
    flags = {"use_graph": 1, "use_time_mamba": 1, "use_patch": 1, "use_decomp": 1}
    flags.update({key: value for key, value in definition.items() if key.startswith("use_")})
    name = candidate(dataset, horizon, variant, seed)
    channels = str(spec["channels"])
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(ROOT / str(spec["root"])), "--data_path", str(spec["path"]),
        "--model_id", f"{dataset}_336_{horizon}_{name}", "--model", str(definition["model"]),
        "--seed", str(seed), "--data", str(spec["data"]), "--features", "M", "--target", str(spec["target"]),
        "--seq_len", "336", "--label_len", "48", "--pred_len", str(horizon),
        "--enc_in", channels, "--dec_in", channels, "--c_out", channels,
        "--patch_len", "4", "--stride", "2", "--d_model", "64", "--d_ff", "128",
        "--d_state", "32", "--d_conv", "2", "--e_layers", "1", "--expand", "2",
        "--mamba_version", "1", "--mamba_bidirectional", "1",
        "--use_graph", str(flags["use_graph"]), "--use_time_mamba", str(flags["use_time_mamba"]),
        "--use_patch", str(flags["use_patch"]), "--use_decomp", str(flags["use_decomp"]), "--moving_avg", "25",
        "--dual_scale_scan_mode", "independent_shared", "--periodic_period", "24",
        "--periodic_local_patch", "0", "--periodic_local_stride", "0", "--periodic_period_stride", "12",
        "--periodic_use_adapter", "1", "--graph_alpha", "0.5", "--graph_top_k", "2",
        "--graph_sample_size", "2000", "--graph_sample_method", "uniform", "--static_graph_mode", "weighted",
        "--graph_cache", "0", "--gc_graph_dim", "16", "--gc_temperature", "1.0",
        "--gc_residual_init", "0.5", "--gc_dynamic_graph", "1", "--gc_symmetric_graph", "1",
        "--gc_input_modulation", "1", "--gc_direction_fusion", "1", "--gc_parallel_residual", "1",
        "--af_hidden_dim", "32", "--af_rank", "16", "--af_mode", "variable_scale_residual",
        "--dropout", "0.1", "--batch_size", "32", "--learning_rate", "0.0005", "--lradj", "type1",
        "--train_epochs", "100", "--patience", "6", "--num_workers", "0", "--gpu", str(gpu),
        "--checkpoints", str(OUTPUT / "checkpoints"), "--des", name, "--itr", "1",
        "--test_after_train", "1", "--timerole_old_intervention", "intact",
    ]


def parse_metrics(output: str) -> tuple[dict[str, object] | None, tuple[float, float] | None]:
    validations = VALIDATION_PATTERN.findall(output)
    validation = json.loads(validations[-1]) if validations else None
    tests = TEST_PATTERN.findall(output)
    test = (float(tests[-1][0]), float(tests[-1][1])) if tests else None
    return validation, test


def execute_job(dataset: str, horizon: int, variant: str, seed: int, args: argparse.Namespace) -> bool:
    destination = record_path(dataset, horizon, variant, seed)
    if args.resume and completed(destination):
        print(f"SKIP {dataset} H={horizon} {variant} seed={seed}", flush=True)
        return True
    if reuse_frozen(dataset, horizon, variant, seed, destination, args.dry_run):
        return True
    command = build_command(dataset, horizon, variant, seed, args.gpu)
    if args.dry_run:
        print("RUN " + shlex.join(command), flush=True)
        return True
    name = candidate(dataset, horizon, variant, seed)
    log_path = OUTPUT / "job_logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"START {dataset} H={horizon} {variant} seed={seed}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=args.timeout_seconds)
        return_code = result.returncode
        output = log_path.read_text(encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return_code = 124
        output = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    duration = round(time.monotonic() - started, 3)
    validation, test = parse_metrics(output)
    payload: dict[str, object] = {
        "status": "completed" if return_code == 0 and validation is not None and test is not None else "failed",
        "dataset": dataset, "horizon": horizon, "pred_len": horizon, "seq_len": 336, "seed": seed,
        "variant": variant, "variant_label": VARIANTS[variant]["label"], "model": VARIANTS[variant]["model"],
        "checkpoint_selected_by": "validation_best_mse", "test_access": "one_shot_after_validation_selection",
        "command": command, "log_path": str(log_path), "return_code": return_code,
        "duration_seconds": duration, "recorded_at": now(),
    }
    if validation is not None:
        payload["validation_best_epoch"] = validation.get("best_epoch", validation.get("epoch"))
        payload["validation_best_mse"] = validation.get("mse", validation.get("best_mse"))
        payload["validation_best_mae"] = validation.get("mae", validation.get("best_mae"))
    if test is not None:
        payload["test_mse"], payload["test_mae"] = test
    atomic_json(destination, payload)
    if payload["status"] != "completed":
        print(f"FAILED {dataset} H={horizon} {variant} seed={seed}; rc={return_code}; log={log_path}", flush=True)
        return False
    print(f"DONE {dataset} H={horizon} {variant} seed={seed}; test MSE={test[0]:.6f}, MAE={test[1]:.6f}", flush=True)
    return True


def selected_jobs(args: argparse.Namespace) -> list[tuple[str, int, str, int]]:
    jobs = [(dataset, horizon, variant, seed) for dataset in args.datasets for horizon in args.horizons for variant in args.variants for seed in args.seeds]
    return jobs[: args.max_jobs] if args.max_jobs else jobs


def write_outputs(jobs: list[tuple[str, int, str, int]]) -> None:
    rows: list[dict[str, object]] = []
    for dataset, horizon, variant, seed in jobs:
        path = record_path(dataset, horizon, variant, seed)
        if completed(path):
            payload = load_json(path)
            rows.append({key: payload.get(key) for key in ("dataset", "horizon", "variant", "variant_label", "seed", "validation_best_mse", "validation_best_mae", "test_mse", "test_mae", "result_source", "test_access")})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "horizon", "variant", "variant_label", "seed", "validation_best_mse", "validation_best_mae", "test_mse", "test_mae", "result_source", "test_access"]
    with (OUTPUT / "results_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    grouped: dict[tuple[str, int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["horizon"]), str(row["variant"]), str(row["variant_label"]))
        grouped.setdefault(key, []).append(row)
    summary_fields = ["dataset", "horizon", "variant", "variant_label", "n", "mse_mean", "mse_std", "mae_mean", "mae_std"]
    with (OUTPUT / "mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader()
        for (dataset, horizon, variant, label), group in sorted(grouped.items()):
            mses = [float(row["test_mse"]) for row in group]; maes = [float(row["test_mae"]) for row in group]
            writer.writerow({"dataset": dataset, "horizon": horizon, "variant": variant, "variant_label": label, "n": len(group), "mse_mean": statistics.mean(mses), "mse_std": statistics.stdev(mses) if len(mses) > 1 else 0.0, "mae_mean": statistics.mean(maes), "mae_std": statistics.stdev(maes) if len(maes) > 1 else 0.0})


def write_status(jobs: list[tuple[str, int, str, int]], state: str, current: tuple[str, int, str, int] | None = None, error: str | None = None) -> None:
    complete = sum(completed(record_path(*job)) for job in jobs)
    payload: dict[str, object] = {"state": state, "total_jobs": len(jobs), "completed_jobs": complete, "remaining_jobs": len(jobs) - complete, "updated_at": now()}
    if current is not None:
        payload["current_job"] = {"dataset": current[0], "horizon": current[1], "variant": current[2], "seed": current[3]}
    if error:
        payload["error"] = error
    atomic_json(OUTPUT / "status.json", payload)


def main() -> int:
    args = parse_args()
    jobs = selected_jobs(args)
    if args.dry_run:
        print(f"DRY RUN: {len(jobs)} jobs", flush=True)
    else:
        for directory in (OUTPUT / "records", OUTPUT / "job_logs", OUTPUT / "checkpoints"):
            directory.mkdir(parents=True, exist_ok=True)
        write_status(jobs, "running")
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}]", flush=True)
        if not args.dry_run:
            write_status(jobs, "running", job)
        try:
            success = execute_job(*job, args)
        except Exception as exc:
            if not args.dry_run:
                write_outputs(jobs); write_status(jobs, "failed", job, str(exc))
            print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        if not success:
            if not args.dry_run:
                write_outputs(jobs); write_status(jobs, "failed", job, "job failed; no automatic retry")
            return 1
        if not args.dry_run:
            write_outputs(jobs)
    if not args.dry_run:
        write_status(jobs, "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
