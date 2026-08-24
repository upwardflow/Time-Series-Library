#!/usr/bin/env python3
"""Run the frozen eight-model Electricity/Solar P0 matrix.

The pilot phase is validation-only. The formal phase selects checkpoints using
validation loss and evaluates the selected checkpoint once through the streaming
test path. Jobs are serial, resumable, atomic, and stop at the first failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_graphmamba_backbone_ablation as base


OUTPUT = ROOT / "logs" / "timerole_p0" / "ecl_solar"
MODELS = (
    "TimeRole", "GraphMambaRecent", "DLinear", "PatchTST",
    "iTransformer", "TimeMixer", "SMamba", "TimeFilter",
)
DISPLAY = {"GraphMambaRecent": "Recent96"}
DATASET_ORDER = ("electricity", "solar")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)


@dataclass(frozen=True)
class Dataset:
    root: Path
    file: str
    channels: int
    target: str
    freq: str
    batch_size: int


DATASETS = {
    "electricity": Dataset(ROOT / "dataset" / "electricity", "electricity.csv", 321, "0", "h", 8),
    "solar": Dataset(ROOT / "dataset" / "solar", "solar.csv", 137, "channel_99", "t", 16),
}


@dataclass(frozen=True)
class Task:
    model: str
    dataset: str
    horizon: int
    seed: int

    @property
    def name(self) -> str:
        return f"p0_{self.model.lower()}_{self.dataset}_l336_h{self.horizon}_s{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pilot-epochs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pilot_epochs < 1 or args.timeout_seconds < 1 or args.max_jobs < 0:
        parser.error("pilot epochs and timeout must be positive; max-jobs cannot be negative")
    args.output_dir = (args.output_dir / args.phase).resolve()
    return args


def preset(model: str, dataset: str, horizon: int) -> dict[str, object]:
    batch = DATASETS[dataset].batch_size
    if model == "DLinear":
        return dict(label_len=48, e_layers=2, d_model=32, d_ff=64, n_heads=4,
                    batch_size=batch, learning_rate=1e-4, epochs=10, patience=3)
    if model == "PatchTST":
        return dict(label_len=48, e_layers=2, d_model=512, d_ff=2048, n_heads=8,
                    batch_size=16, learning_rate=1e-4, epochs=10, patience=3,
                    patch_len=16, stride=8, use_amp=True)
    if model == "iTransformer":
        return dict(label_len=48, e_layers=3, d_model=512, d_ff=512, n_heads=8,
                    batch_size=16, learning_rate=5e-4 if dataset == "electricity" else 1e-4,
                    epochs=10, patience=3)
    if model == "TimeMixer":
        return dict(label_len=0, e_layers=3, d_model=16, d_ff=32, n_heads=4,
                    batch_size=32 if dataset == "electricity" else 16,
                    learning_rate=1e-2, epochs=20, patience=10,
                    down_sampling_layers=3, down_sampling_window=2,
                    down_sampling_method="avg")
    if model == "SMamba":
        if dataset == "electricity":
            learning_rate = 1e-3 if horizon == 96 else 5e-4
            d_state = 16 if horizon == 96 else 32
        else:
            learning_rate, d_state = 5e-5, 2
        return dict(label_len=48, e_layers=3, d_model=512, d_ff=512, n_heads=4,
                    d_state=d_state, batch_size=16 if dataset == "electricity" else 16,
                    learning_rate=learning_rate, epochs=5, patience=3)
    if model == "TimeFilter":
        if dataset == "electricity":
            dropout = 0.5 if horizon == 96 else 0.4
            return dict(label_len=48, e_layers=2, d_model=512, d_ff=512, n_heads=8,
                        batch_size=1, learning_rate=1e-3, epochs=15, patience=3,
                        patch_len=48, stride=48, dropout=dropout, alpha=0.1, top_p=0.5)
        dropout = {96: 0.2, 192: 0.3, 336: 0.3, 720: 0.6}[horizon]
        return dict(label_len=48, e_layers=2, d_model=256, d_ff=512, n_heads=8,
                    batch_size=8, learning_rate=5e-4, epochs=10, patience=3,
                    patch_len=48, stride=48, dropout=dropout, alpha=0.1, top_p=0.5)
    if model in {"TimeRole", "GraphMambaRecent"}:
        return dict(label_len=48, e_layers=1, d_model=64, d_ff=128, n_heads=8,
                    batch_size=batch, learning_rate=5e-4, epochs=100, patience=6,
                    patch_len=4, stride=2, dropout=0.1)
    raise ValueError(model)


def tasks(args: argparse.Namespace) -> list[Task]:
    horizons = [96] if args.phase == "pilot" else args.horizons
    seeds = [args.seeds[0]] if args.phase == "pilot" else args.seeds
    matrix = [Task(model, dataset, horizon, seed)
              for dataset in args.datasets for model in args.models
              for seed in seeds for horizon in horizons]
    return matrix[: args.max_jobs] if args.max_jobs else matrix


def command(task: Task, args: argparse.Namespace) -> list[str]:
    data = DATASETS[task.dataset]
    config = preset(task.model, task.dataset, task.horizon)
    epochs = args.pilot_epochs if args.phase == "pilot" else config["epochs"]
    patience = min(2, epochs) if args.phase == "pilot" else config["patience"]
    result = [
        sys.executable, "-u", str(ROOT / "run.py"),
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(data.root), "--data_path", data.file,
        "--model_id", task.name, "--model", task.model, "--seed", str(task.seed),
        "--data", "custom", "--features", "M", "--target", data.target,
        "--freq", data.freq, "--seq_len", "336",
        "--label_len", str(config["label_len"]), "--pred_len", str(task.horizon),
        "--enc_in", str(data.channels), "--dec_in", str(data.channels), "--c_out", str(data.channels),
        "--d_model", str(config["d_model"]), "--d_ff", str(config["d_ff"]),
        "--n_heads", str(config["n_heads"]), "--e_layers", str(config["e_layers"]),
        "--d_layers", "1", "--factor", "3", "--dropout", str(config.get("dropout", 0.1)),
        "--batch_size", str(config["batch_size"]), "--learning_rate", str(config["learning_rate"]),
        "--train_epochs", str(epochs), "--patience", str(patience), "--lradj", "type1",
        "--num_workers", "0", "--gpu", str(args.gpu),
        "--checkpoints", str(args.output_dir / "checkpoints"),
        "--des", task.name, "--itr", "1",
        "--test_after_train", "0" if args.phase == "pilot" else "1",
    ]
    for key in ("patch_len", "stride", "down_sampling_layers", "down_sampling_window", "alpha", "top_p"):
        if key in config:
            result.extend(("--" + key, str(config[key])))
    if "down_sampling_method" in config:
        result.extend(("--down_sampling_method", str(config["down_sampling_method"])))
    if config.get("use_amp"):
        result.append("--use_amp")
    if task.model == "SMamba":
        result.extend(("--d_state", str(config["d_state"]), "--use_norm", "1"))
    if task.model in {"TimeRole", "GraphMambaRecent"}:
        result.extend((
            "--timerole_recent_len", "96", "--timerole_memory_pool", "16",
            "--timerole_old_intervention", "intact", "--d_state", "32", "--d_conv", "2",
            "--expand", "2", "--mamba_version", "1", "--mamba_bidirectional", "1",
            "--use_graph", "1", "--use_time_mamba", "1", "--use_patch", "1",
            "--use_decomp", "1", "--moving_avg", "25",
            "--dual_scale_scan_mode", "independent_shared", "--periodic_period", "24",
            "--periodic_local_patch", "4", "--periodic_local_stride", "2",
            "--periodic_period_stride", "12", "--periodic_use_adapter", "1",
            "--graph_alpha", "0.5", "--graph_top_k", "2", "--graph_sample_size", "2000",
            "--graph_sample_method", "uniform", "--static_graph_mode", "weighted", "--graph_cache", "1",
        ))
    return result


def record_path(task: Task, args: argparse.Namespace) -> Path:
    return args.output_dir / "records" / f"{task.name}.json"


def completed(path: Path, phase: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "completed":
        return False
    if phase == "pilot":
        return payload.get("split") == "validation" and payload.get("test_accessed") is False
    origin = payload.get("origin_metrics_path")
    return payload.get("split") == "test" and payload.get("test_accessed") is True and isinstance(origin, str) and Path(origin).is_file()


def run_one(task: Task, args: argparse.Namespace) -> int:
    destination = record_path(task, args)
    if completed(destination, args.phase):
        print(f"SKIP completed: {task.name}", flush=True)
        return 0
    run_command = command(task, args)
    print("COMMAND", shlex.join(run_command), flush=True)
    if args.dry_run:
        return 0
    log_path = args.output_dir / "logs" / f"{task.name}.log"
    pattern = base.VALIDATION_PATTERN if args.phase == "pilot" else base.EVALUATION_PATTERN
    started = time.monotonic()
    try:
        code, result, duration = base.execute(
            run_command, log_path, pattern, args.gpu, args.timeout_seconds, cwd=ROOT
        )
    except KeyboardInterrupt:
        base.atomic_write(destination, {
            "status": "interrupted", "phase": args.phase,
            "model": DISPLAY.get(task.model, task.model), "implementation_model": task.model,
            "dataset": task.dataset, "horizon": task.horizon, "seq_len": 336, "seed": task.seed,
            "split": "validation" if args.phase == "pilot" else "test",
            "test_accessed": False if args.phase == "pilot" else None,
            "return_code": 130, "duration_seconds": round(time.monotonic() - started, 3),
            "checkpoint_selected_by": "validation_loss", "command": run_command,
            "execution_precision": "amp_fp16" if "--use_amp" in run_command else "fp32",
            "log_path": str(log_path), "reason": "external_keyboard_interrupt",
            "recorded_at": datetime.now().astimezone().isoformat(),
        })
        print(f"INTERRUPTED {task.name}; no automatic retry", flush=True)
        raise
    if args.phase == "pilot":
        success = code == 0 and isinstance(result, dict)
        split, test_accessed = "validation", False
    else:
        success = (code == 0 and isinstance(result, dict)
                   and result.get("split") == "test" and result.get("test_accessed") is True
                   and isinstance(result.get("origin_metrics_path"), str)
                   and Path(result["origin_metrics_path"]).is_file())
        split, test_accessed = "test", True if success else None
    payload = {
        "status": "completed" if success else "failed", "phase": args.phase,
        "model": DISPLAY.get(task.model, task.model), "implementation_model": task.model,
        "dataset": task.dataset, "horizon": task.horizon, "seq_len": 336, "seed": task.seed,
        "split": split, "test_accessed": test_accessed, "return_code": code,
        "duration_seconds": round(duration, 3), "checkpoint_selected_by": "validation_loss",
        "execution_precision": "amp_fp16" if "--use_amp" in run_command else "fp32",
        "runtime_adaptation": ("fp32_pilot_infeasible_under_21600s_cap"
                               if task.model == "PatchTST" and "--use_amp" in run_command else None),
        "command": run_command, "log_path": str(log_path),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if result:
        payload.update(result)
    base.atomic_write(destination, payload)
    if not success:
        print(f"FAILED {task.name}; no automatic retry", flush=True)
        return 1
    return 0


def summarize(matrix: list[Task], args: argparse.Namespace, status: str) -> None:
    rows = []
    for task in matrix:
        path = record_path(task, args)
        if not completed(path, args.phase):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": DISPLAY.get(task.model, task.model), "dataset": task.dataset,
            "horizon": task.horizon, "seed": task.seed,
            "mse": payload.get("mse", payload.get("best_mse")),
            "mae": payload.get("mae", payload.get("best_mae")),
            "record": str(path),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model", "dataset", "horizon", "seed", "mse", "mae", "record"]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    base.atomic_write(args.output_dir / "status.json", {
        "status": status, "phase": args.phase,
        "dataset": matrix[0].dataset if matrix and len({task.dataset for task in matrix}) == 1 else None,
        "expected": len(matrix), "completed": len(rows),
        "failed": sum(1 for task in matrix if record_path(task, args).is_file() and not completed(record_path(task, args), args.phase)),
        "test_accessed": False if args.phase == "pilot" else True,
        "updated_at": datetime.now().astimezone().isoformat(),
    })


def main() -> int:
    args = parse_args()
    matrix = tasks(args)
    if args.dry_run:
        for task in matrix:
            run_one(task, args)
        print(json.dumps({"phase": args.phase, "jobs": len(matrix), "test_accessed": args.phase == "formal"}))
        return 0
    try:
        for index, task in enumerate(matrix, 1):
            print(f"=== [{index}/{len(matrix)}] {task.name} ===", flush=True)
            if run_one(task, args):
                summarize(matrix, args, "failed")
                return 1
            summarize(matrix, args, "running")
    except KeyboardInterrupt:
        summarize(matrix, args, "interrupted")
        return 130
    summarize(matrix, args, "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
