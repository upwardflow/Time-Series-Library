#!/usr/bin/env python3
"""Run GraphMamba on the selected multivariate forecasting benchmarks.

The default matrix excludes Electricity (ECL) and Traffic, so it contains
6 datasets x 4 prediction lengths = 24 runs. ECL and Traffic remain available
through ``--datasets`` for later runs.
Experiments are run sequentially so this script can be left inside tmux.

Examples:
    python scripts/run_graphmamba_all.py --dry-run
    python scripts/run_graphmamba_all.py
    python scripts/run_graphmamba_all.py --datasets ETTh1 weather solar
    python scripts/run_graphmamba_all.py --pred-lens 96 192 --epochs 1
    python scripts/run_graphmamba_all.py --no-resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "run.py"
DEFAULT_PRED_LENS = (96, 192, 336, 720)
DEFAULT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "solar")


@dataclass(frozen=True)
class DatasetConfig:
    root_path: Path
    data_path: str
    data_type: str
    channels: int
    target: str
    batch_size: int


# M -> M forecasting uses every value column. ``target`` is retained only for
# run.py compatibility and is not used to select or reorder columns in M mode.
DATASETS = {
    "ETTh1": DatasetConfig(
        REPO_ROOT / "dataset" / "ETT-small", "ETTh1.csv", "ETTh1", 7, "OT", 32
    ),
    "ETTh2": DatasetConfig(
        REPO_ROOT / "dataset" / "ETT-small", "ETTh2.csv", "ETTh2", 7, "OT", 32
    ),
    "ETTm1": DatasetConfig(
        REPO_ROOT / "dataset" / "ETT-small", "ETTm1.csv", "ETTm1", 7, "OT", 32
    ),
    "ETTm2": DatasetConfig(
        REPO_ROOT / "dataset" / "ETT-small", "ETTm2.csv", "ETTm2", 7, "OT", 32
    ),
    "weather": DatasetConfig(
        REPO_ROOT / "dataset" / "weather",
        "weather.csv",
        "custom",
        21,
        "CO2 (ppm)",
        32,
    ),
    "solar": DatasetConfig(
        REPO_ROOT / "dataset" / "solar",
        "solar.csv",
        "custom",
        137,
        "channel_99",
        32,
    ),
    "electricity": DatasetConfig(
        REPO_ROOT / "dataset" / "electricity",
        "electricity.csv",
        "custom",
        321,
        "320",
        16,
    ),
    "traffic": DatasetConfig(
        REPO_ROOT / "dataset" / "traffic",
        "traffic.csv",
        "custom",
        862,
        "861",
        8,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GraphMamba long-term forecasting matrix."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DEFAULT_DATASETS),
        help="datasets to run (default: ETT x4, weather, and solar)",
    )
    parser.add_argument(
        "--pred-lens",
        type=int,
        nargs="+",
        choices=DEFAULT_PRED_LENS,
        default=list(DEFAULT_PRED_LENS),
        metavar="N",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the memory-safe per-dataset batch sizes",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patch-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--d-conv", type=int, default=2)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--graph-alpha", type=float, default=0.5)
    parser.add_argument("--graph-top-k", type=int, default=2)
    parser.add_argument("--graph-sample-size", type=int, default=2000)
    parser.add_argument("--mamba-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gpu", type=int, default=0, help="physical CUDA GPU index")
    parser.add_argument("--des", default="GraphMamba_all")
    parser.add_argument(
        "--log-dir", type=Path, default=REPO_ROOT / "logs" / "graphmamba_all"
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip runs carrying a successful completion marker (default: true)",
    )
    parser.add_argument(
        "--keep-graph-cache",
        action="store_true",
        help="keep static adjacency .npy caches after each dataset finishes",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.patch_len < 2 or args.stride < 2:
        parser.error("--patch-len and --stride must be at least 2")
    if args.graph_sample_size < 1:
        parser.error("--graph-sample-size must be at least 1")

    args.datasets = list(dict.fromkeys(args.datasets))
    args.pred_lens = list(dict.fromkeys(args.pred_lens))
    args.log_dir = args.log_dir.resolve()
    return args


def build_command(
    args: argparse.Namespace,
    dataset_name: str,
    config: DatasetConfig,
    pred_len: int,
) -> list[str]:
    batch_size = args.batch_size or config.batch_size
    return [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(config.root_path),
        "--data_path",
        config.data_path,
        "--model_id",
        f"{dataset_name}_96_{pred_len}",
        "--model",
        "GraphMamba",
        "--seed",
        str(args.seed),
        "--data",
        config.data_type,
        "--features",
        "M",
        "--target",
        config.target,
        "--seq_len",
        "96",
        "--label_len",
        "48",
        "--pred_len",
        str(pred_len),
        "--enc_in",
        str(config.channels),
        "--dec_in",
        str(config.channels),
        "--c_out",
        str(config.channels),
        "--patch_len",
        str(args.patch_len),
        "--stride",
        str(args.stride),
        "--d_model",
        str(args.d_model),
        "--d_ff",
        str(args.d_ff),
        "--d_state",
        str(args.d_state),
        "--d_conv",
        str(args.d_conv),
        "--e_layers",
        str(args.e_layers),
        "--expand",
        str(args.expand),
        "--mamba_version",
        str(args.mamba_version),
        "--mamba_bidirectional",
        "1",
        "--use_graph",
        "1",
        "--use_time_mamba",
        "1",
        "--use_patch",
        "1",
        "--use_decomp",
        "1",
        "--moving_avg",
        "25",
        "--graph_alpha",
        str(args.graph_alpha),
        "--graph_top_k",
        str(args.graph_top_k),
        "--graph_sample_size",
        str(args.graph_sample_size),
        "--graph_sample_method",
        "uniform",
        "--static_graph_mode",
        "weighted",
        "--graph_cache",
        "1",
        "--dropout",
        str(args.dropout),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        "0",
        "--des",
        args.des,
        "--itr",
        "1",
    ]


def run_and_log(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return process.wait()


def marker_path(args: argparse.Namespace, dataset_name: str, pred_len: int) -> Path:
    return args.log_dir / dataset_name / f"96_{pred_len}_{args.des}.done.json"


def log_path(args: argparse.Namespace, dataset_name: str, pred_len: int) -> Path:
    return args.log_dir / dataset_name / f"GraphMamba_{dataset_name}_96_{pred_len}_{args.des}.log"


def write_marker(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "completed_at": datetime.now().astimezone().isoformat(),
                "command": command,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def graph_cache_path(
    args: argparse.Namespace, dataset_name: str, config: DatasetConfig
) -> Path:
    stem = Path(config.data_path).stem
    return config.root_path / (
        f"{stem}_adj_train_{config.data_type}_M_uniform_"
        f"{args.graph_sample_size}_seed{args.seed}.npy"
    )


def validate_datasets(selected: list[str]) -> list[str]:
    problems = []
    for name in selected:
        config = DATASETS[name]
        path = config.root_path / config.data_path
        if not path.is_file():
            problems.append(f"missing file: {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as data_file:
            columns = next(csv.reader(data_file), [])
        value_columns = [column for column in columns if column != "date"]
        if "date" not in columns:
            problems.append(f"{path}: missing date column")
        if len(value_columns) != config.channels:
            problems.append(
                f"{path}: expected {config.channels} variables, "
                f"found {len(value_columns)}"
            )
        if config.target not in value_columns:
            problems.append(f"{path}: target column {config.target!r} not found")
    return problems


def main() -> int:
    args = parse_args()
    problems = validate_datasets(args.datasets)
    if problems:
        print("Dataset validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    total = len(args.datasets) * len(args.pred_lens)
    current = 0
    failures: list[str] = []

    print(f"GraphMamba matrix: {len(args.datasets)} datasets x "
          f"{len(args.pred_lens)} horizons = {total} runs")
    print(f"GPU: physical CUDA device {args.gpu}; seed: {args.seed}")

    for dataset_name in args.datasets:
        config = DATASETS[dataset_name]
        for pred_len in args.pred_lens:
            current += 1
            marker = marker_path(args, dataset_name, pred_len)
            experiment_log = log_path(args, dataset_name, pred_len)
            command = build_command(args, dataset_name, config, pred_len)

            print(
                f"\n[{current}/{total}] GraphMamba {dataset_name}: 96 -> {pred_len}"
            )
            if args.resume and marker.exists():
                print(f"Already completed; skipping: {marker}")
                continue
            print("Command:", shlex.join(command))
            print("Log:", experiment_log)
            if args.dry_run:
                continue

            return_code = run_and_log(command, experiment_log, env)
            if return_code == 0:
                write_marker(marker, command)
                print(f"Completed: {dataset_name} 96 -> {pred_len}")
            else:
                run_name = f"{dataset_name}_96_{pred_len} (exit {return_code})"
                failures.append(run_name)
                print(f"Experiment failed: {run_name}", file=sys.stderr)
                if not args.continue_on_error:
                    print("Stopping. Re-run the same command to resume.", file=sys.stderr)
                    return return_code

        cache_path = graph_cache_path(args, dataset_name, config)
        if not args.dry_run and not args.keep_graph_cache and cache_path.exists():
            cache_path.unlink()
            print(f"Removed temporary graph cache: {cache_path}")

    if failures:
        print("\nCompleted with failed runs:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll requested GraphMamba experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
