#!/usr/bin/env python3
"""Run GraphMamba ETT experiments without manually assembling run.py commands.

Examples:
    python scripts/run_graphmamba.py
    python scripts/run_graphmamba.py --dataset ETTh1 --pred-lens 96 192 336 720
    python scripts/run_graphmamba.py --epochs 1 --num-workers 0 --des smoke
    python scripts/run_graphmamba.py --mamba-version 2 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "run.py"
DATA_ROOT = REPO_ROOT / "dataset" / "ETT-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GraphMamba on an ETT dataset.")
    parser.add_argument(
        "--dataset", choices=("ETTh1", "ETTh2"), default="ETTh1"
    )
    parser.add_argument(
        "--pred-lens",
        type=int,
        nargs="+",
        choices=(96, 192, 336, 720),
        default=[96],
        metavar="N",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--graph-alpha", type=float, default=0.5)
    parser.add_argument("--graph-top-k", type=int, default=2)
    parser.add_argument("--mamba-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gpu", type=int, default=0, help="physical CUDA GPU index")
    parser.add_argument("--des", default="GraphMamba")
    parser.add_argument(
        "--log-dir", type=Path, default=REPO_ROOT / "logs" / "graphmamba"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.patch_len < 2 or args.stride < 2:
        parser.error("--patch-len and --stride must be at least 2")
    args.pred_lens = list(dict.fromkeys(args.pred_lens))
    return args


def build_command(args: argparse.Namespace, pred_len: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(DATA_ROOT),
        "--data_path",
        f"{args.dataset}.csv",
        "--model_id",
        f"{args.dataset}_96_{pred_len}",
        "--model",
        "GraphMamba",
        "--seed",
        str(args.seed),
        "--data",
        args.dataset,
        "--features",
        "M",
        "--seq_len",
        "96",
        "--label_len",
        "48",
        "--pred_len",
        str(pred_len),
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
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
        "2000",
        "--graph_sample_method",
        "uniform",
        "--static_graph_mode",
        "weighted",
        "--dropout",
        "0.1",
        "--batch_size",
        str(args.batch_size),
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


def main() -> int:
    args = parse_args()
    data_path = DATA_ROOT / f"{args.dataset}.csv"
    if not data_path.exists():
        print(f"Dataset not found: {data_path}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for index, pred_len in enumerate(args.pred_lens, start=1):
        command = build_command(args, pred_len)
        log_path = args.log_dir / (
            f"GraphMamba_{args.dataset}_96_{pred_len}_{args.des}.log"
        )
        print(f"\n[{index}/{len(args.pred_lens)}] GraphMamba {args.dataset}: 96 -> {pred_len}")
        print("Command:", shlex.join(command))
        print("Log:", log_path)
        if args.dry_run:
            continue

        return_code = run_and_log(command, log_path, env)
        if return_code != 0:
            print(f"Experiment failed with exit code {return_code}", file=sys.stderr)
            if not args.continue_on_error:
                return return_code

    print("\nAll requested GraphMamba experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
