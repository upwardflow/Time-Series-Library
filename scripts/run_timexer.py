#!/usr/bin/env python3
"""Run TimeXer ETT experiments without repeatedly typing long commands.

Examples:
    python scripts/run_timexer.py
    python scripts/run_timexer.py --dataset ETTh1 --pred-lens 96 192 336 720
    python scripts/run_timexer.py --dataset ETTh2 --pred-lens 96 --epochs 1 --num-workers 0 --des smoke
    python scripts/run_timexer.py --dataset ETTh1 --pred-lens 96 192 --dry-run
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

# Effective hyperparameters from the repository's TimeXer_ETTh1.sh and
# TimeXer_ETTh2.sh scripts. Values omitted by the shell scripts are made
# explicit here using run.py defaults.
PRESETS = {
    "ETTh1": {
        96: {"e_layers": 1, "d_model": 256, "d_ff": 2048, "batch_size": 4},
        192: {"e_layers": 2, "d_model": 128, "d_ff": 2048, "batch_size": 4},
        336: {"e_layers": 1, "d_model": 512, "d_ff": 1024, "batch_size": 16},
        720: {"e_layers": 1, "d_model": 256, "d_ff": 1024, "batch_size": 16},
    },
    "ETTh2": {
        96: {"e_layers": 1, "d_model": 256, "d_ff": 1024, "batch_size": 16},
        192: {"e_layers": 1, "d_model": 256, "d_ff": 1024, "batch_size": 32},
        336: {"e_layers": 2, "d_model": 512, "d_ff": 1024, "batch_size": 4},
        720: {"e_layers": 2, "d_model": 256, "d_ff": 1024, "batch_size": 16},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more TimeXer experiments with repository presets."
    )
    parser.add_argument("--dataset", choices=sorted(PRESETS), default="ETTh1")
    parser.add_argument(
        "--pred-lens",
        type=int,
        nargs="+",
        choices=(96, 192, 336, 720),
        default=[96],
        metavar="N",
        help="prediction lengths to run sequentially (default: 96)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="physical GPU exposed through CUDA_VISIBLE_DEVICES (default: 0)",
    )
    parser.add_argument(
        "--des",
        default="Exp",
        help="experiment description; use 'smoke' for short checks (default: Exp)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "timexer",
        help="directory for per-run logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without starting experiments",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue with later horizons if one experiment fails",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    # Preserve the requested order while avoiding accidental duplicate runs.
    args.pred_lens = list(dict.fromkeys(args.pred_lens))
    return args


def build_command(args: argparse.Namespace, pred_len: int) -> list[str]:
    preset = PRESETS[args.dataset][pred_len]
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
        "TimeXer",
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
        "--e_layers",
        str(preset["e_layers"]),
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
    if data_path.exists():
        print(f"Data: {data_path}")
    else:
        print(
            f"Data not found locally: {data_path}\n"
            "The project loader will try to download it from Hugging Face."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    total = len(args.pred_lens)
    for index, pred_len in enumerate(args.pred_lens, start=1):
        command = build_command(args, pred_len)
        log_path = args.log_dir / f"TimeXer_{args.dataset}_96_{pred_len}_{args.des}.log"
        print(f"\n[{index}/{total}] TimeXer {args.dataset}: 96 -> {pred_len}")
        print("Command:", shlex.join(command))
        print("Log:", log_path)

        if args.dry_run:
            continue

        return_code = run_and_log(command, log_path, env)
        if return_code != 0:
            print(f"Experiment failed with exit code {return_code}: {args.dataset}-{pred_len}")
            if not args.continue_on_error:
                return return_code

    print("\nAll requested TimeXer experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
