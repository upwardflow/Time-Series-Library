#!/usr/bin/env python3
"""Run one auditable TimeRole paper experiment on an ETT dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "run.py"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "timerole_experiment"
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
TEST_PATTERN = re.compile(
    r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(
            "TimeRole",
            "TimeRoleRecent",
            "TimeRoleFullHistory",
            "TimeRoleConcat",
            "TimeRoleNoDiff",
            "TimeRoleGlobalGate",
        ),
        required=True,
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--dataset", choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2"), default="ETTh1"
    )
    parser.add_argument("--pred-len", type=int, choices=(96, 192, 336, 720), default=96)
    parser.add_argument("--seq-len", type=int, choices=(96, 336), default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lradj", choices=("type1", "type2", "type3", "cosine"), default="type1")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--graph-alpha", type=float, default=0.5)
    parser.add_argument(
        "--dual-scale-scan-mode",
        choices=("auto", "joint", "independent_shared", "periodic_aligned"),
        default="auto",
    )
    parser.add_argument("--periodic-period", type=int, default=24)
    parser.add_argument(
        "--periodic-local-patch", type=int, default=0,
        help="0 lets run.py load the training-derived period-constrained scale",
    )
    parser.add_argument(
        "--periodic-local-stride", type=int, default=0,
        help="0 uses the derived stride (or half an explicitly supplied patch)",
    )
    parser.add_argument("--periodic-period-stride", type=int, default=12)
    parser.add_argument("--periodic-use-adapter", type=int, choices=(0, 1), default=1)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch size must be positive")
    args.output_dir = args.output_dir.resolve()
    return args


def build_command(args: argparse.Namespace) -> list[str]:
    scan_mode = args.dual_scale_scan_mode
    if scan_mode == "auto":
        scan_mode = (
            "periodic_aligned"
            if args.dataset in {"ETTh1", "ETTh2"}
            and args.periodic_period < args.seq_len
            else "independent_shared"
        )
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", str(REPO_ROOT / "dataset" / "ETT-small"),
        "--data_path", f"{args.dataset}.csv",
        "--model_id", f"{args.dataset}_96_{args.pred_len}_{args.candidate}",
        "--model", args.model,
        "--seed", str(args.seed),
        "--data", args.dataset,
        "--features", "M",
        "--target", "OT",
        "--seq_len", str(args.seq_len), "--label_len", "48", "--pred_len", str(args.pred_len),
        "--enc_in", "7", "--dec_in", "7", "--c_out", "7",
        "--patch_len", "4", "--stride", "2",
        "--d_model", "64", "--d_ff", "128",
        "--d_state", "32", "--d_conv", "2", "--e_layers", "1", "--expand", "2",
        "--mamba_version", "1", "--mamba_bidirectional", "1",
        "--use_graph", "1", "--use_time_mamba", "1",
        "--use_patch", "1", "--use_decomp", "1", "--moving_avg", "25",
        "--dual_scale_scan_mode", scan_mode,
        "--periodic_period", str(args.periodic_period),
        "--periodic_local_patch", str(args.periodic_local_patch),
        "--periodic_local_stride", str(args.periodic_local_stride),
        "--periodic_period_stride", str(args.periodic_period_stride),
        "--periodic_use_adapter", str(args.periodic_use_adapter),
        "--graph_alpha", str(args.graph_alpha), "--graph_top_k", "2",
        "--graph_sample_size", "2000", "--graph_sample_method", "uniform",
        "--static_graph_mode", "weighted", "--graph_cache", "0",
        "--dropout", str(args.dropout), "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate), "--lradj", args.lradj,
        "--train_epochs", str(args.epochs), "--patience", str(args.patience),
        "--num_workers", "0", "--gpu", "0",
        "--des", args.candidate, "--itr", "1",
        "--test_after_train", "1" if args.final_test else "0",
    ]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir / ("final" if args.final_test else "validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{args.candidate}.log"
    record_path = output_dir / f"{args.candidate}.json"
    if record_path.exists() and not args.force:
        print(f"Record exists; use --force to rerun: {record_path}")
        return 0

    command = build_command(args)
    print("Command:", shlex.join(command))
    print("Log:", log_path)
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    validation = None
    test_metrics = None
    started = time.monotonic()
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
            validation_match = VALIDATION_PATTERN.match(line.strip())
            if validation_match:
                validation = json.loads(validation_match.group(1))
            test_match = TEST_PATTERN.match(line.strip())
            if test_match:
                test_metrics = {
                    "test_mse": float(test_match.group(1)),
                    "test_mae": float(test_match.group(2)),
                }
        return_code = process.wait()

    payload = {
        "status": "completed" if return_code == 0 and validation else "failed",
        "model": args.model,
        "candidate": args.candidate,
        "dataset": args.dataset,
        "pred_len": args.pred_len,
        "seed": args.seed,
        "protocol": f"{args.dataset}_96_{args.pred_len}_M",
        "final_test": args.final_test,
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "log_path": str(log_path),
    }
    if validation:
        payload.update(validation)
    if test_metrics:
        payload.update(test_metrics)
    temporary = record_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(record_path)
    print("Record:", record_path)

    if payload["status"] != "completed":
        return return_code or 1
    if args.final_test and test_metrics is None:
        print("Final test requested but no test metric was parsed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
