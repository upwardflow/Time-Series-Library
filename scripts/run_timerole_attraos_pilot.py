#!/usr/bin/env python3
"""Run one validation-only Attraos compatibility pilot with an atomic record."""

from __future__ import annotations

import argparse
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
ATTRAOS = ROOT / "external" / "timerole_p0" / "Attraos"
RUN_PY = ATTRAOS / "run.py"
DEFAULT_OUTPUT = ROOT / "logs" / "timerole_p0" / "closest" / "attraos_pilot"
RESULT_PATTERN = re.compile(r"^EVALUATION_RESULT\s+(\{.*\})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.timeout_seconds < 1:
        parser.error("epochs, patience, and timeout must be positive")
    args.output_dir = args.output_dir.resolve()
    return args


def build_command(args: argparse.Namespace) -> list[str]:
    name = f"p0_attraos_ettm1_h96_s{args.seed}"
    return [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        name,
        "--model",
        "Attraos",
        "--seed",
        str(args.seed),
        "--data",
        "ETTm1",
        "--root_path",
        str(ROOT / "dataset" / "ETT-small"),
        "--data_path",
        "ETTm1.csv",
        "--features",
        "M",
        "--target",
        "OT",
        "--freq",
        "t",
        "--seq_len",
        "336",
        "--label_len",
        "48",
        "--pred_len",
        "96",
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
        "--patch_len",
        "16",
        "--stride",
        "16",
        "--e_layers",
        "1",
        "--batch_size",
        "32",
        "--learning_rate",
        "0.0001",
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--num_workers",
        "0",
        "--gpu",
        str(args.gpu),
        "--checkpoints",
        str(args.output_dir / "checkpoints"),
        "--des",
        name,
        "--itr",
        "1",
        "--test_after_train",
        "0",
        "--evaluation_split",
        "val",
        "--FFT_evolve",
        "True",
        "--multi_res",
        "True",
    ]


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    command = build_command(args)
    print("COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not RUN_PY.is_file():
        print(f"missing Attraos entrypoint: {RUN_PY}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"attraos_ettm1_h96_s{args.seed}.log"
    record_path = args.output_dir / f"attraos_ettm1_h96_s{args.seed}.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["MPLCONFIGDIR"] = str(args.output_dir / "matplotlib-cache")
    result = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ATTRAOS,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = started + args.timeout_seconds
        while True:
            for key, _ in selector.select(timeout=1.0):
                line = key.fileobj.readline()
                if line:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                    match = RESULT_PATTERN.match(line.strip())
                    if match:
                        result = json.loads(match.group(1))
            if process.poll() is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    match = RESULT_PATTERN.match(line.strip())
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

    success = (
        return_code == 0
        and isinstance(result, dict)
        and result.get("split") == "val"
        and result.get("test_accessed") is False
    )
    payload: dict[str, object] = {
        "status": "completed" if success else "failed",
        "stage": "compatibility_pilot",
        "model": "Attraos",
        "dataset": "ETTm1",
        "horizon": 96,
        "seed": args.seed,
        "seq_len": 336,
        "split": "validation",
        "test_accessed": False if success else None,
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "command": command,
        "log_path": str(log_path),
        "official_commit": "b2c7307269a844d6ae2608a0180c22d4a8b711f4",
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if result:
        payload.update(result)
    atomic_write(record_path, payload)
    print("PILOT_RESULT " + json.dumps(payload, sort_keys=True), flush=True)
    if not success:
        print("Attraos pilot failed; no automatic retry", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
