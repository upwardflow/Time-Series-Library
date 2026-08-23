#!/usr/bin/env python3
"""Run recent-only Weather controls for the frozen TimeRole domain audit.

Only horizons 96 and 720 are used as short/long endpoint checks.  The paired
TimeRole validation metrics already exist from the frozen six-dataset run.  This
script never evaluates the test split.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
OUTPUT = ROOT / "logs" / "timerole_weather_controls"
HORIZONS = (96, 720)
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def candidate(horizon: int) -> str:
    return f"q2w_weather_p{horizon}_recent_s2021"


def record_path(horizon: int) -> Path:
    return OUTPUT / "validation" / f"{candidate(horizon)}.json"


def completed(horizon: int) -> dict[str, object] | None:
    path = record_path(horizon)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == "completed" and "best_mse" in payload:
        return payload
    return None


def command(horizon: int, args: argparse.Namespace) -> list[str]:
    name = candidate(horizon)
    return [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", str(ROOT / "dataset" / "weather"),
        "--data_path", "weather.csv",
        "--model_id", f"weather_336_{horizon}_{name}",
        "--model", "GraphMambaRecent",
        "--seed", "2021",
        "--data", "custom",
        "--features", "M",
        "--target", "CO2 (ppm)",
        "--seq_len", "336",
        "--label_len", "48",
        "--pred_len", str(horizon),
        "--enc_in", "21",
        "--dec_in", "21",
        "--c_out", "21",
        "--patch_len", "4",
        "--stride", "2",
        "--d_model", "64",
        "--d_ff", "128",
        "--d_state", "32",
        "--d_conv", "2",
        "--e_layers", "1",
        "--expand", "2",
        "--mamba_version", "1",
        "--mamba_bidirectional", "1",
        "--use_graph", "1",
        "--use_time_mamba", "1",
        "--use_patch", "1",
        "--use_decomp", "1",
        "--moving_avg", "25",
        "--dual_scale_scan_mode", "independent_shared",
        "--periodic_period", "24",
        "--periodic_local_patch", "4",
        "--periodic_local_stride", "2",
        "--periodic_period_stride", "12",
        "--periodic_use_adapter", "1",
        "--graph_alpha", "0.5",
        "--graph_top_k", "2",
        "--graph_sample_size", "2000",
        "--graph_sample_method", "uniform",
        "--static_graph_mode", "weighted",
        "--graph_cache", "1",
        "--dropout", "0.1",
        "--batch_size", "32",
        "--learning_rate", "0.0005",
        "--lradj", "type1",
        "--train_epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--num_workers", "0",
        "--gpu", "0",
        "--des", name,
        "--itr", "1",
        "--test_after_train", "0",
    ]


def run_one(horizon: int, args: argparse.Namespace) -> int:
    if completed(horizon) is not None:
        print(f"already completed: {candidate(horizon)}", flush=True)
        return 0
    run_command = command(horizon, args)
    print(" ".join(run_command), flush=True)
    if args.dry_run:
        return 0
    destination = record_path(horizon)
    log_path = destination.with_suffix(".log")
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    validation = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            run_command,
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
            match = VALIDATION_PATTERN.match(line.strip())
            if match:
                validation = json.loads(match.group(1))
        return_code = process.wait()
    payload: dict[str, object] = {
        "status": "completed" if return_code == 0 and validation else "failed",
        "dataset": "weather",
        "pred_len": horizon,
        "seed": 2021,
        "model": "GraphMambaRecent",
        "final_test": False,
        "test_accessed": False,
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "command": run_command,
        "log_path": str(log_path),
    }
    if validation:
        payload.update(validation)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return 0 if payload["status"] == "completed" else 1


def finalize() -> None:
    full_source = ROOT / "logs" / "timerole_six_dataset_final" / "records"
    rows = []
    for horizon in HORIZONS:
        recent = completed(horizon)
        if recent is None:
            raise RuntimeError(f"missing recent control for horizon {horizon}")
        full = json.loads(
            (full_source / f"sixds_weather_p{horizon}_s2021.json").read_text(encoding="utf-8")
        )
        recent_mse = float(recent["best_mse"])
        recent_mae = float(recent["best_mae"])
        timerole_mse = float(full["validation_best_mse"])
        timerole_mae = float(full["validation_best_mae"])
        rows.append(
            {
                "horizon": horizon,
                "recent_mse": recent_mse,
                "timerole_mse": timerole_mse,
                "mse_improvement_pct": 100 * (recent_mse - timerole_mse) / recent_mse,
                "recent_mae": recent_mae,
                "timerole_mae": timerole_mae,
                "mae_improvement_pct": 100 * (recent_mae - timerole_mae) / recent_mae,
            }
        )
    macro_mse = sum(row["mse_improvement_pct"] for row in rows) / len(rows)
    macro_mae = sum(row["mae_improvement_pct"] for row in rows) / len(rows)
    summary = {
        "dataset": "weather",
        "validation_only": True,
        "rows": rows,
        "macro_mse_improvement_pct": macro_mse,
        "macro_mae_improvement_pct": macro_mae,
        "mse_wins": sum(row["mse_improvement_pct"] > 0 for row in rows),
        "mae_wins": sum(row["mae_improvement_pct"] > 0 for row in rows),
        "gate_pass": (
            all(row["mse_improvement_pct"] > 0 for row in rows)
            and macro_mse >= 1.0
            and macro_mae >= 0.0
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    failed = [horizon for horizon in HORIZONS if run_one(horizon, args)]
    if failed:
        return 1
    if not args.dry_run:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

