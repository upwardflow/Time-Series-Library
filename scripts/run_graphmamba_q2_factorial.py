#!/usr/bin/env python3
"""Run the frozen validation-only 2x2 periodic-backbone x CMRHM audit.

This script never requests test evaluation.  It compares, under one protocol:

* b:  recent-96 GraphMamba with ordinary independent dual patches;
* p:  recent-96 GraphMamba with periodic multi-resolution patches;
* c:  b plus frozen CMRHM;
* pc: p plus frozen CMRHM.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ONE_RUN = ROOT / "scripts" / "run_graphmamba_innovation.py"
OUTPUT = ROOT / "logs" / "graphmamba_q2_factorial"
DATASETS = ("ETTh1", "ETTh2")
HORIZONS = (192, 720)
SEEDS = (2021, 2022, 2023)
VARIANTS = {
    "b": ("GraphMambaRecent", "independent_shared"),
    "p": ("GraphMambaRecent", "periodic_aligned"),
    "c": ("GraphMambaCMRHM", "independent_shared"),
    "pc": ("GraphMambaCMRHM", "periodic_aligned"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def jobs() -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "variant": variant,
            "model": model,
            "scan_mode": scan_mode,
        }
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
        for variant, (model, scan_mode) in VARIANTS.items()
    ]


def candidate(job: dict[str, object]) -> str:
    return (
        f"q2f_{str(job['dataset']).lower()}_p{job['horizon']}_"
        f"{job['variant']}_s{job['seed']}"
    )


def record_path(job: dict[str, object]) -> Path:
    return OUTPUT / "validation" / f"{candidate(job)}.json"


def completed(job: dict[str, object]) -> dict[str, object] | None:
    path = record_path(job)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "completed" or "best_mse" not in payload:
        return None
    return payload


def command(job: dict[str, object], args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ONE_RUN),
        "--dataset",
        str(job["dataset"]),
        "--pred-len",
        str(job["horizon"]),
        "--seq-len",
        "336",
        "--model",
        str(job["model"]),
        "--dual-scale-scan-mode",
        str(job["scan_mode"]),
        "--periodic-period",
        "24",
        "--periodic-local-patch",
        "4",
        "--periodic-local-stride",
        "2",
        "--periodic-period-stride",
        "12",
        "--periodic-use-adapter",
        "1",
        "--candidate",
        candidate(job),
        "--seed",
        str(job["seed"]),
        "--gpu",
        str(args.gpu),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--output-dir",
        str(OUTPUT),
    ]


def audit(job: dict[str, object], payload: dict[str, object]) -> None:
    if payload.get("final_test") is not False:
        raise RuntimeError(f"test-access violation: {candidate(job)}")
    recorded = payload.get("command", [])
    if not isinstance(recorded, list) or "--test_after_train" not in recorded:
        raise RuntimeError(f"missing test guard: {candidate(job)}")
    index = recorded.index("--test_after_train")
    if index + 1 >= len(recorded) or recorded[index + 1] != "0":
        raise RuntimeError(f"test-access violation: {candidate(job)}")


def write_status(done: list[str], failed: list[str], active: str | None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "status.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "protocol": "frozen 2x2 periodic x CMRHM; validation only",
                "total_jobs": len(jobs()),
                "completed_jobs": len(done),
                "failed_jobs": failed,
                "active_or_last_job": active,
                "test_accessed": False,
                "updated_at": datetime.now().astimezone().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    done: list[str] = []
    failed: list[str] = []
    all_jobs = jobs()
    for index, job in enumerate(all_jobs, start=1):
        name = candidate(job)
        write_status(done, failed, name)
        payload = completed(job)
        if payload is not None:
            audit(job, payload)
            done.append(name)
            print(f"[{index}/{len(all_jobs)}] already completed: {name}", flush=True)
            continue
        run_command = command(job, args)
        print(f"[{index}/{len(all_jobs)}] running: {name}", flush=True)
        if args.dry_run:
            print(" ".join(run_command), flush=True)
            continue
        return_code = subprocess.run(run_command, cwd=ROOT).returncode
        payload = completed(job)
        if return_code == 0 and payload is not None:
            audit(job, payload)
            done.append(name)
        else:
            failed.append(name)
            print(f"FAILED: {name}; return_code={return_code}", flush=True)
        write_status(done, failed, name)
    if not args.dry_run:
        write_status(done, failed, None)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

