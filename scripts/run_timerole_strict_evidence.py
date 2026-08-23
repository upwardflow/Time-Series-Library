#!/usr/bin/env python3
"""Run the preregistered 52-job, validation-only TimeRole-v1 audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ONE_RUN = ROOT / "scripts" / "run_graphmamba_innovation.py"
OUTPUT = ROOT / "logs" / "timerole_strict_evidence"
DATASETS = ("ETTm1", "ETTm2")
HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def candidate(job: dict) -> str:
    return (f"{job['group']}_{job['dataset'].lower()}_p{job['horizon']}_"
            f"{job['variant']}_s{job['seed']}")


def jobs() -> list[dict]:
    result = []
    # Group A: seed 2021 is reused; train paired seeds 2022 and 2023.
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in (2022, 2023):
                for variant, model in (("r", "GraphMambaRecent"),
                                       ("c", "TimeRole")):
                    result.append(dict(group="a", dataset=dataset, horizon=horizon,
                                       seed=seed, variant=variant, model=model))
    # Group B: Recent336 and TimeRole336 are reused; only train Raw336.
    for dataset in DATASETS:
        for horizon in HORIZONS:
            result.append(dict(group="b", dataset=dataset, horizon=horizon,
                               seed=2021, variant="raw", model="GraphMamba"))
    # Group C: full TimeRole is reused; train three isolated controls.
    controls = (("cat", "TimeRoleConcat"),
                ("nd", "TimeRoleNoDiff"),
                ("gg", "TimeRoleGlobalGate"))
    for dataset in DATASETS:
        for horizon in (96, 720):
            for variant, model in controls:
                result.append(dict(group="c", dataset=dataset, horizon=horizon,
                                   seed=2021, variant=variant, model=model))
    assert len(result) == 52
    return result


def record_path(job: dict) -> Path:
    return OUTPUT / "validation" / f"{candidate(job)}.json"


def completed(job: dict) -> dict | None:
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


def command(job: dict, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-u", str(ONE_RUN),
        "--dataset", job["dataset"], "--pred-len", str(job["horizon"]),
        "--seq-len", "336", "--model", job["model"],
        "--candidate", candidate(job), "--seed", str(job["seed"]),
        "--gpu", str(args.gpu), "--epochs", str(args.epochs),
        "--patience", str(args.patience), "--output-dir", str(OUTPUT),
    ]


def write_status(done: list[str], failed: list[str], active: str | None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "protocol": "TimeRole-v1 strict evidence audit; validation only",
        "total_new_jobs": 52,
        "completed_new_jobs": len(done),
        "failed_jobs": failed,
        "active_or_last_job": active,
        "test_accessed": False,
        "updated_at": datetime.now().astimezone().isoformat(),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_record(job: dict, payload: dict) -> None:
    command_line = payload.get("command", [])
    if payload.get("final_test") is not False:
        raise RuntimeError(f"final_test integrity failure: {candidate(job)}")
    if "--test_after_train" not in command_line:
        raise RuntimeError(f"missing test_after_train flag: {candidate(job)}")
    index = command_line.index("--test_after_train")
    if index + 1 >= len(command_line) or command_line[index + 1] != "0":
        raise RuntimeError(f"test access integrity failure: {candidate(job)}")


def main() -> int:
    args = parse_args()
    done, failed = [], []
    for index, job in enumerate(jobs(), start=1):
        name = candidate(job)
        write_status(done, failed, name)
        payload = completed(job)
        if payload:
            audit_record(job, payload)
            done.append(name)
            print(f"[{index}/52] already completed: {name}", flush=True)
            continue
        run_command = command(job, args)
        print(f"[{index}/52] running: {name}", flush=True)
        if args.dry_run:
            print(" ".join(run_command), flush=True)
            continue
        return_code = subprocess.run(run_command, cwd=ROOT).returncode
        payload = completed(job)
        if return_code == 0 and payload:
            audit_record(job, payload)
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
