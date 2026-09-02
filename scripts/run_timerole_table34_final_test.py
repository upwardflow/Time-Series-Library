#!/usr/bin/env python3
"""Evaluate the frozen validation-best checkpoints used by paper Tables 3 and 4.

The test split is evaluation-only: every checkpoint was selected by validation MSE
before this script runs. Existing seed-2021 final-test records are reused, while
strict-evidence seeds 2022/2023 and design variants are evaluated once and archived.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_VALIDATION = ROOT / "logs" / "timerole_all_horizons" / "validation"
STRICT_VALIDATION = ROOT / "logs" / "timerole_strict_evidence" / "validation"
SEED2021_TEST = ROOT / "logs" / "timerole_final_test" / "records"
OUTPUT = ROOT / "logs" / "timerole_table34_final_test"
DATASETS = ("ETTm1", "ETTm2")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
TEST_PATTERN = re.compile(r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:")
MODEL_RENAMES = {
    "GraphMambaRecent": "TimeRoleRecent",
    "GraphMamba": "TimeRoleFullHistory",
}


def read_completed(path: Path, require_test: bool = False) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = "test_mse" if require_test else "best_mse"
    return payload if payload.get("status") == "completed" and required in payload else None


def strict_jobs() -> list[dict]:
    jobs = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in (2022, 2023):
                for variant, label in (("r", "recent336"), ("c", "timerole")):
                    candidate = f"a_{dataset.lower()}_p{horizon}_{variant}_s{seed}"
                    jobs.append({"group": "table3", "dataset": dataset, "horizon": horizon,
                                 "seed": seed, "label": label, "candidate": candidate})
    for dataset in DATASETS:
        for horizon in (96, 720):
            for variant, label in (("cat", "Concat"), ("nd", "NoDiff"),
                                   ("gg", "GlobalGate")):
                candidate = f"c_{dataset.lower()}_p{horizon}_{variant}_s2021"
                jobs.append({"group": "table4", "dataset": dataset, "horizon": horizon,
                             "seed": 2021, "label": label, "candidate": candidate})
    assert len(jobs) == 44
    return jobs


def test_command(validation_record: dict) -> list[str]:
    command = list(validation_record["command"])
    if validation_record.get("final_test") is not False:
        raise RuntimeError("Source record is not a validation-only record")
    train_index = command.index("--is_training")
    test_index = command.index("--test_after_train")
    if command[test_index + 1] != "0":
        raise RuntimeError("Source training command accessed the test split")
    model_index = command.index("--model")
    command[model_index + 1] = MODEL_RENAMES.get(
        str(command[model_index + 1]), str(command[model_index + 1])
    )
    command[train_index + 1] = "0"
    command[test_index + 1] = "0"
    return command


def checkpoint_for(candidate: str) -> Path:
    matches = list((ROOT / "checkpoints").glob(f"*_{candidate}_0/checkpoint.pth"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {candidate}, found {len(matches)}")
    return matches[0]


def run_one(job: dict, gpu: int) -> int:
    destination = OUTPUT / "records" / f"{job['candidate']}.json"
    if read_completed(destination, require_test=True):
        print(f"already completed: {job['candidate']}", flush=True)
        return 0
    source = STRICT_VALIDATION / f"{job['candidate']}.json"
    validation_record = read_completed(source)
    if not validation_record:
        raise RuntimeError(f"Missing completed validation record: {source}")
    checkpoint = checkpoint_for(job["candidate"])
    command = test_command(validation_record)
    log_path = OUTPUT / "logs" / f"{job['candidate']}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    metrics = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
            match = TEST_PATTERN.match(line.strip())
            if match:
                metrics = {"test_mse": float(match.group(1)),
                           "test_mae": float(match.group(2))}
        return_code = process.wait()
    payload = {
        **job,
        "status": "completed" if return_code == 0 and metrics else "failed",
        "model": validation_record["model"],
        "checkpoint_selected_by": "validation_best_mse",
        "validation_best_epoch": validation_record["best_epoch"],
        "validation_best_mse": validation_record["best_mse"],
        "validation_best_mae": validation_record["best_mae"],
        "source_validation_record": str(source),
        "checkpoint": str(checkpoint),
        "command": command,
        "log_path": str(log_path),
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    if metrics:
        payload.update(metrics)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)
    return 0 if payload["status"] == "completed" else 1


def seed2021(dataset: str, horizon: int, label: str) -> dict:
    path = SEED2021_TEST / f"{dataset.lower()}_{horizon}_{label}_s2021.json"
    payload = read_completed(path, require_test=True)
    if not payload:
        raise RuntimeError(f"Missing seed-2021 final-test record: {path}")
    return payload


def new_test(candidate: str) -> dict:
    path = OUTPUT / "records" / f"{candidate}.json"
    payload = read_completed(path, require_test=True)
    if not payload:
        raise RuntimeError(f"Missing new final-test record: {path}")
    return payload


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate() -> dict:
    pairs, table3 = [], []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            task = []
            for seed in SEEDS:
                if seed == 2021:
                    recent = seed2021(dataset, horizon, "recent336")
                    timerole = seed2021(dataset, horizon, "timerole")
                else:
                    recent = new_test(f"a_{dataset.lower()}_p{horizon}_r_s{seed}")
                    timerole = new_test(f"a_{dataset.lower()}_p{horizon}_c_s{seed}")
                row = {
                    "dataset": dataset, "horizon": horizon, "seed": seed,
                    "recent_test_mse": recent["test_mse"], "timerole_test_mse": timerole["test_mse"],
                    "mse_improvement_pct": 100 * (recent["test_mse"] - timerole["test_mse"]) / recent["test_mse"],
                    "recent_test_mae": recent["test_mae"], "timerole_test_mae": timerole["test_mae"],
                    "mae_improvement_pct": 100 * (recent["test_mae"] - timerole["test_mae"]) / recent["test_mae"],
                }
                pairs.append(row)
                task.append(row)
            recent_mse = [x["recent_test_mse"] for x in task]
            timerole_mse = [x["timerole_test_mse"] for x in task]
            recent_mae = [x["recent_test_mae"] for x in task]
            timerole_mae = [x["timerole_test_mae"] for x in task]
            table3.append({
                "dataset": dataset, "horizon": horizon,
                "recent_test_mse_mean": statistics.fmean(recent_mse),
                "recent_test_mse_sd": statistics.stdev(recent_mse),
                "timerole_test_mse_mean": statistics.fmean(timerole_mse),
                "timerole_test_mse_sd": statistics.stdev(timerole_mse),
                "mean_mse_improvement_pct": 100 * (statistics.fmean(recent_mse) - statistics.fmean(timerole_mse)) / statistics.fmean(recent_mse),
                "recent_test_mae_mean": statistics.fmean(recent_mae),
                "recent_test_mae_sd": statistics.stdev(recent_mae),
                "timerole_test_mae_mean": statistics.fmean(timerole_mae),
                "timerole_test_mae_sd": statistics.stdev(timerole_mae),
            })

    table4 = []
    for dataset in DATASETS:
        for horizon in (96, 720):
            recent = seed2021(dataset, horizon, "recent336")
            full = seed2021(dataset, horizon, "timerole")
            controls = {
                label: new_test(f"c_{dataset.lower()}_p{horizon}_{code}_s2021")
                for code, label in (("cat", "Concat"), ("nd", "NoDiff"), ("gg", "GlobalGate"))
            }
            table4.append({
                "dataset": dataset, "horizon": horizon,
                "recent_test_mse": recent["test_mse"], "timerole_test_mse": full["test_mse"],
                "concat_test_mse": controls["Concat"]["test_mse"],
                "nodiff_test_mse": controls["NoDiff"]["test_mse"],
                "globalgate_test_mse": controls["GlobalGate"]["test_mse"],
                "recent_test_mae": recent["test_mae"], "timerole_test_mae": full["test_mae"],
                "concat_test_mae": controls["Concat"]["test_mae"],
                "nodiff_test_mae": controls["NoDiff"]["test_mae"],
                "globalgate_test_mae": controls["GlobalGate"]["test_mae"],
            })

    summary = {
        "table3_pairs": len(pairs),
        "table3_mse_wins": sum(x["timerole_test_mse"] < x["recent_test_mse"] for x in pairs),
        "table3_mae_wins": sum(x["timerole_test_mae"] < x["recent_test_mae"] for x in pairs),
        "table3_pair_macro_mse_improvement_pct": statistics.fmean(x["mse_improvement_pct"] for x in pairs),
        "table3_pair_macro_mae_improvement_pct": statistics.fmean(x["mae_improvement_pct"] for x in pairs),
        "table3_task_improvement_min_pct": min(x["mean_mse_improvement_pct"] for x in table3),
        "table3_task_improvement_max_pct": max(x["mean_mse_improvement_pct"] for x in table3),
        "table4_full_vs_recent_wins": sum(x["timerole_test_mse"] < x["recent_test_mse"] for x in table4),
        "table4_full_vs_concat_wins": sum(x["timerole_test_mse"] < x["concat_test_mse"] for x in table4),
        "table4_full_vs_nodiff_wins": sum(x["timerole_test_mse"] < x["nodiff_test_mse"] for x in table4),
        "table4_full_vs_globalgate_wins": sum(x["timerole_test_mse"] < x["globalgate_test_mse"] for x in table4),
        "generated_at": datetime.now().astimezone().isoformat(),
        "selection_rule": "validation_best_mse",
    }
    write_csv(OUTPUT / "table3_pairs.csv", pairs)
    write_csv(OUTPUT / "table3_test_mean_sd.csv", table3)
    write_csv(OUTPUT / "table4_test.csv", table4)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    gpu = int(os.environ.get("TABLE34_GPU", "0"))
    jobs = strict_jobs()
    for index, job in enumerate(jobs, start=1):
        print(f"\n=== [{index}/44] {job['candidate']} ===", flush=True)
        if run_one(job, gpu):
            return 1
    summary = aggregate()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
