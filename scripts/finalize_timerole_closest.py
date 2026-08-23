#!/usr/bin/env python3
"""Audit and summarize the 24-job formal Attraos/DiM ETT comparison."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs" / "timerole_p0" / "closest" / "formal"
OUTPUT = SOURCE / "final"
MODELS = ("Attraos", "DiM")
DATASETS = ("ETTm1", "ETTm2")
HORIZONS = (96, 720)
SEEDS = (2021, 2022, 2023)
OFFICIAL = {
    "Attraos": ("b2c7307269a844d6ae2608a0180c22d4a8b711f4", "no_license_file_found_do_not_redistribute"),
    "DiM": ("73f60a7ff955c17817a115649e97c06fb7d1e143", "MIT"),
}


def slug(model: str, dataset: str, horizon: int, seed: int) -> str:
    return f"closest_{model.lower()}_{dataset.lower()}_l336_h{horizon}_s{seed}"


def record_path(model: str, dataset: str, horizon: int, seed: int) -> Path:
    return SOURCE / "records" / f"{slug(model, dataset, horizon, seed)}.json"


def flag(command: object, option: str) -> str | None:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return None
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    rows: list[dict[str, object]] = []
    problems: list[dict[str, object]] = []
    expected_paths: set[Path] = set()
    for model in MODELS:
        expected_commit, expected_license = OFFICIAL[model]
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    path = record_path(model, dataset, horizon, seed)
                    expected_paths.add(path.resolve())
                    errors = []
                    if not path.is_file():
                        errors.append("missing record")
                        payload = {}
                    else:
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError) as exc:
                            payload = {}
                            errors.append(f"invalid JSON: {exc}")
                    expected = {
                        "status": "completed", "model": model, "dataset": dataset,
                        "horizon": horizon, "seed": seed, "seq_len": 336,
                        "split": "test", "test_accessed": True,
                        "checkpoint_selected_by": "validation_loss_early_stopping",
                        "test_access": "one_shot_after_validation_selection",
                        "official_commit": expected_commit, "license": expected_license,
                    }
                    for key, value in expected.items():
                        if payload.get(key) != value:
                            errors.append(f"{key}: expected {value!r}, found {payload.get(key)!r}")
                    for metric in ("mse", "mae"):
                        value = payload.get(metric)
                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                            errors.append(f"non-finite or missing {metric}")
                    command = payload.get("command")
                    command_expected = {
                        "--model": model, "--data": dataset,
                        "--seq_len": "336", "--pred_len": str(horizon),
                        "--seed": str(seed), "--test_after_train": "1",
                        "--evaluation_split": "test",
                    }
                    for option, value in command_expected.items():
                        if flag(command, option) != value:
                            errors.append(f"command {option}: expected {value!r}, found {flag(command, option)!r}")
                    if errors:
                        problems.append({"record": str(path), "errors": errors})
                    else:
                        rows.append({
                            "model": model, "dataset": dataset, "horizon": horizon,
                            "seed": seed, "mse": float(payload["mse"]),
                            "mae": float(payload["mae"]),
                            "parameter_count": payload.get("parameter_count"),
                            "duration_seconds": payload.get("duration_seconds"),
                            "official_commit": expected_commit, "license": expected_license,
                            "record": str(path),
                        })
    record_dir = SOURCE / "records"
    unexpected = sorted(
        str(path) for path in record_dir.glob("*.json")
        if path.resolve() not in expected_paths
    ) if record_dir.is_dir() else []
    if unexpected:
        problems.append({"record": str(record_dir), "errors": [f"unexpected records: {unexpected}"]})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "dataset", "horizon", "seed", "mse", "mae",
        "parameter_count", "duration_seconds", "official_commit", "license", "record",
    ]
    with (OUTPUT / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["dataset"]), int(row["horizon"])), []).append(row)
    mean_fields = ["model", "dataset", "horizon", "n", "seeds", "mse_mean", "mse_std", "mae_mean", "mae_std"]
    with (OUTPUT / "mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mean_fields); writer.writeheader()
        for key in sorted(grouped):
            group = grouped[key]
            mse = [float(row["mse"]) for row in group]
            mae = [float(row["mae"]) for row in group]
            writer.writerow({
                "model": key[0], "dataset": key[1], "horizon": key[2], "n": len(group),
                "seeds": ";".join(str(row["seed"]) for row in sorted(group, key=lambda row: int(row["seed"]))),
                "mse_mean": statistics.mean(mse), "mse_std": statistics.stdev(mse) if len(mse) > 1 else "",
                "mae_mean": statistics.mean(mae), "mae_std": statistics.stdev(mae) if len(mae) > 1 else "",
            })
    status = {
        "status": "completed" if len(rows) == 24 and not problems else "incomplete",
        "expected": 24, "verified": len(rows), "problems": problems,
        "checkpoint_selected_by": "validation_loss_early_stopping",
        "test_access": "one_shot_after_validation_selection",
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(OUTPUT / "audit.json", status)
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
