#!/usr/bin/env python3
"""Audit and summarize the three-seed TimeRole RGSP ablation records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ETTm1", "ETTm2")
HORIZONS = (96, 720)
VARIANTS = ("no_decomp", "no_patch", "uni_mamba", "no_mamba", "no_graph")
SEEDS = (2021, 2022, 2023)
RECORD_ROOTS = {
    2021: ROOT / "logs" / "graphmamba_backbone_ablation" / "records",
    2022: ROOT / "logs" / "timerole_p0" / "recent_backbone" / "seed2022" / "records",
    2023: ROOT / "logs" / "timerole_p0" / "recent_backbone" / "seed2023" / "records",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs" / "timerole_p0" / "recent_backbone" / "final",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write an interim audit even if some expected records are missing",
    )
    return parser.parse_args()


def option_value(command: list[object], option: str) -> str | None:
    command = [str(item) for item in command]
    if option not in command:
        return None
    index = command.index(option)
    return command[index + 1] if index + 1 < len(command) else None


def record_path(seed: int, dataset: str, horizon: int, variant: str) -> Path:
    return RECORD_ROOTS[seed] / (
        f"table2_{dataset.lower()}_p{horizon}_{variant}_s{seed}.json"
    )


def audit_record(
    path: Path, seed: int, dataset: str, horizon: int, variant: str
) -> tuple[dict[str, object] | None, list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return None, ["missing record"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"unreadable JSON: {exc}"]

    expected = {
        "status": "completed",
        "dataset": dataset,
        "horizon": horizon,
        "variant": variant,
        "seed": seed,
        "return_code": 0,
        "test_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"{key}={payload.get(key)!r}, expected {value!r}")
    # Seed 2021 records predate the explicit top-level model field.  The
    # immutable command remains the authoritative model provenance there.
    if payload.get("model") not in {None, "GraphMambaRecent"}:
        errors.append(
            f"model={payload.get('model')!r}, expected 'GraphMambaRecent'"
        )
    if payload.get("split") not in {"val", "validation"}:
        errors.append(f"split={payload.get('split')!r}, expected validation")

    for metric in ("mse", "mae"):
        value = payload.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{metric} is not finite")
        elif value < 0 or value >= 10:
            errors.append(f"{metric}={value!r} outside [0, 10)")

    command = payload.get("command")
    if not isinstance(command, list):
        errors.append("command is not a list")
    else:
        required_options = {
            "--model": "GraphMambaRecent",
            "--seed": str(seed),
            "--data": dataset,
            "--pred_len": str(horizon),
            "--seq_len": "336",
            "--test_after_train": "0",
            "--evaluation_split": "val",
        }
        for option, expected_value in required_options.items():
            actual = option_value(command, option)
            if actual != expected_value:
                errors.append(f"{option}={actual!r}, expected {expected_value!r}")

    training_record = payload.get("training_record")
    if not isinstance(training_record, str) or not Path(training_record).is_file():
        errors.append("training_record is missing")
    else:
        try:
            training = json.loads(Path(training_record).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"training_record unreadable: {exc}")
        else:
            if training.get("status") != "completed":
                errors.append("training_record status is not completed")
            if training.get("test_accessed") is not False:
                errors.append("training_record does not prove test isolation")

    return payload, errors


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    problems: list[dict[str, object]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            for dataset in DATASETS:
                for horizon in HORIZONS:
                    path = record_path(seed, dataset, horizon, variant)
                    payload, errors = audit_record(
                        path, seed, dataset, horizon, variant
                    )
                    if errors:
                        problems.append({"record": str(path), "errors": errors})
                        continue
                    assert payload is not None
                    rows.append(
                        {
                            "variant": variant,
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                            "validation_mse": payload["mse"],
                            "validation_mae": payload["mae"],
                            "parameter_count": payload["parameter_count"],
                            "record": str(path),
                        }
                    )

    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["variant"]), str(row["dataset"]), int(row["horizon"]))
        grouped.setdefault(key, []).append(row)
    mean_std_rows: list[dict[str, object]] = []
    for (variant, dataset, horizon), group in sorted(grouped.items()):
        mse = [float(row["validation_mse"]) for row in group]
        mae = [float(row["validation_mae"]) for row in group]
        mean_std_rows.append(
            {
                "variant": variant,
                "dataset": dataset,
                "horizon": horizon,
                "n": len(group),
                "validation_mse_mean": statistics.fmean(mse),
                "validation_mse_std": statistics.stdev(mse) if len(mse) > 1 else 0.0,
                "validation_mae_mean": statistics.fmean(mae),
                "validation_mae_std": statistics.stdev(mae) if len(mae) > 1 else 0.0,
            }
        )

    expected = len(SEEDS) * len(VARIANTS) * len(DATASETS) * len(HORIZONS)
    complete = len(rows) == expected and not problems
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "records.csv", rows)
    write_csv(args.output_dir / "mean_std.csv", mean_std_rows)
    audit = {
        "status": "completed" if complete else "incomplete",
        "expected": expected,
        "verified": len(rows),
        "problems": problems,
        "split": "validation",
        "test_accessed": False if rows else None,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0 if complete or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
