#!/usr/bin/env python3
"""Audit and summarize one dataset from the frozen ECL/Solar formal matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("TimeRole", "Recent96", "DLinear", "PatchTST", "iTransformer", "TimeMixer", "SMamba", "TimeFilter")
IMPLEMENTATION = {"Recent96": "GraphMambaRecent"}
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("electricity", "solar"))
    parser.add_argument("--source", type=Path, default=ROOT / "logs" / "timerole_p0" / "ecl_solar" / "formal")
    args = parser.parse_args()
    source = args.source.resolve()
    output = source / "final" / args.dataset
    rows: list[dict[str, object]] = []
    problems: list[dict[str, object]] = []
    expected_paths: set[Path] = set()

    for model in MODELS:
        implementation = IMPLEMENTATION.get(model, model)
        for seed in SEEDS:
            for horizon in HORIZONS:
                name = f"p0_{implementation.lower()}_{args.dataset}_l336_h{horizon}_s{seed}"
                path = source / "records" / f"{name}.json"
                expected_paths.add(path.resolve())
                errors: list[str] = []
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    payload = {}
                    errors.append(f"missing or invalid JSON: {exc}")
                expected = {
                    "status": "completed", "phase": "formal", "model": model,
                    "implementation_model": implementation, "dataset": args.dataset,
                    "horizon": horizon, "seq_len": 336, "seed": seed,
                    "split": "test", "test_accessed": True, "return_code": 0,
                    "checkpoint_selected_by": "validation_loss",
                }
                for key, value in expected.items():
                    if payload.get(key) != value:
                        errors.append(f"{key}: expected {value!r}, found {payload.get(key)!r}")
                for metric in ("mse", "mae"):
                    value = payload.get(metric)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        errors.append(f"non-finite or missing {metric}")
                origin = payload.get("origin_metrics_path")
                if not isinstance(origin, str) or not Path(origin).is_file():
                    errors.append("missing origin metrics")
                if errors:
                    problems.append({"record": str(path), "errors": errors})
                else:
                    rows.append({
                        "model": model, "dataset": args.dataset, "horizon": horizon,
                        "seed": seed, "mse": float(payload["mse"]), "mae": float(payload["mae"]),
                        "duration_seconds": payload.get("duration_seconds"), "record": str(path),
                    })

    record_dir = source / "records"
    for path in sorted(record_dir.glob(f"p0_*_{args.dataset}_l336_h*_s*.json")):
        if path.resolve() not in expected_paths:
            problems.append({"record": str(path), "errors": ["unexpected record"]})

    output.mkdir(parents=True, exist_ok=True)
    fields = ["model", "dataset", "horizon", "seed", "mse", "mae", "duration_seconds", "record"]
    with (output / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), int(row["horizon"])), []).append(row)
    with (output / "mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        fields2 = ["model", "dataset", "horizon", "n", "seeds", "mse_mean", "mse_std", "mae_mean", "mae_std"]
        writer = csv.DictWriter(handle, fieldnames=fields2); writer.writeheader()
        for (model, horizon), group in sorted(groups.items()):
            mse = [float(row["mse"]) for row in group]
            mae = [float(row["mae"]) for row in group]
            writer.writerow({
                "model": model, "dataset": args.dataset, "horizon": horizon, "n": len(group),
                "seeds": ";".join(str(row["seed"]) for row in sorted(group, key=lambda item: int(item["seed"]))),
                "mse_mean": statistics.mean(mse), "mse_std": statistics.stdev(mse),
                "mae_mean": statistics.mean(mae), "mae_std": statistics.stdev(mae),
            })
    audit = {
        "status": "completed" if len(rows) == 96 and not problems else "incomplete",
        "dataset": args.dataset, "expected": 96, "verified": len(rows), "problems": problems,
        "split": "test", "test_accessed": True,
        "checkpoint_selected_by": "validation_loss",
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
