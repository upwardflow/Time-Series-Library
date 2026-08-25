#!/usr/bin/env python3
"""Read-only aggregation and smoke gating for TimeRole simplification."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "timerole_recent_simplification"
VARIANTS = ("R0", "R1", "R2", "R3", "R4", "R5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "phase_b"), default="smoke")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def records(stage: str) -> list[dict[str, object]]:
    rows = []
    directory = OUTPUT / "records" / stage
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload["record_path"] = str(path)
            rows.append(payload)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "candidate", "stage", "role", "dataset", "horizon", "seed",
        "variant", "status", "mse", "mae", "best_epoch", "parameter_count",
        "train_duration_seconds", "milliseconds_per_batch",
        "train_peak_cuda_memory_bytes", "peak_cuda_memory_bytes",
        "memory_correction_mae", "memory_correction_rms", "test_accessed",
        "source_dirty", "record_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def smoke_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    structure_path = OUTPUT / "audit" / "structure_and_rng_audit.json"
    try:
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        structure = {}
    by_variant = {
        str(row.get("variant")): row for row in rows
        if row.get("role") == "timerole" and row.get("dataset") == "ETTm1"
        and row.get("horizon") == 96 and row.get("seed") == 2021
    }
    checks: dict[str, object] = {
        "structure_and_rng_audit_passed": structure.get("status") == "passed",
        "paired_train_orders_equal": structure.get("paired_train_orders_equal") is True,
        "six_variants_present": set(by_variant) == set(VARIANTS),
        "all_completed": all(by_variant.get(v, {}).get("status") == "completed" for v in VARIANTS),
        "no_test_access": all(by_variant.get(v, {}).get("test_accessed") is False for v in VARIANTS),
        "all_metrics_finite": all(
            math.isfinite(float(by_variant.get(v, {}).get(metric, float("nan"))))
            for v in VARIANTS for metric in ("mse", "mae")
        ),
        "all_logs_exist": all(
            Path(str(by_variant.get(v, {}).get("log_path", ""))).is_file()
            for v in VARIANTS
        ),
        "all_source_clean": all(by_variant.get(v, {}).get("source_dirty") is False for v in VARIANTS),
    }
    if set(by_variant) == set(VARIANTS):
        r0_params = int(by_variant["R0"].get("parameter_count", 0))
        checks["simplified_parameters_lower_than_r0"] = all(
            int(by_variant[v].get("parameter_count", r0_params)) < r0_params
            for v in VARIANTS if v != "R0"
        )
        checks["corrections_nonzero"] = all(
            float(by_variant[v].get("memory_correction_rms", 0.0)) > 0.0
            for v in VARIANTS
        )
    else:
        checks["simplified_parameters_lower_than_r0"] = False
        checks["corrections_nonzero"] = False
    passed = all(value is True for value in checks.values())
    return {
        "created_at": now(), "stage": "smoke", "status": "passed" if passed else "failed",
        "test_accessed": False, "checks": checks,
        "variants": {
            variant: {
                key: by_variant.get(variant, {}).get(key)
                for key in (
                    "status", "mse", "mae", "best_epoch", "parameter_count",
                    "train_duration_seconds", "milliseconds_per_batch",
                    "train_peak_cuda_memory_bytes", "peak_cuda_memory_bytes",
                    "memory_correction_mae", "memory_correction_rms",
                )
            }
            for variant in VARIANTS
        },
    }


def main() -> int:
    args = parse_args()
    rows = records(args.stage)
    summary = {
        "created_at": now(), "stage": args.stage, "record_count": len(rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "variant_counts": dict(Counter(str(row.get("variant")) for row in rows)),
        "test_accessed_count": sum(row.get("test_accessed") is not False for row in rows),
    }
    atomic_json(OUTPUT / "summaries" / f"{args.stage}_summary.json", summary)
    write_csv(OUTPUT / "summaries" / f"{args.stage}_summary.csv", rows)
    if args.stage == "smoke":
        gate = smoke_gate(rows)
        atomic_json(OUTPUT / "audit" / "smoke_gate.json", gate)
        summary["gate_status"] = gate["status"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
