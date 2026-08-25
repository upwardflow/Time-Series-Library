#!/usr/bin/env python3
"""Read-only aggregation for saved SCSD validation records."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "graphmamba_scsd_validation"
RECORDS = OUTPUT / "records"
HISTORICAL = ROOT / "logs" / "graphmamba_scan_mode_validation" / "validation"


def phase0_reproduction(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons = []
    old_variant = {"J": "joint", "IS": "independent"}
    for row in rows:
        if row.get("stage") != "phase0" or row.get("variant") not in old_variant:
            continue
        reference_path = HISTORICAL / (
            f"scan_{old_variant[str(row['variant'])]}_"
            f"{str(row['dataset']).lower()}_192_s2021.json"
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        current_mse = float(row["best_mse"])
        current_mae = float(row["best_mae"])
        reference_mse = float(reference["best_mse"])
        reference_mae = float(reference["best_mae"])
        comparisons.append({
            "dataset": row["dataset"], "variant": row["variant"],
            "current_mse": current_mse, "reference_mse": reference_mse,
            "mse_relative_error": abs(current_mse - reference_mse) / reference_mse,
            "current_mae": current_mae, "reference_mae": reference_mae,
            "mae_relative_error": abs(current_mae - reference_mae) / reference_mae,
            "best_epoch_equal": int(row["best_epoch"]) == int(reference["best_epoch"]),
        })
    return comparisons


def main() -> int:
    rows = []
    for path in sorted(RECORDS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "candidate": payload.get("candidate"), "stage": payload.get("stage"),
            "dataset": payload.get("dataset"), "horizon": payload.get("horizon"),
            "seed": payload.get("seed"), "variant": payload.get("variant"),
            "status": payload.get("status"), "best_mse": payload.get("best_mse"),
            "best_mae": payload.get("best_mae"), "best_epoch": payload.get("best_epoch"),
            "parameter_count": payload.get("parameter_count"),
            "duration_seconds": payload.get("duration_seconds"),
            "train_duration_seconds": payload.get("train_duration_seconds"),
            "train_peak_cuda_memory_bytes": payload.get("train_peak_cuda_memory_bytes"),
            "inference_milliseconds_per_batch": payload.get("inference_milliseconds_per_batch"),
            "peak_cuda_memory_bytes": payload.get("peak_cuda_memory_bytes"),
            "test_accessed": payload.get("test_accessed"),
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["candidate", "stage", "dataset", "horizon", "seed", "variant", "status"]
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(str(row["status"]) for row in rows)
    reproduction = phase0_reproduction(rows)
    summary = {
        "record_count": len(rows), "status_counts": dict(counts),
        "test_accessed_any": any(row.get("test_accessed") is True for row in rows),
        "baseline_expected": 3 * 2 * 3 * 5,
        "baseline_completed": sum(row["stage"] == "baseline" and row["status"] == "completed" for row in rows),
        "phase0_completed": sum(row["stage"] == "phase0" and row["status"] == "completed" for row in rows),
        "phase0_reproduction": reproduction,
        "phase0_reproduction_pass": (
            len(reproduction) == 4
            and all(row["mse_relative_error"] <= 1e-6 for row in reproduction)
            and all(row["mae_relative_error"] <= 1e-6 for row in reproduction)
            and all(row["best_epoch_equal"] for row in reproduction)
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
