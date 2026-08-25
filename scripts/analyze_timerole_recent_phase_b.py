#!/usr/bin/env python3
"""Recompute the frozen Phase B TimeRole simplification admission gate."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "timerole_recent_simplification"
RECORDS = OUTPUT / "records" / "phase_b"
RANKING = OUTPUT / "summaries" / "phase_b_candidate_ranking.csv"
GATE = OUTPUT / "audit" / "phase_b_gate.json"
VARIANTS = ("R0", "R1", "R2", "R3", "R4", "R5")
DATASETS = ("ETTm1", "ETTh2", "weather")
HORIZONS = (96, 720)
EPSILON = 1e-8


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def mean(rows: list[dict[str, object]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def pct_delta(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def main() -> int:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in RECORDS.glob("*.json")]
    expected = {(dataset, horizon, variant) for dataset in DATASETS for horizon in HORIZONS for variant in VARIANTS}
    cells = {(str(row["dataset"]), int(row["horizon"]), str(row["variant"])) for row in records}
    if cells != expected or len(records) != len(expected):
        raise SystemExit(f"expected 36 unique cells, found records={len(records)} cells={len(cells)}")
    if any(row.get("status") != "completed" for row in records):
        raise SystemExit("all Phase B records must be completed")
    if any(bool(row.get("test_accessed")) for row in records):
        raise SystemExit("Phase B record accessed the test split")
    if any(bool(row.get("source_dirty")) for row in records):
        raise SystemExit("Phase B record used tracked source modifications")
    source_commits = {str(row.get("git_commit_sha")) for row in records}
    if len(source_commits) != 1:
        raise SystemExit(f"Phase B records span multiple source commits: {source_commits}")
    if any(int(row.get("data_order_seed", -1)) != 2021 for row in records):
        raise SystemExit("Phase B records do not share the frozen data-order seed")
    if any(bool(row.get("validation_shuffle")) for row in records):
        raise SystemExit("Phase B validation loader was shuffled")
    for row in records:
        if not all(math.isfinite(float(row[key])) for key in ("mse", "mae")):
            raise SystemExit(f"non-finite metric in {row.get('candidate')}")

    by_cell = {
        (str(row["dataset"]), int(row["horizon"]), str(row["variant"])): row
        for row in records
    }
    baseline_rows = [by_cell[(dataset, horizon, "R0")] for dataset in DATASETS for horizon in HORIZONS]
    baseline_macro_mse = mean(baseline_rows, "mse")
    baseline_macro_mae = mean(baseline_rows, "mae")
    resource_keys = (
        "parameter_count",
        "milliseconds_per_batch",
        "train_peak_cuda_memory_bytes",
        "peak_cuda_memory_bytes",
    )

    results: list[dict[str, object]] = []
    for variant in VARIANTS:
        rows = [by_cell[(dataset, horizon, variant)] for dataset in DATASETS for horizon in HORIZONS]
        macro_mse = mean(rows, "mse")
        macro_mae = mean(rows, "mae")
        cell_deltas = {
            f"{dataset}-{horizon}": pct_delta(
                float(by_cell[(dataset, horizon, variant)]["mse"]),
                float(by_cell[(dataset, horizon, "R0")]["mse"]),
            )
            for dataset in DATASETS for horizon in HORIZONS
        }
        domain_deltas = {
            dataset: statistics.mean(cell_deltas[f"{dataset}-{horizon}"] for horizon in HORIZONS)
            for dataset in DATASETS
        }
        domain_systematic_degradation = {
            dataset: all(cell_deltas[f"{dataset}-{horizon}"] > 0.0 for horizon in HORIZONS)
            for dataset in DATASETS
        }
        reductions = {
            key: (1.0 - mean(rows, key) / mean(baseline_rows, key)) * 100.0
            for key in resource_keys
        }
        mse_nonlosses = sum(value <= 0.0 for value in cell_deltas.values())
        correction_rms_min = min(float(row["memory_correction_rms"]) for row in rows)
        checks = {
            "macro_mse_within_0_5pct": pct_delta(macro_mse, baseline_macro_mse) <= 0.5,
            "macro_mae_within_0_5pct": pct_delta(macro_mae, baseline_macro_mae) <= 0.5,
            "mse_nonlosses_at_least_4_of_6": mse_nonlosses >= 4,
            "parameter_or_latency_reduction": (
                reductions["parameter_count"] >= 20.0
                or reductions["milliseconds_per_batch"] >= 15.0
            ),
            "no_joint_etth2_weather_systematic_degradation": not (
                domain_systematic_degradation["ETTh2"]
                and domain_systematic_degradation["weather"]
            ),
            "timerole_correction_nonzero": correction_rms_min > EPSILON,
        }
        results.append({
            "variant": variant,
            "macro_mse": macro_mse,
            "macro_mse_delta_pct": pct_delta(macro_mse, baseline_macro_mse),
            "macro_mae": macro_mae,
            "macro_mae_delta_pct": pct_delta(macro_mae, baseline_macro_mae),
            "mse_nonlosses": mse_nonlosses,
            "worst_cell_mse_delta_pct": max(cell_deltas.values()),
            "cell_mse_delta_pct": cell_deltas,
            "domain_mean_mse_delta_pct": domain_deltas,
            "domain_systematic_degradation": domain_systematic_degradation,
            "parameter_reduction_pct": reductions["parameter_count"],
            "latency_reduction_pct": reductions["milliseconds_per_batch"],
            "train_peak_memory_reduction_pct": reductions["train_peak_cuda_memory_bytes"],
            "inference_peak_memory_reduction_pct": reductions["peak_cuda_memory_bytes"],
            "correction_rms_min": correction_rms_min,
            "checks": checks,
            "phase_b_pass": variant != "R0" and all(checks.values()),
        })

    accuracy_order = sorted(results, key=lambda item: float(item["macro_mse"]))
    rank = {str(item["variant"]): index for index, item in enumerate(accuracy_order, 1)}
    for result in results:
        result["macro_mse_rank"] = rank[str(result["variant"])]

    RANKING.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "variant", "macro_mse_rank", "macro_mse", "macro_mse_delta_pct",
        "macro_mae", "macro_mae_delta_pct", "mse_nonlosses",
        "worst_cell_mse_delta_pct", "parameter_reduction_pct",
        "latency_reduction_pct", "train_peak_memory_reduction_pct",
        "inference_peak_memory_reduction_pct", "correction_rms_min", "phase_b_pass",
    )
    with RANKING.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: int(item["macro_mse_rank"])):
            writer.writerow({key: result[key] for key in fields})

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "phase_b",
        "status": "analyzed",
        "record_count": len(records),
        "test_accessed": False,
        "source_dirty": False,
        "source_commits": sorted(source_commits),
        "data_order_seed": 2021,
        "validation_shuffle": False,
        "baseline": "R0",
        "baseline_macro_mse": baseline_macro_mse,
        "baseline_macro_mae": baseline_macro_mae,
        "gate_definition": {
            "macro_mse_max_degradation_pct": 0.5,
            "macro_mae_max_degradation_pct": 0.5,
            "minimum_mse_nonlosses": 4,
            "minimum_parameter_reduction_pct": 20.0,
            "minimum_latency_reduction_pct": 15.0,
            "maximum_phase_c_candidates": 2,
            "correction_nonzero_epsilon": EPSILON,
        },
        "passing_candidates": [str(item["variant"]) for item in results if item["phase_b_pass"]],
        "results": {str(item["variant"]): item for item in results},
        "ranking_csv": str(RANKING),
    }
    atomic_json(GATE, payload)
    print(json.dumps({
        "record_count": len(records),
        "passing_candidates": payload["passing_candidates"],
        "accuracy_order": [str(item["variant"]) for item in accuracy_order],
        "gate_json": str(GATE),
        "ranking_csv": str(RANKING),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
