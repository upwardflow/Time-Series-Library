#!/usr/bin/env python3
"""Derive a period-constrained local patch from training residual correlation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def moving_average_residual(x: np.ndarray, kernel: int) -> np.ndarray:
    pad = (kernel - 1) // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    cumulative = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(padded, axis=0)])
    trend = (cumulative[kernel:] - cumulative[:-kernel]) / kernel
    return x - trend


def lag_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    rows = []
    for lag in range(1, max_lag + 1):
        left = centered[lag:]
        right = centered[:-lag]
        numerator = np.sum(left * right, axis=0)
        denominator = np.sqrt(
            np.sum(left * left, axis=0) * np.sum(right * right, axis=0)
        ).clip(1e-12)
        rows.append(numerator / denominator)
    return np.stack(rows)


def proper_divisors(period: int) -> list[int]:
    return [
        value
        for value in range(2, period // 2 + 1)
        if period % value == 0
    ]


def derive_scale(x: np.ndarray, period: int, decay_ratio: float) -> dict:
    max_lag = period // 2
    acf_by_variable = lag_autocorrelation(x, max_lag)
    aggregate_acf = np.median(acf_by_variable, axis=1)
    lag_one = float(aggregate_acf[0])
    threshold = lag_one * decay_ratio
    crossing = next(
        (
            lag
            for lag in range(2, max_lag + 1)
            if aggregate_acf[lag - 1] <= threshold
        ),
        None,
    )
    used_fallback = crossing is None or lag_one <= 0.0
    raw_length = max_lag if crossing is None else crossing
    if lag_one <= 0.0:
        raw_length = 2

    candidates = proper_divisors(period)
    if not candidates:
        raise ValueError(
            f"period {period} has no proper divisor usable as a local patch"
        )
    ranked = sorted(
        candidates,
        key=lambda value: (abs(math.log(value / raw_length)), value),
    )
    selected = ranked[0]
    return {
        "decay_ratio": decay_ratio,
        "lag_one_correlation": lag_one,
        "crossing_threshold": threshold,
        "raw_correlation_length": raw_length,
        "used_fallback": used_fallback,
        "proper_divisors": candidates,
        "divisor_log_distances": {
            str(value): abs(math.log(value / raw_length)) for value in candidates
        },
        "selected_patch": selected,
        "selected_stride": max(1, selected // 2),
        "aggregate_acf": aggregate_acf.tolist(),
        "acf_by_variable": acf_by_variable.T.tolist(),
    }


def detected_period(dataset: str) -> int:
    path = (
        REPO_ROOT
        / "logs"
        / "graphmamba_period_candidates"
        / f"{dataset}_period_candidates.json"
    )
    payload = json.loads(path.read_text())
    return int(payload["top_candidates"][0]["period"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ETTh1", "ETTh2"), required=True)
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--moving-avg", type=int, default=25)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "graphmamba_local_scale",
    )
    args = parser.parse_args()
    if args.moving_avg < 1 or args.moving_avg % 2 != 1:
        parser.error("moving-avg must be a positive odd integer")
    if args.blocks < 2:
        parser.error("blocks must be at least 2")

    period = args.period or detected_period(args.dataset)
    csv_path = REPO_ROOT / "dataset" / "ETT-small" / f"{args.dataset}.csv"
    raw = pd.read_csv(csv_path).iloc[: 12 * 30 * 24, 1:].to_numpy(dtype=np.float64)
    normalized = (raw - raw.mean(axis=0, keepdims=True)) / raw.std(
        axis=0, keepdims=True
    ).clip(1e-12)
    residual = moving_average_residual(normalized, args.moving_avg)

    primary_ratio = 1.0 / math.e
    primary = derive_scale(residual, period, primary_ratio)
    block_results = [
        derive_scale(block, period, primary_ratio)
        for block in np.array_split(residual, args.blocks)
    ]
    block_agreement = np.mean(
        [
            block["selected_patch"] == primary["selected_patch"]
            for block in block_results
        ]
    )
    sensitivity = {
        label: derive_scale(residual, period, ratio)
        for label, ratio in (
            ("quarter", 0.25),
            ("e_folding", primary_ratio),
            ("half", 0.5),
        )
    }

    payload = {
        "dataset": args.dataset,
        "scope": "training_split_only",
        "train_samples": int(raw.shape[0]),
        "moving_average": args.moving_avg,
        "period": period,
        "rule": "median_acf_e_folding_then_nearest_period_divisor",
        "primary": primary,
        "chronological_blocks": block_results,
        "block_agreement": float(block_agreement),
        "threshold_sensitivity": sensitivity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}_local_scale.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    summary = {
        "dataset": args.dataset,
        "period": period,
        "raw_correlation_length": primary["raw_correlation_length"],
        "selected_patch": primary["selected_patch"],
        "selected_stride": primary["selected_stride"],
        "block_selected_patches": [
            block["selected_patch"] for block in block_results
        ],
        "block_agreement": float(block_agreement),
        "threshold_selected_patches": {
            label: result["selected_patch"]
            for label, result in sensitivity.items()
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
