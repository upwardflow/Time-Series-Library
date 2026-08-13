#!/usr/bin/env python3
"""Estimate stable period candidates from the training split only."""

from __future__ import annotations

import argparse
import json
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


def period_scores(x: np.ndarray, min_period: int, max_period: int) -> list[dict]:
    x = x - x.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(x, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[0])
    non_dc_total = spectrum[1:].sum(axis=0).clip(1e-12)
    variance = np.sum(x * x, axis=0).clip(1e-12)
    rows = []
    for period in range(min_period, max_period + 1):
        target = 1.0 / period
        bin_index = int(np.argmin(np.abs(frequencies - target)))
        spectral = spectrum[bin_index] / non_dc_total
        autocorr = np.sum(x[period:] * x[:-period], axis=0) / variance
        combined = np.sqrt(np.maximum(spectral, 0.0) * np.maximum(autocorr, 0.0))
        rows.append(
            {
                "period": period,
                "spectral_mean": float(spectral.mean()),
                "autocorr_mean": float(autocorr.mean()),
                "combined_mean": float(combined.mean()),
                "spectral_by_variable": spectral.tolist(),
                "autocorr_by_variable": autocorr.tolist(),
            }
        )
    return sorted(rows, key=lambda row: row["combined_mean"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ETTh1", "ETTh2"), required=True)
    parser.add_argument("--moving-avg", type=int, default=25)
    parser.add_argument("--min-period", type=int, default=4)
    parser.add_argument("--max-period", type=int, default=48)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "graphmamba_period_candidates",
    )
    args = parser.parse_args()
    if args.moving_avg < 1 or args.moving_avg % 2 != 1:
        parser.error("moving-avg must be a positive odd integer")
    if not 2 <= args.min_period <= args.max_period:
        parser.error("period bounds are invalid")

    csv_path = REPO_ROOT / "dataset" / "ETT-small" / f"{args.dataset}.csv"
    raw = pd.read_csv(csv_path).iloc[: 12 * 30 * 24, 1:].to_numpy(dtype=np.float64)
    normalized = (raw - raw.mean(axis=0, keepdims=True)) / raw.std(
        axis=0, keepdims=True
    ).clip(1e-12)
    residual = moving_average_residual(normalized, args.moving_avg)
    ranking = period_scores(residual, args.min_period, args.max_period)
    payload = {
        "dataset": args.dataset,
        "scope": "training_split_only",
        "train_samples": int(raw.shape[0]),
        "moving_average": args.moving_avg,
        "top_candidates": ranking[:10],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}_period_candidates.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
