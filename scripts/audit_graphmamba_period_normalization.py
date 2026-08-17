#!/usr/bin/env python3
"""Audit physical daily periods using training data and timestamps only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "ETTh1": (ROOT / "dataset/ETT-small/ETTh1.csv", "ett_hour"),
    "ETTh2": (ROOT / "dataset/ETT-small/ETTh2.csv", "ett_hour"),
    "ETTm1": (ROOT / "dataset/ETT-small/ETTm1.csv", "ett_minute"),
    "ETTm2": (ROOT / "dataset/ETT-small/ETTm2.csv", "ett_minute"),
    "weather": (ROOT / "dataset/weather/weather.csv", "custom"),
    "solar": (ROOT / "dataset/solar/solar.csv", "custom"),
}


def train_end(kind: str, rows: int) -> int:
    if kind == "ett_hour":
        return 12 * 30 * 24
    if kind == "ett_minute":
        return 12 * 30 * 24 * 4
    return int(rows * 0.7)


def moving_average_residual(values: np.ndarray, kernel: int) -> np.ndarray:
    if kernel % 2 != 1:
        raise ValueError("moving-average kernel must be odd")
    pad = (kernel - 1) // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0)]
    )
    trend = (cumulative[kernel:] - cumulative[:-kernel]) / kernel
    return values - trend


def candidate_periods(daily: int) -> list[int]:
    radius = max(4, int(round(0.1 * daily)))
    candidates = set(range(max(2, daily - radius), daily + radius + 1))
    candidates.update({max(2, daily // 2), daily, 2 * daily, 7 * daily})
    return sorted(candidates)


def score_periods(values: np.ndarray, candidates: list[int]) -> list[dict[str, float | int]]:
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(centered.shape[0])
    non_dc = spectrum[1:].sum(axis=0).clip(1e-12)
    rows: list[dict[str, float | int]] = []
    for period in candidates:
        if period >= centered.shape[0] // 2:
            continue
        left = centered[:-period]
        right = centered[period:]
        numerator = np.sum(left * right, axis=0)
        denominator = np.sqrt(
            np.sum(left * left, axis=0) * np.sum(right * right, axis=0)
        ).clip(1e-12)
        autocorr = numerator / denominator
        bin_index = int(np.argmin(np.abs(frequencies - 1.0 / period)))
        spectral = spectrum[bin_index] / non_dc
        combined = np.sqrt(np.maximum(spectral, 0.0) * np.maximum(autocorr, 0.0))
        rows.append(
            {
                "period": int(period),
                "spectral_median": float(np.median(spectral)),
                "autocorr_median": float(np.median(autocorr)),
                "combined_median": float(np.median(combined)),
                "autocorr_q25": float(np.quantile(autocorr, 0.25)),
                "autocorr_q75": float(np.quantile(autocorr, 0.75)),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["combined_median"]), int(row["period"])))


def audit_dataset(name: str) -> dict[str, object]:
    path, kind = DATASETS[name]
    frame = pd.read_csv(path)
    end = train_end(kind, len(frame))
    train = frame.iloc[:end]
    timestamps = pd.to_datetime(train["date"], utc=True)
    deltas_ns = np.diff(timestamps.astype("int64").to_numpy())
    if deltas_ns.size == 0:
        raise ValueError(f"{name}: insufficient timestamps")
    positive_seconds = deltas_ns[deltas_ns > 0] / 1e9
    if positive_seconds.size == 0:
        raise ValueError(f"{name}: no positive timestamp interval")
    median_seconds = float(np.median(positive_seconds))
    delta_seconds = deltas_ns / 1e9
    anomaly_mask = np.abs(delta_seconds - median_seconds) > 1e-6
    anomaly_indices = np.flatnonzero(anomaly_mask)
    anomaly_fraction = float(anomaly_mask.mean())
    if anomaly_fraction > 0.001:
        raise ValueError(
            f"{name}: timestamp anomaly fraction {anomaly_fraction:.6f} exceeds 0.001"
        )
    samples_per_hour_float = 3600.0 / median_seconds
    samples_per_hour = int(round(samples_per_hour_float))
    if not np.isclose(samples_per_hour_float, samples_per_hour, atol=1e-9):
        raise ValueError(f"{name}: sampling interval does not divide one hour")
    daily = 24 * samples_per_hour

    numeric = train.drop(columns=["date"]).apply(pd.to_numeric, errors="raise")
    raw = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError(f"{name}: non-finite value in training split")
    scale = raw.std(axis=0, keepdims=True).clip(1e-12)
    normalized = (raw - raw.mean(axis=0, keepdims=True)) / scale
    residual = moving_average_residual(normalized, daily + 1)
    candidates = candidate_periods(daily)
    ranking = score_periods(residual, candidates)
    rank_by_period = {int(row["period"]): index + 1 for index, row in enumerate(ranking)}
    score_by_period = {int(row["period"]): row for row in ranking}

    block_tops: list[int] = []
    block_daily_ranks: list[int] = []
    for block in np.array_split(residual, 4, axis=0):
        block_ranking = score_periods(block, candidates)
        block_tops.append(int(block_ranking[0]["period"]))
        block_ranks = {
            int(row["period"]): index + 1 for index, row in enumerate(block_ranking)
        }
        block_daily_ranks.append(int(block_ranks[daily]))

    return {
        "dataset": name,
        "scope": "training_split_only",
        "path": str(path.relative_to(ROOT)),
        "train_samples": int(end),
        "variables": int(raw.shape[1]),
        "sampling_interval_seconds": median_seconds,
        "timestamps_regular": bool(anomaly_indices.size == 0),
        "timestamp_anomaly_count": int(anomaly_indices.size),
        "timestamp_anomaly_fraction": anomaly_fraction,
        "timestamp_anomalies": [
            {
                "left_index": int(index),
                "left": str(train["date"].iloc[index]),
                "right": str(train["date"].iloc[index + 1]),
                "delta_seconds": float(delta_seconds[index]),
            }
            for index in anomaly_indices
        ],
        "samples_per_hour": samples_per_hour,
        "physical_daily_period": daily,
        "hourly_resample_factor": samples_per_hour,
        "detrend_kernel": daily + 1,
        "daily_rank": int(rank_by_period[daily]),
        "daily_score": score_by_period[daily],
        "top_candidates": ranking[:10],
        "block_top_periods": block_tops,
        "block_daily_ranks": block_daily_ranks,
        "daily_top_blocks": int(sum(period == daily for period in block_tops)),
    }


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    csv_path = output_dir / "period_audit_summary.csv"
    fields = [
        "dataset",
        "train_samples",
        "variables",
        "sampling_interval_seconds",
        "timestamp_anomaly_count",
        "samples_per_hour",
        "physical_daily_period",
        "hourly_resample_factor",
        "daily_rank",
        "daily_top_blocks",
        "block_top_periods",
        "block_daily_ranks",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    report = [
        "# GraphMamba v2 训练集周期审计",
        "",
        "- 数据使用：仅官方训练区间；未读取 validation/test 目标用于周期选择。",
        "- 周期来源：训练时间戳推导的物理日周期；ACF/FFT 只做支持性审计。",
        "",
        "| Dataset | interval(s) | anomalies | train | vars | samples/hour | daily P | daily rank | daily-top blocks | block tops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['dataset']} | {row['sampling_interval_seconds']:.0f} | "
            f"{row['timestamp_anomaly_count']} | "
            f"{row['train_samples']} | {row['variables']} | {row['samples_per_hour']} | "
            f"{row['physical_daily_period']} | {row['daily_rank']} | "
            f"{row['daily_top_blocks']}/4 | {row['block_top_periods']} |"
        )
    (output_dir / "period_audit_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs/graphmamba_period_normalized_v2/period_audit",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        result = audit_dataset(dataset)
        rows.append(result)
        (args.output_dir / f"{dataset}_period_audit.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{dataset}: interval={result['sampling_interval_seconds']:.0f}s, "
            f"P={result['physical_daily_period']}, rank={result['daily_rank']}, "
            f"blocks={result['block_top_periods']}"
        )
    write_summary(rows, args.output_dir)
    print(f"Saved audit to {args.output_dir}")


if __name__ == "__main__":
    main()
