"""Leakage-safe static graph construction for GraphMamba."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import dcor
import numpy as np
import pandas as pd


def _train_end(data_length: int, dataset_name: str) -> int:
    if dataset_name in {"ETTh1", "ETTh2"}:
        return min(data_length, 12 * 30 * 24)
    if dataset_name in {"ETTm1", "ETTm2"}:
        return min(data_length, 12 * 30 * 24 * 4)
    return int(data_length * 0.7)


def _select_columns(frame: pd.DataFrame, features: str, target: str) -> pd.DataFrame:
    value_columns = [column for column in frame.columns if column != "date"]
    if features == "M":
        return frame[value_columns]
    if target not in value_columns:
        raise ValueError(
            f"Target column {target!r} was not found for features={features!r}"
        )
    if features == "S":
        return frame[[target]]
    if features != "MS":
        raise ValueError("features must be one of: M, S, MS")
    ordered_columns = [column for column in value_columns if column != target] + [target]
    return frame[ordered_columns]


def generate_adjacency(
    data_path: str | Path,
    dataset_name: str,
    features: str = "M",
    target: str = "OT",
    sample_size: int = 2000,
    sample_method: str = "uniform",
    random_seed: int = 2021,
    cache: bool = False,
) -> np.ndarray:
    """Build a distance-correlation graph using training rows only."""
    data_path = Path(data_path)
    cache_name = (
        f"{data_path.stem}_adj_train_{dataset_name}_{features}_"
        f"{sample_method}_{sample_size}_seed{random_seed}.npy"
    )
    cache_path = data_path.with_name(cache_name)
    if cache and cache_path.exists():
        adjacency = np.load(cache_path)
        print(f"[GraphMamba] Loaded static graph: {cache_path}")
        return adjacency

    frame = pd.read_csv(data_path)
    values = _select_columns(frame, features, target).to_numpy(dtype=np.float64)
    train_end = _train_end(len(values), dataset_name)
    values = values[:train_end]

    if sample_size > 0 and len(values) > sample_size:
        if sample_method == "uniform":
            indices = np.linspace(0, len(values) - 1, sample_size, dtype=np.int64)
        elif sample_method == "random":
            rng = np.random.default_rng(random_seed)
            indices = np.sort(rng.choice(len(values), sample_size, replace=False))
        elif sample_method == "recent":
            indices = np.arange(len(values) - sample_size, len(values))
        else:
            raise ValueError("graph_sample_method must be uniform, random, or recent")
        values = values[indices]

    values = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-5)
    n_vars = values.shape[1]
    adjacency = np.eye(n_vars, dtype=np.float32)
    print(
        f"[GraphMamba] Building {n_vars}x{n_vars} static graph from "
        f"{len(values)} training samples"
    )
    for row in range(n_vars):
        for column in range(row + 1, n_vars):
            correlation = float(
                dcor.distance_correlation(values[:, row], values[:, column])
            )
            adjacency[row, column] = correlation
            adjacency[column, row] = correlation

    if cache:
        np.save(cache_path, adjacency)
        print(f"[GraphMamba] Saved static graph: {cache_path}")
    return adjacency
