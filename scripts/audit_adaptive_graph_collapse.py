#!/usr/bin/env python3
"""Audit adaptive-graph collapse across the frozen 48-run ETT baseline matrix."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs" / "graphmamba_darc_universality" / "runs.csv"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUT = ROOT / "logs" / "graphmamba_adaptive_graph_audit"


def entropy(adjacency: torch.Tensor) -> float:
    return (-(adjacency * (adjacency + 1e-12).log()).sum(dim=1)).mean().item()


def locate(candidate: str) -> Path:
    matches = [
        path for path in CHECKPOINTS.glob("*/checkpoint.pth")
        if candidate in path.parent.name and "GraphMambaAF" not in path.parent.name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {candidate}, found {len(matches)}")
    return matches[0]


def audit(row: dict[str, str]) -> dict:
    checkpoint = locate(row["candidate"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embeddings = state["graph_mixer.node_embeddings"].double()
    static = state["graph_mixer.static_adj"].double()
    adaptive = torch.softmax(embeddings @ embeddings.T, dim=1)
    diagonal = adaptive.diagonal()
    offdiag = adaptive.clone()
    offdiag.fill_diagonal_(0)
    top_indices = adaptive.argmax(dim=1)
    identity = torch.arange(adaptive.shape[0])
    result = {
        "dataset": row["dataset"], "pred_len": int(row["pred_len"]),
        "seed": int(row["seed"]), "candidate": row["candidate"],
        "best_mse": float(row["best_mse"]), "test_mse": float(row["test_mse"]),
        "adaptive_entropy": entropy(adaptive), "static_entropy": entropy(static),
        "mean_self_mass": diagonal.mean().item(), "min_self_mass": diagonal.min().item(),
        "max_self_mass": diagonal.max().item(), "mean_max_offdiag": offdiag.max(dim=1).values.mean().item(),
        "top1_self_fraction": (top_indices == identity).double().mean().item(),
        "embedding_norm_mean": embeddings.norm(dim=1).mean().item(),
        "embedding_norm_std": embeddings.norm(dim=1).std(unbiased=False).item(),
        "checkpoint": str(checkpoint),
    }
    return result


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metrics = (
        "adaptive_entropy", "static_entropy", "mean_self_mass", "min_self_mass",
        "max_self_mass", "mean_max_offdiag", "top1_self_fraction", "embedding_norm_mean",
        "best_mse", "test_mse",
    )
    output = []
    for group, values in sorted(groups.items()):
        record = dict(zip(keys, group))
        record["n"] = len(values)
        for metric in metrics:
            tensor = torch.tensor([value[metric] for value in values], dtype=torch.float64)
            record[f"{metric}_mean"] = tensor.mean().item()
            record[f"{metric}_std"] = tensor.std(unbiased=False).item()
        output.append(record)
    return output


def correlation(rows: list[dict], x_key: str, y_key: str) -> float:
    x = torch.tensor([row[x_key] for row in rows], dtype=torch.float64)
    y = torch.tensor([row[y_key] for row in rows], dtype=torch.float64)
    x = x - x.mean(); y = y - y.mean()
    return (x @ y / ((x.norm() * y.norm()) + 1e-12)).item()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    baseline_rows = [
        row for row in csv.DictReader(RUNS.open())
        if row["model"] == "GraphMamba" and row["status"] == "completed"
    ]
    audited = [audit(row) for row in baseline_rows]
    if len(audited) != 48:
        raise RuntimeError(f"Expected 48 baseline checkpoints, found {len(audited)}")
    by_task = aggregate(audited, ("dataset", "pred_len"))
    by_dataset = aggregate(audited, ("dataset",))
    metadata = {
        "n_checkpoints": len(audited),
        "adaptive_entropy_range": [min(r["adaptive_entropy"] for r in audited), max(r["adaptive_entropy"] for r in audited)],
        "mean_self_mass_range": [min(r["mean_self_mass"] for r in audited), max(r["mean_self_mass"] for r in audited)],
        "top1_self_fraction_mean": sum(r["top1_self_fraction"] for r in audited) / len(audited),
        "entropy_validation_mse_correlation": correlation(audited, "adaptive_entropy", "best_mse"),
        "entropy_test_mse_correlation": correlation(audited, "adaptive_entropy", "test_mse"),
        "test_accessed": "Only previously recorded test metrics were joined; no test loader or new test inference was run.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "checkpoints.csv", audited)
    write_csv(OUTPUT / "by_task.csv", by_task)
    write_csv(OUTPUT / "by_dataset.csv", by_dataset)
    (OUTPUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": metadata, "by_dataset": by_dataset}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
