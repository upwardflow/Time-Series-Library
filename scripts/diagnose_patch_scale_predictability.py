#!/usr/bin/env python3
"""Model-agnostic ridge probe for dual patch-scale predictability."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from scripts.diagnose_graphmamba_representations import command_args, locate_record


OUTPUT = ROOT / "logs" / "graphmamba_patch_scale_predictability"
DATASETS = ("ETTm1", "ETTm2")
SCALE_PAIRS = ((2, 4), (4, 8), (8, 16), (16, 32), (24, 48))
PRED_LEN = 720


def trailing_average(x: torch.Tensor, window: int) -> torch.Tensor:
    # B,N,L -> causal moving average with replicated left boundary.
    padded = F.pad(x, (window - 1, 0), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window, stride=1)


def features(x: torch.Tensor, pair: tuple[int, int]) -> torch.Tensor:
    series = x.permute(0, 2, 1)
    short = trailing_average(series, pair[0])
    long = trailing_average(series, pair[1])
    design = torch.cat((short, long), dim=-1)
    ones = torch.ones(*design.shape[:-1], 1, device=x.device, dtype=x.dtype)
    return torch.cat((design, ones), dim=-1)


def fit_probe(train_loader, pair: tuple[int, int], n_vars: int) -> tuple[torch.Tensor, float]:
    n_features = 2 * 96 + 1
    xtx = torch.zeros(n_vars, n_features, n_features, device="cuda")
    xty = torch.zeros(n_vars, n_features, PRED_LEN, device="cuda")
    for batch_x, batch_y, _, _ in train_loader:
        x = batch_x.float().cuda()
        y = batch_y[:, -PRED_LEN:, :].float().cuda().permute(0, 2, 1)
        design = features(x, pair)
        xtx += torch.einsum("bnf,bng->nfg", design, design)
        xty += torch.einsum("bnf,bnh->nfh", design, y)
    scale = xtx[:, :-1, :-1].diagonal(dim1=-2, dim2=-1).mean().item()
    ridge = 1e-3 * scale
    penalty = torch.eye(n_features, device="cuda") * ridge
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(xtx + penalty.unsqueeze(0), xty)
    return weights, ridge


def evaluate_probe(loader, pair: tuple[int, int], weights: torch.Tensor) -> tuple[float, list[float]]:
    se = torch.zeros(weights.shape[0], dtype=torch.float64)
    count = 0
    for batch_x, batch_y, _, _ in loader:
        x = batch_x.float().cuda()
        y = batch_y[:, -PRED_LEN:, :].float().cuda().permute(0, 2, 1)
        prediction = torch.einsum("bnf,nfh->bnh", features(x, pair), weights)
        se += (prediction - y).square().sum(dim=(0, 2)).double().cpu()
        count += y.shape[0] * y.shape[2]
    mse_var = se / count
    return mse_var.mean().item(), mse_var.tolist()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, variables = [], []
    columns = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
    for dataset in DATASETS:
        record, _ = locate_record(dataset, PRED_LEN)
        args = command_args(record["command"])
        args.batch_size = 128
        _, train_loader = data_provider(args, "train")
        _, val_loader = data_provider(args, "val")
        print(f"Patch-scale predictability: {dataset}-720", flush=True)
        dataset_rows = []
        for short, long in SCALE_PAIRS:
            pair = (short, long)
            weights, ridge = fit_probe(train_loader, pair, args.enc_in)
            mse, mse_vars = evaluate_probe(val_loader, pair, weights)
            row = {"dataset": dataset, "short_scale": short, "long_scale": long,
                   "validation_mse": mse, "ridge": ridge}
            dataset_rows.append(row)
            for column, variable_mse in zip(columns, mse_vars):
                variables.append({**row, "variable": column, "variable_mse": variable_mse})
            print(json.dumps(row, sort_keys=True), flush=True)
        current = dataset_rows[0]["validation_mse"]
        for row in dataset_rows:
            row["vs_current_improvement_pct"] = 100 * (current - row["validation_mse"]) / current
        summaries.extend(dataset_rows)
        (OUTPUT / f"{dataset.lower()}_720.json").write_text(
            json.dumps(dataset_rows, indent=2) + "\n"
        )
    for filename, rows in (("summary.csv", summaries), ("variables.csv", variables)):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
