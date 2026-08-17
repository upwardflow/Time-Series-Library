#!/usr/bin/env python3
"""Fit bounded STC coefficients on train predictions and evaluate validation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args, locate_record

OUTPUT = ROOT / "logs" / "graphmamba_train_to_val_stc"
TASKS = (("ETTm1", 96), ("ETTm2", 96), ("ETTm1", 720), ("ETTm2", 720))


def contributions(model: Model, x: torch.Tensor):
    means = x.mean(dim=1, keepdim=True)
    centered = x - means
    stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
    seasonal_input, trend_input = model.decomposition(centered / stdev)
    trend = model.trend_projection(trend_input.permute(0, 2, 1)).permute(0, 2, 1)
    seasonal_input = seasonal_input.permute(0, 2, 1)
    tokens = torch.cat((
        model.long_patch_embedding(seasonal_input) + model.variable_embedding,
        model.short_patch_embedding(seasonal_input) + model.variable_embedding,
    ), dim=-1)
    seasonal = model.head(model.encoder(tokens) + model.graph_mixer(tokens))
    return seasonal * stdev, trend * stdev, means


def empty(n_vars: int):
    return {
        "xtx": torch.zeros((n_vars, 2, 2), dtype=torch.float64),
        "xtr": torch.zeros((n_vars, 2), dtype=torch.float64),
    }


def fit(model, loader, pred_len, n_vars):
    stats = empty(n_vars)
    with torch.no_grad():
        for batch_x, batch_y, *_ in loader:
            x = batch_x.float().cuda(); y = batch_y[:, -pred_len:, :].float().cuda()
            seasonal, trend, means = contributions(model, x)
            baseline = seasonal + trend + means
            residual = y - baseline
            design = torch.stack((seasonal, trend), dim=-1).double().cpu()
            residual = residual.double().cpu()
            stats["xtx"] += torch.einsum("btvi,btvj->vij", design, design)
            stats["xtr"] += torch.einsum("btvi,btv->vi", design, residual)
    eye = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    unconstrained = torch.linalg.solve(stats["xtx"] + 1e-8 * eye, stats["xtr"].unsqueeze(-1)).squeeze(-1)
    # Delta is added to the baseline multiplier 1; bound final multipliers to [0, 2].
    bounded = unconstrained.clamp(-1.0, 1.0)
    return unconstrained, bounded


def evaluate(model, loader, pred_len, coefficients):
    sse = mae = corrected_sse = corrected_mae = count = 0.0
    coefficients = coefficients.cuda().float()
    with torch.no_grad():
        for batch_x, batch_y, *_ in loader:
            x = batch_x.float().cuda(); y = batch_y[:, -pred_len:, :].float().cuda()
            seasonal, trend, means = contributions(model, x)
            baseline = seasonal + trend + means
            corrected = baseline + coefficients[None, None, :, 0] * seasonal + coefficients[None, None, :, 1] * trend
            error = y - baseline; corrected_error = y - corrected
            sse += error.square().sum().item(); mae += error.abs().sum().item()
            corrected_sse += corrected_error.square().sum().item(); corrected_mae += corrected_error.abs().sum().item()
            count += y.numel()
    return {"base_mse": sse/count, "base_mae": mae/count,
        "corrected_mse": corrected_sse/count, "corrected_mae": corrected_mae/count,
        "mse_improvement_pct": 100*(sse-corrected_sse)/sse,
        "mae_improvement_pct": 100*(mae-corrected_mae)/mae}


def diagnose(dataset, pred_len):
    record, checkpoint = locate_record(dataset, pred_len)
    args = command_args(record["command"])
    torch.manual_seed(2021); model = Model(args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    _, train_loader = data_provider(args, "train")
    _, val_loader = data_provider(args, "val")
    unconstrained, bounded = fit(model, train_loader, pred_len, args.enc_in)
    metrics = evaluate(model, val_loader, pred_len, bounded)
    return {"dataset": dataset, "pred_len": pred_len, **metrics,
        "unconstrained_delta": unconstrained.tolist(), "bounded_delta": bounded.tolist(),
        "seasonal_multiplier": (1+bounded[:,0]).tolist(), "trend_multiplier": (1+bounded[:,1]).tolist(),
        "checkpoint": str(checkpoint), "fit_split": "train", "evaluation_split": "val", "test_accessed": False}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); rows=[]
    for dataset, pred_len in TASKS:
        print(f"Train-to-val STC: {dataset}-{pred_len}", flush=True)
        row=diagnose(dataset,pred_len); rows.append(row); print(json.dumps(row,sort_keys=True),flush=True)
        (OUTPUT/f"{dataset.lower()}_{pred_len}.json").write_text(json.dumps(row,indent=2)+"\n")
    with (OUTPUT/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
