#!/usr/bin/env python3
"""Diagnose scale and branch usage in frozen GraphMamba validation checkpoints.

This script is read-only with respect to model parameters. It reuses frozen
seed-2021 baseline checkpoints and never evaluates the test split.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models.GraphMamba import Model


RECORDS = ROOT / "logs" / "graphmamba_darc_universality" / "final"
OUTPUT = ROOT / "logs" / "graphmamba_representation_diagnosis"
TASKS = (
    ("ETTh1", 192), ("ETTm1", 720), ("ETTm2", 720),
    ("ETTh1", 96), ("ETTh2", 96), ("ETTm2", 336),
)


def command_args(command: list[str]) -> Namespace:
    raw = {}
    index = 2
    while index < len(command):
        key = command[index]
        if key.startswith("--") and index + 1 < len(command):
            raw[key[2:]] = command[index + 1]
            index += 2
        else:
            index += 1
    integer = {
        "seed", "seq_len", "label_len", "pred_len", "enc_in", "dec_in",
        "c_out", "patch_len", "stride", "d_model", "d_ff", "d_state",
        "d_conv", "e_layers", "expand", "mamba_version",
        "mamba_bidirectional", "use_graph", "use_time_mamba", "use_patch",
        "use_decomp", "moving_avg", "graph_top_k", "graph_sample_size",
        "graph_cache", "batch_size", "num_workers", "gpu", "use_lag_graph",
        "lag_max_lag", "lag_frequency_bands", "lag_top_k",
        "periodic_period", "periodic_local_patch", "periodic_local_stride",
        "periodic_period_stride", "periodic_use_adapter",
        "periodic_adapter_confidence", "periodic_use_alignment",
        "periodic_use_router", "test_after_train",
    }
    floating = {
        "dropout", "graph_alpha", "learning_rate", "lag_temperature",
        "lag_residual_init", "periodic_router_threshold",
    }
    for key in integer & raw.keys():
        raw[key] = int(raw[key])
    for key in floating & raw.keys():
        raw[key] = float(raw[key])
    raw.update({
        "task_name": "long_term_forecast", "use_gpu": True,
        "gpu_type": "cuda", "use_multi_gpu": False, "devices": "0",
        "static_graph_only": 0, "node_dim": 10, "activation": "gelu",
        "mamba_headdim": 0, "embed": "timeF", "freq": "h",
        "seasonal_patterns": "Monthly", "augmentation_ratio": 0,
    })
    return Namespace(**raw)


def locate_record(dataset: str, pred_len: int) -> tuple[dict, Path]:
    for path in RECORDS.glob("*.json"):
        row = json.loads(path.read_text())
        if (row.get("dataset"), row.get("pred_len"), row.get("seed"), row.get("model")) == (
            dataset, pred_len, 2021, "GraphMamba"
        ):
            matches = list((ROOT / "checkpoints").glob(f"*{row['candidate']}*/checkpoint.pth"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one checkpoint for {row['candidate']}, got {matches}")
            return row, matches[0]
    raise FileNotFoundError(f"No frozen baseline record for {dataset}-{pred_len}")


def diagnose(dataset: str, pred_len: int) -> dict:
    record, checkpoint = locate_record(dataset, pred_len)
    args = command_args(record["command"])
    random.seed(2021); np.random.seed(2021); torch.manual_seed(2021)
    model = Model(args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    _, loader = data_provider(args, "val")

    sums = {key: 0.0 for key in (
        "se", "ae", "se_no_long", "se_no_short", "se_time_only", "se_graph_only", "cos_scale",
        "cos_time_graph", "cancellation", "long_norm", "short_norm",
        "time_norm", "graph_norm", "long_grad", "short_grad",
    )}
    elements = samples = grad_batches = 0
    oracle = {
        split: {"xtx": torch.zeros(2, 2, dtype=torch.float64),
                "xtr": torch.zeros(2, dtype=torch.float64),
                "rtr": 0.0, "count": 0}
        for split in ("calibration", "evaluation")
    }
    long_patches = (args.seq_len - args.patch_len) // args.stride + 2

    for batch_index, (batch_x, batch_y, _, _) in enumerate(loader):
        x = batch_x.float().cuda()
        y = batch_y[:, -args.pred_len:, :].float().cuda()
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev
        seasonal, trend = model.decomposition(normalized)
        trend = model.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)
        long_tokens = model.long_patch_embedding(seasonal)
        short_tokens = model.short_patch_embedding(seasonal)
        tokens = torch.cat((
            long_tokens + model.variable_embedding,
            short_tokens + model.variable_embedding,
        ), dim=-1)
        if batch_index < 8:
            tokens.requires_grad_(True)
        temporal = model.encoder(tokens)
        graph = model.graph_mixer(tokens)
        fused = temporal + graph
        flat = fused.permute(0, 1, 3, 2).flatten(start_dim=-2)
        split = long_patches * args.d_model
        weight, bias = model.head.linear.weight, model.head.linear.bias
        long_pred = F.linear(flat[..., :split], weight[:, :split], bias)
        short_pred = F.linear(flat[..., split:], weight[:, split:], None)
        full = (long_pred + short_pred).permute(0, 2, 1) + trend
        no_long = short_pred.permute(0, 2, 1) + trend
        no_short = long_pred.permute(0, 2, 1) + trend
        time_only = model.head(temporal) + trend
        graph_only = model.head(graph) + trend
        full = full * stdev + means
        no_long = no_long * stdev + means
        no_short = no_short * stdev + means
        time_only = time_only * stdev + means
        graph_only = graph_only * stdev + means
        error = full - y

        # Diagnostic upper bound: split the graph branch into its component
        # parallel to the temporal representation and its orthogonal remainder.
        # Fit two residual coefficients on the first half of validation batches
        # and evaluate the frozen coefficients on the second half.
        coefficient = (graph * temporal).sum(dim=2, keepdim=True) / (
            temporal.square().sum(dim=2, keepdim=True) + 1e-8
        )
        graph_parallel = coefficient * temporal
        graph_orthogonal = graph - graph_parallel
        def head_without_bias(value):
            value = value.permute(0, 1, 3, 2).flatten(start_dim=-2)
            return F.linear(value, model.head.linear.weight, None).permute(0, 2, 1)
        parallel_contribution = head_without_bias(graph_parallel) * stdev
        orthogonal_contribution = head_without_bias(graph_orthogonal) * stdev
        design = torch.stack((parallel_contribution, orthogonal_contribution), dim=-1)
        residual = y - full
        split_name = "calibration" if batch_index < len(loader) // 2 else "evaluation"
        design_2d = design.detach().reshape(-1, 2).double().cpu()
        residual_1d = residual.detach().reshape(-1).double().cpu()
        oracle[split_name]["xtx"] += design_2d.T @ design_2d
        oracle[split_name]["xtr"] += design_2d.T @ residual_1d
        oracle[split_name]["rtr"] += residual_1d.dot(residual_1d).item()
        oracle[split_name]["count"] += residual_1d.numel()
        sums["se"] += error.square().sum().item()
        sums["ae"] += error.abs().sum().item()
        sums["se_no_long"] += (no_long - y).square().sum().item()
        sums["se_no_short"] += (no_short - y).square().sum().item()
        sums["se_time_only"] += (time_only - y).square().sum().item()
        sums["se_graph_only"] += (graph_only - y).square().sum().item()
        elements += y.numel()

        long_repr = fused[..., :long_patches].mean(dim=-1)
        short_repr = fused[..., long_patches:].mean(dim=-1)
        sums["cos_scale"] += F.cosine_similarity(long_repr, short_repr, dim=-1).sum().item()
        sums["cos_time_graph"] += F.cosine_similarity(
            temporal.flatten(2), graph.flatten(2), dim=-1
        ).sum().item()
        tnorm = temporal.flatten(2).norm(dim=-1)
        gnorm = graph.flatten(2).norm(dim=-1)
        sums["cancellation"] += (
            fused.flatten(2).norm(dim=-1) / (tnorm + gnorm + 1e-8)
        ).sum().item()
        sums["time_norm"] += tnorm.sum().item()
        sums["graph_norm"] += gnorm.sum().item()
        sums["long_norm"] += long_repr.norm(dim=-1).sum().item()
        sums["short_norm"] += short_repr.norm(dim=-1).sum().item()
        samples += x.shape[0] * args.enc_in

        if batch_index < 8:
            grad = torch.autograd.grad(error.square().mean(), tokens)[0]
            sums["long_grad"] += grad[..., :long_patches].square().mean().sqrt().item()
            sums["short_grad"] += grad[..., long_patches:].square().mean().sqrt().item()
            grad_batches += 1

    base_mse = sums["se"] / elements
    calibration = oracle["calibration"]
    ridge = 1e-8 * torch.eye(2, dtype=torch.float64)
    delta = torch.linalg.solve(calibration["xtx"] + ridge, calibration["xtr"])
    evaluation = oracle["evaluation"]
    oracle_sse = (
        evaluation["rtr"]
        - 2 * delta.dot(evaluation["xtr"]).item()
        + delta.dot(evaluation["xtx"] @ delta).item()
    )
    evaluation_base_mse = evaluation["rtr"] / evaluation["count"]
    evaluation_oracle_mse = oracle_sse / evaluation["count"]
    return {
        "dataset": dataset, "pred_len": pred_len, "batches": len(loader),
        "val_mse": base_mse, "val_mae": sums["ae"] / elements,
        "no_long_mse": sums["se_no_long"] / elements,
        "no_short_mse": sums["se_no_short"] / elements,
        "time_only_mse": sums["se_time_only"] / elements,
        "graph_only_mse": sums["se_graph_only"] / elements,
        "no_long_delta_pct": 100 * (sums["se_no_long"] / elements - base_mse) / base_mse,
        "no_short_delta_pct": 100 * (sums["se_no_short"] / elements - base_mse) / base_mse,
        "long_short_cosine": sums["cos_scale"] / samples,
        "time_graph_cosine": sums["cos_time_graph"] / samples,
        "fusion_cancellation_ratio": sums["cancellation"] / samples,
        "time_graph_norm_ratio": sums["time_norm"] / max(sums["graph_norm"], 1e-12),
        "long_short_norm_ratio": sums["long_norm"] / max(sums["short_norm"], 1e-12),
        "long_short_grad_ratio": (sums["long_grad"] / grad_batches) / max(sums["short_grad"] / grad_batches, 1e-12),
        "parallel_scale": 1.0 + delta[0].item(),
        "orthogonal_scale": 1.0 + delta[1].item(),
        "split_eval_base_mse": evaluation_base_mse,
        "split_eval_calibrated_mse": evaluation_oracle_mse,
        "split_eval_improvement_pct": 100 * (evaluation_base_mse - evaluation_oracle_mse) / evaluation_base_mse,
        "checkpoint": str(checkpoint),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, pred_len in TASKS:
        print(f"Diagnosing {dataset}-{pred_len}", flush=True)
        row = diagnose(dataset, pred_len)
        rows.append(row)
        (OUTPUT / f"{dataset.lower()}_{pred_len}.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
        print(json.dumps(row, sort_keys=True), flush=True)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
