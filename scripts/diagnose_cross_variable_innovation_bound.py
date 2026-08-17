#!/usr/bin/env python3
"""Split-validation upper bound for temporally residualized graph innovation."""

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
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args, locate_record


OUTPUT = ROOT / "logs" / "graphmamba_cross_variable_bound"
DATASETS = ("ETTm1", "ETTm2")
PRED_LEN = 720


def forecast_without_bias(model: Model, representation: torch.Tensor) -> torch.Tensor:
    flat = representation.permute(0, 1, 3, 2).flatten(start_dim=-2)
    return F.linear(flat, model.head.linear.weight, None).permute(0, 2, 1)


def make_stats(n_vars: int) -> dict:
    return {
        "xtx": torch.zeros(n_vars, dtype=torch.float64),
        "xtr": torch.zeros(n_vars, dtype=torch.float64),
        "rtr": torch.zeros(n_vars, dtype=torch.float64),
        "count": torch.zeros(n_vars, dtype=torch.float64),
    }


def update(stats: dict, contribution: torch.Tensor, residual: torch.Tensor) -> None:
    c = contribution.detach().double().cpu()
    r = residual.detach().double().cpu()
    stats["xtx"] += c.square().sum(dim=(0, 1))
    stats["xtr"] += (c * r).sum(dim=(0, 1))
    stats["rtr"] += r.square().sum(dim=(0, 1))
    stats["count"] += r.shape[0] * r.shape[1]


def evaluate(dataset: str) -> tuple[dict, list[dict]]:
    record, checkpoint = locate_record(dataset, PRED_LEN)
    args = command_args(record["command"])
    torch.manual_seed(2021)
    model = Model(args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    _, loader = data_provider(args, "val")
    calibration = make_stats(args.enc_in)
    evaluation = make_stats(args.enc_in)
    raw_calibration = make_stats(args.enc_in)
    raw_evaluation = make_stats(args.enc_in)

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda()
            y = batch_y[:, -PRED_LEN:, :].float().cuda()
            decoder = torch.cat((
                batch_y[:, : args.label_len].float().cuda(),
                torch.zeros_like(y),
            ), dim=1)
            baseline = model(
                x, batch_x_mark.float().cuda(), decoder, batch_y_mark.float().cuda()
            )
            residual = y - baseline

            means = x.mean(dim=1, keepdim=True)
            centered = x - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            normalized = centered / stdev
            seasonal, _ = model.decomposition(normalized)
            seasonal = seasonal.permute(0, 2, 1)
            tokens = torch.cat((
                model.long_patch_embedding(seasonal) + model.variable_embedding,
                model.short_patch_embedding(seasonal) + model.variable_embedding,
            ), dim=-1)
            temporal = model.encoder(tokens)
            innovation = temporal - tokens

            # Remove each variable's component parallel to its own input-token
            # trajectory. What remains is a temporal-orthogonal innovation that
            # can only help through the existing cross-variable graph operation.
            coefficient = (innovation * tokens).sum(dim=(2, 3), keepdim=True) / (
                tokens.square().sum(dim=(2, 3), keepdim=True) + 1e-8
            )
            cross_innovation = innovation - coefficient * tokens
            cross_graph = model.graph_mixer(cross_innovation)
            raw_graph = model.graph_mixer(innovation)
            contribution = forecast_without_bias(model, cross_graph) * stdev
            raw_contribution = forecast_without_bias(model, raw_graph) * stdev
            target_stats = calibration if batch_index < len(loader) // 2 else evaluation
            target_raw = raw_calibration if batch_index < len(loader) // 2 else raw_evaluation
            update(target_stats, contribution, residual)
            update(target_raw, raw_contribution, residual)

    alpha = calibration["xtr"] / (calibration["xtx"] + 1e-8)
    raw_alpha = raw_calibration["xtr"] / (raw_calibration["xtx"] + 1e-8)
    corrected_sse = evaluation["rtr"] - 2 * alpha * evaluation["xtr"] + alpha.square() * evaluation["xtx"]
    raw_corrected_sse = raw_evaluation["rtr"] - 2 * raw_alpha * raw_evaluation["xtr"] + raw_alpha.square() * raw_evaluation["xtx"]
    base_mse_var = evaluation["rtr"] / evaluation["count"]
    corrected_mse_var = corrected_sse / evaluation["count"]
    raw_corrected_mse_var = raw_corrected_sse / raw_evaluation["count"]
    base_mse = evaluation["rtr"].sum().item() / evaluation["count"].sum().item()
    corrected_mse = corrected_sse.sum().item() / evaluation["count"].sum().item()
    raw_corrected_mse = raw_corrected_sse.sum().item() / evaluation["count"].sum().item()
    columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    variable_rows = []
    for index, column in enumerate(columns):
        variable_rows.append({
            "dataset": dataset,
            "variable": column,
            "alpha": alpha[index].item(),
            "raw_alpha": raw_alpha[index].item(),
            "base_mse": base_mse_var[index].item(),
            "cross_corrected_mse": corrected_mse_var[index].item(),
            "cross_improvement_pct": 100 * (base_mse_var[index] - corrected_mse_var[index]).item() / base_mse_var[index].item(),
            "raw_corrected_mse": raw_corrected_mse_var[index].item(),
        })
    summary = {
        "dataset": dataset,
        "calibration_batches": len(loader) // 2,
        "evaluation_batches": len(loader) - len(loader) // 2,
        "evaluation_base_mse": base_mse,
        "cross_corrected_mse": corrected_mse,
        "cross_improvement_pct": 100 * (base_mse - corrected_mse) / base_mse,
        "raw_innovation_corrected_mse": raw_corrected_mse,
        "raw_innovation_improvement_pct": 100 * (base_mse - raw_corrected_mse) / base_mse,
        "alpha": alpha.tolist(),
        "raw_alpha": raw_alpha.tolist(),
    }
    return summary, variable_rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, variables = [], []
    for dataset in DATASETS:
        print(f"Upper-bound diagnosis: {dataset}-720", flush=True)
        summary, rows = evaluate(dataset)
        summaries.append(summary); variables.extend(rows)
        (OUTPUT / f"{dataset.lower()}_720.json").write_text(
            json.dumps({"summary": summary, "variables": rows}, indent=2) + "\n"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    for filename, rows in (("summary.csv", summaries), ("variables.csv", variables)):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
