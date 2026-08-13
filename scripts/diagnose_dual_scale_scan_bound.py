#!/usr/bin/env python3
"""Frozen-weight upper bound for independent dual-scale Mamba scans."""

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


OUTPUT = ROOT / "logs" / "graphmamba_dual_scale_scan_bound"
DATASETS = ("ETTm1", "ETTm2")
PRED_LEN = 720


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


def diagnose(dataset: str) -> tuple[dict, list[dict]]:
    record, checkpoint = locate_record(dataset, PRED_LEN)
    args = command_args(record["command"])
    torch.manual_seed(2021)
    model = Model(args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    _, loader = data_provider(args, "val")
    calibration, evaluation = make_stats(args.enc_in), make_stats(args.enc_in)
    long_patches = (args.seq_len - args.patch_len) // args.stride + 2
    diagnostic_sums = {key: 0.0 for key in (
        "input_boundary_jump", "within_long_jump", "within_short_jump",
        "joint_boundary_jump", "separate_boundary_jump",
        "long_representation_change", "short_representation_change",
        "joint_norm", "separate_norm",
    )}
    sample_vars = 0

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda()
            y = batch_y[:, -PRED_LEN:, :].float().cuda()
            decoder = torch.cat((
                batch_y[:, : args.label_len].float().cuda(),
                torch.zeros_like(y),
            ), dim=1)
            baseline = model(x, batch_x_mark.float().cuda(), decoder, batch_y_mark.float().cuda())

            means = x.mean(dim=1, keepdim=True)
            centered = x - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            normalized = centered / stdev
            seasonal, _ = model.decomposition(normalized)
            seasonal = seasonal.permute(0, 2, 1)
            long_tokens = model.long_patch_embedding(seasonal) + model.variable_embedding
            short_tokens = model.short_patch_embedding(seasonal) + model.variable_embedding
            tokens = torch.cat((long_tokens, short_tokens), dim=-1)

            joint_temporal = model.encoder(tokens)
            separate_long = model.encoder(long_tokens)
            separate_short = model.encoder(short_tokens)
            separate_temporal = torch.cat((separate_long, separate_short), dim=-1)
            graph = model.graph_mixer(tokens)
            joint_output = model.head(joint_temporal + graph)
            separate_output = model.head(separate_temporal + graph)
            contribution = (separate_output - joint_output) * stdev
            residual = y - baseline
            update(
                calibration if batch_index < len(loader) // 2 else evaluation,
                contribution,
                residual,
            )

            norm = lambda value: value.square().mean(dim=2).sqrt()
            diagnostic_sums["input_boundary_jump"] += norm(long_tokens[..., -1] - short_tokens[..., 0]).sum().item()
            diagnostic_sums["within_long_jump"] += norm(long_tokens[..., 1:] - long_tokens[..., :-1]).mean(dim=-1).sum().item()
            diagnostic_sums["within_short_jump"] += norm(short_tokens[..., 1:] - short_tokens[..., :-1]).mean(dim=-1).sum().item()
            diagnostic_sums["joint_boundary_jump"] += norm(joint_temporal[..., long_patches - 1] - joint_temporal[..., long_patches]).sum().item()
            diagnostic_sums["separate_boundary_jump"] += norm(separate_long[..., -1] - separate_short[..., 0]).sum().item()
            diagnostic_sums["long_representation_change"] += norm(separate_long - joint_temporal[..., :long_patches]).mean(dim=-1).sum().item()
            diagnostic_sums["short_representation_change"] += norm(separate_short - joint_temporal[..., long_patches:]).mean(dim=-1).sum().item()
            diagnostic_sums["joint_norm"] += norm(joint_temporal).mean(dim=-1).sum().item()
            diagnostic_sums["separate_norm"] += norm(separate_temporal).mean(dim=-1).sum().item()
            sample_vars += x.shape[0] * args.enc_in

    alpha = calibration["xtr"] / (calibration["xtx"] + 1e-8)
    corrected_sse = evaluation["rtr"] - 2 * alpha * evaluation["xtr"] + alpha.square() * evaluation["xtx"]
    base_mse_var = evaluation["rtr"] / evaluation["count"]
    corrected_mse_var = corrected_sse / evaluation["count"]
    base_mse = evaluation["rtr"].sum().item() / evaluation["count"].sum().item()
    corrected_mse = corrected_sse.sum().item() / evaluation["count"].sum().item()
    summary = {
        "dataset": dataset,
        "long_patches": long_patches,
        "short_patches": tokens.shape[-1] - long_patches,
        "calibration_batches": len(loader) // 2,
        "evaluation_batches": len(loader) - len(loader) // 2,
        "evaluation_base_mse": base_mse,
        "corrected_mse": corrected_mse,
        "improvement_pct": 100 * (base_mse - corrected_mse) / base_mse,
        "alpha": alpha.tolist(),
        **{key: value / sample_vars for key, value in diagnostic_sums.items()},
    }
    columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    rows = []
    for index, column in enumerate(columns):
        rows.append({
            "dataset": dataset,
            "variable": column,
            "alpha": alpha[index].item(),
            "base_mse": base_mse_var[index].item(),
            "corrected_mse": corrected_mse_var[index].item(),
            "improvement_pct": 100 * (base_mse_var[index] - corrected_mse_var[index]).item() / base_mse_var[index].item(),
        })
    return summary, rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, variables = [], []
    for dataset in DATASETS:
        print(f"Dual-scale scan diagnosis: {dataset}-720", flush=True)
        summary, rows = diagnose(dataset)
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
