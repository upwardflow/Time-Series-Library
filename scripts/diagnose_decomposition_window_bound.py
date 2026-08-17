#!/usr/bin/env python3
"""Frozen validation-only diagnosis for GraphMamba decomposition windows."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from layers.Autoformer_EncDec import series_decomp
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args, locate_record


OUTPUT = ROOT / "logs" / "graphmamba_decomposition_window_bound"
DATASETS = ("ETTm1", "ETTm2")
PRED_LEN = 720
WINDOWS = (7, 13, 25, 49, 95)


def forecast_with_decomposition(model: Model, x: torch.Tensor, decomp) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = x.mean(dim=1, keepdim=True).detach()
    centered = x - means
    stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    normalized = centered / stdev
    seasonal, trend = decomp(normalized)
    trend_output = model.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
    seasonal_input = seasonal.permute(0, 2, 1)
    tokens = torch.cat(
        (
            model.long_patch_embedding(seasonal_input) + model.variable_embedding,
            model.short_patch_embedding(seasonal_input) + model.variable_embedding,
        ),
        dim=-1,
    )
    encoded = model.encoder(tokens) + model.graph_mixer(tokens)
    output = model.head(encoded) + trend_output
    return output * stdev + means, seasonal, trend


def zeros(n_windows: int, n_vars: int) -> dict[str, torch.Tensor]:
    shape = (n_windows, n_vars)
    return {
        "delta2": torch.zeros(shape, dtype=torch.float64),
        "delta_residual": torch.zeros(shape, dtype=torch.float64),
        "direct_sse": torch.zeros(shape, dtype=torch.float64),
        "base_sse": torch.zeros(n_vars, dtype=torch.float64),
        "count": torch.zeros(n_vars, dtype=torch.float64),
    }


def update(stats: dict[str, torch.Tensor], index: int, delta: torch.Tensor,
           residual: torch.Tensor, direct_residual: torch.Tensor) -> None:
    delta = delta.detach().double().cpu()
    residual = residual.detach().double().cpu()
    direct_residual = direct_residual.detach().double().cpu()
    stats["delta2"][index] += delta.square().sum(dim=(0, 1))
    stats["delta_residual"][index] += (delta * residual).sum(dim=(0, 1))
    stats["direct_sse"][index] += direct_residual.square().sum(dim=(0, 1))


def diagnose(dataset: str) -> tuple[list[dict], list[dict], dict]:
    record, checkpoint = locate_record(dataset, PRED_LEN)
    args = command_args(record["command"])
    torch.manual_seed(2021)
    model = Model(args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    decompositions = {window: series_decomp(window).cuda().eval() for window in WINDOWS}
    _, loader = data_provider(args, "val")
    split = len(loader) // 2
    calibration = zeros(len(WINDOWS), args.enc_in)
    evaluation = zeros(len(WINDOWS), args.enc_in)
    energy = torch.zeros((len(WINDOWS), 3), dtype=torch.float64)
    energy_count = 0
    equivalence_max_abs = 0.0

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda()
            y = batch_y[:, -PRED_LEN:, :].float().cuda()
            decoder = torch.cat((
                batch_y[:, : args.label_len].float().cuda(),
                torch.zeros_like(y),
            ), dim=1)
            baseline = model(x, batch_x_mark.float().cuda(), decoder, batch_y_mark.float().cuda())
            residual = y - baseline
            stats = calibration if batch_index < split else evaluation
            stats["base_sse"] += residual.detach().double().cpu().square().sum(dim=(0, 1))
            stats["count"] += residual.shape[0] * residual.shape[1]

            for window_index, window in enumerate(WINDOWS):
                candidate, seasonal, trend = forecast_with_decomposition(
                    model, x, decompositions[window]
                )
                if window == 25:
                    equivalence_max_abs = max(
                        equivalence_max_abs,
                        (candidate - baseline).abs().max().item(),
                    )
                update(stats, window_index, candidate - baseline, residual, y - candidate)
                energy[window_index, 0] += seasonal.detach().double().square().sum().cpu()
                energy[window_index, 1] += trend.detach().double().square().sum().cpu()
                energy[window_index, 2] += (
                    trend[:, 1:] - trend[:, :-1]
                ).detach().double().square().sum().cpu()
            energy_count += x.numel()

    columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    summaries, variables = [], []
    for window_index, window in enumerate(WINDOWS):
        alpha = calibration["delta_residual"][window_index] / (
            calibration["delta2"][window_index] + 1e-12
        )
        corrected_sse = (
            evaluation["base_sse"]
            - 2 * alpha * evaluation["delta_residual"][window_index]
            + alpha.square() * evaluation["delta2"][window_index]
        )
        base_mse = evaluation["base_sse"].sum().item() / evaluation["count"].sum().item()
        direct_mse = evaluation["direct_sse"][window_index].sum().item() / evaluation["count"].sum().item()
        corrected_mse = corrected_sse.sum().item() / evaluation["count"].sum().item()
        summaries.append({
            "dataset": dataset,
            "window": window,
            "calibration_batches": split,
            "evaluation_batches": len(loader) - split,
            "base_mse": base_mse,
            "direct_mse": direct_mse,
            "direct_improvement_pct": 100 * (base_mse - direct_mse) / base_mse,
            "corrected_mse": corrected_mse,
            "corrected_improvement_pct": 100 * (base_mse - corrected_mse) / base_mse,
            "seasonal_energy": energy[window_index, 0].item() / energy_count,
            "trend_energy": energy[window_index, 1].item() / energy_count,
            "trend_roughness": energy[window_index, 2].item() / energy_count,
            "alpha": alpha.tolist(),
        })
        for variable_index, variable in enumerate(columns):
            base_var = evaluation["base_sse"][variable_index] / evaluation["count"][variable_index]
            direct_var = evaluation["direct_sse"][window_index, variable_index] / evaluation["count"][variable_index]
            corrected_var = corrected_sse[variable_index] / evaluation["count"][variable_index]
            variables.append({
                "dataset": dataset,
                "window": window,
                "variable": variable,
                "alpha": alpha[variable_index].item(),
                "base_mse": base_var.item(),
                "direct_mse": direct_var.item(),
                "direct_improvement_pct": 100 * (base_var - direct_var).item() / base_var.item(),
                "corrected_mse": corrected_var.item(),
                "corrected_improvement_pct": 100 * (base_var - corrected_var).item() / base_var.item(),
            })
    metadata = {
        "dataset": dataset,
        "checkpoint": str(checkpoint),
        "validation_batches": len(loader),
        "split_batch": split,
        "window_25_equivalence_max_abs": equivalence_max_abs,
        "test_accessed": False,
    }
    return summaries, variables, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_summaries, all_variables = [], []
    for dataset in DATASETS:
        print(f"Decomposition-window diagnosis: {dataset}-{PRED_LEN}", flush=True)
        summaries, variables, metadata = diagnose(dataset)
        all_summaries.extend(summaries)
        all_variables.extend(variables)
        (OUTPUT / f"{dataset.lower()}_{PRED_LEN}.json").write_text(
            json.dumps({"metadata": metadata, "summary": summaries, "variables": variables}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"metadata": metadata, "summary": summaries}, sort_keys=True), flush=True)
    write_csv(OUTPUT / "summary.csv", all_summaries)
    write_csv(OUTPUT / "variables.csv", all_variables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
