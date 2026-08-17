#!/usr/bin/env python3
"""Frozen upper bound for adaptive graphs forced to encode other variables."""

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


OUTPUT = ROOT / "logs" / "graphmamba_cross_variable_graph_bound"
DATASETS = ("ETTm1", "ETTm2")
PRED_LEN = 720
VARIANTS = ("current", "masked_dot", "cosine", "masked_cosine")


def make_adjacencies(embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
    n_vars = embeddings.shape[0]
    diagonal = torch.eye(n_vars, dtype=torch.bool, device=embeddings.device)
    dot = embeddings @ embeddings.T
    normalized = F.normalize(embeddings, dim=1)
    cosine = normalized @ normalized.T
    return {
        "current": torch.softmax(dot, dim=1),
        "masked_dot": torch.softmax(dot.masked_fill(diagonal, -torch.inf), dim=1),
        "cosine": torch.softmax(cosine, dim=1),
        "masked_cosine": torch.softmax(cosine.masked_fill(diagonal, -torch.inf), dim=1),
    }


def graph_forward(mixer, tokens: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    batch_size, n_vars, d_model, n_patches = tokens.shape
    h = tokens.permute(0, 3, 1, 2).reshape(batch_size * n_patches, n_vars, d_model)
    h = mixer.input_projection(h)
    h = torch.einsum("nm,bmd->bnd", adjacency, h)
    h = mixer.output_projection(mixer.activation(h))
    return h.reshape(batch_size, n_patches, n_vars, d_model).permute(0, 2, 3, 1)


def new_stats(n_variants: int, n_vars: int) -> dict[str, torch.Tensor]:
    return {
        "base_sse": torch.zeros(n_vars, dtype=torch.float64),
        "count": torch.zeros(n_vars, dtype=torch.float64),
        "delta2": torch.zeros((n_variants, n_vars), dtype=torch.float64),
        "delta_residual": torch.zeros((n_variants, n_vars), dtype=torch.float64),
        "direct_sse": torch.zeros((n_variants, n_vars), dtype=torch.float64),
    }


def update(store, index, delta, residual, direct_residual) -> None:
    delta = delta.detach().double().cpu()
    residual = residual.detach().double().cpu()
    direct_residual = direct_residual.detach().double().cpu()
    store["delta2"][index] += delta.square().sum(dim=(0, 1))
    store["delta_residual"][index] += (delta * residual).sum(dim=(0, 1))
    store["direct_sse"][index] += direct_residual.square().sum(dim=(0, 1))


def adjacency_stats(adjacency: torch.Tensor) -> dict:
    entropy = (-(adjacency * (adjacency + 1e-12).log()).sum(dim=1)).mean().item()
    return {
        "row_entropy": entropy,
        "mean_self_mass": adjacency.diagonal().mean().item(),
        "mean_max_mass": adjacency.max(dim=1).values.mean().item(),
        "adjacency": adjacency.cpu().tolist(),
    }


def diagnose(dataset: str) -> tuple[list[dict], list[dict], dict]:
    record, checkpoint = locate_record(dataset, PRED_LEN)
    args = command_args(record["command"])
    torch.manual_seed(2021)
    model = Model(args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    _, loader = data_provider(args, "val")
    split = len(loader) // 2
    calibration = new_stats(len(VARIANTS), args.enc_in)
    evaluation = new_stats(len(VARIANTS), args.enc_in)
    mixer = model.graph_mixer
    adaptive = make_adjacencies(mixer.node_embeddings)
    combined = {
        name: mixer.alpha * mixer.static_adj + (1 - mixer.alpha) * adjacency
        for name, adjacency in adaptive.items()
    }
    equivalence_max_abs = 0.0

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda()
            y = batch_y[:, -PRED_LEN:, :].float().cuda()
            decoder = torch.cat((batch_y[:, : args.label_len].float().cuda(), torch.zeros_like(y)), dim=1)
            baseline = model(x, batch_x_mark.float().cuda(), decoder, batch_y_mark.float().cuda())
            residual = y - baseline
            store = calibration if batch_index < split else evaluation
            store["base_sse"] += residual.detach().double().cpu().square().sum(dim=(0, 1))
            store["count"] += residual.shape[0] * residual.shape[1]

            means = x.mean(dim=1, keepdim=True)
            centered = x - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            normalized = centered / stdev
            seasonal, trend = model.decomposition(normalized)
            trend_output = model.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal = seasonal.permute(0, 2, 1)
            tokens = torch.cat((
                model.long_patch_embedding(seasonal) + model.variable_embedding,
                model.short_patch_embedding(seasonal) + model.variable_embedding,
            ), dim=-1)
            temporal = model.encoder(tokens)
            for index, name in enumerate(VARIANTS):
                graph = graph_forward(mixer, tokens, combined[name])
                candidate = (model.head(temporal + graph) + trend_output) * stdev + means
                if name == "current":
                    equivalence_max_abs = max(equivalence_max_abs, (candidate - baseline).abs().max().item())
                update(store, index, candidate - baseline, residual, y - candidate)

    summaries, variables = [], []
    names = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    base_mse = evaluation["base_sse"].sum().item() / evaluation["count"].sum().item()
    for index, variant in enumerate(VARIANTS):
        coefficient = calibration["delta_residual"][index] / (calibration["delta2"][index] + 1e-12)
        corrected_sse = (
            evaluation["base_sse"] - 2 * coefficient * evaluation["delta_residual"][index]
            + coefficient.square() * evaluation["delta2"][index]
        )
        direct_mse = evaluation["direct_sse"][index].sum().item() / evaluation["count"].sum().item()
        corrected_mse = corrected_sse.sum().item() / evaluation["count"].sum().item()
        summaries.append({
            "dataset": dataset, "variant": variant, "base_mse": base_mse,
            "direct_mse": direct_mse, "direct_improvement_pct": 100 * (base_mse-direct_mse)/base_mse,
            "corrected_mse": corrected_mse,
            "corrected_improvement_pct": 100 * (base_mse-corrected_mse)/base_mse,
            "coefficient": coefficient.tolist(), **{k: v for k, v in adjacency_stats(adaptive[variant]).items() if k != "adjacency"},
        })
        for var, variable in enumerate(names):
            base_var = evaluation["base_sse"][var] / evaluation["count"][var]
            corrected_var = corrected_sse[var] / evaluation["count"][var]
            variables.append({
                "dataset": dataset, "variant": variant, "variable": variable,
                "coefficient": coefficient[var].item(), "base_mse": base_var.item(),
                "corrected_mse": corrected_var.item(),
                "corrected_improvement_pct": 100*(base_var-corrected_var).item()/base_var.item(),
            })
    metadata = {
        "dataset": dataset, "checkpoint": str(checkpoint), "validation_batches": len(loader),
        "split_batch": split, "current_equivalence_max_abs": equivalence_max_abs,
        "test_accessed": False,
        "adjacencies": {name: adjacency_stats(value) for name, value in adaptive.items()},
    }
    return summaries, variables, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_summaries, all_variables = [], []
    for dataset in DATASETS:
        print(f"Cross-variable graph diagnosis: {dataset}-{PRED_LEN}", flush=True)
        summaries, variables, metadata = diagnose(dataset)
        all_summaries.extend(summaries); all_variables.extend(variables)
        (OUTPUT / f"{dataset.lower()}_{PRED_LEN}.json").write_text(
            json.dumps({"metadata": metadata, "summary": summaries, "variables": variables}, indent=2)+"\n", encoding="utf-8"
        )
        print(json.dumps({"metadata": {k:v for k,v in metadata.items() if k != "adjacencies"}, "summary": summaries}, sort_keys=True), flush=True)
    write_csv(OUTPUT / "summary.csv", all_summaries); write_csv(OUTPUT / "variables.csv", all_variables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
