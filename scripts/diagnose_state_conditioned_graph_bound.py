#!/usr/bin/env python3
"""Frozen upper bound for parameter-free sample-conditioned variable graphs."""

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


OUTPUT = ROOT / "logs" / "graphmamba_state_conditioned_graph_bound"
DATASETS = ("ETTm1", "ETTm2")
PRED_LEN = 720
VARIANTS = ("current", "sample", "sample_patch")


def masked_cosine(values: torch.Tensor) -> torch.Tensor:
    values = F.normalize(values, dim=-1)
    similarity = torch.einsum("...nd,...md->...nm", values, values)
    n_vars = values.shape[-2]
    diagonal = torch.eye(n_vars, dtype=torch.bool, device=values.device)
    return torch.softmax(similarity.masked_fill(diagonal, -torch.inf), dim=-1)


def graph_outputs(mixer, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, n_vars, d_model, n_patches = tokens.shape
    h = tokens.permute(0, 3, 1, 2)
    h = mixer.input_projection(h)
    global_adaptive = torch.softmax(
        mixer.node_embeddings @ mixer.node_embeddings.T, dim=1
    )
    sample_adaptive = masked_cosine(h.mean(dim=1))
    patch_adaptive = masked_cosine(h)
    static = mixer.static_adj
    adjacencies = {
        "current": mixer.alpha * static + (1 - mixer.alpha) * global_adaptive,
        "sample": mixer.alpha * static[None] + (1 - mixer.alpha) * sample_adaptive,
        "sample_patch": mixer.alpha * static[None, None] + (1 - mixer.alpha) * patch_adaptive,
    }
    propagated = {
        "current": torch.einsum("nm,bpmd->bpnd", adjacencies["current"], h),
        "sample": torch.einsum("bnm,bpmd->bpnd", adjacencies["sample"], h),
        "sample_patch": torch.einsum("bpnm,bpmd->bpnd", adjacencies["sample_patch"], h),
    }
    outputs = {}
    for name, value in propagated.items():
        value = mixer.output_projection(mixer.activation(value))
        outputs[name] = value.permute(0, 2, 3, 1)
    return outputs, {"sample": sample_adaptive, "sample_patch": patch_adaptive}


def new_stats(n_variants: int, n_vars: int) -> dict[str, torch.Tensor]:
    return {
        "base_sse": torch.zeros(n_vars, dtype=torch.float64),
        "count": torch.zeros(n_vars, dtype=torch.float64),
        "delta2": torch.zeros((n_variants, n_vars), dtype=torch.float64),
        "delta_residual": torch.zeros((n_variants, n_vars), dtype=torch.float64),
        "direct_sse": torch.zeros((n_variants, n_vars), dtype=torch.float64),
    }


def update(store, index, delta, residual, direct_residual) -> None:
    delta = delta.detach().double().cpu(); residual = residual.detach().double().cpu()
    direct_residual = direct_residual.detach().double().cpu()
    store["delta2"][index] += delta.square().sum(dim=(0, 1))
    store["delta_residual"][index] += (delta * residual).sum(dim=(0, 1))
    store["direct_sse"][index] += direct_residual.square().sum(dim=(0, 1))


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
    entropy_sum = {"sample": 0.0, "sample_patch": 0.0}
    entropy_count = {"sample": 0, "sample_patch": 0}
    adjacency_variance_sum = {"sample": 0.0, "sample_patch": 0.0}
    equivalence_max_abs = 0.0

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda(); y = batch_y[:, -PRED_LEN:, :].float().cuda()
            decoder = torch.cat((batch_y[:, : args.label_len].float().cuda(), torch.zeros_like(y)), dim=1)
            baseline = model(x, batch_x_mark.float().cuda(), decoder, batch_y_mark.float().cuda())
            residual = y - baseline
            store = calibration if batch_index < split else evaluation
            store["base_sse"] += residual.detach().double().cpu().square().sum(dim=(0, 1))
            store["count"] += residual.shape[0] * residual.shape[1]

            means = x.mean(dim=1, keepdim=True); centered = x - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seasonal, trend = model.decomposition(centered / stdev)
            trend_output = model.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal = seasonal.permute(0, 2, 1)
            tokens = torch.cat((
                model.long_patch_embedding(seasonal) + model.variable_embedding,
                model.short_patch_embedding(seasonal) + model.variable_embedding,
            ), dim=-1)
            temporal = model.encoder(tokens)
            graphs, dynamic_adjacencies = graph_outputs(model.graph_mixer, tokens)
            for index, name in enumerate(VARIANTS):
                candidate = (model.head(temporal + graphs[name]) + trend_output) * stdev + means
                if name == "current":
                    equivalence_max_abs = max(equivalence_max_abs, (candidate-baseline).abs().max().item())
                update(store, index, candidate-baseline, residual, y-candidate)
            for name, adjacency in dynamic_adjacencies.items():
                flat = adjacency.reshape(-1, args.enc_in, args.enc_in)
                entropy_sum[name] += (-(flat*(flat+1e-12).log()).sum(-1).mean(-1)).sum().item()
                entropy_count[name] += flat.shape[0]
                adjacency_variance_sum[name] += flat.var(dim=0, unbiased=False).mean().item() * flat.shape[0]

    summaries, variables = [], []
    names = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    base_mse = evaluation["base_sse"].sum().item()/evaluation["count"].sum().item()
    for index, variant in enumerate(VARIANTS):
        coefficient = calibration["delta_residual"][index]/(calibration["delta2"][index]+1e-12)
        corrected_sse = evaluation["base_sse"] - 2*coefficient*evaluation["delta_residual"][index] + coefficient.square()*evaluation["delta2"][index]
        direct_mse = evaluation["direct_sse"][index].sum().item()/evaluation["count"].sum().item()
        corrected_mse = corrected_sse.sum().item()/evaluation["count"].sum().item()
        summaries.append({
            "dataset": dataset, "variant": variant, "base_mse": base_mse,
            "direct_mse": direct_mse, "direct_improvement_pct": 100*(base_mse-direct_mse)/base_mse,
            "corrected_mse": corrected_mse, "corrected_improvement_pct": 100*(base_mse-corrected_mse)/base_mse,
            "coefficient": coefficient.tolist(),
            "mean_row_entropy": None if variant == "current" else entropy_sum[variant]/entropy_count[variant],
            "adjacency_element_variance": None if variant == "current" else adjacency_variance_sum[variant]/entropy_count[variant],
        })
        for var, variable in enumerate(names):
            base_var = evaluation["base_sse"][var]/evaluation["count"][var]
            corrected_var = corrected_sse[var]/evaluation["count"][var]
            variables.append({"dataset": dataset, "variant": variant, "variable": variable,
                "coefficient": coefficient[var].item(), "base_mse": base_var.item(), "corrected_mse": corrected_var.item(),
                "corrected_improvement_pct": 100*(base_var-corrected_var).item()/base_var.item()})
    metadata = {"dataset": dataset, "checkpoint": str(checkpoint), "validation_batches": len(loader),
        "split_batch": split, "current_equivalence_max_abs": equivalence_max_abs, "test_accessed": False}
    return summaries, variables, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True); all_summaries=[]; all_variables=[]
    for dataset in DATASETS:
        print(f"State-conditioned graph diagnosis: {dataset}-{PRED_LEN}", flush=True)
        summaries, variables, metadata=diagnose(dataset); all_summaries.extend(summaries); all_variables.extend(variables)
        (OUTPUT/f"{dataset.lower()}_{PRED_LEN}.json").write_text(json.dumps({"metadata":metadata,"summary":summaries,"variables":variables},indent=2)+"\n")
        print(json.dumps({"metadata":metadata,"summary":summaries},sort_keys=True),flush=True)
    write_csv(OUTPUT/"summary.csv",all_summaries); write_csv(OUTPUT/"variables.csv",all_variables); return 0


if __name__ == "__main__":
    raise SystemExit(main())
