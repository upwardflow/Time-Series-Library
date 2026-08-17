#!/usr/bin/env python3
"""Frozen validation-only diagnosis of GraphMamba's two global graph priors."""

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


OUTPUT = ROOT / "logs" / "graphmamba_graph_blend_bound"
DATASETS = ("ETTm1", "ETTm2")
PRED_LEN = 720
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def graph_forward(mixer, tokens: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    batch_size, n_vars, d_model, n_patches = tokens.shape
    h = tokens.permute(0, 3, 1, 2).reshape(batch_size * n_patches, n_vars, d_model)
    h = mixer.input_projection(h)
    h = torch.einsum("nm,bmd->bnd", adjacency, h)
    h = mixer.output_projection(mixer.activation(h))
    return h.reshape(batch_size, n_patches, n_vars, d_model).permute(0, 2, 3, 1)


def stats(n_candidates: int, n_vars: int) -> dict[str, torch.Tensor]:
    return {
        "base_sse": torch.zeros(n_vars, dtype=torch.float64),
        "count": torch.zeros(n_vars, dtype=torch.float64),
        "delta2": torch.zeros((n_candidates, n_vars), dtype=torch.float64),
        "delta_residual": torch.zeros((n_candidates, n_vars), dtype=torch.float64),
        "direct_sse": torch.zeros((n_candidates, n_vars), dtype=torch.float64),
        "joint_xtx": torch.zeros((n_vars, 2, 2), dtype=torch.float64),
        "joint_xtr": torch.zeros((n_vars, 2), dtype=torch.float64),
    }


def update_candidate(store, index, delta, residual, direct_residual) -> None:
    delta = delta.detach().double().cpu()
    residual = residual.detach().double().cpu()
    direct_residual = direct_residual.detach().double().cpu()
    store["delta2"][index] += delta.square().sum(dim=(0, 1))
    store["delta_residual"][index] += (delta * residual).sum(dim=(0, 1))
    store["direct_sse"][index] += direct_residual.square().sum(dim=(0, 1))


def update_joint(store, static_delta, adaptive_delta, residual) -> None:
    x = torch.stack((static_delta, adaptive_delta), dim=-1).detach().double().cpu()
    residual = residual.detach().double().cpu()
    store["joint_xtx"] += torch.einsum("btvi,btvj->vij", x, x)
    store["joint_xtr"] += torch.einsum("btvi,btv->vi", x, residual)


def adjacency_diagnostics(static_adj: torch.Tensor, adaptive_adj: torch.Tensor) -> dict:
    eps = 1e-12
    row_entropy = lambda a: (-(a * (a + eps).log()).sum(dim=1)).mean().item()
    static_neighbors = static_adj.argsort(dim=1, descending=True)[:, :3]
    adaptive_neighbors = adaptive_adj.argsort(dim=1, descending=True)[:, :3]
    overlaps = []
    for row in range(static_adj.shape[0]):
        overlaps.append(len(set(static_neighbors[row].tolist()) & set(adaptive_neighbors[row].tolist())) / 3)
    cosine = torch.nn.functional.cosine_similarity(static_adj.flatten(), adaptive_adj.flatten(), dim=0)
    return {
        "frobenius_distance": torch.linalg.vector_norm(static_adj - adaptive_adj).item(),
        "cosine_similarity": cosine.item(),
        "static_row_entropy": row_entropy(static_adj),
        "adaptive_row_entropy": row_entropy(adaptive_adj),
        "top3_neighbor_overlap": sum(overlaps) / len(overlaps),
        "static_adj": static_adj.cpu().tolist(),
        "adaptive_adj": adaptive_adj.cpu().tolist(),
    }


def diagnose(dataset: str) -> tuple[list[dict], list[dict], dict]:
    record, checkpoint = locate_record(dataset, PRED_LEN)
    args = command_args(record["command"])
    torch.manual_seed(2021)
    model = Model(args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    _, loader = data_provider(args, "val")
    split = len(loader) // 2
    calibration = stats(len(ALPHAS), args.enc_in)
    evaluation = stats(len(ALPHAS), args.enc_in)
    mixer = model.graph_mixer
    static_adj = mixer.static_adj
    adaptive_adj = torch.softmax(
        mixer.node_embeddings @ mixer.node_embeddings.transpose(0, 1), dim=1
    )
    candidates = [alpha * static_adj + (1 - alpha) * adaptive_adj for alpha in ALPHAS]
    equivalence_max_abs = 0.0

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda()
            y = batch_y[:, -PRED_LEN:, :].float().cuda()
            decoder = torch.cat((
                batch_y[:, : args.label_len].float().cuda(), torch.zeros_like(y)
            ), dim=1)
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
            outputs = []
            for index, adjacency in enumerate(candidates):
                graph = graph_forward(mixer, tokens, adjacency)
                normalized_output = model.head(temporal + graph) + trend_output
                candidate = normalized_output * stdev + means
                outputs.append(candidate)
                if ALPHAS[index] == 0.5:
                    equivalence_max_abs = max(
                        equivalence_max_abs, (candidate - baseline).abs().max().item()
                    )
                update_candidate(store, index, candidate - baseline, residual, y - candidate)
            update_joint(store, outputs[-1] - baseline, outputs[0] - baseline, residual)

    summaries, variables = [], []
    base_mse = evaluation["base_sse"].sum().item() / evaluation["count"].sum().item()
    for index, alpha_value in enumerate(ALPHAS):
        alpha_correction = calibration["delta_residual"][index] / (
            calibration["delta2"][index] + 1e-12
        )
        corrected_sse = (
            evaluation["base_sse"]
            - 2 * alpha_correction * evaluation["delta_residual"][index]
            + alpha_correction.square() * evaluation["delta2"][index]
        )
        direct_mse = evaluation["direct_sse"][index].sum().item() / evaluation["count"].sum().item()
        corrected_mse = corrected_sse.sum().item() / evaluation["count"].sum().item()
        summaries.append({
            "dataset": dataset, "graph_alpha": alpha_value, "base_mse": base_mse,
            "direct_mse": direct_mse,
            "direct_improvement_pct": 100 * (base_mse - direct_mse) / base_mse,
            "corrected_mse": corrected_mse,
            "corrected_improvement_pct": 100 * (base_mse - corrected_mse) / base_mse,
            "correction": alpha_correction.tolist(),
        })
        for var in range(args.enc_in):
            base_var = evaluation["base_sse"][var] / evaluation["count"][var]
            corrected_var = corrected_sse[var] / evaluation["count"][var]
            variables.append({
                "dataset": dataset, "graph_alpha": alpha_value, "variable": var,
                "correction": alpha_correction[var].item(), "base_mse": base_var.item(),
                "corrected_mse": corrected_var.item(),
                "corrected_improvement_pct": 100 * (base_var - corrected_var).item() / base_var.item(),
            })

    eye = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    coefficients = torch.linalg.solve(calibration["joint_xtx"] + 1e-8 * eye, calibration["joint_xtr"].unsqueeze(-1)).squeeze(-1)
    joint_sse = evaluation["base_sse"].clone()
    for var in range(args.enc_in):
        beta = coefficients[var]
        joint_sse[var] += beta @ evaluation["joint_xtx"][var] @ beta - 2 * beta @ evaluation["joint_xtr"][var]
    joint_mse = joint_sse.sum().item() / evaluation["count"].sum().item()
    metadata = {
        "dataset": dataset, "checkpoint": str(checkpoint), "validation_batches": len(loader),
        "split_batch": split, "alpha_0.5_equivalence_max_abs": equivalence_max_abs,
        "joint_static_adaptive_mse": joint_mse,
        "joint_static_adaptive_improvement_pct": 100 * (base_mse - joint_mse) / base_mse,
        "joint_coefficients": coefficients.tolist(), "test_accessed": False,
        **adjacency_diagnostics(static_adj, adaptive_adj),
    }
    return summaries, variables, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_summaries, all_variables, all_metadata = [], [], []
    for dataset in DATASETS:
        print(f"Graph-blend diagnosis: {dataset}-{PRED_LEN}", flush=True)
        summaries, variables, metadata = diagnose(dataset)
        all_summaries.extend(summaries); all_variables.extend(variables); all_metadata.append(metadata)
        (OUTPUT / f"{dataset.lower()}_{PRED_LEN}.json").write_text(
            json.dumps({"metadata": metadata, "summary": summaries, "variables": variables}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"metadata": metadata, "summary": summaries}, sort_keys=True), flush=True)
    write_csv(OUTPUT / "summary.csv", all_summaries)
    write_csv(OUTPUT / "variables.csv", all_variables)
    (OUTPUT / "metadata.json").write_text(json.dumps(all_metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
