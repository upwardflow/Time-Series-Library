#!/usr/bin/env python3
"""Read-only diagnosis of opposite TIRGE responses on ETTm1/ETTm2-720."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models.GraphMamba import Model as BaselineModel
from models.GraphMambaRG import Model as TIRGEModel
from scripts.diagnose_graphmamba_representations import command_args, locate_record
from utils.graph_utils import generate_adjacency


OUTPUT = ROOT / "logs" / "graphmamba_tirge_contrast"
RG_RECORDS = ROOT / "logs" / "graphmamba_rg_validation" / "validation"
DATA_ROOT = ROOT / "dataset" / "ETT-small"
DATASETS = ("ETTm1", "ETTm2")
PRED_LEN = 720


def rg_checkpoint(dataset: str) -> tuple[dict, Path]:
    record_path = RG_RECORDS / f"rg_{dataset.lower()}_720_deve.json"
    record = json.loads(record_path.read_text())
    matches = list((ROOT / "checkpoints").glob(f"*{record['candidate']}*/checkpoint.pth"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one TIRGE checkpoint for {dataset}, got {matches}")
    return record, matches[0]


def spectral_statistics(values: np.ndarray) -> dict:
    standardized = (values - values.mean(0)) / (values.std(0) + 1e-8)
    power = np.abs(np.fft.rfft(standardized, axis=0)) ** 2
    power[0] = 0
    total = power.sum(0) + 1e-12
    bins = power.shape[0]
    low = power[1 : max(2, bins // 20)].sum(0) / total
    cumulative = np.cumsum(power, axis=0) / total
    median_bins = np.argmax(cumulative >= 0.5, axis=0)
    spectral_entropy = -np.sum(
        (power / total) * np.log(power / total + 1e-12), axis=0
    ) / np.log(max(bins - 1, 2))
    daily_bin = int(round(len(values) / 96))
    daily_slice = power[max(1, daily_bin - 1) : daily_bin + 2].sum(0) / total
    weekly_bin = int(round(len(values) / (96 * 7)))
    weekly_slice = power[max(1, weekly_bin - 1) : weekly_bin + 2].sum(0) / total
    return {
        "low_5pct_power_mean": float(low.mean()),
        "median_frequency_bin_mean": float(median_bins.mean()),
        "spectral_entropy_mean": float(spectral_entropy.mean()),
        "daily_band_power_mean": float(daily_slice.mean()),
        "weekly_band_power_mean": float(weekly_slice.mean()),
        "per_variable_spectral_entropy": spectral_entropy.tolist(),
    }


def raw_statistics(dataset: str) -> dict:
    frame = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
    columns = [column for column in frame.columns if column != "date"]
    values = frame[columns].to_numpy(np.float64)[: 12 * 30 * 24 * 4]
    standardized = (values - values.mean(0)) / (values.std(0) + 1e-8)
    pearson = np.corrcoef(standardized, rowvar=False)
    distance = generate_adjacency(
        DATA_ROOT / f"{dataset}.csv", dataset, "M", "OT", 2000, "uniform", 2021, False
    ).astype(np.float64)
    offdiag = ~np.eye(len(columns), dtype=bool)
    eigenvalues = np.linalg.eigvalsh(distance)
    return {
        "columns": columns,
        "train_rows": len(values),
        "pearson_abs_offdiag_mean": float(np.abs(pearson[offdiag]).mean()),
        "pearson_signed_offdiag_mean": float(pearson[offdiag].mean()),
        "distance_corr_offdiag_mean": float(distance[offdiag].mean()),
        "distance_corr_offdiag_std": float(distance[offdiag].std()),
        "distance_graph_effective_rank": float(
            np.exp(-np.sum((eigenvalues.clip(0) / (eigenvalues.clip(0).sum() + 1e-12)) *
                           np.log(eigenvalues.clip(0) / (eigenvalues.clip(0).sum() + 1e-12) + 1e-12)))
        ),
        "mean_abs_first_difference": float(np.abs(np.diff(standardized, axis=0)).mean()),
        **spectral_statistics(values),
        "pearson": pearson.tolist(),
        "distance_correlation": distance.tolist(),
    }


def effective_adjacency(model: BaselineModel) -> torch.Tensor:
    mixer = model.graph_mixer
    adaptive = torch.softmax(
        mixer.node_embeddings @ mixer.node_embeddings.transpose(0, 1), dim=1
    )
    return mixer.alpha * mixer.static_adj + (1.0 - mixer.alpha) * adaptive


def graph_smoothness(x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    # x: B,N,D,P. Return normalized directed graph variation per batch.
    pairwise = (x[:, :, None] - x[:, None, :]).square().mean(dim=(3, 4))
    variation = (pairwise * adjacency[None]).sum(dim=(1, 2)) / adjacency.sum()
    energy = x.square().mean(dim=(1, 2, 3)) + 1e-8
    return variation / energy


def model_statistics(dataset: str) -> tuple[dict, list[dict], list[dict]]:
    baseline_record, baseline_checkpoint = locate_record(dataset, PRED_LEN)
    _, tirge_checkpoint = rg_checkpoint(dataset)
    args = command_args(baseline_record["command"])
    torch.manual_seed(2021)
    baseline = BaselineModel(args).cuda().eval()
    baseline.load_state_dict(torch.load(baseline_checkpoint, map_location="cuda", weights_only=True))
    torch.manual_seed(2021)
    tirge = TIRGEModel(args).cuda().eval()
    tirge.load_state_dict(torch.load(tirge_checkpoint, map_location="cuda", weights_only=True))
    _, loader = data_provider(args, "val")

    n_vars = args.enc_in
    quarters = 4
    base_se_var = torch.zeros(n_vars, dtype=torch.float64)
    rg_se_var = torch.zeros(n_vars, dtype=torch.float64)
    base_ae_var = torch.zeros(n_vars, dtype=torch.float64)
    rg_ae_var = torch.zeros(n_vars, dtype=torch.float64)
    base_se_q = torch.zeros(quarters, dtype=torch.float64)
    rg_se_q = torch.zeros(quarters, dtype=torch.float64)
    var_count = 0
    quarter_count = torch.zeros(quarters, dtype=torch.float64)
    sums = {key: 0.0 for key in (
        "token_norm", "innovation_norm", "innovation_token_cosine",
        "raw_graph_norm", "innovation_graph_norm", "raw_graph_temporal_cosine",
        "innovation_graph_temporal_cosine", "token_smoothness",
        "innovation_smoothness", "raw_fusion_cancellation",
        "innovation_fusion_cancellation",
        "trained_tirge_token_norm", "trained_tirge_innovation_norm",
        "trained_tirge_graph_norm", "trained_tirge_graph_temporal_cosine",
        "trained_tirge_fusion_cancellation", "trained_tirge_innovation_smoothness",
    )}
    sample_vars = 0
    adjacency = effective_adjacency(baseline)

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x = batch_x.float().cuda()
            y = batch_y[:, -PRED_LEN:, :].float().cuda()
            marks_x = batch_x_mark.float().cuda()
            marks_y = batch_y_mark.float().cuda()
            zeros = torch.zeros_like(y)
            decoder = torch.cat((batch_y[:, : args.label_len].float().cuda(), zeros), dim=1)
            base_pred = baseline(x, marks_x, decoder, marks_y)
            rg_pred = tirge(x, marks_x, decoder, marks_y)
            base_error = base_pred - y
            rg_error = rg_pred - y
            base_se_var += base_error.square().sum(dim=(0, 1)).double().cpu()
            rg_se_var += rg_error.square().sum(dim=(0, 1)).double().cpu()
            base_ae_var += base_error.abs().sum(dim=(0, 1)).double().cpu()
            rg_ae_var += rg_error.abs().sum(dim=(0, 1)).double().cpu()
            var_count += x.shape[0] * PRED_LEN
            for quarter in range(quarters):
                start, end = quarter * (PRED_LEN // quarters), (quarter + 1) * (PRED_LEN // quarters)
                base_se_q[quarter] += base_error[:, start:end].square().sum().double().cpu()
                rg_se_q[quarter] += rg_error[:, start:end].square().sum().double().cpu()
                quarter_count[quarter] += base_error[:, start:end].numel()

            means = x.mean(dim=1, keepdim=True)
            centered = x - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            normalized = centered / stdev
            seasonal, _ = baseline.decomposition(normalized)
            seasonal = seasonal.permute(0, 2, 1)
            tokens = torch.cat((
                baseline.long_patch_embedding(seasonal) + baseline.variable_embedding,
                baseline.short_patch_embedding(seasonal) + baseline.variable_embedding,
            ), dim=-1)
            temporal = baseline.encoder(tokens)
            innovation = temporal - tokens
            graph_raw = baseline.graph_mixer(tokens)
            graph_innovation = baseline.graph_mixer(innovation)
            rg_seasonal, _ = tirge.decomposition(normalized)
            rg_seasonal = rg_seasonal.permute(0, 2, 1)
            rg_tokens = torch.cat((
                tirge.long_patch_embedding(rg_seasonal) + tirge.variable_embedding,
                tirge.short_patch_embedding(rg_seasonal) + tirge.variable_embedding,
            ), dim=-1)
            rg_temporal = tirge.encoder(rg_tokens)
            rg_innovation = rg_temporal - rg_tokens
            rg_graph = tirge.graph_mixer(rg_innovation)
            flatten = lambda value: value.flatten(2)
            token_norm = flatten(tokens).norm(dim=-1)
            innovation_norm = flatten(innovation).norm(dim=-1)
            raw_graph_norm = flatten(graph_raw).norm(dim=-1)
            innovation_graph_norm = flatten(graph_innovation).norm(dim=-1)
            sums["token_norm"] += token_norm.sum().item()
            sums["innovation_norm"] += innovation_norm.sum().item()
            sums["innovation_token_cosine"] += F.cosine_similarity(flatten(innovation), flatten(tokens), dim=-1).sum().item()
            sums["raw_graph_norm"] += raw_graph_norm.sum().item()
            sums["innovation_graph_norm"] += innovation_graph_norm.sum().item()
            sums["raw_graph_temporal_cosine"] += F.cosine_similarity(flatten(graph_raw), flatten(temporal), dim=-1).sum().item()
            sums["innovation_graph_temporal_cosine"] += F.cosine_similarity(flatten(graph_innovation), flatten(temporal), dim=-1).sum().item()
            sums["token_smoothness"] += graph_smoothness(tokens, adjacency).sum().item()
            sums["innovation_smoothness"] += graph_smoothness(innovation, adjacency).sum().item()
            sums["raw_fusion_cancellation"] += (flatten(temporal + graph_raw).norm(dim=-1) / (flatten(temporal).norm(dim=-1) + raw_graph_norm + 1e-8)).sum().item()
            sums["innovation_fusion_cancellation"] += (flatten(temporal + graph_innovation).norm(dim=-1) / (flatten(temporal).norm(dim=-1) + innovation_graph_norm + 1e-8)).sum().item()
            trained_token_norm = flatten(rg_tokens).norm(dim=-1)
            trained_innovation_norm = flatten(rg_innovation).norm(dim=-1)
            trained_graph_norm = flatten(rg_graph).norm(dim=-1)
            sums["trained_tirge_token_norm"] += trained_token_norm.sum().item()
            sums["trained_tirge_innovation_norm"] += trained_innovation_norm.sum().item()
            sums["trained_tirge_graph_norm"] += trained_graph_norm.sum().item()
            sums["trained_tirge_graph_temporal_cosine"] += F.cosine_similarity(flatten(rg_graph), flatten(rg_temporal), dim=-1).sum().item()
            sums["trained_tirge_fusion_cancellation"] += (flatten(rg_temporal + rg_graph).norm(dim=-1) / (flatten(rg_temporal).norm(dim=-1) + trained_graph_norm + 1e-8)).sum().item()
            sums["trained_tirge_innovation_smoothness"] += graph_smoothness(rg_innovation, effective_adjacency(tirge)).sum().item()
            sample_vars += x.shape[0] * n_vars

    columns = raw_statistics(dataset)["columns"]
    variable_rows = []
    for index, column in enumerate(columns):
        base_mse = (base_se_var[index] / var_count).item()
        rg_mse = (rg_se_var[index] / var_count).item()
        variable_rows.append({
            "dataset": dataset, "variable": column,
            "baseline_mse": base_mse, "tirge_mse": rg_mse,
            "mse_improvement_pct": 100 * (base_mse - rg_mse) / base_mse,
            "baseline_mae": (base_ae_var[index] / var_count).item(),
            "tirge_mae": (rg_ae_var[index] / var_count).item(),
        })
    quarter_rows = []
    for quarter in range(quarters):
        base_mse = (base_se_q[quarter] / quarter_count[quarter]).item()
        rg_mse = (rg_se_q[quarter] / quarter_count[quarter]).item()
        quarter_rows.append({
            "dataset": dataset, "quarter": quarter + 1,
            "horizon_start": quarter * 180 + 1, "horizon_end": (quarter + 1) * 180,
            "baseline_mse": base_mse, "tirge_mse": rg_mse,
            "mse_improvement_pct": 100 * (base_mse - rg_mse) / base_mse,
        })
    summary = {"dataset": dataset, "validation_batches": len(loader)}
    summary.update({key: value / sample_vars for key, value in sums.items()})
    summary["innovation_token_norm_ratio"] = summary["innovation_norm"] / summary["token_norm"]
    summary["innovation_raw_graph_norm_ratio"] = summary["innovation_graph_norm"] / summary["raw_graph_norm"]
    summary["effective_adjacency"] = adjacency.cpu().tolist()
    tirge_adjacency = effective_adjacency(tirge)
    summary["trained_tirge_innovation_token_norm_ratio"] = summary["trained_tirge_innovation_norm"] / summary["trained_tirge_token_norm"]
    summary["trained_tirge_adjacency_change_frobenius"] = (tirge_adjacency - adjacency).norm().item()
    summary["trained_tirge_effective_adjacency"] = tirge_adjacency.cpu().tolist()
    return summary, variable_rows, quarter_rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_rows, model_rows, variable_rows, quarter_rows = [], [], [], []
    for dataset in DATASETS:
        print(f"Diagnosing {dataset}-720", flush=True)
        raw = raw_statistics(dataset)
        model, variables, quarters = model_statistics(dataset)
        raw_rows.append(raw); model_rows.append(model)
        variable_rows.extend(variables); quarter_rows.extend(quarters)
        (OUTPUT / f"{dataset.lower()}_diagnosis.json").write_text(
            json.dumps({"raw": raw, "model": model, "variables": variables, "quarters": quarters}, indent=2) + "\n"
        )
        print(json.dumps({"dataset": dataset, **{k: v for k, v in raw.items() if not isinstance(v, list)}, **{k: v for k, v in model.items() if not isinstance(v, list)}}, sort_keys=True), flush=True)
    for name, rows in (("variable_errors.csv", variable_rows), ("horizon_errors.csv", quarter_rows)):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
