#!/usr/bin/env python3
"""Measure matched forward latency, parameters, and CUDA memory."""

from __future__ import annotations

import csv
import gc
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import PatchTSTHistoryCorrection, PatchTSTIRPA, PatchTSTRecent


OUTPUT = ROOT / "logs" / "patchtst_irpa_timerole_2x2x3" / "efficiency.csv"
MODELS = {
    "PatchTSTRecent": PatchTSTRecent.Model,
    "PatchTSTIRPA": PatchTSTIRPA.Model,
    "PatchTSTHistoryCorrection": PatchTSTHistoryCorrection.Model,
}


def config(channels: int, horizon: int, layers: int, heads: int):
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=960, pred_len=horizon,
        enc_in=channels, d_model=512, d_ff=2048, n_heads=heads,
        e_layers=layers, factor=3, dropout=0.1, activation="gelu",
        moving_avg=25, irpa_revise_len=96, irpa_topk=3,
        timerole_hidden_dim=32, timerole_memory_pool=16,
    )


def measure(model_class, cfg, batch_size=32, warmup=10, repeats=30):
    torch.manual_seed(2021)
    model = model_class(cfg).cuda().eval()
    x = torch.randn(batch_size, 960, cfg.enc_in, device="cuda")
    x_dec = torch.zeros(batch_size, 48 + cfg.pred_len, cfg.enc_in, device="cuda")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, None, x_dec, None)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(x, None, x_dec, None)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated()
        if output.shape != (batch_size, cfg.pred_len, cfg.enc_in):
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    result = {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "latency_ms_mean": statistics.mean(timings),
        "latency_ms_sample_std": statistics.stdev(timings),
        "latency_ms_median": statistics.median(timings),
        "peak_memory_mib": peak / 2**20,
        "incremental_peak_mib": (peak - baseline) / 2**20,
    }
    del output, x_dec, x, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    rows = []
    datasets = (("ETTm1", 7, 3, 8), ("Weather", 21, 2, 4))
    for dataset, channels, layers, heads in datasets:
        for horizon in (96, 720):
            cfg = config(channels, horizon, layers, heads)
            for name, model_class in MODELS.items():
                row = {"dataset": dataset, "horizon": horizon, "model": name,
                       "batch_size": 32, "warmup": 10, "repeats": 30}
                row.update(measure(model_class, cfg))
                rows.append(row)
                print(row, flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {OUTPUT}")


if __name__ == "__main__":
    main()
