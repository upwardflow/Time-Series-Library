#!/usr/bin/env python3
"""Save learned periodic-V1 gates and validation-input confidence diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs" / "graphmamba_periodic_v1_validation" / "diagnostics",
    )
    args_cli = parser.parse_args()
    record = json.loads(args_cli.record.read_text())
    args = command_args(record["command"])
    candidates = list((ROOT / "checkpoints").glob(f"*{record['candidate']}*"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one checkpoint directory, got {candidates}")
    checkpoint = candidates[0] / "checkpoint.pth"

    torch.manual_seed(args.seed)
    model = Model(args).eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    _, loader = data_provider(args, "val")

    confidence_sum = torch.zeros(args.enc_in, dtype=torch.float64)
    confidence_sq = torch.zeros(args.enc_in, dtype=torch.float64)
    above = torch.zeros(args.enc_in, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for batch_x, _, _, _ in loader:
            x = batch_x.float()
            centered = x - x.mean(dim=1, keepdim=True)
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            seasonal, _ = model.decomposition(centered / stdev)
            confidence = model._period_confidence(seasonal.permute(0, 2, 1))
            confidence_sum += confidence.double().sum(dim=0)
            confidence_sq += confidence.double().square().sum(dim=0)
            above += (confidence >= model.periodic_router_threshold).double().sum(dim=0)
            count += confidence.shape[0]

    mean = confidence_sum / count
    std = (confidence_sq / count - mean.square()).clamp_min(0).sqrt()
    affine = model.periodic_scale_conditioner(model.periodic_scale_descriptors)
    gain, bias = affine.chunk(2, dim=-1)
    exchange = torch.tanh(model.periodic_exchange_logit)
    period_weight_at_mean = torch.sigmoid(
        model.periodic_router_gain
        * (mean.float() - model.periodic_router_threshold)
    )
    payload = {
        "candidate": record["candidate"],
        "dataset": record["dataset"],
        "checkpoint": str(checkpoint),
        "validation_samples": count,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "period_confidence_mean": mean.tolist(),
        "period_confidence_std": std.tolist(),
        "fraction_above_threshold": (above / count).tolist(),
        "scale_gain_abs_mean": torch.tanh(gain).abs().mean(dim=-1).tolist(),
        "scale_bias_abs_mean": bias.abs().mean(dim=-1).tolist(),
        "exchange_gate_abs_mean": exchange.abs().mean(dim=-1).tolist(),
        "exchange_gate_signed_mean": exchange.mean(dim=-1).tolist(),
        "router_gain": float(model.periodic_router_gain),
        "period_weight_at_variable_mean": period_weight_at_mean.tolist(),
    }
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    output = args_cli.output_dir / f"{record['candidate']}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
