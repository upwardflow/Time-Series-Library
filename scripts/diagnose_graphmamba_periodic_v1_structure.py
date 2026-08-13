#!/usr/bin/env python3
"""Audit GraphMamba periodic V1 geometry, routing, and gradient paths."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.GraphMamba import Model


class SharedSpyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x):
        self.calls.append(tuple(x.shape))
        return 1.1 * x


def config(**overrides) -> Namespace:
    values = dict(
        task_name="long_term_forecast", seq_len=96, pred_len=24, enc_in=3,
        d_model=16, patch_len=4, stride=2, use_decomp=1, use_patch=1,
        use_time_mamba=1, use_graph=1, use_lag_graph=0,
        dual_scale_scan_mode="periodic_aligned", moving_avg=25, dropout=0.0,
        d_ff=32, d_state=8, d_conv=2, expand=2, activation="gelu",
        mamba_version=1, mamba_headdim=0, mamba_bidirectional=1, e_layers=1,
        root_path="/path/that/does/not/exist", data_path="none.csv", data="custom",
        features="M", target="OT", graph_sample_size=8,
        graph_sample_method="uniform", seed=2021, graph_cache=0,
        graph_top_k=2, static_graph_mode="weighted", node_dim=4,
        graph_alpha=0.5, static_graph_only=0, periodic_period=24,
        periodic_local_patch=4, periodic_local_stride=2,
        periodic_period_stride=12, periodic_use_adapter=1,
        periodic_adapter_confidence=0,
        periodic_use_alignment=1, periodic_use_router=1,
        periodic_router_threshold=0.15,
    )
    values.update(overrides)
    return Namespace(**values)


def main() -> None:
    torch.manual_seed(2021)
    model = Model(config())
    spy = SharedSpyEncoder()
    model.encoder = spy

    time = torch.arange(96, dtype=torch.float32)
    periodic = torch.sin(2 * torch.pi * time / 24)
    noise = torch.randn(96)
    x = torch.stack(
        [
            torch.stack([periodic, periodic + 0.05 * noise, noise], dim=-1),
            torch.stack([periodic.roll(3), noise, 0.5 * periodic + noise], dim=-1),
        ]
    )
    seasonal = model.decomposition(x)[0].permute(0, 2, 1)
    confidence = model._period_confidence(seasonal)
    forecast = model(x, None, None, None)
    loss = forecast.square().mean()
    loss.backward()

    gradient_norms = {}
    for name in (
        "periodic_scale_conditioner.2.weight",
        "periodic_exchange_logit",
        "periodic_exchange_projection.weight",
        "periodic_router_gain",
    ):
        parameter = dict(model.named_parameters())[name]
        gradient_norms[name] = (
            None if parameter.grad is None else float(parameter.grad.norm())
        )

    payload = {
        "local_patch_count": model.local_patch_count,
        "period_patch_count": model.period_patch_count,
        "encoder_calls": spy.calls,
        "forecast_shape": list(forecast.shape),
        "finite_forecast": bool(torch.isfinite(forecast).all()),
        "local_alignment_row_error": float(
            (model.local_from_period_alignment.sum(dim=-1) - 1).abs().max()
        ),
        "period_alignment_row_error": float(
            (model.period_from_local_alignment.sum(dim=-1) - 1).abs().max()
        ),
        "confidence": confidence.tolist(),
        "gradient_norms": gradient_norms,
    }

    torch.manual_seed(2021)
    enabled = Model(config()).eval()
    disabled_config = config(
        periodic_use_adapter=0,
        periodic_use_alignment=0,
        periodic_use_router=0,
    )
    torch.manual_seed(2021)
    disabled = Model(disabled_config).eval()
    assert enabled.state_dict().keys() == disabled.state_dict().keys()
    identical_state = all(
        torch.equal(enabled.state_dict()[key], disabled.state_dict()[key])
        for key in enabled.state_dict()
    )
    enabled.encoder = SharedSpyEncoder()
    disabled.encoder = SharedSpyEncoder()
    with torch.no_grad():
        enabled_output = enabled(x, None, None, None)
        disabled_output = disabled(x, None, None, None)
    payload["enabled_disabled_identical_state"] = identical_state
    payload["zero_init_max_output_delta"] = float(
        (enabled_output - disabled_output).abs().max()
    )
    expected_calls = [(2, 3, 16, 48), (2, 3, 16, 8)]
    assert spy.calls == expected_calls, (spy.calls, expected_calls)
    assert forecast.shape == (2, 24, 3)
    assert torch.isfinite(forecast).all()
    assert payload["local_alignment_row_error"] < 1e-6
    assert payload["period_alignment_row_error"] < 1e-6
    assert confidence[0, 0] > confidence[0, 2]
    assert gradient_norms["periodic_scale_conditioner.2.weight"] > 0
    assert gradient_norms["periodic_exchange_logit"] > 0
    assert gradient_norms["periodic_router_gain"] > 0
    # Expected at a zero exchange gate: projection learns after the gate moves.
    assert gradient_norms["periodic_exchange_projection.weight"] == 0
    assert identical_state
    assert payload["zero_init_max_output_delta"] == 0

    output_dir = ROOT / "logs" / "graphmamba_periodic_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "structural_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
