#!/usr/bin/env python3
"""Structural audit for GraphMamba's same-phase cross-period scan."""

from __future__ import annotations

import json
import argparse
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
        self.calls: list[torch.Tensor] = []

    def forward(self, x, scan_mode=None):
        self.calls.append(x.detach().clone())
        return 1.1 * x


class ZeroGraph(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def config(**overrides) -> Namespace:
    values = dict(
        task_name="long_term_forecast", seq_len=96, pred_len=24, enc_in=3,
        d_model=16, patch_len=4, stride=2, use_decomp=1, use_patch=1,
        use_time_mamba=1, use_graph=1, dual_scale_scan_mode="periodic_phase",
        moving_avg=25, dropout=0.0, d_ff=32, d_state=8, d_conv=2,
        expand=2, activation="gelu", mamba_version=1, mamba_headdim=0,
        mamba_bidirectional=1, e_layers=1,
        root_path="/path/that/does/not/exist", data_path="none.csv", data="custom",
        features="M", target="OT", graph_sample_size=8,
        graph_sample_method="uniform", seed=2021, graph_cache=0,
        graph_top_k=2, static_graph_mode="weighted", node_dim=4,
        graph_alpha=0.5, static_graph_only=0, periodic_period=24,
        periodic_local_patch=4, periodic_local_stride=2,
        periodic_period_stride=12, periodic_use_adapter=1,
    )
    values.update(overrides)
    return Namespace(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(2021)
    model = Model(config())
    candidate_parameter_count = sum(p.numel() for p in model.parameters())
    cuda_smoke = None
    if args.cuda_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke test requested but CUDA is unavailable")
        cuda_model = model.cuda().train()
        cuda_input = torch.randn(2, 96, 3, device="cuda", requires_grad=True)
        cuda_output = cuda_model(cuda_input, None, None, None)
        cuda_output.square().mean().backward()
        cuda_smoke = {
            "output_shape": list(cuda_output.shape),
            "finite_output": bool(torch.isfinite(cuda_output).all()),
            "finite_input_gradient": bool(torch.isfinite(cuda_input.grad).all()),
        }
        assert cuda_smoke["finite_output"]
        assert cuda_smoke["finite_input_gradient"]
        model = model.cpu()
    spy = SharedSpyEncoder()
    model.encoder = spy
    model.graph_mixer = ZeroGraph()

    captured_values = []

    def capture_phase_values(_module, inputs):
        captured_values.append(inputs[0].detach().clone())

    model.period_phase_value_embedding.register_forward_pre_hook(capture_phase_values)

    seasonal = torch.arange(96, dtype=torch.float32).reshape(1, 1, 96)
    seasonal = seasonal.expand(2, 3, 96).clone()
    states = model._periodic_multiscale_states(seasonal)

    phase_values = captured_values[0].squeeze(-1)
    expected = torch.arange(96, dtype=torch.float32).reshape(4, 24).transpose(0, 1)
    index_error = float((phase_values[0, 0] - expected).abs().max())

    x = torch.randn(2, 96, 3, requires_grad=True)
    forecast = model(x, None, None, None)
    forecast.square().mean().backward()
    phase_gradient = model.period_phase_value_embedding.weight.grad
    head_gradient = model.head.linear.weight.grad

    torch.manual_seed(2021)
    accepted_a = Model(config(dual_scale_scan_mode="periodic_aligned"))
    torch.manual_seed(2021)
    accepted_b = Model(config(dual_scale_scan_mode="periodic_aligned"))
    accepted_state_unchanged = (
        accepted_a.state_dict().keys() == accepted_b.state_dict().keys()
        and all(
            torch.equal(accepted_a.state_dict()[key], accepted_b.state_dict()[key])
            for key in accepted_a.state_dict()
        )
    )
    accepted_has_candidate_parameters = any(
        key.startswith("period_phase_") for key in accepted_a.state_dict()
    )

    call_shapes = [list(tensor.shape) for tensor in spy.calls[:2]]
    phase_call = spy.calls[1]
    phase_batch_mapping_error = 0.0
    embedded = captured_values[0]
    for phase in range(24):
        expected_tokens = model.period_phase_value_embedding(
            embedded[:, :, phase]
        ).permute(0, 1, 3, 2)
        # Variable embeddings and the zero-initialized scale adapter are added
        # before the encoder. Compare each folded batch row to its source phase.
        expected_tokens = expected_tokens + model.variable_embedding
        actual = phase_call[phase::24]
        phase_batch_mapping_error = max(
            phase_batch_mapping_error,
            float((actual - expected_tokens).abs().max()),
        )

    payload = {
        "mode": model.dual_scale_scan_mode,
        "period": model.periodic_period,
        "cycles": model.period_cycle_count,
        "local_patch_count": model.local_patch_count,
        "phase_token_count": model.period_patch_count,
        "index_error": index_error,
        "phase_3_sequence": phase_values[0, 0, 3].tolist(),
        "encoder_call_shapes": call_shapes,
        "states_shape": list(states.shape),
        "forecast_shape": list(forecast.shape),
        "finite_forecast": bool(torch.isfinite(forecast).all()),
        "finite_input_gradient": bool(torch.isfinite(x.grad).all()),
        "phase_projection_gradient_norm": float(phase_gradient.norm()),
        "head_gradient_norm": float(head_gradient.norm()),
        "phase_batch_mapping_error": phase_batch_mapping_error,
        "shared_encoder_call_count": len(spy.calls),
        "accepted_state_reproducible": accepted_state_unchanged,
        "accepted_has_candidate_parameters": accepted_has_candidate_parameters,
        "candidate_parameter_count": candidate_parameter_count,
        "accepted_parameter_count": sum(p.numel() for p in accepted_a.parameters()),
        "cuda_smoke": cuda_smoke,
    }

    assert index_error == 0.0
    assert payload["phase_3_sequence"] == [3.0, 27.0, 51.0, 75.0]
    assert call_shapes == [[2, 3, 16, 48], [48, 3, 16, 4]]
    assert states.shape == (2, 3, 16, 72)
    assert forecast.shape == (2, 24, 3)
    assert payload["finite_forecast"] and payload["finite_input_gradient"]
    assert payload["phase_projection_gradient_norm"] > 0
    assert payload["head_gradient_norm"] > 0
    assert phase_batch_mapping_error < 1e-6
    assert accepted_state_unchanged
    assert not accepted_has_candidate_parameters

    output_dir = ROOT / "logs" / "graphmamba_cross_period_phase"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "structural_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
