#!/usr/bin/env python3
"""Numerically audit the preregistered physical-time-normalized candidate."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.GraphMambaPeriodNorm import Model as PeriodNormModel
from models.GraphMambaRecent import Model as RecentModel


OUT = ROOT / "logs/graphmamba_period_normalized_v2/structure_audit.json"


def config(model: str, factor: int, n_vars: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        task_name="long_term_forecast",
        seq_len=336,
        pred_len=192,
        enc_in=n_vars,
        d_model=64,
        d_ff=128,
        d_state=32,
        d_conv=2,
        expand=2,
        dropout=0.0,
        activation="gelu",
        mamba_version=1,
        mamba_headdim=32,
        mamba_bidirectional=1,
        e_layers=1,
        use_decomp=1,
        use_patch=1,
        use_time_mamba=1,
        use_graph=1,
        moving_avg=25,
        patch_len=4,
        stride=2,
        dual_scale_scan_mode="independent_shared",
        periodic_period=24,
        periodic_local_patch=4,
        periodic_local_stride=2,
        periodic_period_stride=12,
        periodic_use_adapter=1,
        period_norm_factor=factor,
        period_norm_recent_len=96,
        root_path=str(ROOT / "data"),
        data_path="__structure_audit_missing__.csv",
        data="custom",
        features="M",
        target="OT",
        graph_sample_size=2000,
        graph_sample_method="uniform",
        graph_cache=0,
        graph_top_k=2,
        graph_alpha=0.5,
        static_graph_mode="weighted",
        static_graph_only=0,
        node_dim=10,
        seed=2021,
    )


def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def audit_factor(factor: int, device: torch.device) -> dict:
    torch.manual_seed(2021)
    candidate = PeriodNormModel(config("GraphMambaPeriodNorm", factor)).to(device)
    baseline = RecentModel(config("GraphMambaRecent", factor)).to(device)

    ramp = torch.arange(336, dtype=torch.float32, device=device).view(1, 336, 1)
    pooled = candidate._hourly_pool(ramp).flatten()
    expected = ramp.flatten().view(-1, factor).mean(dim=1)
    pooling_exact = bool(torch.equal(pooled, expected))

    encoder_shapes: list[list[int]] = []
    hook = candidate.encoder.register_forward_pre_hook(
        lambda _module, inputs: encoder_shapes.append(list(inputs[0].shape))
    )
    x = torch.randn(2, 336, 7, device=device, requires_grad=True)
    output = candidate(x, None, None, None)
    loss = output.square().mean()
    loss.backward()
    hook.remove()

    named_gradients = {
        name: bool(param.grad is not None and torch.isfinite(param.grad).all())
        for name, param in candidate.named_parameters()
        if any(
            key in name
            for key in (
                "local_patch_embedding.projection",
                "period_patch_embedding.projection",
                "encoder",
                "periodic_scale_conditioner",
                "head.linear",
            )
        )
    }
    candidate_params = trainable_parameters(candidate)
    baseline_params = trainable_parameters(baseline)
    expected_local = (96 + 2 - 4) // 2 + 1
    hourly_len = 336 // factor
    expected_period = (hourly_len + 12 - 24) // 12 + 1
    result = {
        "factor": factor,
        "hourly_length": candidate.hourly_len,
        "pooling_exact": pooling_exact,
        "local_patch_count": candidate.local_patch_count,
        "period_patch_count": candidate.period_patch_count,
        "expected_patch_counts": [expected_local, expected_period],
        "encoder_call_shapes": encoder_shapes,
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "selected_gradients_finite": named_gradients,
        "all_selected_gradients_finite": bool(named_gradients) and all(named_gradients.values()),
        "candidate_trainable_parameters": candidate_params,
        "baseline_trainable_parameters": baseline_params,
        "parameter_ratio": candidate_params / baseline_params,
    }
    result["passed"] = all(
        [
            result["pooling_exact"],
            [result["local_patch_count"], result["period_patch_count"]]
            == result["expected_patch_counts"],
            len(encoder_shapes) == 2,
            encoder_shapes[0][-1] == expected_local,
            encoder_shapes[1][-1] == expected_period,
            result["output_shape"] == [2, 192, 7],
            result["output_finite"],
            result["loss_finite"],
            result["all_selected_gradients_finite"],
            result["parameter_ratio"] <= 1.05,
        ]
    )
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Mamba structural audit")
    device = torch.device("cuda:0")
    results = [audit_factor(factor, device) for factor in (1, 4, 6)]
    payload = {
        "device": torch.cuda.get_device_name(device),
        "factors": results,
        "passed": all(result["passed"] for result in results),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
