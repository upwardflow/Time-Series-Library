#!/usr/bin/env python3
"""Audit TimeRole R0-R5 structure and paired data-order contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models.GraphMambaRecent import Model as RecentModel
from models.TimeRole import Model as TimeRoleModel


OUTPUT = ROOT / "logs" / "timerole_recent_simplification" / "audit"
VARIANTS = {
    "R0": dict(use_decomp=1, dual_scale_selection="dual", mamba_bidirectional=1, use_graph=1, use_time_mamba=1),
    "R1": dict(use_decomp=0, dual_scale_selection="fine", mamba_bidirectional=1, use_graph=1, use_time_mamba=1),
    "R2": dict(use_decomp=0, dual_scale_selection="coarse", mamba_bidirectional=1, use_graph=1, use_time_mamba=1),
    "R3": dict(use_decomp=0, dual_scale_selection="fine", mamba_bidirectional=0, use_graph=1, use_time_mamba=1),
    "R4": dict(use_decomp=0, dual_scale_selection="fine", mamba_bidirectional=1, use_graph=0, use_time_mamba=1),
    "R5": dict(use_decomp=0, dual_scale_selection="fine", mamba_bidirectional=1, use_graph=1, use_time_mamba=0),
}


def config(variant: str) -> Namespace:
    flags = VARIANTS[variant]
    return Namespace(
        task_name="long_term_forecast", seq_len=336, label_len=48,
        pred_len=96, enc_in=7, dec_in=7, c_out=7, d_model=16,
        patch_len=4, stride=2, use_patch=1, moving_avg=25,
        graph_mamba_fusion="fixed_sum", dual_scale_scan_mode="independent_shared",
        periodic_period=24, periodic_local_patch=4, periodic_local_stride=2,
        periodic_period_stride=12, periodic_use_adapter=0,
        d_ff=32, d_state=8, d_conv=2, expand=2, dropout=0.0,
        activation="gelu", mamba_version=1, mamba_headdim=0, e_layers=1,
        root_path=str(ROOT / "dataset" / "ETT-small"), data_path="ETTm1.csv",
        data="ETTm1", features="M", target="OT", freq="t", embed="timeF",
        seasonal_patterns="Monthly", graph_sample_size=32,
        graph_sample_method="uniform", seed=2021, graph_cache=0,
        graph_alpha=0.5, graph_top_k=2, static_graph_mode="weighted",
        gc_graph_dim=16, gc_temperature=1.0, gc_residual_init=0.5,
        gc_dynamic_graph=1, gc_symmetric_graph=1, gc_input_modulation=1,
        gc_direction_fusion=1, gc_parallel_residual=1,
        timerole_recent_len=96, timerole_memory_pool=16,
        timerole_hidden_dim=32, timerole_old_intervention="intact",
        timerole_noise_std=1.0, batch_size=32, num_workers=0,
        augmentation_ratio=0,
        **flags,
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class IdentityRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[int]] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(list(value.shape))
        return value


def digest_order(loader, epochs: int = 2) -> list[str]:
    hashes = []
    for _ in range(epochs):
        values = list(loader.sampler)
        payload = ",".join(map(str, values)).encode("ascii")
        hashes.append(hashlib.sha256(payload).hexdigest())
    return hashes


def common_initialization_equal(recent: RecentModel, timerole: TimeRoleModel) -> bool:
    recent_state = recent.state_dict()
    timerole_state = timerole.state_dict()
    return all(
        name in timerole_state and torch.equal(value, timerole_state[name])
        for name, value in recent_state.items()
    )


def main() -> int:
    rows: dict[str, object] = {
        "status": "running", "test_accessed": False,
        "scan_mode": "independent_shared", "variants": {},
    }
    train_orders: dict[str, list[str]] = {}
    sample = torch.randn(2, 336, 7)

    for variant, expected in VARIANTS.items():
        cfg = config(variant)
        torch.manual_seed(cfg.seed)
        recent = RecentModel(cfg)
        torch.manual_seed(cfg.seed)
        timerole = TimeRoleModel(cfg)
        assert common_initialization_equal(recent, timerole), variant
        recent_parameter_count = parameter_count(recent)
        timerole_parameter_count = parameter_count(timerole)
        assert timerole_parameter_count > recent_parameter_count, variant

        has_long = hasattr(timerole, "long_patch_embedding")
        has_short = hasattr(timerole, "short_patch_embedding")
        assert has_long == (expected["dual_scale_selection"] in {"dual", "coarse"})
        assert has_short == (expected["dual_scale_selection"] in {"dual", "fine"})
        assert hasattr(timerole, "decomposition") == bool(expected["use_decomp"])
        assert hasattr(timerole, "graph_mixer") == bool(expected["use_graph"])
        assert hasattr(timerole, "encoder") == bool(expected["use_time_mamba"])

        temporal_recorder = None
        graph_recorder = None
        if hasattr(timerole, "encoder"):
            temporal_recorder = IdentityRecorder()
            timerole.encoder = temporal_recorder
        if hasattr(timerole, "graph_mixer"):
            graph_recorder = IdentityRecorder()
            timerole.graph_mixer = graph_recorder
        timerole.eval()
        with torch.no_grad():
            output = timerole(sample, None, None, None)
        assert list(output.shape) == [2, 96, 7]
        assert torch.isfinite(output).all()

        _, train_loader = data_provider(cfg, "train")
        train_orders[variant] = digest_order(train_loader)
        _, val_loader = data_provider(cfg, "val")
        assert val_loader.sampler.__class__.__name__ == "SequentialSampler"

        rows["variants"][variant] = {
            "active_components": {
                "decomposition": bool(expected["use_decomp"]),
                "scale": expected["dual_scale_selection"],
                "mamba": bool(expected["use_time_mamba"]),
                "mamba_bidirectional": bool(
                    expected["mamba_bidirectional"] and expected["use_time_mamba"]
                ),
                "graph": bool(expected["use_graph"]),
                "timerole": True,
            },
            "has_long_patch_embedding": has_long,
            "has_short_patch_embedding": has_short,
            "parameter_count": timerole_parameter_count,
            "recent_only_parameter_count": recent_parameter_count,
            "recent_timerole_common_initialization_equal": True,
            "output_shape": list(output.shape),
            "output_finite": True,
            "temporal_call_shapes": (
                [] if temporal_recorder is None else temporal_recorder.calls
            ),
            "graph_call_shapes": (
                [] if graph_recorder is None else graph_recorder.calls
            ),
            "train_order_sha256_by_epoch": train_orders[variant],
            "validation_sampler": val_loader.sampler.__class__.__name__,
        }

    reference = train_orders["R0"]
    assert all(order == reference for order in train_orders.values())
    assert len(rows["variants"]["R0"]["temporal_call_shapes"]) == 2
    assert len(rows["variants"]["R4"]["graph_call_shapes"]) == 0
    assert len(rows["variants"]["R5"]["temporal_call_shapes"]) == 0
    rows["paired_train_orders_equal"] = True
    rows["train_order_epochs_checked"] = len(reference)
    rows["status"] = "passed"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "structure_and_rng_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
