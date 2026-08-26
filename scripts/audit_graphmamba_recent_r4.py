#!/usr/bin/env python3
"""Audit the frozen GraphMambaRecentR4 structural contract without test data."""

from __future__ import annotations

import copy
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.GraphMambaRecent import Model as RecentModel
from models.GraphMambaRecentR4 import Model as R4Model
from models.GraphMambaRecentR4 import R4_STRUCTURE


class IdentityEncoder(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def config() -> Namespace:
    # Deliberately conflict with every frozen switch.  The R4 wrapper must
    # override these values while leaving this caller-owned object untouched.
    return Namespace(
        task_name="long_term_forecast",
        seq_len=336,
        label_len=48,
        pred_len=96,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=16,
        patch_len=4,
        stride=2,
        use_patch=0,
        moving_avg=25,
        graph_mamba_fusion="graph_residual_gate",
        dual_scale_scan_mode="joint",
        dual_scale_selection="coarse",
        d_ff=32,
        d_state=8,
        d_conv=2,
        expand=2,
        dropout=0.0,
        activation="gelu",
        mamba_version=1,
        mamba_headdim=0,
        mamba_bidirectional=0,
        e_layers=1,
        use_decomp=1,
        use_graph=1,
        use_time_mamba=0,
        root_path=str(ROOT / "dataset" / "ETT-small"),
        data_path="ETTm1.csv",
        data="ETTm1",
        features="M",
        target="OT",
        graph_sample_size=32,
        graph_sample_method="uniform",
        seed=2021,
        graph_cache=0,
        timerole_recent_len=48,
    )


def manual_r4_config(source: Namespace) -> Namespace:
    result = copy.copy(source)
    for name, value in R4_STRUCTURE.items():
        setattr(result, name, value)
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> int:
    supplied = config()
    original = vars(supplied).copy()

    torch.manual_seed(supplied.seed)
    r4 = R4Model(supplied)
    assert vars(supplied) == original, "R4 mutated the caller-owned config"

    torch.manual_seed(supplied.seed)
    manual = RecentModel(manual_r4_config(supplied))

    r4_state = r4.state_dict()
    manual_state = manual.state_dict()
    assert r4_state.keys() == manual_state.keys()
    assert all(torch.equal(r4_state[name], manual_state[name]) for name in r4_state)

    assert r4.variant == "R4"
    assert r4.input_seq_len == 336
    assert r4.recent_len == 96
    assert r4.seq_len == 96
    assert not r4.use_decomp
    assert r4.use_patch
    assert r4.dual_scale_scan_mode == "independent_shared"
    assert r4.dual_scale_selection == "fine"
    assert r4.use_time_mamba
    assert not r4.use_graph
    assert hasattr(r4, "short_patch_embedding")
    assert not hasattr(r4, "long_patch_embedding")
    assert hasattr(r4, "encoder")
    assert not hasattr(r4, "graph_mixer")
    assert not hasattr(r4, "decomposition")
    for forbidden in (
        "recent_context",
        "memory_context",
        "memory_decoder",
        "memory_scale",
        "last_memory_correction",
    ):
        assert not hasattr(r4, forbidden), forbidden
    r4_parameter_count = parameter_count(r4)

    # Replacing Mamba with the same identity map in both models makes this a
    # deterministic CPU audit of wrapper behavior, not a kernel benchmark.
    r4.encoder = IdentityEncoder()
    manual.encoder = IdentityEncoder()
    r4.eval()
    manual.eval()

    sample = torch.randn(2, 336, 7)
    with torch.no_grad():
        r4_output = r4(sample, None, None, None)
        manual_output = manual(sample, None, None, None)
    assert torch.equal(r4_output, manual_output)
    assert list(r4_output.shape) == [2, 96, 7]
    assert torch.isfinite(r4_output).all()

    changed_old = sample.clone()
    changed_old[:, :-96, :] += 1000.0 * torch.randn_like(changed_old[:, :-96, :])
    changed_recent = sample.clone()
    changed_recent[:, -1, :] += 0.25
    with torch.no_grad():
        old_output = r4(changed_old, None, None, None)
        recent_output = r4(changed_recent, None, None, None)
    assert torch.equal(r4_output, old_output), "old history leaked into R4"
    assert not torch.equal(r4_output, recent_output), "recent suffix has no effect"

    result = {
        "status": "passed",
        "variant": r4.variant,
        "test_accessed": False,
        "caller_config_unchanged": True,
        "manual_r4_state_exact": True,
        "manual_r4_output_exact": True,
        "old_history_invariant": True,
        "recent_suffix_effective": True,
        "single_forecast_head": True,
        "parameter_count": r4_parameter_count,
        "output_shape": list(r4_output.shape),
        "frozen_structure": dict(R4_STRUCTURE),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
