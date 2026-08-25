#!/usr/bin/env python3
"""Audit SCSD Phase-0 encoder identity and patch-grid shape contracts."""

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


OUTPUT = ROOT / "logs" / "graphmamba_scsd_validation" / "audit"


def config(mode: str, selection: str = "dual") -> Namespace:
    return Namespace(
        task_name="long_term_forecast", seq_len=96, pred_len=96, enc_in=7,
        d_model=16, patch_len=4, stride=2, use_decomp=1, use_patch=1,
        use_time_mamba=1, use_graph=0, graph_mamba_fusion="fixed_sum",
        dual_scale_scan_mode=mode, dual_scale_selection=selection,
        periodic_period=24, moving_avg=25, d_ff=32, d_state=8, d_conv=2,
        expand=2, dropout=0.0, activation="gelu", mamba_version=1,
        mamba_headdim=0, mamba_bidirectional=1, e_layers=1,
        root_path=str(ROOT / "does_not_exist"), data_path="none.csv",
        data="custom", features="M", target="OT", graph_sample_size=16,
        graph_sample_method="uniform", seed=2021, graph_cache=0,
    )


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes: list[list[int]] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.shapes.append(list(value.shape))
        return value


def parameter_ids(module: nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def main() -> int:
    torch.manual_seed(2021)
    shared = Model(config("independent_shared"))
    assert not hasattr(shared, "fine_encoder")

    torch.manual_seed(2021)
    unshared = Model(config("independent_unshared"))
    assert unshared.encoder is not unshared.fine_encoder
    assert parameter_ids(unshared.encoder).isdisjoint(
        parameter_ids(unshared.fine_encoder)
    )
    shared_count = sum(p.numel() for p in shared.parameters())
    unshared_count = sum(p.numel() for p in unshared.parameters())
    assert unshared_count > shared_count
    for name, value in shared.state_dict().items():
        assert torch.equal(value, unshared.state_dict()[name]), name

    rows: dict[str, object] = {
        "shared_encoder_object_count": 1,
        "shared_parameter_count": shared_count,
        "unshared_encoder_object_count": 2,
        "unshared_parameter_count": unshared_count,
        "unshared_parameter_sets_disjoint": True,
        "shared_and_unshared_common_initialization_equal": True,
        "variants": {},
    }
    sample = torch.randn(2, 96, 7)
    for mode, selection in (
        ("joint", "dual"),
        ("independent_shared", "dual"),
        ("independent_unshared", "dual"),
        ("independent_shared", "coarse"),
        ("independent_shared", "fine"),
    ):
        torch.manual_seed(2021)
        model = Model(config(mode, selection))
        parameter_count = sum(p.numel() for p in model.parameters())
        coarse = Recorder()
        model.encoder = coarse
        fine = None
        if mode == "independent_unshared":
            fine = Recorder()
            model.fine_encoder = fine
        with torch.no_grad():
            output = model(sample, None, None, None)
        assert list(output.shape) == [2, 96, 7]
        key = f"{mode}:{selection}"
        rows["variants"][key] = {
            "coarse_or_shared_calls": coarse.shapes,
            "fine_calls": [] if fine is None else fine.shapes,
            "output_shape": list(output.shape),
            "parameter_count": parameter_count,
        }

    joint_calls = rows["variants"]["joint:dual"]["coarse_or_shared_calls"]
    shared_calls = rows["variants"]["independent_shared:dual"]["coarse_or_shared_calls"]
    unshared_fine = rows["variants"]["independent_unshared:dual"]["fine_calls"]
    assert len(joint_calls) == 1
    assert len(shared_calls) == 2
    assert len(unshared_fine) == 1
    assert rows["variants"]["independent_shared:coarse"]["coarse_or_shared_calls"][0][-1] == 48
    assert rows["variants"]["independent_shared:fine"]["coarse_or_shared_calls"][0][-1] == 96

    rows["status"] = "passed"
    rows["test_accessed"] = False
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "structure_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
