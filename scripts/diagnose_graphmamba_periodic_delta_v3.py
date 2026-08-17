#!/usr/bin/env python3
"""Audit period-normalized Mamba delta scaling without accessing data splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.GraphMamba import Model
from scripts.diagnose_graphmamba_periodic_v1_structure import config


class DeltaSpyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, delta_scale=None):
        self.calls.append(
            {
                "shape": list(x.shape),
                "delta_scale": None
                if delta_scale is None
                else float(delta_scale.detach()),
            }
        )
        return x if delta_scale is None else x * delta_scale


def common_state_is_identical(first: Model, second: Model) -> bool:
    first_state = first.state_dict()
    second_state = second.state_dict()
    shared_keys = first_state.keys() & second_state.keys()
    return all(torch.equal(first_state[key], second_state[key]) for key in shared_keys)


def main() -> None:
    torch.manual_seed(2021)
    unit = Model(config(periodic_delta_mode="unit")).eval()
    torch.manual_seed(2021)
    learned = Model(config(periodic_delta_mode="learned")).eval()
    shared_init = common_state_is_identical(unit, learned)

    unit_spy = DeltaSpyEncoder()
    learned_spy = DeltaSpyEncoder()
    unit.encoder = unit_spy
    learned.encoder = learned_spy
    x = torch.randn(2, 96, 3)
    with torch.no_grad():
        unit_output = unit(x, None, None, None)
        learned_output = learned(x, None, None, None)

    initial_delta = float((unit_output - learned_output).abs().max())
    learned_initial_calls = list(learned_spy.calls)
    learned.periodic_delta_exponent.data.fill_(torch.atanh(torch.tensor(0.5)))
    learned_spy.calls.clear()
    active_output = learned(x, None, None, None)
    active_output.square().mean().backward()

    payload = {
        "period_normalized_strides": learned.periodic_scale_descriptors[:, 1].tolist(),
        "relative_stride_ratio": float(
            learned.periodic_scale_descriptors[1, 1]
            / learned.periodic_scale_descriptors[0, 1]
        ),
        "unit_calls": unit_spy.calls,
        "learned_initial_calls": learned_initial_calls,
        "active_calls": learned_spy.calls,
        "shared_initialization": shared_init,
        "unit_learned_initial_max_delta": initial_delta,
        "active_output_max_delta": float(
            (active_output.detach() - unit_output).abs().max()
        ),
        "delta_exponent_gradient": float(
            learned.periodic_delta_exponent.grad
        ),
        "finite_delta_exponent_gradient": bool(
            torch.isfinite(learned.periodic_delta_exponent.grad)
        ),
    }
    output = ROOT / "logs" / "graphmamba_periodic_delta_v3" / "structural_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
