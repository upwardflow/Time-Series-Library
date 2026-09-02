#!/usr/bin/env python3
"""Audit TimeRole length parameterization without loading any dataset."""

from __future__ import annotations

import json
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TimeRole import RecentPredictor as RecentModel
from models.TimeRole import Model as TimeRoleModel


def build_model(model_type, cfg):
    """Construct a model without letting graph setup messages pollute JSON output."""
    with redirect_stdout(io.StringIO()):
        return model_type(cfg)


def config(seq_len: int = 336, recent_len: int = 96, pool: int = 16):
    return SimpleNamespace(
        task_name="long_term_forecast",
        data="ETTm1",
        seq_len=seq_len,
        pred_len=96,
        enc_in=7,
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
        root_path=str(ROOT / "dataset" / "ETT-small"),
        data_path="ETTm1.csv",
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
        timerole_recent_len=recent_len,
        timerole_memory_pool=pool,
        timerole_hidden_dim=32,
        timerole_old_intervention="intact",
        timerole_noise_std=1.0,
    )


def expect_error(model_type, cfg, fragment: str) -> bool:
    try:
        build_model(model_type, cfg)
    except ValueError as exc:
        return fragment in str(exc)
    return False


def main() -> int:
    torch.manual_seed(2021)
    explicit = build_model(TimeRoleModel, config())
    torch.manual_seed(2021)
    implicit_cfg = config()
    del implicit_cfg.timerole_recent_len
    implicit = build_model(TimeRoleModel, implicit_cfg)
    explicit_state = explicit.state_dict()
    implicit_state = implicit.state_dict()
    default_state_equal = explicit_state.keys() == implicit_state.keys() and all(
        torch.equal(explicit_state[key], implicit_state[key])
        for key in explicit_state
    )

    cases = []
    for seq_len, recent_len, pool in (
        (192, 96, 16),
        (336, 96, 16),
        (720, 96, 16),
        (960, 96, 16),
        (336, 48, 16),
        (336, 192, 16),
        (336, 96, 8),
        (336, 96, 24),
    ):
        model = build_model(TimeRoleModel, config(seq_len, recent_len, pool))
        recent = build_model(RecentModel, config(seq_len, recent_len, pool))
        cases.append(
            {
                "seq_len": seq_len,
                "recent_len": recent_len,
                "pool": pool,
                "old_len": model.old_len,
                "memory_tokens": model.memory_tokens,
                "recent_backbone_seq_len": recent.seq_len,
                "passed": (
                    model.old_len == seq_len - recent_len
                    and model.memory_tokens == (seq_len - recent_len) // pool
                    and recent.seq_len == recent_len
                ),
            }
        )

    invalid = {
        "zero_recent": expect_error(
            RecentModel, config(336, 0, 16), "must be positive"
        ),
        "recent_exceeds_input": expect_error(
            RecentModel, config(192, 336, 16), "seq_len >="
        ),
        "no_old_history": expect_error(
            TimeRoleModel, config(96, 96, 16), "seq_len >"
        ),
        "zero_pool": expect_error(
            TimeRoleModel, config(336, 96, 0), "must be positive"
        ),
        "nondivisible_old_history": expect_error(
            TimeRoleModel, config(336, 96, 17), "must be divisible"
        ),
    }
    result = {
        "default_implicit_equals_explicit": default_state_equal,
        "valid_cases": cases,
        "invalid_cases_rejected": invalid,
        "test_accessed": False,
    }
    result["passed"] = (
        default_state_equal
        and all(case["passed"] for case in cases)
        and all(invalid.values())
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
