"""Mamba encoder blocks used by GraphMamba."""

from __future__ import annotations

import importlib.util

import torch
import torch.nn as nn
from einops import rearrange


def _resolve_mamba_class(mamba_version: int):
    if mamba_version == 1:
        from mamba_ssm import Mamba

        return Mamba
    if mamba_version == 2:
        if importlib.util.find_spec("causal_conv1d") is None:
            raise ImportError(
                "GraphMamba with mamba_version=2 requires causal-conv1d. "
                "Install a causal-conv1d build compatible with the current "
                "PyTorch and CUDA versions, or use --mamba_version 1."
            )
        from mamba_ssm import Mamba2

        return Mamba2
    raise ValueError(f"mamba_version must be 1 or 2, got {mamba_version}")


class MambaEncoderLayer(nn.Module):
    """Pre-norm Mamba and FFN block for tensors shaped [B, N, D, P]."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        mamba_version: int = 1,
        mamba_headdim: int = 0,
        scan_mode: str = "time",
        bidirectional: bool = True,
    ):
        super().__init__()
        if scan_mode not in {"time", "variable"}:
            raise ValueError("scan_mode must be 'time' or 'variable'")

        self.scan_mode = scan_mode
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.bidirectional = bidirectional

        mamba_cls = _resolve_mamba_class(mamba_version)
        mamba_kwargs = {
            "d_model": d_model,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
        }
        if mamba_version == 2:
            d_inner = d_model * expand
            headdim = mamba_headdim or d_inner // 8
            if headdim <= 0 or d_inner % headdim != 0:
                raise ValueError(
                    f"Mamba2 requires d_model * expand ({d_inner}) to be divisible "
                    f"by mamba_headdim ({headdim})"
                )
            mamba_kwargs["headdim"] = headdim

        if self.bidirectional:
            self.mamba_fwd = mamba_cls(**mamba_kwargs)
            self.mamba_bwd = mamba_cls(**mamba_kwargs)
        else:
            self.mamba = mamba_cls(**mamba_kwargs)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
        )

    def forward(self, x, scan_mode: str | None = None):
        batch_size, n_vars, _, n_patches = x.shape
        active_scan_mode = scan_mode or self.scan_mode
        if active_scan_mode == "variable":
            sequence = rearrange(x, "b n d p -> (b p) n d")
        elif active_scan_mode == "time":
            sequence = rearrange(x, "b n d p -> (b n) p d")
        else:
            raise ValueError("scan_mode must be 'time' or 'variable'")

        normalized = self.norm1(sequence)
        if self.bidirectional:
            forward_out = self.mamba_fwd(normalized)
            backward_out = self.mamba_bwd(normalized.flip(dims=[1])).flip(dims=[1])
            mixed = forward_out + backward_out
        else:
            mixed = self.mamba(normalized)

        sequence = sequence + self.dropout(mixed)
        sequence = sequence + self.dropout(self.ffn(self.norm2(sequence)))

        if active_scan_mode == "variable":
            return rearrange(
                sequence,
                "(b p) n d -> b n d p",
                b=batch_size,
                p=n_patches,
            )
        return rearrange(
            sequence,
            "(b n) p d -> b n d p",
            b=batch_size,
            n=n_vars,
        )


class MambaEncoder(nn.Module):
    def __init__(self, layers, norm_layer: nn.Module | None = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, scan_mode: str | None = None):
        for layer in self.layers:
            x = layer(x, scan_mode=scan_mode)
        if self.norm is not None:
            x = x.permute(0, 1, 3, 2)
            x = self.norm(x)
            x = x.permute(0, 1, 3, 2)
        return x
