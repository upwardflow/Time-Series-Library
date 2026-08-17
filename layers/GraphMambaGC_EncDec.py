"""Graph-conditioned bidirectional Mamba encoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from layers.GraphMamba_EncDec import _resolve_mamba_class


class GraphConditionedMambaEncoderLayer(nn.Module):
    """Condition temporal scanning and directional fusion on graph context."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        activation: str,
        mamba_version: int,
        mamba_headdim: int,
        bidirectional: bool,
        graph_direction_fusion: bool,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.graph_direction_fusion = graph_direction_fusion and bidirectional

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
                    f"Mamba2 requires d_model * expand ({d_inner}) to be "
                    f"divisible by mamba_headdim ({headdim})"
                )
            mamba_kwargs["headdim"] = headdim

        if bidirectional:
            self.mamba_fwd = mamba_cls(**mamba_kwargs)
            self.mamba_bwd = mamba_cls(**mamba_kwargs)
        else:
            self.mamba = mamba_cls(**mamba_kwargs)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        if self.graph_direction_fusion:
            self.direction_gate = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.direction_gate.weight)
            nn.init.zeros_(self.direction_gate.bias)

        self.dropout = nn.Dropout(dropout)
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, graph_context):
        batch_size, n_vars, _, _ = x.shape
        sequence = rearrange(x, "b n d p -> (b n) p d")
        context = rearrange(graph_context, "b n d p -> (b n) p d")
        normalized = self.norm1(sequence)

        if self.bidirectional:
            forward_out = self.mamba_fwd(normalized)
            backward_out = self.mamba_bwd(normalized.flip(dims=[1])).flip(dims=[1])
            if self.graph_direction_fusion:
                gate = torch.sigmoid(self.direction_gate(self.context_norm(context)))
                # gate=0.5 at initialization reproduces the scale of a sum.
                mixed = 2.0 * (gate * forward_out + (1.0 - gate) * backward_out)
            else:
                mixed = forward_out + backward_out
        else:
            mixed = self.mamba(normalized)

        sequence = sequence + self.dropout(mixed)
        sequence = sequence + self.dropout(self.ffn(self.norm2(sequence)))
        return rearrange(
            sequence,
            "(b n) p d -> b n d p",
            b=batch_size,
            n=n_vars,
        )


class GraphConditionedMambaEncoder(nn.Module):
    def __init__(self, layers, norm_layer: nn.Module | None = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, graph_context):
        for layer in self.layers:
            x = layer(x, graph_context)
        if self.norm is not None:
            x = x.permute(0, 1, 3, 2)
            x = self.norm(x)
            x = x.permute(0, 1, 3, 2)
        return x
