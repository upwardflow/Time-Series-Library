"""Sample-adaptive graph conditioning for GraphMambaGC."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sparsify_static_adjacency(
    adjacency: np.ndarray,
    top_k: int,
    mode: str,
) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("static adjacency must be square")
    if mode not in {"weighted", "binary"}:
        raise ValueError("static graph mode must be 'weighted' or 'binary'")

    n_vars = adjacency.shape[0]
    top_k = max(0, min(int(top_k), n_vars - 1))
    without_diag = adjacency.copy()
    np.fill_diagonal(without_diag, -np.inf)
    sparse = np.zeros_like(adjacency, dtype=np.float32)
    if top_k:
        indices = np.argsort(without_diag, axis=1)[:, -top_k:]
        for row in range(n_vars):
            if mode == "binary":
                sparse[row, indices[row]] = 1.0
            else:
                sparse[row, indices[row]] = adjacency[row, indices[row]]
    np.fill_diagonal(sparse, 1.0)
    return sparse / np.clip(sparse.sum(axis=1, keepdims=True), 1e-6, None)


class SampleAdaptiveGraphConditioner(nn.Module):
    """Build a patch-wise variable graph and condition temporal tokens with it.

    Input and output tensors use ``[batch, variables, d_model, patches]``.
    The dynamic graph has shape ``[batch, patches, variables, variables]``;
    consequently two samples may use different variable relations and the
    relation may also change between temporal patches.
    """

    def __init__(self, config, static_adj: np.ndarray | None = None):
        super().__init__()
        self.d_model = int(config.d_model)
        self.n_vars = int(config.enc_in)
        graph_dim = int(getattr(config, "gc_graph_dim", 16))
        if graph_dim < 1:
            raise ValueError("gc_graph_dim must be positive")

        if static_adj is None:
            static_adj = np.eye(self.n_vars, dtype=np.float32)
        static_adj = np.asarray(static_adj, dtype=np.float32)
        if static_adj.shape != (self.n_vars, self.n_vars):
            raise ValueError(
                f"expected static adjacency {(self.n_vars, self.n_vars)}, "
                f"got {static_adj.shape}"
            )
        static_adj = _sparsify_static_adjacency(
            static_adj,
            top_k=int(getattr(config, "graph_top_k", 2)),
            mode=getattr(config, "static_graph_mode", "weighted"),
        )
        self.register_buffer("static_adj", torch.from_numpy(static_adj))

        self.static_weight = float(getattr(config, "graph_alpha", 0.5))
        if not 0.0 <= self.static_weight <= 1.0:
            raise ValueError("graph_alpha must be between 0 and 1")
        self.temperature = float(getattr(config, "gc_temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError("gc_temperature must be positive")
        self.use_dynamic_graph = bool(getattr(config, "gc_dynamic_graph", 1))
        self.symmetric_dynamic = bool(getattr(config, "gc_symmetric_graph", 1))
        self.use_input_modulation = bool(
            getattr(config, "gc_input_modulation", 1)
        )

        self.query = nn.Linear(self.d_model, graph_dim, bias=False)
        if not self.symmetric_dynamic:
            self.key = nn.Linear(self.d_model, graph_dim, bias=False)
        self.value = nn.Linear(self.d_model, self.d_model, bias=False)
        self.modulation_gate = nn.Linear(2 * self.d_model, self.d_model)
        self.context_norm = nn.LayerNorm(self.d_model)
        # Keep graph conditioning deterministic so a zero-initialized adapter
        # preserves the baseline model's dropout stream exactly.
        self.dropout = nn.Identity()

        residual_init = float(getattr(config, "gc_residual_init", 0.5))
        if not 0.0 <= residual_init < 1.0:
            raise ValueError("gc_residual_init must lie in [0, 1)")
        self.residual_logit = nn.Parameter(
            torch.tensor(math.atanh(residual_init), dtype=torch.float32)
        )

    def forward(self, x):
        batch_size, n_vars, d_model, _ = x.shape
        if n_vars != self.n_vars or d_model != self.d_model:
            raise ValueError(
                f"expected variables/model dim {(self.n_vars, self.d_model)}, "
                f"got {(n_vars, d_model)}"
            )

        tokens = x.permute(0, 3, 1, 2)  # [B, P, N, D]
        if self.use_dynamic_graph:
            query = F.normalize(self.query(tokens), dim=-1)
            key = (
                query
                if self.symmetric_dynamic
                else F.normalize(self.key(tokens), dim=-1)
            )
            logits = torch.einsum("bpig,bpjg->bpij", query, key)
            dynamic_adj = torch.softmax(logits / self.temperature, dim=-1)
            adjacency = (
                self.static_weight * self.static_adj[None, None, :, :]
                + (1.0 - self.static_weight) * dynamic_adj
            )
        else:
            adjacency = self.static_adj[None, None, :, :].expand(
                batch_size, tokens.shape[1], -1, -1
            )

        graph_context = torch.einsum("bpij,bpjd->bpid", adjacency, tokens)
        graph_context = self.context_norm(self.value(graph_context))
        if self.use_input_modulation:
            gate = torch.sigmoid(
                self.modulation_gate(torch.cat([tokens, graph_context], dim=-1))
            )
            strength = torch.tanh(self.residual_logit)
            conditioned = tokens + strength * gate * self.dropout(graph_context)
        else:
            conditioned = tokens

        conditioned = conditioned.permute(0, 2, 3, 1)
        graph_context = graph_context.permute(0, 2, 3, 1)
        return conditioned, graph_context
