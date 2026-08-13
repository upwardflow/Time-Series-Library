"""Patch embedding, graph mixing, and prediction head for GraphMamba."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    """Convert [B, N, L] series into [B, N, D, P] patch tokens."""

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        dropout: float,
        n_vars: int,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_vars = n_vars
        self.padding = nn.ReplicationPad1d((0, stride))
        self.projection = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.projection(x) + self.position_embedding(x)
        x = x.reshape(-1, self.n_vars, x.shape[-2], x.shape[-1])
        return self.dropout(x.permute(0, 1, 3, 2))


class ParallelGraphMixer(nn.Module):
    """Blend a train-data prior graph with a learnable variable graph."""

    def __init__(self, config, static_adj: np.ndarray | None = None):
        super().__init__()
        self.d_model = config.d_model
        self.n_vars = config.enc_in

        if static_adj is None:
            static_adj = np.eye(self.n_vars, dtype=np.float32)
        static_adj = np.asarray(static_adj, dtype=np.float32)
        if static_adj.shape != (self.n_vars, self.n_vars):
            raise ValueError(
                "Static adjacency shape must match enc_in: "
                f"expected {(self.n_vars, self.n_vars)}, got {static_adj.shape}"
            )

        top_k = int(getattr(config, "graph_top_k", 2))
        top_k = max(0, min(top_k, self.n_vars - 1))
        graph_mode = getattr(config, "static_graph_mode", "weighted")
        if graph_mode not in {"weighted", "binary"}:
            raise ValueError("static_graph_mode must be 'weighted' or 'binary'")

        without_diag = static_adj.copy()
        np.fill_diagonal(without_diag, -np.inf)
        sparse_adj = np.zeros_like(static_adj, dtype=np.float32)
        if top_k:
            indices = np.argsort(without_diag, axis=1)[:, -top_k:]
            for row in range(self.n_vars):
                if graph_mode == "binary":
                    sparse_adj[row, indices[row]] = 1.0
                else:
                    sparse_adj[row, indices[row]] = static_adj[row, indices[row]]
        np.fill_diagonal(sparse_adj, 1.0)
        sparse_adj /= np.clip(sparse_adj.sum(axis=1, keepdims=True), 1e-6, None)
        self.register_buffer("static_adj", torch.from_numpy(sparse_adj))

        node_dim = int(getattr(config, "node_dim", 10))
        self.node_embeddings = nn.Parameter(torch.randn(self.n_vars, node_dim))
        self.input_projection = nn.Linear(self.d_model, self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

        self.alpha = float(getattr(config, "graph_alpha", 0.3))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("graph_alpha must be between 0 and 1")
        self.static_only = bool(getattr(config, "static_graph_only", 0))

    def forward(self, x):
        batch_size, n_vars, d_model, n_patches = x.shape
        if n_vars != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {n_vars}")

        h = x.permute(0, 3, 1, 2).reshape(batch_size * n_patches, n_vars, d_model)
        h = self.input_projection(h)
        adaptive_adj = torch.softmax(
            self.node_embeddings @ self.node_embeddings.transpose(0, 1), dim=1
        )
        adjacency = (
            self.static_adj
            if self.static_only
            else self.alpha * self.static_adj + (1.0 - self.alpha) * adaptive_adj
        )
        h = torch.einsum("nm,bmd->bnd", adjacency, h)
        h = self.output_projection(self.dropout(self.activation(h)))
        return h.reshape(batch_size, n_patches, n_vars, d_model).permute(0, 2, 3, 1)


class FlattenHead(nn.Module):
    def __init__(self, input_features: int, pred_len: int, dropout: float):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_features, pred_len)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.linear(self.dropout(self.flatten(x)))
        return x.permute(0, 2, 1)
