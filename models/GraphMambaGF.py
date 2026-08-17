"""GraphMamba with geometry-aware temporal/graph fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.GraphMamba import Model as GraphMamba


class GeometryAwareFusion(nn.Module):
    """Calibrate graph components parallel and orthogonal to temporal features."""

    def __init__(self, n_vars: int, long_patches: int):
        super().__init__()
        self.long_patches = long_patches
        # Axes: variable, patch scale, graph geometry, singleton feature axis.
        self.scale_logits = nn.Parameter(torch.zeros(1, n_vars, 2, 2, 1))

    def forward(self, temporal: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        projection = (graph * temporal).sum(dim=2, keepdim=True) / (
            temporal.square().sum(dim=2, keepdim=True) + 1e-8
        )
        parallel = projection * temporal
        orthogonal = graph - parallel
        scales = 2.0 * torch.sigmoid(self.scale_logits)
        long_parallel = parallel[..., : self.long_patches] * scales[:, :, 0, 0].unsqueeze(2)
        short_parallel = parallel[..., self.long_patches :] * scales[:, :, 1, 0].unsqueeze(2)
        long_orthogonal = orthogonal[..., : self.long_patches] * scales[:, :, 0, 1].unsqueeze(2)
        short_orthogonal = orthogonal[..., self.long_patches :] * scales[:, :, 1, 1].unsqueeze(2)
        calibrated_graph = torch.cat(
            (long_parallel + long_orthogonal, short_parallel + short_orthogonal),
            dim=-1,
        )
        return temporal + calibrated_graph


class Model(GraphMamba):
    """Paired GraphMamba whose only addition is a 4*N-variable fusion table."""

    def __init__(self, configs):
        super().__init__(configs)
        long_patches = (self.seq_len - self.patch_len) // self.stride + 2
        cpu_rng_state = torch.get_rng_state()
        self.geometry_fusion = GeometryAwareFusion(self.n_vars, long_patches)
        torch.set_rng_state(cpu_rng_state)

    def forecast(self, x_enc):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        normalized = centered / stdev

        seasonal, trend = self.decomposition(normalized)
        trend_output = self.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)
        long_tokens = self.long_patch_embedding(seasonal)
        short_tokens = self.short_patch_embedding(seasonal)
        tokens = torch.cat(
            (long_tokens + self.variable_embedding,
             short_tokens + self.variable_embedding),
            dim=-1,
        )
        temporal = self.encoder(tokens)
        graph = self.graph_mixer(tokens)
        output = self.head(self.geometry_fusion(temporal, graph)) + trend_output
        return output * stdev + means
