"""GraphMamba with temporal-innovation residual graph encoding (TIRGE)."""

from __future__ import annotations

import torch

from models.GraphMamba import Model as GraphMamba


class Model(GraphMamba):
    """Route only temporal innovation through the graph encoder.

    The original parallel GraphMamba computes ``T(x) + G(x)``. TIRGE computes
    ``T(x) + G(T(x) - x)`` so the graph branch models the representation update
    introduced by temporal Mamba instead of independently re-encoding all input
    tokens. No parameters or losses are added.
    """

    def forecast(self, x_enc):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        normalized = centered / stdev

        seasonal, trend = self.decomposition(normalized)
        trend_output = self.trend_projection(trend.permute(0, 2, 1))
        trend_output = trend_output.permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)

        long_tokens = self.long_patch_embedding(seasonal)
        short_tokens = self.short_patch_embedding(seasonal)
        tokens = torch.cat(
            (
                long_tokens + self.variable_embedding,
                short_tokens + self.variable_embedding,
            ),
            dim=-1,
        )
        temporal_output = self.encoder(tokens)
        temporal_innovation = temporal_output - tokens
        graph_residual = self.graph_mixer(temporal_innovation)
        output = self.head(temporal_output + graph_residual) + trend_output
        return output * stdev + means
