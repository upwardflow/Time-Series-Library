"""GraphMamba with bounded variable-wise seasonal/trend calibration."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.GraphMamba import Model as GraphMambaModel


class Model(GraphMambaModel):
    """Calibrate decomposition contributions without changing the backbone."""

    def __init__(self, configs):
        super().__init__(configs)
        if not self.use_decomp:
            raise ValueError("GraphMambaSTC requires use_decomp=1")
        # Zero initialization gives exact GraphMamba output at step zero.
        # 1 + tanh(theta) constrains each multiplier to (0, 2).
        self.seasonal_calibration = nn.Parameter(torch.zeros(self.n_vars))
        self.trend_calibration = nn.Parameter(torch.zeros(self.n_vars))

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
        graph_output = self.graph_mixer(tokens)
        seasonal_output = self.head(temporal_output + graph_output)

        seasonal_scale = (1.0 + torch.tanh(self.seasonal_calibration))[None, None, :]
        trend_scale = (1.0 + torch.tanh(self.trend_calibration))[None, None, :]
        output = seasonal_scale * seasonal_output + trend_scale * trend_output
        return output * stdev + means
