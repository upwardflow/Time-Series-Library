"""No paired-decoder-difference control for frozen CMRHM-v1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.GraphMambaCMRHM import Model as GraphMambaCMRHM


class Model(GraphMambaCMRHM):
    """Inject the conditioned decoder output without its recent-only control."""

    def forecast(self, x_enc):
        recent = x_enc[:, -self.recent_len :, :]
        base_output = super(GraphMambaCMRHM, self).forecast(recent)
        means = recent.mean(dim=1, keepdim=True).detach()
        centered = recent - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        recent_normalized = (centered / stdev).permute(0, 2, 1)
        old_normalized = ((x_enc[:, : self.old_len, :] - means) / stdev).permute(0, 2, 1)
        memory = F.avg_pool1d(
            old_normalized, kernel_size=self.memory_pool, stride=self.memory_pool
        )
        recent_state = self.recent_context(recent_normalized)
        memory_state = self.memory_context(memory)
        memory_delta = self.memory_decoder(
            F.gelu(recent_state + memory_state)
        ).permute(0, 2, 1)
        scale = torch.tanh(self.memory_scale)[None, None, :]
        return base_output + scale * memory_delta * stdev

