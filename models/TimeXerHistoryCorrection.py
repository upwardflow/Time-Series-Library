"""TimeXer with TimeRole's compressed-history correction mechanism."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TimeXerRecent import Model as RecentTimeXer


class Model(RecentTimeXer):
    """Add only the paired conditional contribution of compressed old history."""

    def __init__(self, configs):
        if int(configs.seq_len) != 336:
            raise ValueError("TimeXerHistoryCorrection requires seq_len=336")
        super().__init__(configs)
        self.memory_pool = 16
        self.old_len = self.input_seq_len - self.recent_len
        if self.old_len % self.memory_pool:
            raise ValueError("Old-history length must be divisible by memory_pool")
        self.memory_tokens = self.old_len // self.memory_pool
        hidden_dim = int(
            getattr(configs, "timerole_hidden_dim", 32)
        )

        # Do not perturb the backbone/dropout/data-loader RNG trajectory relative
        # to TimeXerRecent. The added branch is initially exactly inactive.
        cpu_rng_state = torch.get_rng_state()
        self.recent_context = nn.Linear(self.recent_len, hidden_dim, bias=False)
        self.memory_context = nn.Linear(self.memory_tokens, hidden_dim, bias=False)
        self.memory_decoder = nn.Linear(hidden_dim, self.pred_len, bias=False)
        self.memory_scale = nn.Parameter(torch.zeros(self.n_vars))
        torch.set_rng_state(cpu_rng_state)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        recent = x_enc[:, -self.recent_len :, :]
        base_output = super().forward(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask
        )

        means = recent.mean(dim=1, keepdim=True).detach()
        centered = recent - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        recent_normalized = (centered / stdev).permute(0, 2, 1)
        old_normalized = (
            (x_enc[:, : self.old_len, :] - means) / stdev
        ).permute(0, 2, 1)
        memory = F.avg_pool1d(
            old_normalized, kernel_size=self.memory_pool, stride=self.memory_pool
        )

        recent_state = self.recent_context(recent_normalized)
        memory_state = self.memory_context(memory)
        without_memory = self.memory_decoder(F.gelu(recent_state))
        with_memory = self.memory_decoder(F.gelu(recent_state + memory_state))
        memory_delta = (with_memory - without_memory).permute(0, 2, 1)
        scale = torch.tanh(self.memory_scale)[None, None, :]
        return base_output + scale * memory_delta * stdev
