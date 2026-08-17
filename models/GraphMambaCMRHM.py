"""GraphMamba with Conditioned Multi-Resolution History Memory (CMRHM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GraphMambaRecent import Model as RecentGraphMamba


class Model(RecentGraphMamba):
    """Use paired with/without-memory decoding to isolate old-history impact."""

    def __init__(self, configs):
        if int(configs.seq_len) != 336:
            raise ValueError("GraphMambaCMRHM currently requires seq_len=336")
        super().__init__(configs)
        self.memory_pool = 16
        self.old_len = self.input_seq_len - self.recent_len
        if self.old_len % self.memory_pool:
            raise ValueError("Old-history length must be divisible by memory_pool")
        self.memory_tokens = self.old_len // self.memory_pool
        hidden_dim = int(getattr(configs, "cmrhm_hidden_dim", 32))

        # Preserve the RNG state so baseline initialization, data shuffling, and
        # backbone dropout follow the strict control's stochastic trajectory.
        cpu_rng_state = torch.get_rng_state()
        self.recent_context = nn.Linear(self.recent_len, hidden_dim, bias=False)
        self.memory_context = nn.Linear(self.memory_tokens, hidden_dim, bias=False)
        self.memory_decoder = nn.Linear(hidden_dim, self.pred_len, bias=False)
        self.memory_scale = nn.Parameter(torch.zeros(self.n_vars))
        self.old_history_intervention = getattr(
            configs, "cmrhm_old_intervention", "intact"
        )
        self.memory_noise_std = float(getattr(configs, "cmrhm_noise_std", 1.0))
        if self.old_history_intervention not in {
            "intact",
            "batch_shuffle",
            "temporal_shuffle",
            "reverse",
            "recent_mean",
            "noise",
        }:
            raise ValueError(
                f"Unsupported CMRHM old-history intervention: "
                f"{self.old_history_intervention}"
            )
        self.last_memory_correction = None
        torch.set_rng_state(cpu_rng_state)

    def _intervene_old_history(
        self,
        old: torch.Tensor,
        means: torch.Tensor,
        stdev: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.old_history_intervention
        if mode == "intact":
            return old
        if mode == "batch_shuffle":
            return torch.roll(old, shifts=1, dims=0) if old.shape[0] > 1 else old
        if mode == "temporal_shuffle":
            return old[:, torch.randperm(old.shape[1], device=old.device), :]
        if mode == "reverse":
            return torch.flip(old, dims=(1,))
        if mode == "recent_mean":
            return means.expand_as(old)
        if mode == "noise":
            return means + self.memory_noise_std * stdev * torch.randn_like(old)
        raise AssertionError(f"Unhandled intervention: {mode}")

    def forecast(self, x_enc):
        recent = x_enc[:, -self.recent_len :, :]
        base_output = super().forecast(recent)

        means = recent.mean(dim=1, keepdim=True).detach()
        centered = recent - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        recent_normalized = (centered / stdev).permute(0, 2, 1)
        old = self._intervene_old_history(
            x_enc[:, : self.old_len, :], means, stdev
        )
        old_normalized = ((old - means) / stdev).permute(0, 2, 1)
        memory = F.avg_pool1d(
            old_normalized, kernel_size=self.memory_pool, stride=self.memory_pool
        )

        recent_state = self.recent_context(recent_normalized)
        memory_state = self.memory_context(memory)
        without_memory = self.memory_decoder(F.gelu(recent_state))
        with_memory = self.memory_decoder(F.gelu(recent_state + memory_state))
        memory_delta = (with_memory - without_memory).permute(0, 2, 1)
        scale = torch.tanh(self.memory_scale)[None, None, :]
        correction = scale * memory_delta * stdev
        if not self.training:
            self.last_memory_correction = correction.detach()
        return base_output + correction
