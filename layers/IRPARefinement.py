"""Independent IRPA-compatible long-history refinement for controlled studies.

This implementation follows the four-path IRPA algorithm described by Tong and
Yuan (AAAI 2025) and its public reference implementation, but is written for the
current repository's model interface. It is used only for same-budget comparison
experiments; it does not claim to reproduce the paper's dataset-specific optimal
lookback lengths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp


class IRPARefinement(nn.Module):
    """Map a long input to a refined recent-length input and forecast auxiliary."""

    def __init__(self, configs):
        super().__init__()
        self.input_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.revise_len = int(getattr(configs, "irpa_revise_len", 96))
        self.topk = int(getattr(configs, "irpa_topk", 3))
        self.channels = int(configs.enc_in)
        self.stride = self.revise_len // 2
        self.decomposition = series_decomp(int(configs.moving_avg))

        if self.revise_len <= 1 or self.stride <= 0:
            raise ValueError("IRPA revise length must be greater than one")
        if self.input_len < self.revise_len:
            raise ValueError("IRPA input must be at least one revise-length patch")
        self.patch_count = 1 + (self.input_len - self.revise_len) // self.stride
        self.pred_patch_count = math.ceil(self.pred_len / self.revise_len)
        candidate_count = self.patch_count - self.pred_patch_count
        if self.patch_count - 1 < self.topk:
            raise ValueError("IRPA history has fewer searchable patches than top-k")
        if candidate_count < 1:
            raise ValueError(
                "IRPA prediction auxiliary has no history patch with enough "
                "subsequent observations; increase seq_len"
            )

        self.trend_weight = nn.Parameter(torch.ones(self.channels, 1))
        self.seasonal_weight = nn.Parameter(torch.ones(self.channels, 1))
        self.seasonal_projection = nn.Linear(self.revise_len, self.revise_len)
        self.trend_projection = nn.Linear(self.revise_len, self.revise_len)
        self.similarity_projection = nn.Linear(
            (self.topk + 1) * self.revise_len, self.pred_len
        )
        self.followup_projection = nn.Linear(self.pred_len, self.pred_len)
        self._initialize_average_projections()

    def _initialize_average_projections(self) -> None:
        with torch.no_grad():
            self.seasonal_projection.weight.fill_(1.0 / self.revise_len)
            self.trend_projection.weight.fill_(1.0 / self.revise_len)
            self.similarity_projection.weight.fill_(
                1.0 / ((self.topk + 1) * self.revise_len)
            )
            self.followup_projection.weight.fill_(1.0 / self.pred_len)

    def _patch(self, values: torch.Tensor) -> torch.Tensor:
        # [B, N, L] -> [B, N, P, revise_len]
        return values.unfold(-1, self.revise_len, self.stride)

    @staticmethod
    def _pearson_to_last(patches: torch.Tensor) -> torch.Tensor:
        centered = patches - patches.mean(dim=-1, keepdim=True)
        reference = centered[:, :, -1:, :]
        numerator = (centered * reference).sum(dim=-1)
        denominator = torch.linalg.vector_norm(centered, dim=-1) * torch.linalg.vector_norm(
            reference, dim=-1
        )
        return numerator / denominator.clamp_min(1e-6)

    def _periodic_weight(self, patch_count: int, device, dtype) -> torch.Tensor:
        positions = torch.arange(patch_count, device=device, dtype=dtype)
        phase = positions / max(1, patch_count - 1)
        phase = phase * int(patch_count / 5)
        return 1.0 - 0.1 * torch.sin(torch.pi * phase).square()

    @staticmethod
    def _gather_patches(patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        expanded = indices.unsqueeze(-1).expand(*indices.shape, patches.shape[-1])
        return torch.gather(patches, dim=2, index=expanded)

    def forward(self, x_enc: torch.Tensor):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        normalized = centered / stdev

        seasonal, trend = self.decomposition(normalized)
        seasonal_patches = self._patch(seasonal.permute(0, 2, 1))
        trend_patches = self._patch(trend.permute(0, 2, 1))
        raw_patches = self._patch(normalized.permute(0, 2, 1))

        similarity = self._pearson_to_last(seasonal_patches)
        similarity = similarity * self._periodic_weight(
            similarity.shape[-1], similarity.device, similarity.dtype
        )

        nearest = similarity[:, :, :-1].argmax(dim=2, keepdim=True)
        seasonal_match = self._gather_patches(seasonal_patches, nearest).squeeze(2)
        seasonal_candidate = seasonal_match + seasonal_patches[:, :, 0, :]
        seasonal_gate = torch.sigmoid(self.seasonal_weight)[None, :, :]
        seasonal_refined = (
            seasonal_patches[:, :, -1, :] * seasonal_gate
            + seasonal_candidate * (1.0 - seasonal_gate)
        )

        top_indices = torch.topk(
            similarity[:, :, :-1], self.topk, dim=2
        ).indices
        trend_matches = self._gather_patches(trend_patches, top_indices)
        trend_candidate = torch.sigmoid(trend_matches).mean(dim=2)
        trend_gate = torch.sigmoid(self.trend_weight)[None, :, :]
        trend_refined = (
            trend_candidate * (1.0 - trend_gate)
            + trend_patches[:, :, -1, :] * trend_gate
        )

        similar_raw = self._gather_patches(raw_patches, top_indices)
        similar_raw = torch.cat(
            [similar_raw, raw_patches[:, :, -1:, :]], dim=2
        ).flatten(start_dim=2)

        eligible = similarity[:, :, : -self.pred_patch_count]
        source = eligible.argmax(dim=2)
        offsets = torch.arange(
            1, self.pred_patch_count + 1, device=x_enc.device
        )[None, None, :]
        followup_indices = source.unsqueeze(-1) + offsets
        followup = self._gather_patches(seasonal_patches, followup_indices)
        followup = followup.flatten(start_dim=2)[:, :, : self.pred_len]

        refined = self.seasonal_projection(seasonal_refined)
        refined = refined + self.trend_projection(trend_refined)
        auxiliary = self.similarity_projection(similar_raw)
        auxiliary = auxiliary + self.followup_projection(followup)
        return refined, auxiliary, means, stdev
