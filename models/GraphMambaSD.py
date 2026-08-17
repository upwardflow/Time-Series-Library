"""GraphMamba with scale-disagreement forecast calibration."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from layers.GraphMamba_EncDec import MambaEncoder, MambaEncoderLayer
from layers.GraphMamba_Layers import FlattenHead, ParallelGraphMixer, PatchEmbedding
from utils.graph_utils import generate_adjacency


class ScaleDisagreementHead(nn.Module):
    """Calibrate long/short scale contributions without duplicating predictors."""

    def __init__(self, baseline_head: FlattenHead, long_features: int, n_vars: int, pred_len: int):
        super().__init__()
        self.baseline_head = baseline_head
        self.long_features = long_features
        self.scale_logits = nn.Parameter(torch.zeros(1, pred_len, n_vars))

    def forward(self, x):
        tokens = x.permute(0, 1, 3, 2)
        flat = self.baseline_head.dropout(self.baseline_head.flatten(tokens))
        weight = self.baseline_head.linear.weight
        bias = self.baseline_head.linear.bias
        long_pred = torch.nn.functional.linear(
            flat[..., : self.long_features], weight[:, : self.long_features], bias
        )
        short_pred = torch.nn.functional.linear(
            flat[..., self.long_features :], weight[:, self.long_features :], None
        )
        baseline = long_pred + short_pred
        disagreement = (long_pred - short_pred).permute(0, 2, 1)
        correction = torch.tanh(self.scale_logits) * disagreement
        return baseline.permute(0, 2, 1) + correction


class Model(nn.Module):
    """Baseline-paired GraphMamba with a zero-init scale-disagreement head."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError("GraphMambaSD supports forecasting tasks only")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        patch_len, stride = configs.patch_len, configs.stride
        short_patch_len, short_stride = patch_len // 2, stride // 2
        if short_patch_len < 1 or short_stride < 1:
            raise ValueError("GraphMambaSD requires patch_len >= 2 and stride >= 2")

        self.decomposition = series_decomp(configs.moving_avg)
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)
        self.long_patch_embedding = PatchEmbedding(
            self.d_model, patch_len, stride, configs.dropout, self.n_vars
        )
        self.short_patch_embedding = PatchEmbedding(
            self.d_model, short_patch_len, short_stride, configs.dropout, self.n_vars
        )
        self.long_patches = (self.seq_len - patch_len) // stride + 2
        short_patches = (self.seq_len - short_patch_len) // short_stride + 2
        n_patches = self.long_patches + short_patches

        self.variable_embedding = nn.Parameter(torch.zeros(1, self.n_vars, self.d_model, 1))
        nn.init.normal_(self.variable_embedding, std=0.02)
        self.encoder = MambaEncoder(
            [MambaEncoderLayer(
                d_model=self.d_model, d_ff=configs.d_ff, d_state=configs.d_state,
                d_conv=configs.d_conv, expand=configs.expand, dropout=configs.dropout,
                activation=configs.activation, mamba_version=configs.mamba_version,
                mamba_headdim=configs.mamba_headdim, scan_mode="time",
                bidirectional=bool(configs.mamba_bidirectional),
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(self.d_model),
        )
        data_path = Path(configs.root_path) / configs.data_path
        static_adj = generate_adjacency(
            data_path=data_path, dataset_name=configs.data, features=configs.features,
            target=configs.target, sample_size=configs.graph_sample_size,
            sample_method=configs.graph_sample_method, random_seed=configs.seed,
            cache=bool(configs.graph_cache),
        ) if data_path.exists() else None
        self.graph_mixer = ParallelGraphMixer(configs, static_adj=static_adj)
        baseline_head = FlattenHead(
            input_features=self.d_model * n_patches,
            pred_len=self.pred_len,
            dropout=configs.dropout,
        )
        # Add the zero-init adapter after the complete baseline construction and
        # preserve the RNG stream used by shuffling/dropout.
        cpu_rng_state = torch.get_rng_state()
        self.head = ScaleDisagreementHead(
            baseline_head,
            long_features=self.d_model * self.long_patches,
            n_vars=self.n_vars,
            pred_len=self.pred_len,
        )
        torch.set_rng_state(cpu_rng_state)

    def forecast(self, x_enc):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev
        seasonal, trend = self.decomposition(normalized)
        trend_output = self.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)
        long_tokens = self.long_patch_embedding(seasonal)
        short_tokens = self.short_patch_embedding(seasonal)
        tokens = torch.cat([
            long_tokens + self.variable_embedding,
            short_tokens + self.variable_embedding,
        ], dim=-1)
        fused = self.encoder(tokens) + self.graph_mixer(tokens)
        output = self.head(fused) + trend_output
        return output * stdev + means

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
