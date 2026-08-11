"""GraphMamba for multivariate long-term time-series forecasting."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from layers.GraphMamba_EncDec import MambaEncoder, MambaEncoderLayer
from layers.GraphMamba_Layers import FlattenHead, ParallelGraphMixer, PatchEmbedding
from utils.graph_utils import generate_adjacency


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(
                "GraphMamba currently supports long_term_forecast and "
                "short_term_forecast only"
            )

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.short_patch_len = self.patch_len // 2
        self.short_stride = self.stride // 2
        if self.short_patch_len < 1 or self.short_stride < 1:
            raise ValueError("GraphMamba requires patch_len >= 2 and stride >= 2")
        if self.patch_len > self.seq_len or self.short_patch_len > self.seq_len:
            raise ValueError("GraphMamba patch lengths cannot exceed seq_len")

        self.use_decomp = bool(configs.use_decomp)
        self.use_patch = bool(configs.use_patch)
        self.use_time_mamba = bool(configs.use_time_mamba)
        self.use_graph = bool(configs.use_graph)
        if not self.use_time_mamba and not self.use_graph:
            raise ValueError("At least one of use_time_mamba or use_graph must be enabled")

        if self.use_decomp:
            self.decomposition = series_decomp(configs.moving_avg)
            self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

        if self.use_patch:
            self.long_patch_embedding = PatchEmbedding(
                self.d_model,
                self.patch_len,
                self.stride,
                configs.dropout,
                self.n_vars,
            )
            self.short_patch_embedding = PatchEmbedding(
                self.d_model,
                self.short_patch_len,
                self.short_stride,
                configs.dropout,
                self.n_vars,
            )
            long_patches = (self.seq_len - self.patch_len) // self.stride + 2
            short_patches = (
                (self.seq_len - self.short_patch_len) // self.short_stride + 2
            )
            n_patches = long_patches + short_patches
        else:
            self.pointwise_embedding = nn.Linear(1, self.d_model)
            n_patches = self.seq_len

        self.variable_embedding = nn.Parameter(
            torch.zeros(1, self.n_vars, self.d_model, 1)
        )
        nn.init.normal_(self.variable_embedding, std=0.02)

        if self.use_time_mamba:
            encoder_layers = [
                MambaEncoderLayer(
                    d_model=self.d_model,
                    d_ff=configs.d_ff,
                    d_state=configs.d_state,
                    d_conv=configs.d_conv,
                    expand=configs.expand,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    mamba_version=configs.mamba_version,
                    mamba_headdim=configs.mamba_headdim,
                    scan_mode="time",
                    bidirectional=bool(configs.mamba_bidirectional),
                )
                for _ in range(configs.e_layers)
            ]
            self.encoder = MambaEncoder(
                encoder_layers,
                norm_layer=nn.LayerNorm(self.d_model),
            )

        if self.use_graph:
            data_path = Path(configs.root_path) / configs.data_path
            if data_path.exists():
                static_adj = generate_adjacency(
                    data_path=data_path,
                    dataset_name=configs.data,
                    features=configs.features,
                    target=configs.target,
                    sample_size=configs.graph_sample_size,
                    sample_method=configs.graph_sample_method,
                    random_seed=configs.seed,
                    cache=bool(configs.graph_cache),
                )
            else:
                print(
                    f"[GraphMamba] Dataset not found at {data_path}; "
                    "using an identity static graph"
                )
                static_adj = None
            self.graph_mixer = ParallelGraphMixer(configs, static_adj=static_adj)

        self.head = FlattenHead(
            input_features=self.d_model * n_patches,
            pred_len=self.pred_len,
            dropout=configs.dropout,
        )

    def forecast(self, x_enc):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        normalized = centered / stdev

        if self.use_decomp:
            seasonal, trend = self.decomposition(normalized)
            trend_output = self.trend_projection(trend.permute(0, 2, 1))
            trend_output = trend_output.permute(0, 2, 1)
        else:
            seasonal = normalized
            trend_output = 0

        seasonal = seasonal.permute(0, 2, 1)
        if self.use_patch:
            long_tokens = self.long_patch_embedding(seasonal)
            short_tokens = self.short_patch_embedding(seasonal)
            tokens = torch.cat(
                [
                    long_tokens + self.variable_embedding,
                    short_tokens + self.variable_embedding,
                ],
                dim=-1,
            )
        else:
            tokens = self.pointwise_embedding(seasonal.unsqueeze(-1))
            tokens = tokens.permute(0, 1, 3, 2) + self.variable_embedding

        temporal_output = self.encoder(tokens) if self.use_time_mamba else 0
        graph_output = self.graph_mixer(tokens) if self.use_graph else 0
        output = self.head(temporal_output + graph_output) + trend_output
        return output * stdev + means

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
