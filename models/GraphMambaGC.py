"""Sample-Adaptive Graph-Conditioned Bidirectional Mamba forecasting model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from layers.GraphMamba_Layers import FlattenHead, ParallelGraphMixer, PatchEmbedding
from layers.GraphMambaGC_EncDec import (
    GraphConditionedMambaEncoder,
    GraphConditionedMambaEncoderLayer,
)
from layers.GraphMambaGC_Layers import SampleAdaptiveGraphConditioner
from utils.graph_utils import generate_adjacency


class Model(nn.Module):
    """GraphMamba whose variable graph conditions temporal state-space scans."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(
                "GraphMambaGC supports long_term_forecast and "
                "short_term_forecast only"
            )
        if not bool(configs.use_patch) or not bool(configs.use_decomp):
            raise ValueError("GraphMambaGC v1 requires patching and decomposition")
        if not bool(configs.use_time_mamba) or not bool(configs.use_graph):
            raise ValueError("GraphMambaGC v1 requires both Mamba and graph conditioning")

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        short_patch_len = self.patch_len // 2
        short_stride = self.stride // 2
        if short_patch_len < 1 or short_stride < 1:
            raise ValueError("GraphMambaGC requires patch_len >= 2 and stride >= 2")
        if self.patch_len > self.seq_len or short_patch_len > self.seq_len:
            raise ValueError("GraphMambaGC patch lengths cannot exceed seq_len")

        self.decomposition = series_decomp(configs.moving_avg)
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)
        self.long_patch_embedding = PatchEmbedding(
            self.d_model,
            self.patch_len,
            self.stride,
            configs.dropout,
            self.n_vars,
        )
        self.short_patch_embedding = PatchEmbedding(
            self.d_model,
            short_patch_len,
            short_stride,
            configs.dropout,
            self.n_vars,
        )
        long_patches = (self.seq_len - self.patch_len) // self.stride + 2
        short_patches = (self.seq_len - short_patch_len) // short_stride + 2
        n_patches = long_patches + short_patches

        self.variable_embedding = nn.Parameter(
            torch.zeros(1, self.n_vars, self.d_model, 1)
        )
        nn.init.normal_(self.variable_embedding, std=0.02)

        # Construct the entire baseline temporal core before any new module.
        # With direction fusion disabled this preserves GraphMamba's seeded
        # initialization order exactly; the graph conditioner is an adapter.
        layers = [
            GraphConditionedMambaEncoderLayer(
                d_model=self.d_model,
                d_ff=configs.d_ff,
                d_state=configs.d_state,
                d_conv=configs.d_conv,
                expand=configs.expand,
                dropout=configs.dropout,
                activation=configs.activation,
                mamba_version=configs.mamba_version,
                mamba_headdim=configs.mamba_headdim,
                bidirectional=bool(configs.mamba_bidirectional),
                graph_direction_fusion=bool(configs.gc_direction_fusion),
            )
            for _ in range(configs.e_layers)
        ]
        self.encoder = GraphConditionedMambaEncoder(
            layers,
            norm_layer=nn.LayerNorm(self.d_model),
        )

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
                f"[GraphMambaGC] Dataset not found at {data_path}; "
                "using an identity static graph"
            )
            static_adj = None
        self.use_parallel_residual = bool(
            getattr(configs, "gc_parallel_residual", 1)
        )
        if self.use_parallel_residual:
            self.graph_residual = ParallelGraphMixer(configs, static_adj=static_adj)
        self.head = FlattenHead(
            input_features=self.d_model * n_patches,
            pred_len=self.pred_len,
            dropout=configs.dropout,
        )
        # New parameters are intentionally initialized only after every
        # baseline parameter to make paired-seed comparisons meaningful.
        self.graph_conditioner = SampleAdaptiveGraphConditioner(
            configs, static_adj=static_adj
        )

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
            [
                long_tokens + self.variable_embedding,
                short_tokens + self.variable_embedding,
            ],
            dim=-1,
        )

        conditioned_tokens, graph_context = self.graph_conditioner(tokens)
        encoded = self.encoder(conditioned_tokens, graph_context)
        if self.use_parallel_residual:
            encoded = encoded + self.graph_residual(tokens)
        output = self.head(encoded) + trend_output
        return output * stdev + means

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
