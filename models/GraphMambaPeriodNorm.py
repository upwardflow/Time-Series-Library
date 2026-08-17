"""Physical-time-normalized dual-path GraphMamba validation candidate."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.GraphMamba_EncDec import MambaEncoder, MambaEncoderLayer
from layers.GraphMamba_Layers import FlattenHead, ParallelGraphMixer, PatchEmbedding
from utils.graph_utils import generate_adjacency


class Model(nn.Module):
    """Combine native-resolution local tokens with hourly-normalized day tokens."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(
                "GraphMambaPeriodNorm supports forecasting tasks only"
            )
        if not all(
            bool(getattr(configs, name))
            for name in ("use_decomp", "use_patch", "use_time_mamba")
        ):
            raise ValueError(
                "GraphMambaPeriodNorm requires decomposition, patching, and temporal Mamba"
            )

        self.input_seq_len = int(configs.seq_len)
        self.recent_len = int(getattr(configs, "period_norm_recent_len", 96))
        self.resample_factor = int(getattr(configs, "period_norm_factor", 1))
        self.pred_len = int(configs.pred_len)
        self.n_vars = int(configs.enc_in)
        self.d_model = int(configs.d_model)
        self.use_graph = bool(configs.use_graph)
        self.period = int(getattr(configs, "periodic_period", 24))
        self.local_patch = int(getattr(configs, "periodic_local_patch", 4))
        self.local_stride = int(getattr(configs, "periodic_local_stride", 2))
        self.period_stride = int(
            getattr(configs, "periodic_period_stride", max(self.period // 2, 1))
        )

        if self.input_seq_len != 336:
            raise ValueError("GraphMambaPeriodNorm v2 currently requires seq_len=336")
        if not 1 <= self.recent_len <= self.input_seq_len:
            raise ValueError("period_norm_recent_len must lie within the input window")
        if self.resample_factor < 1 or self.input_seq_len % self.resample_factor:
            raise ValueError("period_norm_factor must divide seq_len exactly")
        self.hourly_len = self.input_seq_len // self.resample_factor
        if not (
            1 <= self.local_stride <= self.local_patch <= self.recent_len
            and 1 <= self.period_stride <= self.period <= self.hourly_len
        ):
            raise ValueError("invalid local or hourly-period patch geometry")

        self.local_decomposition = series_decomp(configs.moving_avg)
        self.period_decomposition = series_decomp(configs.moving_avg)
        self.trend_projection = nn.Linear(self.recent_len, self.pred_len)

        self.local_patch_embedding = PatchEmbedding(
            self.d_model,
            self.local_patch,
            self.local_stride,
            configs.dropout,
            self.n_vars,
        )
        self.period_patch_embedding = PatchEmbedding(
            self.d_model,
            self.period,
            self.period_stride,
            configs.dropout,
            self.n_vars,
        )
        self.local_patch_count = (
            self.recent_len + self.local_stride - self.local_patch
        ) // self.local_stride + 1
        self.period_patch_count = (
            self.hourly_len + self.period_stride - self.period
        ) // self.period_stride + 1
        self.total_patch_count = self.local_patch_count + self.period_patch_count

        self.variable_embedding = nn.Parameter(
            torch.zeros(1, self.n_vars, self.d_model, 1)
        )
        nn.init.normal_(self.variable_embedding, std=0.02)

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
                    f"[GraphMambaPeriodNorm] Dataset not found at {data_path}; "
                    "using an identity static graph"
                )
                static_adj = None
            self.graph_mixer = ParallelGraphMixer(configs, static_adj=static_adj)

        self.head = FlattenHead(
            input_features=self.d_model * self.total_patch_count,
            pred_len=self.pred_len,
            dropout=configs.dropout,
        )

        self.periodic_use_adapter = bool(
            getattr(configs, "periodic_use_adapter", 1)
        )
        descriptors = torch.tensor(
            [
                [self.local_patch / self.period, self.local_stride / self.period],
                [1.0, self.period_stride / self.period],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("periodic_scale_descriptors", descriptors)
        self.periodic_scale_conditioner = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 2 * self.d_model),
        )
        nn.init.zeros_(self.periodic_scale_conditioner[-1].weight)
        nn.init.zeros_(self.periodic_scale_conditioner[-1].bias)

    def _hourly_pool(self, normalized_full: torch.Tensor) -> torch.Tensor:
        values = normalized_full.permute(0, 2, 1)
        pooled = F.avg_pool1d(
            values,
            kernel_size=self.resample_factor,
            stride=self.resample_factor,
        )
        return pooled.permute(0, 2, 1)

    def _apply_scale_adapter(
        self, tokens: torch.Tensor, scale_index: int
    ) -> torch.Tensor:
        if not self.periodic_use_adapter:
            return tokens
        affine = self.periodic_scale_conditioner(
            self.periodic_scale_descriptors[scale_index]
        )
        gain, bias = affine.chunk(2, dim=-1)
        gain = torch.tanh(gain)[None, None, :, None]
        bias = bias[None, None, :, None]
        return tokens + tokens * gain + bias

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        if x_enc.size(1) != self.input_seq_len:
            raise ValueError(
                f"Expected input length {self.input_seq_len}, got {x_enc.size(1)}"
            )
        recent = x_enc[:, -self.recent_len :, :]
        means = recent.mean(dim=1, keepdim=True).detach()
        centered_recent = recent - means
        stdev = torch.sqrt(
            torch.var(centered_recent, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        normalized_recent = centered_recent / stdev
        normalized_full = (x_enc - means) / stdev

        local_seasonal, local_trend = self.local_decomposition(normalized_recent)
        hourly = self._hourly_pool(normalized_full)
        period_seasonal, _ = self.period_decomposition(hourly)
        trend_output = self.trend_projection(local_trend.permute(0, 2, 1))
        trend_output = trend_output.permute(0, 2, 1)

        local_tokens = self.local_patch_embedding(
            local_seasonal.permute(0, 2, 1)
        ) + self.variable_embedding
        period_tokens = self.period_patch_embedding(
            period_seasonal.permute(0, 2, 1)
        ) + self.variable_embedding
        local_tokens = self._apply_scale_adapter(local_tokens, 0)
        period_tokens = self._apply_scale_adapter(period_tokens, 1)

        local_temporal = self.encoder(local_tokens)
        period_temporal = self.encoder(period_tokens)
        if self.use_graph:
            all_tokens = torch.cat([local_tokens, period_tokens], dim=-1)
            graph = self.graph_mixer(all_tokens)
            local_graph, period_graph = torch.split(
                graph,
                [self.local_patch_count, self.period_patch_count],
                dim=-1,
            )
            local_state = local_temporal + local_graph
            period_state = period_temporal + period_graph
        else:
            local_state, period_state = local_temporal, period_temporal

        fused = torch.cat([local_state, period_state], dim=-1)
        output = self.head(fused) + trend_output
        return output * stdev + means

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
