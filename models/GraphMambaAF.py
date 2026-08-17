"""GraphMamba with dual-domain adaptive residual calibration."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from layers.GraphMamba_EncDec import MambaEncoder, MambaEncoderLayer
from layers.GraphMamba_Layers import FlattenHead, ParallelGraphMixer, PatchEmbedding
from utils.graph_utils import generate_adjacency


class ReliabilityAdaptiveFusion(nn.Module):
    """Estimate local graph-residual reliability without disturbing baseline init."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("af_hidden_dim must be positive")
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # sigmoid(0) * 2 = 1, exactly reproducing temporal + graph.
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, temporal, graph):
        temporal_tokens = temporal.permute(0, 1, 3, 2)
        graph_tokens = graph.permute(0, 1, 3, 2)
        temporal_norm = self.norm_t(temporal_tokens)
        graph_norm = self.norm_g(graph_tokens)
        reliability = 2.0 * torch.sigmoid(
            self.projection(
                torch.cat(
                    [temporal_norm, graph_norm, torch.abs(temporal_norm - graph_norm)],
                    dim=-1,
                )
            )
        )
        fused = temporal_tokens + reliability * graph_tokens
        return fused.permute(0, 1, 3, 2)


class VariableScaleFusion(nn.Module):
    """Learn graph reliability for each variable and temporal patch scale."""

    def __init__(self, n_vars: int, long_patches: int):
        super().__init__()
        self.long_patches = long_patches
        self.reliability_logits = nn.Parameter(torch.zeros(1, n_vars, 2, 1))

    def forward(self, temporal, graph):
        reliability = 2.0 * torch.sigmoid(self.reliability_logits)
        long_scale = reliability[:, :, 0:1, :]
        short_scale = reliability[:, :, 1:2, :]
        long_graph = graph[..., : self.long_patches] * long_scale
        short_graph = graph[..., self.long_patches :] * short_scale
        calibrated_graph = torch.cat([long_graph, short_graph], dim=-1)
        return temporal + calibrated_graph


class FixedAdditiveFusion(nn.Module):
    """Baseline temporal-plus-graph fusion used for residual-only ablation."""

    def forward(self, temporal, graph):
        return temporal + graph


class DirectResidualCorrection(nn.Module):
    """Learn only the forecast error unexplained by the main model path."""

    def __init__(self, seq_len: int, pred_len: int, n_vars: int):
        super().__init__()
        self.temporal_map = nn.Linear(seq_len, pred_len)
        self.variable_horizon_scale = nn.Parameter(
            torch.ones(1, pred_len, n_vars)
        )
        nn.init.zeros_(self.temporal_map.weight)
        nn.init.zeros_(self.temporal_map.bias)

    def forward(self, normalized):
        correction = self.temporal_map(normalized.permute(0, 2, 1))
        return correction.permute(0, 2, 1) * self.variable_horizon_scale


class LowRankResidualCorrection(nn.Module):
    """Factorized history-to-forecast correction with a compact latent basis."""

    def __init__(self, seq_len: int, pred_len: int, n_vars: int, rank: int):
        super().__init__()
        if rank < 1 or rank > min(seq_len, pred_len):
            raise ValueError("af_rank must be in [1, min(seq_len, pred_len)]")
        self.history_to_basis = nn.Linear(seq_len, rank, bias=False)
        self.basis_to_horizon = nn.Linear(rank, pred_len)
        self.variable_scale = nn.Parameter(torch.ones(1, 1, n_vars))
        # Only the second factor is zero initialized: the complete residual is
        # exactly zero while the first factor can receive gradients immediately.
        nn.init.zeros_(self.basis_to_horizon.weight)
        nn.init.zeros_(self.basis_to_horizon.bias)

    def forward(self, normalized):
        hidden = self.history_to_basis(normalized.permute(0, 2, 1))
        correction = self.basis_to_horizon(hidden).permute(0, 2, 1)
        return correction * self.variable_scale


class Model(nn.Module):
    """Paired GraphMamba with representation- and forecast-domain calibration."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(
                "GraphMambaAF supports long_term_forecast and short_term_forecast only"
            )

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        short_patch_len = self.patch_len // 2
        short_stride = self.stride // 2
        if short_patch_len < 1 or short_stride < 1:
            raise ValueError("GraphMambaAF requires patch_len >= 2 and stride >= 2")

        self.decomposition = series_decomp(configs.moving_avg)
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)
        self.long_patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, configs.dropout, self.n_vars
        )
        self.short_patch_embedding = PatchEmbedding(
            self.d_model, short_patch_len, short_stride, configs.dropout, self.n_vars
        )
        long_patches = (self.seq_len - self.patch_len) // self.stride + 2
        short_patches = (self.seq_len - short_patch_len) // short_stride + 2
        n_patches = long_patches + short_patches

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
            encoder_layers, norm_layer=nn.LayerNorm(self.d_model)
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
                f"[GraphMambaAF] Dataset not found at {data_path}; "
                "using an identity static graph"
            )
            static_adj = None
        self.graph_mixer = ParallelGraphMixer(configs, static_adj=static_adj)
        self.head = FlattenHead(
            input_features=self.d_model * n_patches,
            pred_len=self.pred_len,
            dropout=configs.dropout,
        )
        # Adapter is initialized after every baseline module, preserving the
        # seeded parameters. Restore the RNG afterwards as well, so DataLoader
        # shuffling and baseline dropout use the exact same random stream.
        cpu_rng_state = torch.get_rng_state()
        fusion_mode = getattr(configs, "af_mode", "variable_scale")
        self.residual_correction = None
        if fusion_mode == "local":
            self.adaptive_fusion = ReliabilityAdaptiveFusion(
                self.d_model, int(getattr(configs, "af_hidden_dim", 32))
            )
        elif fusion_mode in {
            "variable_scale", "variable_scale_residual", "variable_scale_lowrank"
        }:
            self.adaptive_fusion = VariableScaleFusion(
                self.n_vars, long_patches=long_patches
            )
            if fusion_mode == "variable_scale_residual":
                self.residual_correction = DirectResidualCorrection(
                    self.seq_len, self.pred_len, self.n_vars
                )
            elif fusion_mode == "variable_scale_lowrank":
                self.residual_correction = LowRankResidualCorrection(
                    self.seq_len,
                    self.pred_len,
                    self.n_vars,
                    int(getattr(configs, "af_rank", 16)),
                )
        elif fusion_mode == "residual_only":
            self.adaptive_fusion = FixedAdditiveFusion()
            self.residual_correction = DirectResidualCorrection(
                self.seq_len, self.pred_len, self.n_vars
            )
        else:
            raise ValueError(
                "af_mode must be 'local', 'variable_scale', "
                "'variable_scale_residual', 'variable_scale_lowrank', or "
                "'residual_only'"
            )
        torch.set_rng_state(cpu_rng_state)

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
        temporal_output = self.encoder(tokens)
        graph_output = self.graph_mixer(tokens)
        fused = self.adaptive_fusion(temporal_output, graph_output)
        output = self.head(fused) + trend_output
        if self.residual_correction is not None:
            output = output + self.residual_correction(normalized)
        return output * stdev + means

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
