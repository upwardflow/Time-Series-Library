"""GraphMamba for multivariate long-term time-series forecasting."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from layers.GraphMamba_EncDec import MambaEncoder, MambaEncoderLayer
from layers.GraphMamba_Layers import FlattenHead, ParallelGraphMixer, PatchEmbedding
from utils.graph_utils import generate_adjacency


class GraphResidualMambaFusion(nn.Module):
    """Use the graph state as a base and gate the temporal Mamba increment."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        initial_mamba_weight: float,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("agf_hidden_dim must be positive")
        if not 0.0 < initial_mamba_weight < 1.0:
            raise ValueError("agf_initial_mamba_weight must be in (0, 1)")

        self.temporal_norm = nn.LayerNorm(d_model)
        self.graph_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        initial_logit = torch.logit(torch.tensor(initial_mamba_weight)).item()
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, initial_logit)
        self.last_gate = None

    def forward(self, temporal: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        # Pool only for reliability estimation. The fused states retain the full
        # patch resolution used by the forecasting head.
        temporal_summary = self.temporal_norm(temporal.mean(dim=-1))
        graph_summary = self.graph_norm(graph.mean(dim=-1))
        descriptor = torch.cat(
            [
                temporal_summary,
                graph_summary,
                torch.abs(temporal_summary - graph_summary),
            ],
            dim=-1,
        )
        mamba_weight = torch.sigmoid(self.gate(descriptor)).unsqueeze(-1)
        self.last_gate = mamba_weight.detach() if not self.training else None
        return graph + mamba_weight * temporal


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
        self.graph_mamba_fusion = getattr(
            configs, "graph_mamba_fusion", "fixed_sum"
        )
        if self.graph_mamba_fusion not in {
            "fixed_sum",
            "graph_residual_gate",
        }:
            raise ValueError(
                "graph_mamba_fusion must be 'fixed_sum' or "
                "'graph_residual_gate'"
            )
        requested_scan_mode = getattr(configs, "dual_scale_scan_mode", "auto")
        if requested_scan_mode == "auto":
            configured_period = int(getattr(configs, "periodic_period", 24))
            hourly_ett = getattr(configs, "data", "") in {"ETTh1", "ETTh2"}
            requested_scan_mode = (
                "periodic_aligned"
                if hourly_ett and configured_period < self.seq_len
                else "independent_shared"
            )
        self.dual_scale_scan_mode = requested_scan_mode
        if self.dual_scale_scan_mode not in {
            "joint",
            "independent_shared",
            "independent_unshared",
            "periodic_aligned",
        }:
            raise ValueError(
                "dual_scale_scan_mode must be 'joint', 'independent_shared', "
                "'independent_unshared', or 'periodic_aligned' "
                "(configuration may also request 'auto')"
            )
        self.dual_scale_selection = getattr(
            configs, "dual_scale_selection", "dual"
        )
        if self.dual_scale_selection not in {"dual", "coarse", "fine"}:
            raise ValueError(
                "dual_scale_selection must be 'dual', 'coarse', or 'fine'"
            )
        self.use_periodic_multiscale = self.dual_scale_scan_mode == "periodic_aligned"
        if self.use_periodic_multiscale and self.dual_scale_selection != "dual":
            raise ValueError(
                "periodic_aligned currently requires dual_scale_selection='dual'"
            )
        if self.use_periodic_multiscale and not (
            self.use_patch and self.use_time_mamba
        ):
            raise ValueError(
                "periodic_aligned requires use_patch and use_time_mamba"
            )
        if not self.use_time_mamba and not self.use_graph:
            raise ValueError("At least one of use_time_mamba or use_graph must be enabled")
        if self.graph_mamba_fusion == "graph_residual_gate" and not (
            self.use_time_mamba and self.use_graph
        ):
            raise ValueError(
                "graph_residual_gate requires both Mamba and graph branches"
            )
        if self.graph_mamba_fusion == "graph_residual_gate" and self.use_periodic_multiscale:
            raise ValueError(
                "graph_residual_gate currently requires a non-periodic scan mode"
            )
        if self.use_decomp:
            self.decomposition = series_decomp(configs.moving_avg)
            self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

        if self.use_patch:
            if self.use_periodic_multiscale:
                self.periodic_period = int(getattr(configs, "periodic_period", 24))
                self.periodic_local_patch = int(
                    getattr(configs, "periodic_local_patch", self.patch_len)
                )
                self.periodic_local_stride = int(
                    getattr(configs, "periodic_local_stride", self.stride)
                )
                self.periodic_period_stride = int(
                    getattr(
                        configs,
                        "periodic_period_stride",
                        max(self.periodic_period // 2, 1),
                    )
                )
                geometry = (
                    self.periodic_period,
                    self.periodic_local_patch,
                    self.periodic_local_stride,
                    self.periodic_period_stride,
                )
                if any(value < 1 for value in geometry):
                    raise ValueError("Periodic patch geometry must be positive")
                if self.periodic_period > self.seq_len:
                    raise ValueError(
                        "periodic_period cannot exceed seq_len"
                    )
                if self.periodic_local_patch > self.seq_len:
                    raise ValueError("periodic_local_patch cannot exceed seq_len")
                if (
                    self.periodic_local_patch < 2
                    or self.periodic_local_patch > self.periodic_period // 2
                    or self.periodic_period % self.periodic_local_patch != 0
                ):
                    raise ValueError(
                        "periodic_local_patch must be a proper divisor of "
                        "periodic_period between 2 and half the period"
                    )
                self.local_patch_embedding = PatchEmbedding(
                    self.d_model,
                    self.periodic_local_patch,
                    self.periodic_local_stride,
                    configs.dropout,
                    self.n_vars,
                )
                self.period_patch_embedding = PatchEmbedding(
                    self.d_model,
                    self.periodic_period,
                    self.periodic_period_stride,
                    configs.dropout,
                    self.n_vars,
                )
                self.local_patch_count = (
                    self.seq_len
                    + self.periodic_local_stride
                    - self.periodic_local_patch
                ) // self.periodic_local_stride + 1
                self.period_patch_count = (
                    self.seq_len
                    + self.periodic_period_stride
                    - self.periodic_period
                ) // self.periodic_period_stride + 1
                n_patches = self.local_patch_count + self.period_patch_count
            else:
                if self.dual_scale_selection in {"dual", "coarse"}:
                    self.long_patch_embedding = PatchEmbedding(
                        self.d_model,
                        self.patch_len,
                        self.stride,
                        configs.dropout,
                        self.n_vars,
                    )
                if self.dual_scale_selection in {"dual", "fine"}:
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
                patch_counts = {
                    "dual": long_patches + short_patches,
                    "coarse": long_patches,
                    "fine": short_patches,
                }
                n_patches = patch_counts[self.dual_scale_selection]
        else:
            self.pointwise_embedding = nn.Linear(1, self.d_model)
            n_patches = self.seq_len

        self.variable_embedding = nn.Parameter(
            torch.zeros(1, self.n_vars, self.d_model, 1)
        )
        nn.init.normal_(self.variable_embedding, std=0.02)

        if self.use_time_mamba:
            def build_encoder():
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
                return MambaEncoder(
                    encoder_layers,
                    norm_layer=nn.LayerNorm(self.d_model),
                )

            self.encoder = build_encoder()
            if (
                self.dual_scale_scan_mode == "independent_unshared"
                and self.dual_scale_selection == "dual"
            ):
                # Preserve the RNG trajectory of all downstream modules so the
                # only paired structural difference is the extra fine encoder.
                cpu_rng_state = torch.get_rng_state()
                self.fine_encoder = build_encoder()
                torch.set_rng_state(cpu_rng_state)

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

        # The scale adapter is the only periodic extension retained after the
        # validation gate. It is shared across scales and conditioned by fixed,
        # physically interpretable patch-length and stride descriptors.
        if self.use_periodic_multiscale:
            self.periodic_use_adapter = bool(
                getattr(configs, "periodic_use_adapter", 1)
            )
            descriptors = torch.tensor(
                [
                    [
                        self.periodic_local_patch / self.periodic_period,
                        self.periodic_local_stride / self.periodic_period,
                    ],
                    [1.0, self.periodic_period_stride / self.periodic_period],
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

        self.branch_fusion = None
        self.last_fusion_gate = None
        if self.graph_mamba_fusion == "graph_residual_gate":
            # The candidate adapter must not change seeded initialization or the
            # subsequent data-loader/dropout RNG trajectory of the base model.
            cpu_rng_state = torch.get_rng_state()
            self.branch_fusion = GraphResidualMambaFusion(
                d_model=self.d_model,
                hidden_dim=int(getattr(configs, "agf_hidden_dim", 32)),
                initial_mamba_weight=float(
                    getattr(configs, "agf_initial_mamba_weight", 0.1)
                ),
            )
            torch.set_rng_state(cpu_rng_state)

    def _apply_periodic_scale_adapter(
        self,
        tokens: torch.Tensor,
        scale_index: int,
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

    def _periodic_multiscale_states(self, seasonal: torch.Tensor):
        local_tokens = self.local_patch_embedding(seasonal) + self.variable_embedding
        period_tokens = self.period_patch_embedding(seasonal) + self.variable_embedding
        local_tokens = self._apply_periodic_scale_adapter(local_tokens, 0)
        period_tokens = self._apply_periodic_scale_adapter(period_tokens, 1)

        local_temporal = self.encoder(local_tokens)
        period_temporal = self.encoder(period_tokens)

        all_tokens = torch.cat([local_tokens, period_tokens], dim=-1)
        if self.use_graph:
            graph = self.graph_mixer(all_tokens)
            local_graph, period_graph = torch.split(
                graph, [self.local_patch_count, self.period_patch_count], dim=-1
            )
            local_state = local_temporal + local_graph
            period_state = period_temporal + period_graph
        else:
            local_state, period_state = local_temporal, period_temporal
        return torch.cat([local_state, period_state], dim=-1)

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
            if self.use_periodic_multiscale:
                fused_output = self._periodic_multiscale_states(seasonal)
            else:
                long_tokens = None
                short_tokens = None
                if self.dual_scale_selection in {"dual", "coarse"}:
                    long_tokens = (
                        self.long_patch_embedding(seasonal)
                        + self.variable_embedding
                    )
                if self.dual_scale_selection in {"dual", "fine"}:
                    short_tokens = (
                        self.short_patch_embedding(seasonal)
                        + self.variable_embedding
                    )
                if self.dual_scale_selection == "dual":
                    assert long_tokens is not None and short_tokens is not None
                    tokens = torch.cat([long_tokens, short_tokens], dim=-1)
                elif self.dual_scale_selection == "coarse":
                    assert long_tokens is not None
                    tokens = long_tokens
                else:
                    assert short_tokens is not None
                    tokens = short_tokens
        else:
            tokens = self.pointwise_embedding(seasonal.unsqueeze(-1))
            tokens = tokens.permute(0, 1, 3, 2) + self.variable_embedding

        if not self.use_periodic_multiscale:
            if self.use_time_mamba:
                if self.use_patch:
                    if self.dual_scale_selection == "coarse":
                        assert long_tokens is not None
                        temporal_output = self.encoder(long_tokens)
                    elif self.dual_scale_selection == "fine":
                        assert short_tokens is not None
                        fine_encoder = getattr(self, "fine_encoder", self.encoder)
                        temporal_output = fine_encoder(short_tokens)
                    elif self.dual_scale_scan_mode == "joint":
                        temporal_output = self.encoder(tokens)
                    elif self.dual_scale_scan_mode == "independent_unshared":
                        assert long_tokens is not None and short_tokens is not None
                        temporal_output = torch.cat(
                            [
                                self.encoder(long_tokens),
                                self.fine_encoder(short_tokens),
                            ],
                            dim=-1,
                        )
                    else:
                        assert long_tokens is not None and short_tokens is not None
                        # Long and short patches are two sampling grids over the
                        # same history, not consecutive pieces of one sequence.
                        # Reuse the encoder parameters but reset state per scale.
                        temporal_output = torch.cat(
                            [self.encoder(long_tokens), self.encoder(short_tokens)],
                            dim=-1,
                        )
                else:
                    temporal_output = self.encoder(tokens)
            else:
                temporal_output = 0
            graph_output = self.graph_mixer(tokens) if self.use_graph else 0
            if self.branch_fusion is not None:
                fused_output = self.branch_fusion(temporal_output, graph_output)
                self.last_fusion_gate = self.branch_fusion.last_gate
            else:
                fused_output = temporal_output + graph_output
        output = self.head(fused_output) + trend_output
        return output * stdev + means

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
