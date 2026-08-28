"""SimpleTM for long-term multivariate forecasting.

Adapted from the official ICLR 2025 implementation:
https://github.com/vsingh-group/SimpleTM

The implementation is kept self-contained so it does not replace the shared
Time-Series-Library attention or encoder layers used by other models.
"""

from math import sqrt

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted


class WaveletEmbedding(nn.Module):
    def __init__(self, channels, *, reconstruct=False, learnable=True,
                 wavelet="db1", levels=3, kernel_size=None):
        super().__init__()
        self.channels = channels
        self.reconstruct = reconstruct
        self.levels = levels

        if kernel_size is None:
            filters = pywt.Wavelet(wavelet)
            low = filters.rec_lo if reconstruct else filters.dec_lo
            high = filters.rec_hi if reconstruct else filters.dec_hi
            low = torch.tensor(low[::-1], dtype=torch.float32)
            high = torch.tensor(high[::-1], dtype=torch.float32)
            self.low = nn.Parameter(
                low[None, None, :].repeat(channels, 1, 1),
                requires_grad=learnable,
            )
            self.high = nn.Parameter(
                high[None, None, :].repeat(channels, 1, 1),
                requires_grad=learnable,
            )
            self.kernel_size = low.numel()
        else:
            self.kernel_size = int(kernel_size)
            self.low = nn.Parameter(
                torch.empty(channels, 1, self.kernel_size),
                requires_grad=learnable,
            )
            self.high = nn.Parameter(
                torch.empty(channels, 1, self.kernel_size),
                requires_grad=learnable,
            )
            nn.init.xavier_uniform_(self.low)
            nn.init.xavier_uniform_(self.high)
            with torch.no_grad():
                self.low.div_(torch.norm(self.low, dim=-1, keepdim=True))
                self.high.div_(torch.norm(self.high, dim=-1, keepdim=True))

    def forward(self, x):
        if self.reconstruct:
            return self._reconstruct(x)
        return self._decompose(x)

    def _decompose(self, x):
        approximation = x
        coefficients = []
        dilation = 1
        for _ in range(self.levels):
            padding = dilation * (self.kernel_size - 1)
            padding_right = (self.kernel_size * dilation) // 2
            padded = F.pad(
                approximation,
                (padding - padding_right, padding_right),
                mode="circular",
            )
            detail = F.conv1d(
                padded, self.high, dilation=dilation, groups=self.channels
            )
            approximation = F.conv1d(
                padded, self.low, dilation=dilation, groups=self.channels
            )
            coefficients.append(detail)
            dilation *= 2
        coefficients.append(approximation)
        return torch.stack(list(reversed(coefficients)), dim=-2)

    def _reconstruct(self, coefficients):
        dilation = 2 ** (self.levels - 1)
        approximation = coefficients[:, :, 0, :]
        details = coefficients[:, :, 1:, :]
        for level in range(self.levels):
            detail = details[:, :, level, :]
            padding = dilation * (self.kernel_size - 1)
            padding_left = (dilation * self.kernel_size) // 2
            pad = (padding_left, padding - padding_left)
            approximation = (
                F.conv1d(
                    F.pad(approximation, pad, mode="circular"),
                    self.low,
                    groups=self.channels,
                    dilation=dilation,
                )
                + F.conv1d(
                    F.pad(detail, pad, mode="circular"),
                    self.high,
                    groups=self.channels,
                    dilation=dilation,
                )
            ) / 2
            dilation //= 2
        return approximation


class GeometricAttention(nn.Module):
    def __init__(self, alpha, dropout):
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        embedding_dim = queries.shape[-1]
        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)
        query_norm = torch.sum(queries ** 2, dim=-1).permute(0, 2, 1).unsqueeze(-1)
        key_norm = torch.sum(keys ** 2, dim=-1).permute(0, 2, 1).unsqueeze(-2)
        wedge_norm = torch.sqrt(
            F.relu(query_norm * key_norm - dot_product ** 2) + 1e-8
        )
        scores = (
            (1.0 - self.alpha) * dot_product + self.alpha * wedge_norm
        ) / sqrt(embedding_dim)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", weights, values)
        return output.contiguous(), scores.abs().mean()


class GeometricAttentionLayer(nn.Module):
    def __init__(self, channels, d_model, *, alpha, attention_dropout,
                 projection_dropout, learnable_wavelet, wavelet, levels,
                 kernel_size):
        super().__init__()
        wavelet_args = dict(
            channels=channels,
            learnable=learnable_wavelet,
            wavelet=wavelet,
            levels=levels,
            kernel_size=kernel_size,
        )
        self.decompose = WaveletEmbedding(**wavelet_args)
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(projection_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(projection_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(projection_dropout)
        )
        self.attention = GeometricAttention(alpha, attention_dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            WaveletEmbedding(reconstruct=True, **wavelet_args),
        )

    def forward(self, x):
        queries = self.query_projection(self.decompose(x)).permute(0, 3, 2, 1)
        keys = self.key_projection(self.decompose(x)).permute(0, 3, 2, 1)
        values = self.value_projection(self.decompose(x)).permute(0, 3, 2, 1)
        output, regularizer = self.attention(queries, keys, values)
        output = self.output_projection(output.permute(0, 3, 2, 1))
        return output, regularizer


class SimpleTMEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        attended, regularizer = self.attention(x)
        x = self.norm1(x + self.dropout(attended))
        residual = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        residual = self.dropout(self.conv2(residual).transpose(-1, 1))
        return self.norm2(x + residual), regularizer


class Model(nn.Module):
    """SimpleTM model using the standard TSL forecasting interface."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError("SimpleTM currently supports forecasting only")

        self.pred_len = configs.pred_len
        self.use_norm = bool(configs.use_norm)
        self.embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        layer_args = dict(
            channels=configs.dec_in,
            d_model=configs.d_model,
            alpha=configs.simpletm_alpha,
            attention_dropout=configs.dropout,
            projection_dropout=configs.simpletm_geom_dropout,
            learnable_wavelet=bool(configs.simpletm_learnable_wavelet),
            wavelet=configs.simpletm_wavelet,
            levels=configs.simpletm_levels,
            kernel_size=configs.simpletm_kernel_size,
        )
        self.layers = nn.ModuleList([
            SimpleTMEncoderLayer(
                GeometricAttentionLayer(**layer_args),
                configs.d_model,
                configs.d_ff,
                configs.dropout,
                configs.activation,
            )
            for _ in range(configs.e_layers)
        ])
        self.norm = nn.LayerNorm(configs.d_model)
        self.projector = nn.Linear(configs.d_model, configs.pred_len)
        self.last_attention_regularizer = None

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = centered / stdev

        variable_count = x_enc.shape[-1]
        encoded = self.embedding(x_enc, None)
        regularizers = []
        for layer in self.layers:
            encoded, regularizer = layer(encoded)
            regularizers.append(regularizer)
        encoded = self.norm(encoded)
        self.last_attention_regularizer = torch.stack(regularizers).sum()
        prediction = self.projector(encoded).permute(0, 2, 1)[:, :, :variable_count]

        if self.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1)
            prediction = prediction + means[:, 0, :].unsqueeze(1)
        return prediction

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
