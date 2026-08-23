"""S-Mamba baseline for multivariate long-term forecasting.

This implementation follows the official S-Mamba design: variables are treated
as tokens, a bidirectional Mamba encoder learns inter-variable dependencies, and
a feed-forward sublayer models each token before direct horizon projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from layers.Embed import DataEmbedding_inverted


class BidirectionalMambaLayer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=2,
            expand=1,
        )
        self.backward_mamba = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=2,
            expand=1,
        )
        self.ffn_in = nn.Conv1d(configs.d_model, configs.d_ff, kernel_size=1)
        self.ffn_out = nn.Conv1d(configs.d_ff, configs.d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu

    def forward(self, x):
        forward = self.forward_mamba(x)
        backward = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        x = self.norm1(x + forward + backward)
        y = self.ffn_in(x.transpose(1, 2))
        y = self.dropout(self.activation(y))
        y = self.dropout(self.ffn_out(y).transpose(1, 2))
        return self.norm2(x + y)


class Model(nn.Module):
    """Official-design S-Mamba adapted to this repository's model interface."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.use_norm = bool(configs.use_norm)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.encoder = nn.ModuleList(
            [BidirectionalMambaLayer(configs) for _ in range(configs.e_layers)]
        )
        self.encoder_norm = nn.LayerNorm(configs.d_model)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_enc = centered / stdev

        n_vars = x_enc.shape[-1]
        encoded = self.enc_embedding(x_enc, x_mark_enc)
        for layer in self.encoder:
            encoded = layer(encoded)
        encoded = self.encoder_norm(encoded)
        output = self.projector(encoded).permute(0, 2, 1)[:, :, :n_vars]

        if self.use_norm:
            output = output * stdev[:, 0, :].unsqueeze(1)
            output = output + means[:, 0, :].unsqueeze(1)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise NotImplementedError(
                f"SMamba does not implement task {self.task_name!r} in this baseline adapter"
            )
        return self.forecast(x_enc, x_mark_enc)[:, -self.pred_len :, :]
