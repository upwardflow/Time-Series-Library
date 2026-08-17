"""TimeXer control that reads 336 points but uses only the recent 96."""

from __future__ import annotations

import copy

from models.TimeXer import Model as TimeXerModel


class Model(TimeXerModel):
    """Align samples with long-memory variants without changing TimeXer."""

    recent_len = 96

    def __init__(self, configs):
        self.input_seq_len = int(configs.seq_len)
        if self.input_seq_len < self.recent_len:
            raise ValueError("TimeXerRecent requires seq_len >= 96")
        recent_configs = copy.copy(configs)
        recent_configs.seq_len = self.recent_len
        super().__init__(recent_configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return super().forward(
            x_enc[:, -self.recent_len :, :],
            x_mark_enc[:, -self.recent_len :, :],
            x_dec,
            x_mark_dec,
            mask,
        )
