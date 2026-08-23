"""PatchTST control that reads a long sample but models only the recent 96 points."""

from __future__ import annotations

import copy

from models.PatchTST import Model as PatchTSTModel


class Model(PatchTSTModel):
    recent_len = 96

    def __init__(self, configs):
        self.input_seq_len = int(configs.seq_len)
        if self.input_seq_len < self.recent_len:
            raise ValueError("PatchTSTRecent requires seq_len >= 96")
        recent_configs = copy.copy(configs)
        recent_configs.seq_len = self.recent_len
        super().__init__(recent_configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        recent_marks = (
            x_mark_enc[:, -self.recent_len :, :]
            if x_mark_enc is not None
            else None
        )
        return super().forecast(
            x_enc[:, -self.recent_len :, :], recent_marks, x_dec, x_mark_dec
        )
