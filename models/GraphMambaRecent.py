"""GraphMamba strict control that reads a long window but uses only recent 96."""

from __future__ import annotations

import copy

from models.GraphMamba import Model as GraphMambaModel


class Model(GraphMambaModel):
    """Align data indices with long-memory models without changing the backbone."""

    recent_len = 96

    def __init__(self, configs):
        self.input_seq_len = int(configs.seq_len)
        if self.input_seq_len < self.recent_len:
            raise ValueError("GraphMambaRecent requires seq_len >= 96")
        recent_configs = copy.copy(configs)
        recent_configs.seq_len = self.recent_len
        super().__init__(recent_configs)

    def forecast(self, x_enc):
        return super().forecast(x_enc[:, -self.recent_len :, :])
