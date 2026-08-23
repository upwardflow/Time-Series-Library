"""GraphMamba strict control that reads a long window but uses a recent suffix."""

from __future__ import annotations

import copy

from models.GraphMamba import Model as GraphMambaModel


class Model(GraphMambaModel):
    """Align data indices with long-memory models without changing the backbone."""

    def __init__(self, configs):
        self.input_seq_len = int(configs.seq_len)
        self.recent_len = int(
            getattr(
                configs,
                "timerole_recent_len",
                getattr(configs, "cmrhm_recent_len", 96),
            )
        )
        if self.recent_len < 1:
            raise ValueError("timerole_recent_len must be positive")
        if self.input_seq_len < self.recent_len:
            raise ValueError(
                "GraphMambaRecent requires seq_len >= timerole_recent_len"
            )
        recent_configs = copy.copy(configs)
        recent_configs.seq_len = self.recent_len
        super().__init__(recent_configs)

    def forecast(self, x_enc):
        return super().forecast(x_enc[:, -self.recent_len :, :])
