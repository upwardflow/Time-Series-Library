"""CMRHM with adaptive graph-residual/Mamba-increment fusion."""

from __future__ import annotations

import copy

from models.GraphMambaCMRHM import Model as GraphMambaCMRHM


class Model(GraphMambaCMRHM):
    """Activate adaptive fusion without changing the established CMRHM model."""

    def __init__(self, configs):
        adaptive_configs = copy.copy(configs)
        adaptive_configs.graph_mamba_fusion = "graph_residual_gate"
        adaptive_configs.agf_hidden_dim = int(
            getattr(configs, "af_hidden_dim", 32)
        )
        adaptive_configs.agf_initial_mamba_weight = 0.1
        super().__init__(adaptive_configs)
