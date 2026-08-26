"""Frozen R4 recent-only backbone selected by the TimeRole Phase C gate.

R4 is intentionally exposed as a separate model instead of changing
``GraphMambaRecent``.  This keeps historical controls reproducible and gives
the later short--long interaction study an explicit, auditable backbone.
"""

from __future__ import annotations

import copy
from types import MappingProxyType

from models.GraphMambaRecent import Model as GraphMambaRecentModel


R4_STRUCTURE = MappingProxyType(
    {
        "timerole_recent_len": 96,
        "use_decomp": 0,
        "use_patch": 1,
        "dual_scale_scan_mode": "independent_shared",
        "dual_scale_selection": "fine",
        "use_time_mamba": 1,
        "mamba_bidirectional": 1,
        "use_graph": 0,
        "graph_mamba_fusion": "fixed_sum",
    }
)


class Model(GraphMambaRecentModel):
    """GraphMambaRecent with the Phase-C-selected R4 structure frozen."""

    variant = "R4"
    frozen_structure = R4_STRUCTURE

    def __init__(self, configs):
        # Never mutate argparse/config objects owned by the caller.  Apart from
        # the structural switches below, all training and capacity settings
        # remain explicit experiment inputs.
        r4_configs = copy.copy(configs)
        for name, value in self.frozen_structure.items():
            setattr(r4_configs, name, value)
        super().__init__(r4_configs)
