"""PatchTST with an IRPA-compatible long-history refinement and auxiliary."""

from __future__ import annotations

import copy

import torch

from layers.IRPARefinement import IRPARefinement
from models.PatchTST import Model as PatchTSTModel


class Model(PatchTSTModel):
    recent_len = 96

    def __init__(self, configs):
        self.input_seq_len = int(configs.seq_len)
        refine_len = int(getattr(configs, "irpa_revise_len", self.recent_len))
        if refine_len != self.recent_len:
            raise ValueError("PatchTSTIRPA comparison requires revise_len=96")
        backbone_configs = copy.copy(configs)
        backbone_configs.seq_len = self.recent_len
        super().__init__(backbone_configs)
        # Keep the backbone/dropout/data-loader random trajectory paired with the
        # recent-only control.  The adapter still receives deterministic weights,
        # but its construction does not advance the experiment RNG state.
        cpu_rng_state = torch.get_rng_state()
        self.irpa = IRPARefinement(configs)
        torch.set_rng_state(cpu_rng_state)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        refined, auxiliary, means, stdev = self.irpa(x_enc)
        enc_out, n_vars = self.patch_embedding(refined)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        ).permute(0, 1, 3, 2)
        prediction = self.head(enc_out).permute(0, 2, 1)
        prediction = prediction + auxiliary.permute(0, 2, 1)
        return prediction * stdev + means
