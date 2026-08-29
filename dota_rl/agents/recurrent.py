"""LSTM core with per-timestep done-masking for TBPTT (spec 5.2)."""
from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentCore(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.cell = nn.LSTMCell(in_dim, hidden)
        self.hidden = hidden

    def initial_state(self, batch: int, device=None):
        z = torch.zeros(batch, self.hidden, device=device)
        return z.clone(), z.clone()

    def step(self, x: torch.Tensor, state):
        h, c = self.cell(x, state)
        return h, (h, c)

    def forward_sequence(self, x_seq: torch.Tensor, done_seq: torch.Tensor, state):
        """x_seq: (T, B, D)  done_seq: (T, B) -- done AFTER step t.

        Hidden state is reset to zero at the start of t+1 when step t ended the
        episode. Returns outputs (T, B, H) and the final state.
        """
        T, B, _ = x_seq.shape
        h, c = state
        outs = []
        for t in range(T):
            h, (h, c) = self.step(x_seq[t], (h, c))
            outs.append(h)
            keep = (1.0 - done_seq[t].float()).unsqueeze(-1)
            h = h * keep
            c = c * keep
        return torch.stack(outs, dim=0), (h, c)
