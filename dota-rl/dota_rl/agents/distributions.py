"""Masked categorical + diagonal Gaussian used by the multi-head policy."""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG_INF = -1e9


class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor | None):
        if mask is not None:
            mask = mask.bool()
            # rows with no legal option -> allow all (keeps dist well-defined;
            # such a head is never consumed because its action type is masked)
            empty = ~mask.any(dim=-1, keepdim=True)
            mask = mask | empty
            logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
        self.logits = logits
        self.logp = F.log_softmax(logits, dim=-1)
        self.probs = self.logp.exp()

    def sample(self) -> torch.Tensor:
        return torch.multinomial(self.probs.clamp_min(1e-12), 1).squeeze(-1)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        return self.logp.gather(-1, a.long().unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> torch.Tensor:
        return -(self.probs * self.logp).sum(-1)


class DiagGaussian:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std.clamp(-5.0, 2.0)
        self.std = self.log_std.exp()

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        var = self.std**2
        lp = -0.5 * (((x - self.mean) ** 2) / var + 2 * self.log_std + torch.log(torch.tensor(2 * torch.pi)))
        return lp.sum(-1)

    def entropy(self) -> torch.Tensor:
        return (0.5 + 0.5 * torch.log(torch.tensor(2 * torch.pi)) + self.log_std).sum(-1)
