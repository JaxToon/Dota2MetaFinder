"""Generalised Advantage Estimation with explicit terminal/bootstrap masks
(spec 10, step 6). ``terminated`` cuts the value bootstrap; ``truncated`` keeps
it (time-limit is not a real terminal).
"""
from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,       # (T, B)
    values: torch.Tensor,        # (T, B)
    terminated: torch.Tensor,    # (T, B) bool -- real episode end
    truncated: torch.Tensor,     # (T, B) bool -- time-limit cutoff
    last_value: torch.Tensor,    # (B,)  bootstrap value for step T
    gamma: float = 0.99,
    lam: float = 0.95,
):
    T, B = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        nonterminal = (~terminated[t]).float()
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        # on truncation: use delta (bootstrapped) but stop the GAE recursion
        cont = (~(terminated[t] | truncated[t])).float()
        last_gae = delta + gamma * lam * cont * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns
