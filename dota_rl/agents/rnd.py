"""Random Network Distillation curiosity (spec 7, stages E3-E4).

A fixed random *target* net and a trainable *predictor* net map a compact
observation summary to an embedding. The predictor's error is the intrinsic
reward: high for novel states, shrinking as they are revisited. The bonus is
normalised by a running std and its weight is annealed toward zero late in
training (E4) so it does not fight league stability once the policy converges.

Introduce this ONLY after entropy-only exploration stalls (spec 7 / 22).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..env.protocol import AGENT_SCALAR_DIM
from ..env.running_norm import RunningNorm

RND_IN_DIM = AGENT_SCALAR_DIM + 4 + 4 + 3  # scalars + entity summary + ability + item means


def _summary(obs: dict) -> torch.Tensor:
    """(B, RND_IN_DIM) compact, permutation-invariant obs digest."""
    s = obs["agent_scalars"]
    ent = obs["entities"]                       # (B, N, 8)
    ent_sum = torch.stack([
        ent[..., 0].mean(-1),                   # presence
        ent[..., 3].mean(-1),                   # visibility
        ent[..., 6].mean(-1),                   # hp
        ent[..., 7].mean(-1),                   # distance
    ], dim=-1)
    ab = obs["abilities"][..., 0].mean(-1, keepdim=True)  # readiness
    ab = torch.cat([ab, obs["abilities"][..., 1].mean(-1, keepdim=True),
                    obs["abilities"][..., 2].mean(-1, keepdim=True),
                    obs["abilities"][..., 3].mean(-1, keepdim=True)], dim=-1)
    it = obs["items"][..., :3].mean(1)          # (B, 3)
    return torch.cat([s, ent_sum, ab, it], dim=-1)


class RND(nn.Module):
    def __init__(self, hidden: int = 128, emb: int = 64, lr: float = 1e-4, device: str = "cpu"):
        super().__init__()
        def net():
            return nn.Sequential(nn.Linear(RND_IN_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, emb))
        self.target = net().to(device)
        self.predictor = net().to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_norm = RunningNorm()
        self.rew_norm = RunningNorm()
        self.device = device

    def _feat(self, obs: dict) -> torch.Tensor:
        x = _summary(obs).to(self.device)
        z = (x - float(self.obs_norm.mean)) / (self.obs_norm.std + 1e-8)
        return z.clamp(-5.0, 5.0)

    @torch.no_grad()
    def intrinsic_reward(self, obs_seq: dict) -> torch.Tensor:
        """obs_seq batched (T, B, ...) -> (T, B) normalised novelty bonus."""
        T = obs_seq["agent_scalars"].shape[0]
        B = obs_seq["agent_scalars"].shape[1]
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_seq.items() if k != "masks"}
        self.obs_norm.update(_summary(flat).cpu().numpy())
        feat = self._feat(flat)
        err = (self.predictor(feat) - self.target(feat)).pow(2).mean(-1)
        err = err.reshape(T, B).cpu().numpy()
        self.rew_norm.update(err.reshape(-1))
        return torch.as_tensor(err / (self.rew_norm.std + 1e-8), dtype=torch.float32)

    def update(self, obs_seq: dict, epochs: int = 1) -> float:
        T = obs_seq["agent_scalars"].shape[0]
        B = obs_seq["agent_scalars"].shape[1]
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_seq.items() if k != "masks"}
        feat = self._feat(flat)
        with torch.no_grad():
            tgt = self.target(feat)
        last = 0.0
        for _ in range(epochs):
            loss = (self.predictor(feat) - tgt).pow(2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            last = float(loss.item())
        return last

    def state_dict_extra(self) -> dict:
        return {"predictor": self.predictor.state_dict(),
                "obs_norm": self.obs_norm.state_dict(),
                "rew_norm": self.rew_norm.state_dict()}


def coef_schedule(coef0: float, coef1: float, progress: float) -> float:
    """Linear anneal (E4). progress 0..1."""
    p = float(np.clip(progress, 0.0, 1.0))
    return coef0 + (coef1 - coef0) * p
