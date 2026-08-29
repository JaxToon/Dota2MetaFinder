"""Observation encoder: scalar MLP + attention-pooled entity/ability sets
(spec 5.1). Produces a single fixed-width feature vector for the LSTM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env.protocol import (
    ABILITY_FEAT_DIM,
    AGENT_SCALAR_DIM,
    ENTITY_FEAT_DIM,
    HERO_EMBED_DIM,
    ITEM_FEAT_DIM,
    EFFECT_EMBED_DIM,
)
from .embeddings import MetadataEmbedding

NUM_HERO_IDS = 16      # room for new heroes/creeps without resizing tensors
NUM_EFFECT_IDS = 16


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.key = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)  present: (B, N) bool
        scores = (self.key(x) @ self.query) / (x.shape[-1] ** 0.5)  # (B, N)
        scores = scores.masked_fill(~present.bool(), float("-inf"))
        scores = torch.where(present.bool().any(-1, keepdim=True), scores, torch.zeros_like(scores))
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)             # (B, N, 1)
        return (attn * x).sum(dim=1)                               # (B, D)


class ObsEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.scalar = mlp([AGENT_SCALAR_DIM, 128, 128])

        self.hero_emb = MetadataEmbedding(NUM_HERO_IDS, HERO_EMBED_DIM)
        self.effect_emb = MetadataEmbedding(NUM_EFFECT_IDS, EFFECT_EMBED_DIM)

        self.entity_mlp = mlp([ENTITY_FEAT_DIM + HERO_EMBED_DIM, 128, 128])
        self.entity_pool = AttentionPool(128)

        self.ability_mlp = mlp([ABILITY_FEAT_DIM + EFFECT_EMBED_DIM, 64, 64])
        self.item_mlp = mlp([ITEM_FEAT_DIM, 32, 32])

        self.proj = mlp([128 + 128 + 64 + 32, out_dim, out_dim])
        self.out_dim = out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        s = self.scalar(obs["agent_scalars"])

        ent = torch.cat([obs["entities"], self.hero_emb(obs["entity_hero_ids"])], dim=-1)
        ent = self.entity_mlp(ent)
        present = obs["entities"][..., 0] > 0.5
        ent = self.entity_pool(ent, present)

        ab = torch.cat([obs["abilities"], self.effect_emb(obs["ability_effect_ids"])], dim=-1)
        ab = self.ability_mlp(ab).mean(dim=1)

        it = self.item_mlp(obs["items"]).mean(dim=1)

        return self.proj(torch.cat([s, ent, ab, it], dim=-1))
