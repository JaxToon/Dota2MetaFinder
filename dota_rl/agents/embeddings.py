"""Attribute-style embeddings for heroes/creeps and ability effects (spec 5.3).

Corrected expectation: embeddings give *architectural compatibility* (new
hero/item metadata loads without changing tensor dims) -- they do NOT guarantee
zero-shot optimal play. New ids map to a learnable vector initialised from an
"attribute prior" so an unseen id starts near a sensible region of the space.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MetadataEmbedding(nn.Module):
    def __init__(self, num_ids: int, dim: int, attribute_dim: int = 4):
        super().__init__()
        self.num_ids = num_ids
        self.dim = dim
        self.table = nn.Embedding(num_ids + 1, dim)  # +1 = OOV / new-content slot
        # optional attribute projection: metadata vector -> embedding space
        self.attr_proj = nn.Linear(attribute_dim, dim)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, ids: torch.Tensor, attributes: torch.Tensor | None = None) -> torch.Tensor:
        ids = ids.long().clamp(0, self.num_ids)  # unknown ids fold into OOV slot
        emb = self.table(ids)
        if attributes is not None:
            emb = emb + self.attr_proj(attributes)
        return emb
