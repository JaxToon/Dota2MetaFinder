"""Welford running mean/variance for reward and advantage normalisation.

Spec 6.1/6.2: raw net-worth deltas can dominate; z-score then clip.
"""
from __future__ import annotations

import numpy as np


class RunningNorm:
    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.size

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-12)))

    def normalize(self, x, clip: float | None = None):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / (self.std + 1e-8)
        if clip is not None:
            z = np.clip(z, -clip, clip)
        return z

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]
