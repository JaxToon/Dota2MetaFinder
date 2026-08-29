"""Minimal Elo rating (spec 9.2). Glicko is a drop-in later; Elo is enough for
the MVP league anchor set.
"""
from __future__ import annotations


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def update(r_a: float, r_b: float, score_a: float, k: float = 24.0) -> tuple[float, float]:
    e_a = expected_score(r_a, r_b)
    r_a2 = r_a + k * (score_a - e_a)
    r_b2 = r_b + k * ((1.0 - score_a) - (1.0 - e_a))
    return r_a2, r_b2


class EloBook:
    def __init__(self, base: float = 1000.0):
        self.base = base
        self.ratings: dict[str, float] = {}

    def get(self, name: str) -> float:
        return self.ratings.setdefault(name, self.base)

    def record_series(self, a: str, b: str, wins_a: int, draws: int, losses_a: int,
                      k: float = 24.0) -> None:
        for _ in range(wins_a):
            self._one(a, b, 1.0, k)
        for _ in range(draws):
            self._one(a, b, 0.5, k)
        for _ in range(losses_a):
            self._one(a, b, 0.0, k)

    def _one(self, a: str, b: str, score_a: float, k: float) -> None:
        ra, rb = self.get(a), self.get(b)
        ra, rb = update(ra, rb, score_a, k)
        self.ratings[a], self.ratings[b] = ra, rb
