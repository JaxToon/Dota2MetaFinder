"""Hero-pair synergy from same-team co-occurrence in the local match store.

Raw pair win rate is confounded by each hero's solo strength, so synergy is
reported as ``pair_wr - expected_wr``, where ``expected_wr`` combines the two
solo win rates on the log-odds scale (independent-contribution baseline).
Positive synergy => the pair wins more than the sum of its parts.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from . import store
from .heroes import all_heroes, resolve


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class SynergyMatrix:
    solo_wr: dict[int, float]
    solo_games: dict[int, int]
    pair_games: dict[tuple[int, int], int]
    pair_wins: dict[tuple[int, int], int]
    shrink: float = 40.0
    _base: float = 0.5

    @staticmethod
    def _key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def expected_wr(self, a: int, b: int) -> float:
        # independent contributions on the log-odds scale, relative to base rate
        la = _logit(self.solo_wr.get(a, self._base)) - _logit(self._base)
        lb = _logit(self.solo_wr.get(b, self._base)) - _logit(self._base)
        return _sigmoid(_logit(self._base) + la + lb)

    def synergy(self, a, b) -> float:
        a, b = resolve(a).id, resolve(b).id
        k = self._key(a, b)
        g = self.pair_games.get(k, 0)
        if g == 0:
            return 0.0
        wr_adj = (self.pair_wins[k] + self.shrink * 0.5) / (g + self.shrink)
        return round(wr_adj - self.expected_wr(a, b), 4)

    def pair_winrate(self, a, b) -> tuple[float, int]:
        a, b = resolve(a).id, resolve(b).id
        k = self._key(a, b)
        g = self.pair_games.get(k, 0)
        return (self.pair_wins[k] / g if g else 0.0), g

    def best_partners(self, hero, n: int = 10, min_games: int = 30):
        h = resolve(hero).id
        names = all_heroes()
        rows = []
        for other in self.solo_wr:
            if other == h:
                continue
            g = self.pair_games.get(self._key(h, other), 0)
            if g < min_games:
                continue
            rows.append((names[other].localized_name, self.synergy(h, other), g))
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[:n]


def build(con=None, *, min_avg_rank: int | None = None,
          shrink: float = 40.0) -> SynergyMatrix:
    con = con or store.connect()
    solo_g: dict[int, int] = {}
    solo_w: dict[int, int] = {}
    pair_g: dict[tuple[int, int], int] = {}
    pair_w: dict[tuple[int, int], int] = {}
    total_g = total_w = 0

    for heroes, won in store.iter_teams(con, min_avg_rank=min_avg_rank):
        total_g += 1
        total_w += int(won)
        for h in heroes:
            solo_g[h] = solo_g.get(h, 0) + 1
            solo_w[h] = solo_w.get(h, 0) + int(won)
        for a, b in itertools.combinations(sorted(set(heroes)), 2):
            pair_g[(a, b)] = pair_g.get((a, b), 0) + 1
            pair_w[(a, b)] = pair_w.get((a, b), 0) + int(won)

    base = (total_w / total_g) if total_g else 0.5
    solo_wr = {h: (solo_w[h] + shrink * 0.5) / (solo_g[h] + shrink) for h in solo_g}
    m = SynergyMatrix(solo_wr, solo_g, pair_g, pair_w, shrink)
    m._base = base
    return m
