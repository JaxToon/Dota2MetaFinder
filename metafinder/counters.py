"""Hero-vs-hero counter matrix.

Primary source: OpenDota ``/heroes/{id}/matchups`` (aggregated head-to-head over
a rolling window). Optional second opinion from the local match store.

``advantage[a][b]`` = hero a's win rate when a and b are on opposite teams,
centred on 0 (positive => a beats b).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import opendota, store
from .heroes import all_heroes, resolve


@dataclass
class CounterMatrix:
    adv: dict[int, dict[int, float]]      # adv[a][b] in [-0.5, 0.5]
    games: dict[int, dict[int, int]]
    shrink: float = 100.0

    def advantage(self, a, b) -> float:
        a, b = resolve(a).id, resolve(b).id
        return self.adv.get(a, {}).get(b, 0.0)

    def worst_against(self, hero, n: int = 8) -> list[tuple[str, float, int]]:
        h = resolve(hero).id
        names = all_heroes()
        rows = [(names[b].localized_name, self.adv[h][b], self.games[h][b])
                for b in self.adv.get(h, {})]
        rows.sort(key=lambda r: r[1])
        return rows[:n]

    def best_against(self, hero, n: int = 8) -> list[tuple[str, float, int]]:
        return self.worst_against(hero, 10_000)[::-1][:n]


def from_opendota(*, shrink: float = 75.0, progress=None) -> CounterMatrix:
    heroes = list(all_heroes())
    adv: dict[int, dict[int, float]] = {}
    games: dict[int, dict[int, int]] = {}
    for i, hid in enumerate(heroes):
        if progress:
            progress(i + 1, len(heroes))
        adv[hid], games[hid] = {}, {}
        for m in opendota.hero_matchups(hid):
            g, w = m["games_played"], m["wins"]
            if g == 0:
                continue
            wr_adj = (w + shrink * 0.5) / (g + shrink)
            adv[hid][m["hero_id"]] = round(wr_adj - 0.5, 4)
            games[hid][m["hero_id"]] = g
    return CounterMatrix(adv, games, shrink)


def from_store(con=None, *, min_avg_rank: int | None = None,
               shrink: float = 60.0) -> CounterMatrix:
    con = con or store.connect()
    g: dict[tuple[int, int], int] = {}
    w: dict[tuple[int, int], int] = {}
    for team_a, team_b, a_won in store.iter_matchups(con, min_avg_rank=min_avg_rank):
        for x in team_a:
            for y in team_b:
                g[(x, y)] = g.get((x, y), 0) + 1
                w[(x, y)] = w.get((x, y), 0) + (1 if a_won else 0)
                g[(y, x)] = g.get((y, x), 0) + 1
                w[(y, x)] = w.get((y, x), 0) + (0 if a_won else 1)
    adv: dict[int, dict[int, float]] = {}
    games: dict[int, dict[int, int]] = {}
    for (x, y), n in g.items():
        wr_adj = (w[(x, y)] + shrink * 0.5) / (n + shrink)
        adv.setdefault(x, {})[y] = round(wr_adj - 0.5, 4)
        games.setdefault(x, {})[y] = n
    return CounterMatrix(adv, games, shrink)
