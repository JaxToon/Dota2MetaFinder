"""Win-probability confidence intervals and episode summary stats (spec 9.3)."""
from __future__ import annotations

import math

import numpy as np


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize_episodes(infos: list[dict]) -> dict:
    if not infos:
        return {}
    wins = sum(1 for i in infos if i["outcome"] > 0)
    losses = sum(1 for i in infos if i["outcome"] < 0)
    n = len(infos)
    lo, hi = wilson_interval(wins, n)
    arr = lambda key: np.array([i[key] for i in infos], dtype=float)
    return {
        "n_games": n,
        "winrate": wins / n,
        "winrate_ci_low": lo,
        "winrate_ci_high": hi,
        "lossrate": losses / n,
        "kills_mean": float(arr("agent_kills").mean()),
        "deaths_mean": float(arr("agent_deaths").mean()),
        "net_worth_mean": float(arr("net_worth").mean()),
        "enemy_tower_hp_mean": float(arr("enemy_tower_hp").mean()),
        "length_mean": float(arr("t").mean()),
    }
