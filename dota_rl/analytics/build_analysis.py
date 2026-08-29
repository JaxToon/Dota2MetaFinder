"""Build-order / ability-timing / power-spike analysis (spec 15, repo §11).

Turns a set of episode traces into interpretable "meta candidates":
  * when (game time, HP, mana, enemy state) the agent casts each ability / uses
    each item -- the learned usage policy,
  * power-spike windows: dV around level-ups and first item use,
  * combat-trade discipline: does it engage when ahead on HP?
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..env.mock_env import ABILITIES, ITEMS
from .value_events import Trace, _describe


def ability_timing(traces: list[Trace]) -> dict:
    casts: dict[int, list[dict]] = defaultdict(list)
    for tr in traces:
        for s in tr.steps:
            if s["ability_slot"] >= 0:
                casts[s["ability_slot"]].append(s)
    out = {}
    for slot, rows in sorted(casts.items()):
        kind = ABILITIES[slot]["kind"] if slot < len(ABILITIES) else "?"
        out[f"ability_{slot} ({kind})"] = {
            "uses_per_episode": round(len(rows) / max(1, len(traces)), 2),
            "at_game_time": _describe([r["t"] for r in rows]),
            "at_self_hp": _describe([r["hp_ratio"] for r in rows]),
            "at_mana": _describe([r["mana_ratio"] for r in rows]),
            "at_enemy_hp": _describe([r["enemy_hp_ratio"] for r in rows if r["enemy_hp_ratio"] is not None]),
            "at_enemy_dist": _describe([r["enemy_dist"] for r in rows if r["enemy_dist"] is not None]),
        }
    return out


def item_timing(traces: list[Trace]) -> dict:
    uses: dict[int, list[dict]] = defaultdict(list)
    first_use: dict[int, list[float]] = defaultdict(list)
    for tr in traces:
        seen = set()
        for s in tr.steps:
            if s["item_slot"] >= 0:
                uses[s["item_slot"]].append(s)
                if s["item_slot"] not in seen:
                    first_use[s["item_slot"]].append(s["t"])
                    seen.add(s["item_slot"])
    out = {}
    for slot in sorted(set(uses) | set(first_use)):
        kind = ITEMS[slot]["kind"] if slot < len(ITEMS) else "?"
        out[f"item_{slot} ({kind})"] = {
            "uses_per_episode": round(len(uses[slot]) / max(1, len(traces)), 2),
            "first_use_game_time": _describe(first_use[slot]),
            "at_self_hp": _describe([r["hp_ratio"] for r in uses[slot]]),
        }
    return out


def power_spikes(traces: list[Trace], k: int = 3, m: int = 6) -> dict:
    """dV around level-ups and first item use -- the learned power spikes."""
    spike: dict[str, list[float]] = defaultdict(list)
    ctx: dict[str, list[dict]] = defaultdict(list)
    for tr in traces:
        s = tr.steps
        for t in range(k, len(s) - m):
            dv = s[t + m]["value"] - s[t - k]["value"]
            for tag in s[t]["events"]:
                if tag in ("level_up", "hero_kill", "tower_damage") or tag.startswith("use_item"):
                    spike[tag].append(dv)
                    ctx[tag].append({"level": s[t]["level"], "t": s[t]["t"],
                                     "net_worth": s[t]["net_worth"]})
    return {
        tag: {**_describe(v), "at": {key: round(float(np.mean([c[key] for c in ctx[tag]])), 1)
                                    for key in ("level", "t", "net_worth")}}
        for tag, v in sorted(spike.items(), key=lambda kv: -np.mean(kv[1]) if kv[1] else 0)
    }


def trade_discipline(traces: list[Trace]) -> dict:
    """When the agent attacks / casts at the enemy hero, is it ahead on HP?"""
    engaged_hp_adv, passed_hp_adv = [], []
    for tr in traces:
        for s in tr.steps:
            if not s["enemy_visible"] or s["enemy_hp_ratio"] is None:
                continue
            hp_adv = s["hp_ratio"] - s["enemy_hp_ratio"]
            aggressive = s["action_type"] in ("ATTACK", "CAST")
            (engaged_hp_adv if aggressive else passed_hp_adv).append(hp_adv)
    return {
        "hp_advantage_when_engaging": _describe(engaged_hp_adv),
        "hp_advantage_when_not_engaging": _describe(passed_hp_adv),
        "reads_as": ("disciplined: engages with a HP lead"
                     if _describe(engaged_hp_adv).get("mean", 0)
                     > _describe(passed_hp_adv).get("mean", 0)
                     else "aggressive: engages regardless of HP"),
    }


def summarize(traces: list[Trace]) -> dict:
    wins = sum(1 for t in traces if t.outcome > 0)
    return {
        "episodes": len(traces),
        "winrate": round(wins / max(1, len(traces)), 3),
        "mean_episode_len": round(float(np.mean([len(t.steps) for t in traces])), 1),
        "ability_timing": ability_timing(traces),
        "item_timing": item_timing(traces),
        "power_spikes": power_spikes(traces),
        "trade_discipline": trade_discipline(traces),
    }
