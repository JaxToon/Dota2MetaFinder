"""Meta discovery: rich episode traces + event-aligned value-head analysis
(spec 15).

We do NOT read a raw V(s) jump as causal. For each event we compare
V(s_{t-k}) vs V(s_{t+m}) and report the *distribution* of deltas plus covariates,
against a matched baseline from windows where the event did not occur.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents.batch import obs_to_tensors
from ..agents.policy import ActorCritic
from ..env.action import Action
from ..env.mock_env import ABILITIES, ITEMS, MAX_HERO_HP, MockDotaEnv
from ..env.protocol import ActionType
from ..env.reward import RewardConfig


@dataclass
class Trace:
    """Per-step record of one deterministic episode."""
    steps: list[dict] = field(default_factory=list)
    outcome: int = 0

    # --- back-compat views (older code / tests) --------------------------
    @property
    def values(self) -> list[float]:
        return [s["value"] for s in self.steps]

    @property
    def events(self) -> list[list[str]]:
        return [s["events"] for s in self.steps]

    @property
    def covariates(self) -> list[dict]:
        return self.steps


def rollout_trace(policy: ActorCritic, phase: str = "F1", seed: int = 0,
                  device: str = "cpu", reward_cfg: RewardConfig | None = None,
                  deterministic: bool = True) -> Trace:
    env = MockDotaEnv(phase=phase, reward_cfg=reward_cfg or RewardConfig(), seed=seed)
    obs, _ = env.reset(seed=seed)
    state = policy.core.initial_state(1, device)
    tr = Trace()
    prev = {"kills": 0, "deaths": 0, "etower": env.enemy_tower_hp, "level": 1, "nw_bracket": 0}
    done = False
    while not done:
        obs_t = obs_to_tensors([obs], device)
        with torch.no_grad():
            value = policy.value_only(obs_t, state).item()
            actions, _, _, state = policy.act(obs_t, state, deterministic=deterministic)
        a = {k: v.squeeze(0).cpu().numpy() for k, v in actions.items()}
        at = ActionType(int(a["type"]))

        # entity slot 0 in the obs is the enemy hero
        ent = obs["entities"][0]
        enemy_visible = bool(ent[0] > 0.5 and ent[3] > 0.5)
        rec_pre = {
            "value": value,
            "pos": [float(env.agent.pos[0]), float(env.agent.pos[1])],
            "hp_ratio": float(env.agent.hp / MAX_HERO_HP),
            "mana_ratio": float(env.agent.mana / 300.0),
            "level": int(env.agent.level),
            "net_worth": float(env.agent.gold + env.agent.xp),
            "action_type": at.name,
            "ability_slot": int(a["ability"]) if at == ActionType.CAST else -1,
            "item_slot": int(a["item"]) if at == ActionType.ITEM else -1,
            "enemy_visible": enemy_visible,
            "enemy_hp_ratio": float(ent[6]) if enemy_visible else None,
            "enemy_dist": float(ent[7]) if enemy_visible else None,
            "enemy_tower_frac": float(env.enemy_tower_hp / env.enemy_tower_max),
        }

        obs, _, term, trunc, info = env.step(
            Action.from_heads(a["type"], a["entity"], a["ability"], a["item"], a["coord"]))
        done = term or trunc

        tags: list[str] = []
        if at == ActionType.CAST and rec_pre["ability_slot"] >= 0:
            tags.append(f"cast_ability_{rec_pre['ability_slot']}")
        if at == ActionType.ITEM and rec_pre["item_slot"] >= 0:
            tags.append(f"use_item_{rec_pre['item_slot']}")
        if info["agent_kills"] > prev["kills"]:
            tags.append("hero_kill")
        if info["agent_deaths"] > prev["deaths"]:
            tags.append("hero_death")
        if info["enemy_tower_hp"] < prev["etower"] - 1:
            tags.append("tower_damage")
        if info.get("net_worth", 0) // 500 > prev["nw_bracket"]:
            tags.append("networth_spike")
        if env.agent.level > prev["level"]:
            tags.append("level_up")

        rec_pre["events"] = tags
        rec_pre["t"] = info["t"]
        tr.steps.append(rec_pre)
        prev = {"kills": info["agent_kills"], "deaths": info["agent_deaths"],
                "etower": info["enemy_tower_hp"], "level": env.agent.level,
                "nw_bracket": int(info.get("net_worth", 0) // 500)}
    tr.outcome = int(info["outcome"])
    return tr


def _describe(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p90": round(float(np.percentile(a, 90)), 4)}


def analyze_events(traces: list[Trace], k: int = 3, m: int = 4) -> dict:
    """Event-aligned dV = V(s_{t+m}) - V(s_{t-k}), grouped by event tag, with a
    matched baseline and per-event covariate means (spec 15)."""
    deltas: dict[str, list[float]] = defaultdict(list)
    cov: dict[str, list[dict]] = defaultdict(list)
    baseline: list[float] = []

    for tr in traces:
        s = tr.steps
        n = len(s)
        for t in range(k, n - m):
            dv = s[t + m]["value"] - s[t - k]["value"]
            if s[t]["events"]:
                for tag in s[t]["events"]:
                    deltas[tag].append(dv)
                    cov[tag].append({
                        "level": s[t]["level"], "net_worth": s[t]["net_worth"],
                        "self_hp": s[t]["hp_ratio"], "enemy_hp": s[t]["enemy_hp_ratio"],
                        "enemy_dist": s[t]["enemy_dist"],
                    })
            else:
                baseline.append(dv)

    base = _describe(baseline)
    out = {}
    for tag, dv in deltas.items():
        d = _describe(dv)
        c = cov[tag]
        d["delta_vs_baseline"] = round(d["mean"] - base.get("mean", 0.0), 4)
        d["covariates"] = {
            key: round(float(np.nanmean([x[key] for x in c if x[key] is not None])), 3)
            if any(x[key] is not None for x in c) else None
            for key in ("level", "net_worth", "self_hp", "enemy_hp", "enemy_dist")
        }
        out[tag] = d
    return {
        "window": {"k_before": k, "m_after": m},
        "baseline_delta_v": base,
        "events": dict(sorted(out.items(), key=lambda kv: -abs(kv[1].get("delta_vs_baseline", 0)))),
        "note": "dV = V(s_{t+m}) - V(s_{t-k}); 'delta_vs_baseline' is the causal-ish "
                "signal after subtracting matched no-event windows.",
    }
