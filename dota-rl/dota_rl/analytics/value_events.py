"""Meta discovery: event-aligned value-head analysis (spec 15).

Rather than reading a raw V(s) jump as causal, we compare V(s_{t-k}) vs
V(s_{t+m}) around an event and report the *distribution* of deltas grouped by
context, plus a matched baseline from windows where the event did not occur.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents.batch import obs_to_tensors
from ..agents.policy import ActorCritic
from ..env.mock_env import MockDotaEnv
from ..env.reward import RewardConfig


@dataclass
class Trace:
    values: list[float] = field(default_factory=list)
    events: list[list[str]] = field(default_factory=list)   # event tags per step
    covariates: list[dict] = field(default_factory=list)


def rollout_trace(policy: ActorCritic, phase: str = "F1", seed: int = 0,
                  device: str = "cpu", reward_cfg: RewardConfig | None = None) -> Trace:
    env = MockDotaEnv(phase=phase, reward_cfg=reward_cfg or RewardConfig(), seed=seed)
    obs, _ = env.reset(seed=seed)
    state = policy.core.initial_state(1, device)
    tr = Trace()
    prev = {"kills": 0, "deaths": 0, "enemy_tower_hp": None, "gold_bracket": 0}
    done = False
    while not done:
        obs_t = obs_to_tensors([obs], device)
        with torch.no_grad():
            v = policy.value_only(obs_t, state).item()
            actions, _, _, state = policy.act(obs_t, state, deterministic=True)
        from ..env.action import Action
        a = {k: val.squeeze(0).cpu().numpy() for k, val in actions.items()}
        obs, _, term, trunc, info = env.step(
            Action.from_heads(a["type"], a["entity"], a["ability"], a["item"], a["coord"]))
        done = term or trunc

        tags = []
        if info["agent_kills"] > prev["kills"]:
            tags.append("hero_kill")
        if info["agent_deaths"] > prev["deaths"]:
            tags.append("hero_death")
        if prev["enemy_tower_hp"] is not None and info["enemy_tower_hp"] < prev["enemy_tower_hp"] - 1:
            tags.append("tower_damage")
        gb = int(info["net_worth"] // 500)
        if gb > prev["gold_bracket"]:
            tags.append("networth_spike")

        tr.values.append(v)
        tr.events.append(tags)
        tr.covariates.append({
            "t": info["t"], "net_worth": info["net_worth"],
            "kills": info["agent_kills"], "deaths": info["agent_deaths"],
            "enemy_tower_hp": info["enemy_tower_hp"],
        })
        prev = {"kills": info["agent_kills"], "deaths": info["agent_deaths"],
                "enemy_tower_hp": info["enemy_tower_hp"], "gold_bracket": gb}
    return tr


def analyze_events(traces: list[Trace], k: int = 3, m: int = 3) -> dict:
    event_deltas: dict[str, list[float]] = defaultdict(list)
    baseline_deltas: list[float] = []
    for tr in traces:
        n = len(tr.values)
        for t in range(k, n - m):
            window_deltas = tr.values[t + m] - tr.values[t - k]
            if tr.events[t]:
                for tag in tr.events[t]:
                    event_deltas[tag].append(window_deltas)
            else:
                baseline_deltas.append(window_deltas)

    def describe(xs):
        a = np.asarray(xs, dtype=float)
        if a.size == 0:
            return {"n": 0}
        return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
                "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90))}

    return {
        "baseline": describe(baseline_deltas),
        "events": {tag: describe(v) for tag, v in event_deltas.items()},
        "note": "delta_v is V(s_{t+m}) - V(s_{t-k}); compare event rows to baseline.",
    }
