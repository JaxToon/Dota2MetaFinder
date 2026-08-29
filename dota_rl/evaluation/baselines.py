"""Fixed opponent / sanity policies for the evaluation harness (spec 9.2).

These act from the public observation dict only -- same interface a real agent
sees. They give the "weak/random" and "scripted baseline" anchors the Elo
league needs.
"""
from __future__ import annotations

import numpy as np

from ..env.action import Action
from ..env.observation import Observation
from ..env.protocol import ActionType


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs: Observation) -> Action:
        m = obs["masks"]
        legal_types = np.flatnonzero(m["action_type"])
        t = ActionType(int(self.rng.choice(legal_types)))
        ent = int(self.rng.choice(np.flatnonzero(m["entity"]))) if m["entity"].any() else -1
        ab = int(self.rng.choice(np.flatnonzero(m["ability"]))) if m["ability"].any() else -1
        it = int(self.rng.choice(np.flatnonzero(m["item"]))) if m["item"].any() else -1
        coord = self.rng.uniform(0, 1, size=2)
        return Action.from_heads(t, ent, ab, it, coord)


class ScriptedLaner:
    """Greedy laner: last-hit in range, poke the enemy hero, else walk to lane."""

    name = "scripted"

    def __init__(self, seed: int = 0, aggression: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.aggression = aggression

    def reset(self):
        pass

    def act(self, obs: Observation) -> Action:
        m = obs["masks"]
        ents = obs["entities"]          # (N, 8): present,is_enemy,is_creep,visible,x,y,hp,dist
        agent_xy = obs["agent_scalars"][5:7]

        if m["action_type"][ActionType.ATTACK]:
            cand = np.flatnonzero(m["entity"] & (ents[:, 0] > 0.5) & (ents[:, 1] > 0.5))
            if cand.size:
                # prefer the lowest-hp target (last-hit), tie-break on distance
                order = sorted(cand, key=lambda s: (ents[s, 6], ents[s, 7]))
                return Action.from_heads(ActionType.ATTACK, int(order[0]), -1, -1, agent_xy)

        if m["action_type"][ActionType.CAST] and self.rng.random() < self.aggression:
            cand = np.flatnonzero(m["entity"] & (ents[:, 0] > 0.5) & (ents[:, 1] > 0.5))
            ab = np.flatnonzero(m["ability"])
            if cand.size and ab.size:
                return Action.from_heads(ActionType.CAST, int(cand[0]), int(ab[0]), -1, agent_xy)

        # walk toward the nearest visible creep, else lane centre
        creeps = np.flatnonzero((ents[:, 0] > 0.5) & (ents[:, 2] > 0.5) & (ents[:, 6] > 0))
        if creeps.size:
            s = min(creeps, key=lambda s: ents[s, 7])
            target = agent_xy + np.array([ents[s, 4] * 2 - 1, ents[s, 5] * 2 - 1])
        else:
            target = np.array([0.5, 0.5])
        target = np.clip(target + self.rng.uniform(-0.02, 0.02, size=2), 0, 1)
        return Action.from_heads(ActionType.MOVE, -1, -1, -1, target)


BASELINES = {"random": RandomPolicy, "scripted": ScriptedLaner}
