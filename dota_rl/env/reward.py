"""Reward system (spec 6).

r_t = clip( w_eco*z(dnw) + w_xp*z(dxp) + w_com*combat + w_obj*dobj
            + w_str*dstr + w_term*terminal , r_min, r_max )

Design rules from the spec:
  * weights live in versioned config, never hard-coded;
  * economy / xp deltas are z-scored with a running estimator so raw net-worth
    cannot dominate;
  * event terms (combat, structure) are small and bounded;
  * the terminal win/loss term is large and always preserved;
  * team blending: r = (1-tau)*r_agent + tau*r_team.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .running_norm import RunningNorm


@dataclass
class RewardConfig:
    w_eco: float = 1.0
    w_xp: float = 0.5
    w_com: float = 1.0
    w_obj: float = 1.0
    w_str: float = 1.0
    w_term: float = 1.0
    terminal_win: float = 10.0
    clip: float = 5.0
    team_blend_tau: float = 0.0
    normalize: bool = True
    # which components are active -- start minimal, enable by ablation (spec 6.1)
    enabled: dict = field(default_factory=lambda: {
        "eco": True, "xp": True, "com": True, "obj": False,
        "str": True, "shape": True, "term": True,
    })


@dataclass
class RewardInputs:
    d_net_worth: float = 0.0
    d_xp: float = 0.0
    combat_event: float = 0.0        # already-bounded kill(+)/death(-) event value
    d_objective: float = 0.0         # bounded objective-control delta
    d_structure: float = 0.0         # normalised structure damage in [-1, 1]
    potential_delta: float = 0.0     # gamma*Phi(s') - Phi(s), potential-based shaping (spec 6)
    terminal: int = 0               # +1 win, -1 loss, 0 not terminal
    team_d_net_worth: float = 0.0
    team_combat_event: float = 0.0


class RewardShaper:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self._nw = RunningNorm()
        self._xp = RunningNorm()

    def _components(self, x: RewardInputs) -> dict:
        cfg = self.cfg
        if cfg.normalize:
            self._nw.update(np.array([x.d_net_worth]))
            self._xp.update(np.array([x.d_xp]))
            eco = float(self._nw.normalize(x.d_net_worth, clip=cfg.clip))
            xp = float(self._xp.normalize(x.d_xp, clip=cfg.clip))
        else:
            eco, xp = x.d_net_worth, x.d_xp
        term = 0.0
        if x.terminal != 0:
            term = cfg.terminal_win * (1.0 if x.terminal > 0 else -1.0)
        return {
            "eco": cfg.w_eco * eco,
            "xp": cfg.w_xp * xp,
            "com": cfg.w_com * float(np.clip(x.combat_event, -2.0, 2.0)),
            "obj": cfg.w_obj * float(np.clip(x.d_objective, -1.0, 1.0)),
            "str": cfg.w_str * float(np.clip(x.d_structure, -1.0, 1.0)),
            "shape": float(np.clip(x.potential_delta, -1.0, 1.0)),
            "term": float(term),
        }

    def step(self, agent: RewardInputs) -> tuple[float, dict]:
        comps = self._components(agent)
        r_agent = sum(v for k, v in comps.items() if self.cfg.enabled.get(k, True))
        r_agent = float(np.clip(r_agent, -self.cfg.clip, self.cfg.clip))

        tau = self.cfg.team_blend_tau
        if tau > 0.0:
            team = RewardInputs(
                d_net_worth=agent.team_d_net_worth,
                d_xp=agent.d_xp,
                combat_event=agent.team_combat_event,
                d_objective=agent.d_objective,
                d_structure=agent.d_structure,
                terminal=agent.terminal,
            )
            tcomps = self._components(team)
            r_team = sum(v for k, v in tcomps.items() if self.cfg.enabled.get(k, True))
            r_team = float(np.clip(r_team, -self.cfg.clip, self.cfg.clip))
            r = (1.0 - tau) * r_agent + tau * r_team
        else:
            r = r_agent
        return float(np.clip(r, -self.cfg.clip, self.cfg.clip)), comps

    def state_dict(self) -> dict:
        return {"nw": self._nw.state_dict(), "xp": self._xp.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        self._nw.load_state_dict(d["nw"])
        self._xp.load_state_dict(d["xp"])
