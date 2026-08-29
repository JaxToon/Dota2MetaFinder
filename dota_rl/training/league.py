"""Self-play & league training (spec section 7: E1 self-play, E2 historical pool).

The learner always controls the agent side. The opponent hero is driven, per
episode, by one of:
  * ``self``     -- a frozen copy of the current learner (refreshed each iter),
  * ``history``  -- a snapshot sampled from the historical pool,
  * ``scripted`` -- the built-in scripted laner (kept in the mix for stability).

The 80/20-ish split is a configurable experiment variable, not a fixed law
(spec section 7 note). Opponent forward passes are batched across envs that share
a snapshot so self-play roughly doubles rollout cost, not more.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents.batch import obs_to_tensors
from ..agents.policy import ActorCritic
from ..env.action import Action
from ..evaluation.baselines import ScriptedLaner


@dataclass
class LeagueConfig:
    enabled: bool = False
    ratio_self: float = 0.5
    ratio_history: float = 0.3
    ratio_scripted: float = 0.2
    pool_capacity: int = 8
    snapshot_every: int = 10          # iterations
    pfsp: bool = True                 # prioritise hard historical opponents
    deterministic_opponent: bool = False


class FrozenPolicy:
    """An eval-only ActorCritic with per-env recurrent state, batched inference."""

    def __init__(self, weights: dict, encoder_dim: int, lstm_hidden: int,
                 device: str = "cpu", deterministic: bool = False, tag: str = "snap"):
        self.net = ActorCritic(encoder_dim=encoder_dim, lstm_hidden=lstm_hidden).to(device)
        self.net.load_state_dict(weights)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.deterministic = deterministic
        self.tag = tag
        self._state: dict[int, tuple] = {}

    def reset(self, env_idx: int) -> None:
        self._state.pop(env_idx, None)

    @torch.no_grad()
    def act_batch(self, obs_by_env: dict[int, dict]) -> dict[int, Action]:
        if not obs_by_env:
            return {}
        idx = sorted(obs_by_env)
        for i in idx:
            if i not in self._state:
                self._state[i] = self.net.core.initial_state(1, self.device)
        h = torch.cat([self._state[i][0] for i in idx], dim=0)
        c = torch.cat([self._state[i][1] for i in idx], dim=0)
        obs_t = obs_to_tensors([obs_by_env[i] for i in idx], self.device)
        actions, _, _, (h2, c2) = self.net.act(obs_t, (h, c), deterministic=self.deterministic)
        for k, i in enumerate(idx):
            self._state[i] = (h2[k:k + 1], c2[k:k + 1])
        a_type = actions["type"].cpu().numpy()
        ent = actions["entity"].cpu().numpy()
        ab = actions["ability"].cpu().numpy()
        it = actions["item"].cpu().numpy()
        coord = actions["coord"].cpu().numpy()
        return {i: Action.from_heads(a_type[k], ent[k], ab[k], it[k], coord[k])
                for k, i in enumerate(idx)}


@dataclass
class _PoolEntry:
    policy: FrozenPolicy
    iteration: int
    games: int = 0
    losses_vs_learner: int = 0        # times this snapshot lost to the current learner

    @property
    def difficulty(self) -> float:
        # PFSP-style: snapshots the learner beats often are less useful
        if self.games == 0:
            return 1.0
        win_rate_vs_learner = 1.0 - self.losses_vs_learner / self.games
        return float(np.clip(win_rate_vs_learner, 0.05, 1.0))


class League:
    def __init__(self, cfg: LeagueConfig, model_cfg: dict, num_envs: int,
                 device: str = "cpu", seed: int = 0):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.n = num_envs
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.pool: list[_PoolEntry] = []
        self.self_policy: FrozenPolicy | None = None
        self._scripted = [ScriptedLaner(seed=1000 + i) for i in range(num_envs)]
        # per-env assignment: ("self", None) | ("hist", entry) | ("scripted", ScriptedLaner)
        self._assign: list[tuple] = [("scripted", self._scripted[i]) for i in range(num_envs)]

    # ---------------------------------------------------------------- snapshots
    def _make_frozen(self, policy: ActorCritic, tag: str) -> FrozenPolicy:
        return FrozenPolicy(
            copy.deepcopy(policy.state_dict()),
            encoder_dim=int(self.model_cfg.get("encoder_dim", 256)),
            lstm_hidden=int(self.model_cfg.get("lstm_hidden", 256)),
            device=self.device, deterministic=self.cfg.deterministic_opponent, tag=tag,
        )

    def refresh_self(self, policy: ActorCritic) -> None:
        self.self_policy = self._make_frozen(policy, "self")

    def maybe_snapshot(self, policy: ActorCritic, iteration: int) -> bool:
        if iteration % self.cfg.snapshot_every != 0:
            return False
        entry = _PoolEntry(self._make_frozen(policy, f"iter{iteration}"), iteration)
        self.pool.append(entry)
        if len(self.pool) > self.cfg.pool_capacity:
            self.pool.pop(0)  # drop the oldest
        return True

    # ------------------------------------------------------------- assignment
    def _sample_kind(self) -> str:
        c = self.cfg
        if not self.pool:
            r = self.rng.random() * (c.ratio_self + c.ratio_scripted + 1e-9)
            return "self" if r < c.ratio_self else "scripted"
        r = self.rng.random()
        if r < c.ratio_self or self.self_policy is None:
            return "self" if self.self_policy is not None else "scripted"
        if r < c.ratio_self + c.ratio_history:
            return "hist"
        return "scripted"

    def _sample_hist(self) -> _PoolEntry:
        if self.cfg.pfsp:
            w = np.array([e.difficulty for e in self.pool], dtype=float)
            w = w / w.sum()
            return self.pool[int(self.rng.choice(len(self.pool), p=w))]
        return self.pool[int(self.rng.integers(len(self.pool)))]

    def on_episode_start(self, env_idx: int) -> None:
        kind = self._sample_kind()
        if kind == "self" and self.self_policy is not None:
            self.self_policy.reset(env_idx)
            self._assign[env_idx] = ("self", None)
        elif kind == "hist":
            entry = self._sample_hist()
            entry.policy.reset(env_idx)
            self._assign[env_idx] = ("hist", entry)
        else:
            self._scripted[env_idx].reset()
            self._assign[env_idx] = ("scripted", self._scripted[env_idx])

    def assign_all(self) -> None:
        for i in range(self.n):
            self.on_episode_start(i)

    # --------------------------------------------------------------- inference
    def opponent_actions(self, enemy_obs: list[dict]) -> list[Action]:
        """One opponent Action per env, given each env's mirrored enemy obs."""
        out: list[Action | None] = [None] * self.n
        by_policy: dict[int, dict[int, dict]] = {}
        policies: dict[int, FrozenPolicy] = {}

        for i, (kind, ref) in enumerate(self._assign):
            if kind == "scripted":
                out[i] = ref.act(enemy_obs[i])
            else:
                pol = self.self_policy if kind == "self" else ref.policy
                key = id(pol)
                policies[key] = pol
                by_policy.setdefault(key, {})[i] = enemy_obs[i]

        for key, obs_by_env in by_policy.items():
            for i, act in policies[key].act_batch(obs_by_env).items():
                out[i] = act
        return out  # type: ignore[return-value]

    def record_result(self, env_idx: int, agent_outcome: int) -> None:
        kind, ref = self._assign[env_idx]
        if kind == "hist" and ref is not None:
            ref.games += 1
            if agent_outcome > 0:
                ref.losses_vs_learner += 1

    def summary(self) -> dict:
        kinds = [k for k, _ in self._assign]
        return {
            "league/pool_size": len(self.pool),
            "league/envs_vs_self": kinds.count("self"),
            "league/envs_vs_hist": kinds.count("hist"),
            "league/envs_vs_scripted": kinds.count("scripted"),
        }
