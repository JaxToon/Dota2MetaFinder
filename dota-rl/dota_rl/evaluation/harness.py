"""Evaluation harness (spec 9). Fixed seeds + frozen opponent = regression set;
a separate rotating seed range reduces benchmark overfitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..agents.batch import obs_to_tensors
from ..agents.policy import ActorCritic
from ..env.action import Action
from ..env.mock_env import MockDotaEnv
from ..env.reward import RewardConfig
from .baselines import RandomPolicy, ScriptedLaner
from .metrics import summarize_episodes


class TorchPolicyRunner:
    def __init__(self, policy: ActorCritic, device="cpu", deterministic=True):
        self.policy = policy
        self.device = device
        self.deterministic = deterministic
        self._state = None

    def reset(self):
        self._state = self.policy.core.initial_state(1, self.device)

    def act(self, obs) -> Action:
        obs_t = obs_to_tensors([obs], self.device)
        actions, _, _, self._state = self.policy.act(
            obs_t, self._state, deterministic=self.deterministic
        )
        a = {k: v.squeeze(0).cpu().numpy() for k, v in actions.items()}
        return Action.from_heads(a["type"], a["entity"], a["ability"], a["item"], a["coord"])


@dataclass
class EvalConfig:
    phase: str = "F1"
    games_per_opponent: int = 30
    regression_seeds: tuple[int, ...] = tuple(range(9000, 9030))
    rotating_seed_base: int = 20000
    deterministic: bool = True


def _play_episode(env: MockDotaEnv, runner, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    if hasattr(runner, "reset"):
        runner.reset()
    info = {}
    done = False
    while not done:
        action = runner.act(obs)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info


def evaluate_policy(policy: ActorCritic, cfg: EvalConfig, *, device="cpu",
                    reward_cfg: RewardConfig | None = None, rotating: bool = False) -> dict:
    policy.eval()
    seeds = (list(range(cfg.rotating_seed_base, cfg.rotating_seed_base + cfg.games_per_opponent))
             if rotating else list(cfg.regression_seeds)[: cfg.games_per_opponent])
    runner = TorchPolicyRunner(policy, device, cfg.deterministic)
    infos = []
    for s in seeds:
        env = MockDotaEnv(phase=cfg.phase, reward_cfg=reward_cfg or RewardConfig(), seed=s)
        infos.append(_play_episode(env, runner, s))
    out = {"vs_scripted_bot": summarize_episodes(infos)}

    # sanity: agent should crush a random policy
    rnd_infos = []
    for s in seeds:
        env = MockDotaEnv(phase=cfg.phase, reward_cfg=reward_cfg or RewardConfig(), seed=s)
        rnd_infos.append(_play_episode(env, RandomPolicy(seed=s), s))
    out["random_vs_scripted_bot"] = summarize_episodes(rnd_infos)
    return out
