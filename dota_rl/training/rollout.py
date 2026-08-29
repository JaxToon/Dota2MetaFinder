"""Parallel rollout collection (spec 3.1 Rollout Manager, 10 step 5).

Runs N mock envs for T steps with the recurrent policy, carrying LSTM state and
resetting it on episode boundaries. Emits (T, B, ...) tensors for the PPO update.
"""
from __future__ import annotations

import numpy as np
import torch

from ..agents.batch import obs_to_tensors, stack_time
from ..env.action import Action
from ..env.mock_env import MockDotaEnv


class RolloutCollector:
    def __init__(self, make_env, num_envs: int, horizon: int, device="cpu", league=None):
        self.envs: list[MockDotaEnv] = [make_env(i) for i in range(num_envs)]
        self.n = num_envs
        self.T = horizon
        self.device = device
        self.league = league          # dota_rl.training.league.League | None
        self._obs = [e.reset(seed=1000 + i)[0] for i, e in enumerate(self.envs)]
        if self.league is not None:
            self.league.assign_all()
        self._state = None
        self.ep_returns: list[float] = []
        self.ep_lengths: list[int] = []
        self.ep_outcomes: list[int] = []
        self._run_ret = np.zeros(num_envs)
        self._run_len = np.zeros(num_envs, dtype=int)

    def collect(self, policy):
        policy.eval()
        if self._state is None:
            self._state = policy.core.initial_state(self.n, self.device)
        state0 = tuple(s.detach().clone() for s in self._state)

        obs_steps, act_steps = [], []
        logp_b, val_b, rew_b, term_b, trunc_b, done_b = [], [], [], [], [], []

        for _ in range(self.T):
            obs_t = obs_to_tensors(self._obs, self.device)
            actions, logp, value, self._state = policy.act(obs_t, self._state)

            a_type = actions["type"].cpu().numpy()
            ent = actions["entity"].cpu().numpy()
            ab = actions["ability"].cpu().numpy()
            it = actions["item"].cpu().numpy()
            coord = actions["coord"].cpu().numpy()

            rewards = np.zeros(self.n, dtype=np.float32)
            terminated = np.zeros(self.n, dtype=bool)
            truncated = np.zeros(self.n, dtype=bool)
            next_obs: list = [None] * self.n

            opp_actions = None
            if self.league is not None:
                opp_actions = self.league.opponent_actions(
                    [env.enemy_observation() for env in self.envs]
                )

            for i, env in enumerate(self.envs):
                action = Action.from_heads(a_type[i], ent[i], ab[i], it[i], coord[i])
                oa = opp_actions[i] if opp_actions is not None else None
                o, r, done_term, done_trunc, info = env.step(action, opponent_action=oa)
                rewards[i] = r
                terminated[i] = done_term
                truncated[i] = done_trunc
                self._run_ret[i] += r
                self._run_len[i] += 1
                if done_term or done_trunc:
                    self.ep_returns.append(float(self._run_ret[i]))
                    self.ep_lengths.append(int(self._run_len[i]))
                    self.ep_outcomes.append(int(info["outcome"]))
                    self._run_ret[i] = 0.0
                    self._run_len[i] = 0
                    if self.league is not None:
                        self.league.record_result(i, int(info["outcome"]))
                    o = env.reset()[0]
                    if self.league is not None:
                        self.league.on_episode_start(i)
                next_obs[i] = o

            obs_steps.append(obs_t)
            act_steps.append({k: v.detach() for k, v in actions.items()})
            logp_b.append(logp.detach())
            val_b.append(value.detach())
            rew_b.append(torch.as_tensor(rewards, device=self.device))
            term_b.append(torch.as_tensor(terminated, device=self.device))
            trunc_b.append(torch.as_tensor(truncated, device=self.device))
            done_b.append(torch.as_tensor(terminated | truncated, device=self.device))
            self._obs = next_obs

            # reset the live LSTM state on episodes that just ended
            done_mask = torch.as_tensor(terminated | truncated, device=self.device).float().unsqueeze(-1)
            self._state = tuple(s * (1.0 - done_mask) for s in self._state)

        with torch.no_grad():
            last_value = policy.value_only(obs_to_tensors(self._obs, self.device), self._state)

        return {
            "obs": stack_time(obs_steps),
            "actions": {k: torch.stack([a[k] for a in act_steps], 0) for k in act_steps[0]},
            "logp": torch.stack(logp_b, 0),
            "values": torch.stack(val_b, 0),
            "rewards": torch.stack(rew_b, 0).float(),
            "terminated": torch.stack(term_b, 0).bool(),
            "truncated": torch.stack(trunc_b, 0).bool(),
            "dones": torch.stack(done_b, 0).bool(),
            "last_value": last_value,
            "state0": state0,
        }

    def drain_episode_stats(self):
        out = {
            "episodes": len(self.ep_returns),
            "return_mean": float(np.mean(self.ep_returns)) if self.ep_returns else 0.0,
            "length_mean": float(np.mean(self.ep_lengths)) if self.ep_lengths else 0.0,
            "winrate": float(np.mean([o > 0 for o in self.ep_outcomes])) if self.ep_outcomes else 0.0,
        }
        self.ep_returns.clear()
        self.ep_lengths.clear()
        self.ep_outcomes.clear()
        return out
