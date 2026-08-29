"""Spec section 7: self-play opponent + historical league pool."""
import numpy as np
import torch

from dota_rl.agents.policy import ActorCritic
from dota_rl.env.action import Action
from dota_rl.env.mock_env import MockDotaEnv
from dota_rl.env.observation import validate_observation
from dota_rl.evaluation.baselines import ScriptedLaner
from dota_rl.training.league import FrozenPolicy, League, LeagueConfig

MC = {"encoder_dim": 32, "lstm_hidden": 32}


def _policy():
    torch.manual_seed(0)
    return ActorCritic(encoder_dim=32, lstm_hidden=32)


def test_mirrored_enemy_observation_is_contract_valid():
    env = MockDotaEnv(phase="F1", seed=3)
    env.reset(seed=3)
    for _ in range(40):
        eo = env.enemy_observation()
        validate_observation(eo)
        env.step(Action.from_heads(1, -1, -1, -1, np.array([0.5, 0.5])))


def test_external_opponent_actually_plays():
    opp = ScriptedLaner(seed=1)
    env = MockDotaEnv(phase="F1", seed=5, opponent=lambda eo: opp.act(eo))
    obs, _ = env.reset(seed=5)
    ag = ScriptedLaner(seed=2)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(ag.act(obs))
        done = term or trunc
    # a farming opponent must have accumulated net worth
    assert info["enemy_net_worth"] > 200


def test_frozen_policy_batched_inference():
    fp = FrozenPolicy(_policy().state_dict(), 32, 32)
    env = MockDotaEnv(phase="F1", seed=1)
    env.reset(seed=1)
    obs_by_env = {0: env.enemy_observation(), 2: env.enemy_observation()}
    acts = fp.act_batch(obs_by_env)
    assert set(acts) == {0, 2}
    assert all(isinstance(a, Action) for a in acts.values())
    # recurrent state is retained per env index
    assert 0 in fp._state and 2 in fp._state


def test_league_assignment_ratios_and_pool_cap():
    cfg = LeagueConfig(enabled=True, ratio_self=0.5, ratio_history=0.3,
                       ratio_scripted=0.2, pool_capacity=3, snapshot_every=1)
    lg = League(cfg, MC, num_envs=64, device="cpu", seed=0)
    pol = _policy()
    lg.refresh_self(pol)
    for it in range(1, 8):
        lg.maybe_snapshot(pol, it)
    assert len(lg.pool) == 3                      # capacity enforced
    lg.assign_all()
    kinds = [k for k, _ in lg._assign]
    assert kinds.count("self") > 0
    assert kinds.count("hist") > 0
    assert kinds.count("scripted") > 0


def test_league_self_play_episode_completes():
    cfg = LeagueConfig(enabled=True, snapshot_every=1)
    lg = League(cfg, MC, num_envs=4, device="cpu", seed=0)
    pol = _policy()
    lg.refresh_self(pol)
    lg.assign_all()
    env = MockDotaEnv(phase="F2", seed=7, max_steps=150)
    obs, _ = env.reset(seed=7)
    from dota_rl.evaluation.harness import TorchPolicyRunner
    runner = TorchPolicyRunner(pol, deterministic=False)
    runner.reset()
    done = False
    steps = 0
    while not done:
        opp = lg.opponent_actions([env.enemy_observation()] + [env.enemy_observation()] * 3)[0]
        obs, r, term, trunc, info = env.step(runner.act(obs), opponent_action=opp)
        done = term or trunc
        steps += 1
    assert steps <= 150 and info["outcome"] in (-1, 0, 1)
