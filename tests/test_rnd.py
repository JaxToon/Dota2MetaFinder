"""Spec 7 E3-E4: RND intrinsic reward + predictor learning + anneal."""
import torch

from dota_rl.agents.policy import ActorCritic
from dota_rl.agents.rnd import RND, coef_schedule
from dota_rl.env.mock_env import MockDotaEnv
from dota_rl.env.reward import RewardConfig
from dota_rl.training.rollout import RolloutCollector


def _batch():
    torch.manual_seed(0)
    pol = ActorCritic(encoder_dim=32, lstm_hidden=32)
    coll = RolloutCollector(
        lambda i: MockDotaEnv(phase="F1", reward_cfg=RewardConfig(), seed=10 + i, max_steps=80),
        num_envs=4, horizon=16)
    return coll.collect(pol)


def test_intrinsic_reward_shape_and_finite():
    b = _batch()
    rnd = RND(device="cpu")
    r_int = rnd.intrinsic_reward(b["obs"])
    assert r_int.shape == b["rewards"].shape
    assert torch.isfinite(r_int).all()


def test_predictor_error_decreases_on_repeated_states():
    b = _batch()
    rnd = RND(lr=1e-3, device="cpu")
    first = rnd.update(b["obs"], epochs=1)
    for _ in range(30):
        rnd.update(b["obs"], epochs=1)
    last = rnd.update(b["obs"], epochs=1)
    assert last < first * 0.7            # familiar states -> lower novelty


def test_coef_anneal():
    assert coef_schedule(0.5, 0.0, 0.0) == 0.5
    assert coef_schedule(0.5, 0.0, 1.0) == 0.0
    assert abs(coef_schedule(0.5, 0.0, 0.5) - 0.25) < 1e-9
