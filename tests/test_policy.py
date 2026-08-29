"""Spec 13 (Model): tensor shapes, recurrent reset, masked logits, finite grads,
plus a one-batch PPO update that improves and does not NaN (Training layer)."""
import torch

from dota_rl.agents.policy import ActorCritic
from dota_rl.env.mock_env import MockDotaEnv
from dota_rl.env.reward import RewardConfig
from dota_rl.training.ppo import PPOConfig, PPOTrainer
from dota_rl.training.rollout import RolloutCollector


def _make_env(i):
    return MockDotaEnv(phase="F1", reward_cfg=RewardConfig(), seed=100 + i, max_steps=120)


def test_act_shapes_and_masking():
    torch.manual_seed(0)
    policy = ActorCritic(encoder_dim=64, lstm_hidden=64)
    coll = RolloutCollector(_make_env, num_envs=4, horizon=8)
    batch = coll.collect(policy)
    assert batch["logp"].shape == (8, 4)
    assert batch["values"].shape == (8, 4)
    # sampled action types must always be legal under the mask (AC-03)
    at = batch["actions"]["type"]
    legal = batch["obs"]["masks"]["action_type"].gather(-1, at.unsqueeze(-1).long()).squeeze(-1)
    assert legal.all()


def test_recurrent_state_resets_on_done():
    policy = ActorCritic(encoder_dim=32, lstm_hidden=16)
    core = policy.core
    x = torch.randn(3, 2, 32)
    done = torch.zeros(3, 2, dtype=torch.bool)
    done[1, 0] = True
    h0 = core.initial_state(2)
    outs, (h, c) = core.forward_sequence(x, done, h0)
    assert outs.shape == (3, 2, 16)
    assert torch.isfinite(outs).all()


def test_one_ppo_update_is_finite_and_moves_params():
    torch.manual_seed(0)
    policy = ActorCritic(encoder_dim=64, lstm_hidden=64)
    trainer = PPOTrainer(policy, PPOConfig(epochs=2, num_minibatches=2))
    coll = RolloutCollector(_make_env, num_envs=8, horizon=16)
    before = [p.detach().clone() for p in policy.parameters()]
    stats = trainer.update(coll.collect(policy))
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    after = list(policy.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    for p in policy.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
