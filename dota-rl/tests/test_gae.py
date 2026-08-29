"""Spec 10 step 6: GAE with terminal / bootstrap masks."""
import torch

from dota_rl.training.gae import compute_gae


def test_gae_matches_discounted_return_when_lambda_one_no_terminal():
    T, B = 5, 2
    rewards = torch.ones(T, B)
    values = torch.zeros(T, B)
    term = torch.zeros(T, B, dtype=torch.bool)
    trunc = torch.zeros(T, B, dtype=torch.bool)
    last_v = torch.zeros(B)
    adv, ret = compute_gae(rewards, values, term, trunc, last_v, gamma=1.0, lam=1.0)
    # return at t = sum of future rewards
    expected = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]).unsqueeze(1).repeat(1, B)
    assert torch.allclose(ret, expected)


def test_terminal_cuts_bootstrap():
    T, B = 3, 1
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.0], [0.0], [10.0]])
    term = torch.tensor([[False], [True], [False]])
    trunc = torch.zeros(T, B, dtype=torch.bool)
    last_v = torch.tensor([100.0])
    adv, ret = compute_gae(rewards, values, term, trunc, last_v, gamma=0.99, lam=0.95)
    # step 1 is terminal: delta = r - V(1) = 1.0, and recursion stops
    assert abs(adv[1, 0].item() - 1.0) < 1e-6


def test_truncation_bootstraps_but_stops_recursion():
    T, B = 2, 1
    rewards = torch.tensor([[0.0], [0.0]])
    values = torch.tensor([[0.0], [1.0]])
    term = torch.zeros(T, B, dtype=torch.bool)
    trunc = torch.tensor([[False], [True]])
    last_v = torch.tensor([5.0])
    adv, ret = compute_gae(rewards, values, term, trunc, last_v, gamma=1.0, lam=1.0)
    # last step truncated: delta = 0 + 1*5 - 1 = 4
    assert abs(adv[1, 0].item() - 4.0) < 1e-6
