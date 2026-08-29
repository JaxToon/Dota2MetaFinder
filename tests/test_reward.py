"""Spec 6: normalisation, clipping, terminal objective preserved."""
import numpy as np

from dota_rl.env.reward import RewardConfig, RewardInputs, RewardShaper


def test_large_networth_delta_is_bounded_by_clip():
    sh = RewardShaper(RewardConfig(clip=5.0))
    for _ in range(50):
        sh.step(RewardInputs(d_net_worth=np.random.uniform(-20, 20)))
    r, _ = sh.step(RewardInputs(d_net_worth=1e6))
    assert -5.0 <= r <= 5.0


def test_terminal_reward_dominates_and_signs_correctly():
    sh = RewardShaper(RewardConfig(terminal_win=10.0, clip=10.0))
    win, _ = sh.step(RewardInputs(terminal=1))
    loss, _ = sh.step(RewardInputs(terminal=-1))
    assert win > 5.0 and loss < -5.0


def test_disabled_component_is_ignored():
    cfg = RewardConfig()
    cfg.enabled["str"] = False
    sh = RewardShaper(cfg)
    r, comps = sh.step(RewardInputs(d_structure=1.0))
    assert comps["str"] != 0.0          # component still computed/logged
    assert abs(r) < 1e-6                 # but not summed into reward


def test_team_blend_interpolates():
    cfg = RewardConfig(team_blend_tau=1.0, normalize=False)
    sh = RewardShaper(cfg)
    r, _ = sh.step(RewardInputs(d_net_worth=0.0, team_d_net_worth=2.0))
    assert r > 0
