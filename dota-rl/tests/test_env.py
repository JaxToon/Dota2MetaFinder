"""Spec 13: reset/step invariants, deterministic seed smoke test, termination."""
import numpy as np

from dota_rl.env.action import Action
from dota_rl.env.mock_env import MockDotaEnv
from dota_rl.env.protocol import ActionType


def _greedy_attack_or_move(obs, rng):
    m = obs["masks"]
    if m["action_type"][ActionType.ATTACK]:
        s = int(np.flatnonzero(m["entity"])[0])
        return Action.from_heads(ActionType.ATTACK, s, -1, -1, obs["agent_scalars"][5:7])
    return Action.from_heads(ActionType.MOVE, -1, -1, -1, np.array([0.5, 0.5]))


def test_deterministic_same_seed():
    def run(seed):
        env = MockDotaEnv(phase="F1", seed=seed)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(0)
        total = 0.0
        for _ in range(200):
            obs, r, term, trunc, _ = env.step(_greedy_attack_or_move(obs, rng))
            total += r
            if term or trunc:
                break
        return total

    assert run(123) == run(123)

    # different seeds must produce a different world (creep layout)
    e1 = MockDotaEnv(phase="F1", seed=123); e1.reset(seed=123)
    e2 = MockDotaEnv(phase="F1", seed=124); e2.reset(seed=124)
    p1 = np.array([c.pos for c in e1.creeps])
    p2 = np.array([c.pos for c in e2.creeps])
    assert not np.allclose(p1, p2)


def test_episode_terminates_and_reset_required():
    env = MockDotaEnv(phase="F1", seed=2, max_steps=50)
    obs, _ = env.reset(seed=2)
    done = False
    steps = 0
    while not done and steps < 100:
        obs, r, term, trunc, info = env.step(
            Action.from_heads(ActionType.MOVE, -1, -1, -1, np.array([0.9, 0.9])))
        done = term or trunc
        steps += 1
    assert done and steps <= 50
    try:
        env.step(Action.from_heads(ActionType.STOP, -1, -1, -1, np.zeros(2)))
        assert False, "step after done should raise"
    except RuntimeError:
        pass


def test_last_hitting_creeps_earns_gold():
    env = MockDotaEnv(phase="F0", seed=7)
    obs, _ = env.reset(seed=7)
    start_nw = env.agent.gold
    for _ in range(400):
        m = obs["masks"]
        if m["action_type"][ActionType.ATTACK]:
            s = int(np.flatnonzero(m["entity"])[0])
            a = Action.from_heads(ActionType.ATTACK, s, -1, -1, obs["agent_scalars"][5:7])
        else:
            # walk to lane centre where creeps are
            a = Action.from_heads(ActionType.MOVE, -1, -1, -1, np.array([0.5, 0.5]))
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    assert env.agent.gold > start_nw
