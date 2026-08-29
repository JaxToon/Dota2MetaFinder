"""Spec 4.2 / testing strategy: normalisation bounds + missing-data behaviour."""
import numpy as np
import pytest

from dota_rl.env.mock_env import MockDotaEnv
from dota_rl.env.observation import empty_observation, validate_observation


def test_empty_observation_is_valid_shape():
    obs = empty_observation()
    # all-masked action_type is illegal by contract -> validate must reject it
    with pytest.raises(ValueError):
        validate_observation(obs)


@pytest.mark.parametrize("phase", ["F0", "F1"])
def test_mock_env_observations_stay_in_bounds(phase):
    env = MockDotaEnv(phase=phase, seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        validate_observation(obs)
        floats = np.concatenate([
            obs["agent_scalars"].ravel(), obs["abilities"].ravel(),
            obs["items"].ravel(), obs["entities"].ravel()])
        assert np.isfinite(floats).all()
        assert floats.min() >= -1.0001 and floats.max() <= 1.0001
        from dota_rl.env.action import Action
        from dota_rl.env.protocol import ActionType
        m = obs["masks"]
        t = int(rng.choice(np.flatnonzero(m["action_type"])))
        ent = int(np.flatnonzero(m["entity"])[0]) if m["entity"].any() else -1
        ab = int(np.flatnonzero(m["ability"])[0]) if m["ability"].any() else -1
        it = int(np.flatnonzero(m["item"])[0]) if m["item"].any() else -1
        obs, r, term, trunc, _ = env.step(
            Action.from_heads(t, ent, ab, it, rng.uniform(0, 1, 2)))
        assert np.isfinite(r)
        if term or trunc:
            obs, _ = env.reset()


def test_missing_enemy_in_f0():
    env = MockDotaEnv(phase="F0", seed=1)
    obs, _ = env.reset(seed=1)
    # slot 0 (enemy hero) must be absent in F0
    assert obs["entities"][0, 0] == 0.0
    assert obs["entity_hero_ids"][0] == 0
