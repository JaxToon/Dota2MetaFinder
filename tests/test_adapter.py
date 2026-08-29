"""Spec 3.1 / M0: the real-Dota adapter speaks the wire protocol and yields a
contract-valid Observation -- verified against an in-process fake game loop.
"""
import threading

import numpy as np

from dota_rl.env.action import Action
from dota_rl.env.dota_adapter import DotaSandboxEnv, action_to_wire, parse_wire_observation
from dota_rl.env.observation import validate_observation
from dota_rl.env.protocol import OBS_SCHEMA_VERSION, ActionType
from dota_rl.transport.zmq_bridge import BridgeConfig, InProcessBridge


def _wire_obs(step_id: int, terminated=False):
    return {
        "type": "obs", "step_id": step_id, "schema_version": OBS_SCHEMA_VERSION,
        "reward": 0.1, "terminated": terminated, "truncated": False, "outcome": 0,
        "observation": {
            "episode_id": "abc", "patch_id": "live", "step_id": step_id, "sim_time": step_id,
            "agent": {"hp_ratio": 0.8, "mana_ratio": 0.5, "level_norm": 0.2,
                      "gold_norm": 0.1, "xp_norm": 0.1, "pos_norm": [0.3, 0.4], "facing": 0.1},
            "abilities": [{"available": True, "cooldown_ratio": 0.0, "mana_ok": True, "effect_id": 1}],
            "inventory": [{"ready": True}],
            "entities": [
                {"slot": 0, "is_enemy": True, "is_creep": False, "visible": True,
                 "pos_norm": [0.6, 0.6], "hp_ratio": 0.9, "dist_norm": 0.3, "hero_id": 1},
                {"slot": 5, "is_enemy": True, "visible": True, "pos_norm": [0.9, 0.9],
                 "hp_ratio": 1.0, "dist_norm": 0.8, "hero_id": 3},
                {"slot": 6, "is_enemy": False, "visible": True, "pos_norm": [0.1, 0.1],
                 "hp_ratio": 1.0, "dist_norm": 0.9, "hero_id": 3},
            ],
            "masks": {"action_type": [1, 1, 1, 0, 0], "entity": [1, 0, 0, 0, 0, 1, 0, 0],
                      "ability": [1, 0, 0], "item": [1, 0, 0, 0, 0, 0], "coordinate": True},
        },
    }


def test_parse_wire_observation_is_contract_valid():
    obs, r, term, trunc, info = parse_wire_observation(_wire_obs(1))
    validate_observation(obs)
    assert r == 0.1 and not term and not trunc
    assert obs["masks"]["action_type"][ActionType.STOP]
    assert obs["entities"][0][1] == 1.0        # slot 0 is enemy


def test_action_round_trips_to_wire():
    a = Action.from_heads(ActionType.CAST, 0, 1, -1, np.array([0.5, 0.5]))
    w = action_to_wire(a)
    assert w["action_type"] == int(ActionType.CAST) and w["ability_slot"] == 1


def test_adapter_reset_step_over_inprocess_bridge():
    bridge = InProcessBridge(BridgeConfig(recv_timeout_ms=1000, strict_order=False))
    N = 5

    def fake_game():
        bridge.game_send_obs(_wire_obs(0))
        for sid in range(1, N + 1):
            act = bridge.game_recv_action(timeout=2.0)
            assert act["type"] == "action"
            bridge.game_send_obs(_wire_obs(sid, terminated=(sid == N)))

    t = threading.Thread(target=fake_game, daemon=True)
    t.start()

    env = DotaSandboxEnv(bridge=bridge)
    obs, info = env.reset()
    validate_observation(obs)
    done = False
    steps = 0
    while not done:
        obs, r, term, trunc, info = env.step(
            Action.from_heads(ActionType.MOVE, -1, -1, -1, np.array([0.5, 0.5])))
        done = term or trunc
        steps += 1
    assert steps == N
    t.join(timeout=2.0)
