"""Real Dota 2 sandbox adapter (spec 3.1, ticket #10, milestone M0).

Implements the SAME ``reset()/step()`` contract as ``MockDotaEnv`` -- so the
learner, rollout, PPO and evaluation code are unchanged -- on top of a transport
bridge to the in-game Lua bot (``dota_rl/transport/zmq_bridge.py``).

Status: the protocol + parsing + contract validation are implemented and tested
against ``InProcessBridge``. Two things gate real use (see docs/M0_real_adapter.md):
  1. run ``scripts/api_probe.lua`` in a custom lobby and confirm which fields the
     public Bot API exposes (fogged enemy HP, enemy cooldowns, ...);
  2. finish ``dota2/scripts/vscripts/bots/bot_generic.lua`` to emit the wire
     observation this parser expects, and apply actions via ``rl_action_handler``.
If the probe shows too many fields are unavailable, stay on ``MockDotaEnv``.
"""
from __future__ import annotations

import numpy as np

from ..transport.zmq_bridge import BridgeError, InProcessBridge
from .action import Action
from .observation import empty_observation, encode_agent_scalars, validate_observation
from .protocol import (
    ACTION_SCHEMA_VERSION,
    NUM_ABILITIES,
    NUM_ENTITY_SLOTS,
    NUM_ITEMS,
    ActionType,
)


class DotaAdapterNotAvailable(RuntimeError):
    pass


def parse_wire_observation(msg: dict) -> tuple[dict, float, bool, bool, dict]:
    """Map a wire message (schemas.proto ``StepResult``) to the Observation
    contract. Field names below match the proto; adjust only if the Lua probe
    shows the public API names differ.
    """
    w = msg.get("observation", msg)
    obs = empty_observation()

    ag = w.get("agent", {})
    pos = ag.get("pos_norm", [0.0, 0.0])
    obs["agent_scalars"] = encode_agent_scalars(
        hp_ratio=float(np.clip(ag.get("hp_ratio", 0.0), 0, 1)),
        mana_ratio=float(np.clip(ag.get("mana_ratio", 0.0), 0, 1)),
        level_norm=float(np.clip(ag.get("level_norm", 0.0), 0, 1)),
        gold_norm=float(np.clip(ag.get("gold_norm", 0.0), 0, 1)),
        xp_norm=float(np.clip(ag.get("xp_norm", 0.0), 0, 1)),
        pos_x=float(np.clip(pos[0], 0, 1)), pos_y=float(np.clip(pos[1], 0, 1)),
        facing=float(ag.get("facing", 0.0)) % 1.0,
        sim_time_norm=float(np.clip(w.get("sim_time", 0.0) / 3600.0, 0, 1)),
    )

    for i, ab in enumerate((w.get("abilities") or [])[:NUM_ABILITIES]):
        obs["abilities"][i] = [
            float(bool(ab.get("available", False))),
            float(np.clip(ab.get("cooldown_ratio", 0.0), 0, 1)),
            float(bool(ab.get("mana_ok", False))),
            (i + 1) / (NUM_ABILITIES + 1),
        ]
        obs["ability_effect_ids"][i] = int(ab.get("effect_id", i + 1))

    for i in range(NUM_ITEMS):
        it = (w.get("inventory") or w.get("items") or [])
        present = i < len(it)
        ready = present and bool(it[i].get("ready", it[i].get("available", False)))
        obs["items"][i] = [float(present), float(ready), (i + 1) / (NUM_ITEMS + 1)]

    for e in (w.get("entities") or [])[:NUM_ENTITY_SLOTS]:
        s = int(e.get("slot", -1))
        if not (0 <= s < NUM_ENTITY_SLOTS):
            continue
        p = e.get("pos_norm", [0.0, 0.0])
        obs["entities"][s] = [
            1.0, float(bool(e.get("is_enemy", False))), float(bool(e.get("is_creep", False))),
            float(bool(e.get("visible", False))),
            float(np.clip(p[0], 0, 1)), float(np.clip(p[1], 0, 1)),
            float(np.clip(e.get("hp_ratio", 0.0), 0, 1)),
            float(np.clip(e.get("dist_norm", 1.0), 0, 1)),
        ]
        obs["entity_hero_ids"][s] = int(e.get("hero_id", 0))

    m = w.get("masks", {})
    obs["masks"] = {
        "action_type": _mask(m.get("action_type"), len(ActionType), default_first=True),
        "entity": _mask(m.get("entity"), NUM_ENTITY_SLOTS),
        "ability": _mask(m.get("ability"), NUM_ABILITIES),
        "item": _mask(m.get("item"), NUM_ITEMS),
        "coordinate": np.array(bool(m.get("coordinate", True))),
    }
    # STOP is always legal; guarantee a non-empty action_type mask
    obs["masks"]["action_type"][ActionType.STOP] = True

    reward = float(msg.get("reward", 0.0))
    terminated = bool(msg.get("terminated", False))
    truncated = bool(msg.get("truncated", False))
    info = {"episode_id": w.get("episode_id"), "patch_id": w.get("patch_id"),
            "schema_version": w.get("schema_version"), "t": int(w.get("step_id", 0)),
            "outcome": int(msg.get("outcome", 0))}
    return obs, reward, terminated, truncated, info


def _mask(v, n: int, default_first: bool = False) -> np.ndarray:
    if v is None:
        out = np.zeros(n, dtype=bool)
        if default_first:
            out[0] = True
        return out
    a = np.zeros(n, dtype=bool)
    for i, x in enumerate(v[:n]):
        a[i] = bool(x)
    if not a.any():
        a[0] = True
    return a


def action_to_wire(a: Action) -> dict:
    return {
        "action_type": int(a.action_type),
        "target_entity_slot": int(a.target_entity_slot),
        "target_xy": list(a.target_xy) if a.target_xy is not None else None,
        "ability_slot": int(a.ability_slot),
        "item_slot": int(a.item_slot),
        "schema_version": ACTION_SCHEMA_VERSION,
    }


class DotaSandboxEnv:
    """Gym-like env over a transport bridge. Same contract as MockDotaEnv."""

    def __init__(self, bridge=None, validate: bool = True):
        if bridge is None:
            raise DotaAdapterNotAvailable(
                "pass a bridge (ZmqBridge once the Lua side is wired, or "
                "InProcessBridge for tests). See docs/M0_real_adapter.md.")
        self.bridge = bridge
        self.validate = validate
        self._pending: dict | None = None

    def reset(self, seed: int | None = None):
        msg = self.bridge.recv_observation()
        obs, _, _, _, info = parse_wire_observation(msg)
        if self.validate:
            validate_observation(obs)
        return obs, info

    def step(self, action: Action, opponent_action: Action | None = None):
        self.bridge.send_action(action_to_wire(action))
        msg = self.bridge.recv_observation()
        obs, reward, terminated, truncated, info = parse_wire_observation(msg)
        if self.validate:
            validate_observation(obs)
        return obs, reward, terminated, truncated, info

    def enemy_observation(self):  # opponent is the real enemy team in-engine
        raise DotaAdapterNotAvailable("self-play mirroring is a mock-env feature")
