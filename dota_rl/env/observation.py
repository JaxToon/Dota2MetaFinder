"""Observation schema, normalisation rules and validation (spec 4.1 / 4.2).

Rules enforced:
  * every quantity normalised by a meaningful denominator -> [0, 1] or [-1, 1];
  * unavailable info is represented explicitly (present / visible flags), never
    silently fabricated;
  * fixed entity ordering (slot layout) so the set encoder does not depend on
    incidental iteration order;
  * unit-tested range + missing-data behaviour (see tests/test_observation.py).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import (
    ABILITY_FEAT_DIM,
    AGENT_SCALAR_DIM,
    ENTITY_FEAT_DIM,
    ITEM_FEAT_DIM,
    NUM_ABILITIES,
    NUM_ACTION_TYPES,
    NUM_ENTITY_SLOTS,
    NUM_ITEMS,
)

Observation = dict[str, Any]

MASK_KEYS = ("action_type", "entity", "ability", "item", "coordinate")


def empty_observation() -> Observation:
    """A well-formed all-zero observation (everything absent / masked)."""
    return {
        "agent_scalars": np.zeros(AGENT_SCALAR_DIM, dtype=np.float32),
        "abilities": np.zeros((NUM_ABILITIES, ABILITY_FEAT_DIM), dtype=np.float32),
        "ability_effect_ids": np.zeros(NUM_ABILITIES, dtype=np.int64),
        "items": np.zeros((NUM_ITEMS, ITEM_FEAT_DIM), dtype=np.float32),
        "entities": np.zeros((NUM_ENTITY_SLOTS, ENTITY_FEAT_DIM), dtype=np.float32),
        "entity_hero_ids": np.zeros(NUM_ENTITY_SLOTS, dtype=np.int64),
        "masks": {
            "action_type": np.zeros(NUM_ACTION_TYPES, dtype=bool),
            "entity": np.zeros(NUM_ENTITY_SLOTS, dtype=bool),
            "ability": np.zeros(NUM_ABILITIES, dtype=bool),
            "item": np.zeros(NUM_ITEMS, dtype=bool),
            "coordinate": np.array(False),
        },
    }


def encode_agent_scalars(
    hp_ratio: float,
    mana_ratio: float,
    level_norm: float,
    gold_norm: float,
    xp_norm: float,
    pos_x: float,
    pos_y: float,
    facing: float,
    sim_time_norm: float,
) -> np.ndarray:
    return np.array(
        [hp_ratio, mana_ratio, level_norm, gold_norm, xp_norm,
         pos_x, pos_y, facing, sim_time_norm],
        dtype=np.float32,
    )


_EXPECTED_SHAPES = {
    "agent_scalars": (AGENT_SCALAR_DIM,),
    "abilities": (NUM_ABILITIES, ABILITY_FEAT_DIM),
    "ability_effect_ids": (NUM_ABILITIES,),
    "items": (NUM_ITEMS, ITEM_FEAT_DIM),
    "entities": (NUM_ENTITY_SLOTS, ENTITY_FEAT_DIM),
    "entity_hero_ids": (NUM_ENTITY_SLOTS,),
}
_MASK_SHAPES = {
    "action_type": (NUM_ACTION_TYPES,),
    "entity": (NUM_ENTITY_SLOTS,),
    "ability": (NUM_ABILITIES,),
    "item": (NUM_ITEMS,),
    "coordinate": (),
}


def validate_observation(obs: Observation) -> None:
    """Raise ``ValueError`` if the observation breaks the contract."""
    for key, shape in _EXPECTED_SHAPES.items():
        if key not in obs:
            raise ValueError(f"observation missing key: {key}")
        arr = np.asarray(obs[key])
        if arr.shape != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key}: contains non-finite values")

    floats = np.concatenate(
        [np.asarray(obs["agent_scalars"]).reshape(-1),
         np.asarray(obs["abilities"]).reshape(-1),
         np.asarray(obs["items"]).reshape(-1),
         np.asarray(obs["entities"]).reshape(-1)]
    )
    if floats.size and (floats.min() < -1.0001 or floats.max() > 1.0001):
        raise ValueError(
            f"float features out of [-1, 1]: min={floats.min()}, max={floats.max()}"
        )

    masks = obs.get("masks")
    if masks is None:
        raise ValueError("observation missing 'masks'")
    for key, shape in _MASK_SHAPES.items():
        if key not in masks:
            raise ValueError(f"mask missing key: {key}")
        if np.asarray(masks[key]).shape != shape:
            raise ValueError(f"mask {key}: expected shape {shape}")

    # A structural invariant: if ATTACK / CAST / ITEM are allowed there must be a
    # legal sub-target, otherwise the sampled action is guaranteed invalid.
    at = np.asarray(masks["action_type"])
    from .protocol import ActionType

    if at[ActionType.ATTACK] and not np.any(masks["entity"]):
        raise ValueError("ATTACK allowed but no entity target is legal")
    if at[ActionType.CAST] and not (np.any(masks["ability"]) and np.any(masks["entity"])):
        raise ValueError("CAST allowed but no ability/target is legal")
    if at[ActionType.ITEM] and not np.any(masks["item"]):
        raise ValueError("ITEM allowed but no item is legal")
    if not np.any(at):
        raise ValueError("action_type mask is all-False (agent cannot act)")
