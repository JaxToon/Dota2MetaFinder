"""Invalid-action masking (spec 4.3 "the action mask is mandatory").

Without masking, PPO puts probability mass on impossible actions and the credit
assignment problem gets materially harder. Every head therefore receives a
boolean mask; a head with no legal option gets an all-True fallback so the
distribution stays well-defined (that head is never consumed anyway because the
owning action type is masked out).
"""
from __future__ import annotations

import numpy as np

from .protocol import (
    NUM_ABILITIES,
    NUM_ACTION_TYPES,
    NUM_ENTITY_SLOTS,
    NUM_ITEMS,
    ActionType,
)


def build_masks(
    entity_present: np.ndarray,
    entity_is_enemy: np.ndarray,
    entity_alive: np.ndarray,
    entity_in_range: np.ndarray,
    ability_ready: np.ndarray,
    item_ready: np.ndarray,
) -> dict:
    entity_present = np.asarray(entity_present, dtype=bool)
    entity_is_enemy = np.asarray(entity_is_enemy, dtype=bool)
    entity_alive = np.asarray(entity_alive, dtype=bool)
    entity_in_range = np.asarray(entity_in_range, dtype=bool)
    ability_ready = np.asarray(ability_ready, dtype=bool)
    item_ready = np.asarray(item_ready, dtype=bool)

    attackable = entity_present & entity_is_enemy & entity_alive & entity_in_range
    castable_target = entity_present & entity_is_enemy & entity_alive  # abilities have their own range check in-engine
    any_ability = bool(ability_ready.any())
    any_item = bool(item_ready.any())

    at = np.zeros(NUM_ACTION_TYPES, dtype=bool)
    at[ActionType.STOP] = True
    at[ActionType.MOVE] = True
    at[ActionType.ATTACK] = bool(attackable.any())
    at[ActionType.CAST] = any_ability and bool(castable_target.any())
    at[ActionType.ITEM] = any_item

    entity_mask = attackable | castable_target
    if not entity_mask.any():
        entity_mask = np.ones(NUM_ENTITY_SLOTS, dtype=bool)  # fallback, unused

    ability_mask = ability_ready.copy()
    if not ability_mask.any():
        ability_mask = np.ones(NUM_ABILITIES, dtype=bool)

    item_mask = item_ready.copy()
    if not item_mask.any():
        item_mask = np.ones(NUM_ITEMS, dtype=bool)

    return {
        "action_type": at,
        "entity": entity_mask,
        "ability": ability_mask,
        "item": item_mask,
        "coordinate": np.array(True),
    }
