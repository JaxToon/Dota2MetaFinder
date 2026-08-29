"""Spec 4.3: action masking is mandatory and must be internally consistent."""
import numpy as np

from dota_rl.env.masks import build_masks
from dota_rl.env.protocol import NUM_ABILITIES, NUM_ITEMS, NUM_ENTITY_SLOTS, ActionType


def test_no_targets_means_only_move_and_stop():
    m = build_masks(
        entity_present=np.zeros(NUM_ENTITY_SLOTS, bool),
        entity_is_enemy=np.zeros(NUM_ENTITY_SLOTS, bool),
        entity_alive=np.zeros(NUM_ENTITY_SLOTS, bool),
        entity_in_range=np.zeros(NUM_ENTITY_SLOTS, bool),
        ability_ready=np.zeros(NUM_ABILITIES, bool),
        item_ready=np.zeros(NUM_ITEMS, bool),
    )
    assert m["action_type"][ActionType.STOP]
    assert m["action_type"][ActionType.MOVE]
    assert not m["action_type"][ActionType.ATTACK]
    assert not m["action_type"][ActionType.CAST]
    assert not m["action_type"][ActionType.ITEM]


def test_attack_requires_in_range_enemy():
    present = np.zeros(NUM_ENTITY_SLOTS, bool); present[0] = True
    enemy = present.copy(); alive = present.copy()
    in_range = np.zeros(NUM_ENTITY_SLOTS, bool)
    m = build_masks(present, enemy, alive, in_range,
                    np.zeros(NUM_ABILITIES, bool), np.zeros(NUM_ITEMS, bool))
    assert not m["action_type"][ActionType.ATTACK]
    in_range[0] = True
    m = build_masks(present, enemy, alive, in_range,
                    np.zeros(NUM_ABILITIES, bool), np.zeros(NUM_ITEMS, bool))
    assert m["action_type"][ActionType.ATTACK]
    assert m["entity"][0]


def test_cast_needs_ability_and_target():
    present = np.zeros(NUM_ENTITY_SLOTS, bool); present[0] = True
    ready = np.zeros(NUM_ABILITIES, bool); ready[1] = True
    m = build_masks(present, present.copy(), present.copy(), present.copy(),
                    ready, np.zeros(NUM_ITEMS, bool))
    assert m["action_type"][ActionType.CAST]
    assert m["ability"][1] and not m["ability"][0]
