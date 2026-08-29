"""Environment contract: schema versions and shared dimensions.

This is the single source of truth for tensor shapes. Every observation and
action carries ``schema_version`` / ``patch_id`` so a patch that changes tensor
semantics forces a new contract version (spec sections 4, 12).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

OBS_SCHEMA_VERSION = 1
ACTION_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Fixed dimensions for the compact MVP observation (spec 4.1 / 5.1, kept small
# per blueprint section 3A revision note: ~500-2000 floats, not ~20k).
# ---------------------------------------------------------------------------
NUM_ABILITIES = 3          # ability slots per hero
NUM_ITEMS = 6              # inventory slots
NUM_ENTITY_SLOTS = 8       # fixed-size "nearby_entities" set (attention pooled)

AGENT_SCALAR_DIM = 9       # see observation.encode_agent_scalars
ABILITY_FEAT_DIM = 4       # available, cooldown_ratio, mana_ok, effect_id_norm
ITEM_FEAT_DIM = 3          # present, ready, effect_id_norm
ENTITY_FEAT_DIM = 8        # present, is_enemy, is_creep, visible, x, y, hp_ratio, dist

EFFECT_EMBED_DIM = 8       # learned embedding width for ability effects
HERO_EMBED_DIM = 8         # learned embedding width for hero/creep identity


class ActionType(IntEnum):
    STOP = 0
    MOVE = 1
    ATTACK = 2
    CAST = 3
    ITEM = 4


NUM_ACTION_TYPES = len(ActionType)

# Which sub-heads each action type consumes. Used to build the joint log-prob
# and to mask entropy contributions (spec 5.1 "multi-head categorical").
ACTION_USES_ENTITY = {
    ActionType.STOP: False,
    ActionType.MOVE: False,
    ActionType.ATTACK: True,
    ActionType.CAST: True,
    ActionType.ITEM: False,
}
ACTION_USES_ABILITY = {
    ActionType.STOP: False,
    ActionType.MOVE: False,
    ActionType.ATTACK: False,
    ActionType.CAST: True,
    ActionType.ITEM: False,
}
ACTION_USES_ITEM = {
    ActionType.STOP: False,
    ActionType.MOVE: False,
    ActionType.ATTACK: False,
    ActionType.CAST: False,
    ActionType.ITEM: True,
}
ACTION_USES_COORD = {
    ActionType.STOP: False,
    ActionType.MOVE: True,
    ActionType.ATTACK: False,
    ActionType.CAST: False,
    ActionType.ITEM: False,
}


@dataclass
class EpisodeMeta:
    episode_id: str
    patch_id: str
    schema_version: int = OBS_SCHEMA_VERSION
    action_schema_version: int = ACTION_SCHEMA_VERSION
    seed: int = 0
    curriculum_phase: str = "F0"
    extra: dict = field(default_factory=dict)
