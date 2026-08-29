"""Action contract (spec 4.3).

A single structured action with a hierarchical grammar rather than one monolithic
categorical over the whole Cartesian product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import (
    ACTION_USES_ABILITY,
    ACTION_USES_COORD,
    ACTION_USES_ENTITY,
    ACTION_USES_ITEM,
    ActionType,
)


@dataclass
class Action:
    action_type: int
    target_entity_slot: int = -1     # index into the fixed entity slot layout
    target_xy: tuple[float, float] | None = None
    ability_slot: int = -1
    item_slot: int = -1

    @classmethod
    def from_heads(
        cls,
        a_type: int,
        entity: int,
        ability: int,
        item: int,
        coord: np.ndarray | tuple[float, float],
    ) -> "Action":
        t = ActionType(int(a_type))
        return cls(
            action_type=int(t),
            target_entity_slot=int(entity) if ACTION_USES_ENTITY[t] else -1,
            target_xy=(float(coord[0]), float(coord[1])) if ACTION_USES_COORD[t] else None,
            ability_slot=int(ability) if ACTION_USES_ABILITY[t] else -1,
            item_slot=int(item) if ACTION_USES_ITEM[t] else -1,
        )

    def is_valid(self, masks: dict) -> bool:
        t = ActionType(self.action_type)
        if not masks["action_type"][t]:
            return False
        if ACTION_USES_ENTITY[t]:
            if self.target_entity_slot < 0 or not masks["entity"][self.target_entity_slot]:
                return False
        if ACTION_USES_ABILITY[t]:
            if self.ability_slot < 0 or not masks["ability"][self.ability_slot]:
                return False
        if ACTION_USES_ITEM[t]:
            if self.item_slot < 0 or not masks["item"][self.item_slot]:
                return False
        return True
