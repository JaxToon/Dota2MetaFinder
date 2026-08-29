"""Patch manifest + versioning (spec 12, 16).

A ``PatchManifest`` captures the game data that a Dota patch can change and that
the environment / observation semantics depend on: hero base stats, the ability
table, the item table. Every episode already carries ``patch_id``; a manifest
whose *structure* changes (ability count, item count) must bump
``obs_schema_version`` because tensor semantics change (spec 4.2 / 12).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from .protocol import NUM_ABILITIES, NUM_ITEMS


@dataclass(frozen=True)
class HeroStats:
    max_hp: float = 600.0
    max_mana: float = 300.0
    attack_damage: float = 22.0
    move_speed: float = 0.045
    attack_range: float = 0.14


@dataclass(frozen=True)
class PatchManifest:
    patch_id: str = "sandbox_mock_v1"
    hero: HeroStats = field(default_factory=HeroStats)
    abilities: tuple[dict, ...] = ()
    items: tuple[dict, ...] = ()

    # ---- identity / versioning ----------------------------------------
    def content_hash(self) -> str:
        blob = json.dumps({
            "hero": self.hero.__dict__,
            "abilities": [dict(sorted(a.items())) for a in self.abilities],
            "items": [dict(sorted(i.items())) for i in self.items],
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def structural_key(self) -> tuple:
        """What must not change without an obs_schema_version bump."""
        return (len(self.abilities), len(self.items),
                tuple(a["kind"] for a in self.abilities),
                tuple(i["kind"] for i in self.items))

    def bumped(self, **overrides) -> "PatchManifest":
        return replace(self, **overrides)


def default_manifest() -> PatchManifest:
    from . import mock_env as m
    return PatchManifest(
        patch_id="sandbox_mock_v1",
        hero=HeroStats(max_hp=m.MAX_HERO_HP, max_mana=m.MAX_HERO_MANA,
                       attack_damage=m.HERO_ATTACK_DMG, move_speed=m.MOVE_SPEED,
                       attack_range=m.ATTACK_RANGE),
        abilities=tuple(dict(a) for a in m.ABILITIES[:NUM_ABILITIES]),
        items=tuple(dict(i) for i in m.ITEMS[:NUM_ITEMS]),
    )
