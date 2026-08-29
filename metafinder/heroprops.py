"""Curated hero property tags -- the knowledge layer that makes build & lane
advice react to the *specific* heroes in a game.

Two kinds of tag:
  THREAT_* -- what this hero forces the other team to itemise against
  BRINGS_* -- what this hero gives its own team (for "your lineup lacks X")

Not every hero is listed; anything missing falls back to a light guess from its
OpenDota roles (see ``tags_for``). These are deliberately coarse -- they drive
"consider Pipe / BKB / sentries", not a full guide.
"""
from __future__ import annotations

from .heroes import resolve

# threat tags
MAGIC = "magic_burst"          # heavy magical nuke / DoT
PHYSICAL = "physical_carry"     # right-click damage threat
PURE = "pure_damage"
LOCKDOWN = "hard_lockdown"      # reliable single-target disable (hex, long stun, leash)
AOE_DISABLE = "aoe_disable"     # teamfight AoE disable
INVIS = "invisibility"
SILENCE = "silence"
BREAK = "break"
HEAL = "team_heal"
LIFESTEAL = "lifesteal_sustain"
EVASION = "grants_evasion"
SUMMONS = "summons_push"
BLINK_PICKOFF = "blink_pickoff"
MANA_BURN = "mana_burn"
TANKY = "tanky"
ILLUSION = "illusion_based"
MOBILE = "very_mobile"
# team-contribution tags
INITIATION = "brings_initiation"
SAVE = "brings_save"
DISPEL = "brings_dispel"
AOE_DMG = "brings_aoe"

_TABLE: dict[str, set[str]] = {
    "Anti-Mage": {PHYSICAL, MANA_BURN, MOBILE, ILLUSION},
    "Juggernaut": {PHYSICAL, MOBILE, HEAL},
    "Phantom Assassin": {PHYSICAL, EVASION, MOBILE},
    "Spectre": {PHYSICAL, LIFESTEAL, TANKY},
    "Faceless Void": {PHYSICAL, AOE_DISABLE, LOCKDOWN},
    "Terrorblade": {PHYSICAL, ILLUSION, SUMMONS},
    "Phantom Lancer": {PHYSICAL, ILLUSION, SUMMONS, MOBILE},
    "Slark": {PHYSICAL, MOBILE, LIFESTEAL},
    "Morphling": {PHYSICAL, MOBILE, MAGIC},
    "Medusa": {PHYSICAL, MANA_BURN, AOE_DISABLE, TANKY},
    "Sniper": {PHYSICAL, MAGIC},
    "Drow Ranger": {PHYSICAL},
    "Luna": {PHYSICAL, AOE_DMG},
    "Gyrocopter": {PHYSICAL, AOE_DMG, MAGIC},
    "Troll Warlord": {PHYSICAL, LOCKDOWN},
    "Ursa": {PHYSICAL, LIFESTEAL},
    "Lifestealer": {PHYSICAL, LIFESTEAL, TANKY, BREAK},
    "Wraith King": {PHYSICAL, LOCKDOWN, LIFESTEAL, SUMMONS, TANKY},
    "Sven": {PHYSICAL, LOCKDOWN, AOE_DMG},
    "Chaos Knight": {PHYSICAL, LOCKDOWN, ILLUSION, TANKY},
    "Lycan": {PHYSICAL, SUMMONS, TANKY},
    "Bloodseeker": {PHYSICAL, SILENCE, LIFESTEAL},
    "Riki": {PHYSICAL, INVIS, SILENCE},
    "Weaver": {PHYSICAL, MOBILE},
    "Clinkz": {PHYSICAL, INVIS},
    "Razor": {PHYSICAL, MAGIC},
    "Templar Assassin": {PHYSICAL, MAGIC, INVIS},
    "Naga Siren": {PHYSICAL, ILLUSION, AOE_DISABLE, SAVE},
    "Broodmother": {PHYSICAL, SUMMONS},
    "Meepo": {PHYSICAL, LOCKDOWN, SUMMONS},
    "Ember Spirit": {PHYSICAL, MAGIC, MOBILE},
    "Monkey King": {PHYSICAL, AOE_DISABLE, MOBILE},
    "Kez": {PHYSICAL, MOBILE},
    "Lina": {MAGIC, LOCKDOWN, AOE_DMG},
    "Lion": {MAGIC, LOCKDOWN, MANA_BURN},
    "Shadow Shaman": {MAGIC, LOCKDOWN, SUMMONS},
    "Zeus": {MAGIC, AOE_DMG, PURE},
    "Skywrath Mage": {MAGIC, SILENCE, PURE},
    "Invoker": {MAGIC, AOE_DISABLE, AOE_DMG, LOCKDOWN},
    "Tinker": {MAGIC, AOE_DMG, MOBILE},
    "Queen of Pain": {MAGIC, SILENCE, MOBILE},
    "Leshrac": {MAGIC, AOE_DMG},
    "Pugna": {MAGIC, PURE},
    "Death Prophet": {MAGIC, SILENCE, SUMMONS},
    "Necrophos": {MAGIC, HEAL, PURE, TANKY},
    "Warlock": {MAGIC, HEAL, AOE_DISABLE, SUMMONS},
    "Witch Doctor": {MAGIC, LOCKDOWN, HEAL, AOE_DMG},
    "Dazzle": {MAGIC, HEAL, SAVE},
    "Oracle": {MAGIC, HEAL, SAVE, DISPEL},
    "Ancient Apparition": {MAGIC, LOCKDOWN},  # anti-heal built in
    "Jakiro": {MAGIC, AOE_DMG, AOE_DISABLE},
    "Grimstroke": {MAGIC, SILENCE, LOCKDOWN, SAVE},
    "Silencer": {MAGIC, SILENCE, MANA_BURN},
    "Disruptor": {MAGIC, LOCKDOWN, AOE_DISABLE, SAVE},
    "Bane": {MAGIC, LOCKDOWN, PURE},
    "Shadow Demon": {MAGIC, LOCKDOWN, ILLUSION, SAVE, DISPEL},
    "Pudge": {MAGIC, LOCKDOWN, PURE, TANKY},
    "Batrider": {MAGIC, LOCKDOWN, INITIATION, MOBILE},
    "Dark Willow": {MAGIC, LOCKDOWN, AOE_DISABLE},
    "Hoodwink": {MAGIC, LOCKDOWN},
    "Snapfire": {MAGIC, LOCKDOWN, AOE_DMG, SAVE},
    "Ringmaster": {MAGIC, LOCKDOWN, AOE_DISABLE},
    "Muerta": {MAGIC, PHYSICAL},
    "Nature's Prophet": {PHYSICAL, MAGIC, SUMMONS, MOBILE},
    "Chen": {HEAL, SUMMONS, SAVE},
    "Enchantress": {MAGIC, SUMMONS, HEAL},
    "Enigma": {AOE_DISABLE, SUMMONS, INITIATION},
    "Tidehunter": {AOE_DISABLE, INITIATION, TANKY, DISPEL},
    "Magnus": {AOE_DISABLE, INITIATION, AOE_DMG, PHYSICAL},
    "Earthshaker": {AOE_DISABLE, AOE_DMG, INITIATION, BLINK_PICKOFF},
    "Sand King": {AOE_DISABLE, AOE_DMG, INVIS, INITIATION},
    "Axe": {LOCKDOWN, TANKY, INITIATION},
    "Legion Commander": {LOCKDOWN, PHYSICAL, INITIATION},
    "Centaur Warrunner": {AOE_DISABLE, TANKY, INITIATION, SAVE},
    "Slardar": {LOCKDOWN, BREAK, TANKY, INITIATION},
    "Spirit Breaker": {LOCKDOWN, TANKY, INITIATION},
    "Night Stalker": {LOCKDOWN, SILENCE, TANKY},
    "Beastmaster": {LOCKDOWN, SUMMONS, INITIATION},
    "Clockwerk": {LOCKDOWN, INITIATION, MOBILE},
    "Bristleback": {MAGIC, TANKY},
    "Timbersaw": {MAGIC, TANKY, MOBILE},
    "Dragon Knight": {MAGIC, TANKY, LOCKDOWN},
    "Abaddon": {HEAL, DISPEL, SAVE, TANKY},
    "Undying": {TANKY, HEAL, AOE_DISABLE},
    "Underlord": {MAGIC, AOE_DMG, TANKY, SAVE},
    "Mars": {AOE_DISABLE, INITIATION, TANKY, PHYSICAL},
    "Doom": {LOCKDOWN, SILENCE, BREAK, PURE, TANKY},
    "Primal Beast": {LOCKDOWN, AOE_DISABLE, TANKY, INITIATION},
    "Pangolier": {AOE_DISABLE, MOBILE, PHYSICAL},
    "Brewmaster": {AOE_DISABLE, SUMMONS, TANKY, INITIATION},
    "Tusk": {LOCKDOWN, INITIATION, SAVE, MOBILE},
    "Earth Spirit": {LOCKDOWN, SILENCE, INITIATION, MOBILE},
    "Void Spirit": {MAGIC, SILENCE, LOCKDOWN, MOBILE},
    "Storm Spirit": {MAGIC, LOCKDOWN, MOBILE},
    "Puck": {MAGIC, SILENCE, AOE_DISABLE, MOBILE},
    "Windranger": {PHYSICAL, MAGIC, LOCKDOWN},
    "Mirana": {LOCKDOWN, INVIS, SAVE},
    "Vengeful Spirit": {LOCKDOWN, PHYSICAL, SAVE},
    "Crystal Maiden": {MAGIC, LOCKDOWN, AOE_DISABLE},
    "Rubick": {MAGIC, LOCKDOWN, SAVE},
    "Keeper of the Light": {MAGIC, AOE_DMG, SAVE},
    "Io": {HEAL, SAVE, DISPEL},
    "Winter Wyvern": {HEAL, AOE_DISABLE, SAVE},
    "Treant Protector": {HEAL, LOCKDOWN, INVIS, SAVE, TANKY},
    "Omniknight": {HEAL, DISPEL, SAVE},
    "Dark Seer": {AOE_DISABLE, INITIATION, ILLUSION},
    "Visage": {PHYSICAL, SUMMONS, SILENCE},
    "Venomancer": {MAGIC, SUMMONS, AOE_DMG},
    "Viper": {MAGIC, PHYSICAL, BREAK},
    "Huskar": {MAGIC, PHYSICAL, LIFESTEAL, TANKY},
    "Lone Druid": {PHYSICAL, SUMMONS, TANKY},
    "Arc Warden": {PHYSICAL, MAGIC, SUMMONS, MOBILE},
    "Techies": {MAGIC, PURE, AOE_DMG},
    "Bounty Hunter": {INVIS, PHYSICAL, MOBILE},
    "Nyx Assassin": {MAGIC, LOCKDOWN, MANA_BURN, INVIS},
    "Phoenix": {MAGIC, AOE_DISABLE, HEAL, AOE_DMG},
    "Dawnbreaker": {HEAL, AOE_DISABLE, TANKY, SAVE},
    "Marci": {LOCKDOWN, SAVE, PHYSICAL, DISPEL},
    "Elder Titan": {AOE_DISABLE, BREAK, AOE_DMG},
    "Kunkka": {AOE_DISABLE, AOE_DMG, PHYSICAL},
    "Lich": {MAGIC, AOE_DMG, SAVE},
    "Ogre Magi": {MAGIC, LOCKDOWN, TANKY},
    "Shadow Fiend": {PHYSICAL, MAGIC, AOE_DMG},
    "Outworld Destroyer": {MAGIC, PURE, MANA_BURN, SAVE},
}

_ROLE_DEFAULT = {
    "Carry": {PHYSICAL}, "Nuker": {MAGIC}, "Disabler": {LOCKDOWN},
    "Initiator": {INITIATION}, "Durable": {TANKY}, "Escape": {MOBILE},
    "Pusher": {SUMMONS}, "Support": set(),
}


def tags_for(hero) -> set[str]:
    h = resolve(hero)
    if h.localized_name in _TABLE:
        return set(_TABLE[h.localized_name])
    out: set[str] = set()
    for r in h.roles:
        out |= _ROLE_DEFAULT.get(r, set())
    return out


def team_tags(heroes) -> dict[str, int]:
    """Count how many heroes on a side carry each tag."""
    counts: dict[str, int] = {}
    for hh in heroes:
        for t in tags_for(hh):
            counts[t] = counts.get(t, 0) + 1
    return counts


def has(heroes, tag: str) -> list:
    return [h for h in heroes if tag in tags_for(h)]
