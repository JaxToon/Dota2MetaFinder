"""Item build advice.

Two layers:
  * ``core_build(hero)``  -- the hero's popular item progression by game stage,
    straight from OpenDota ``/heroes/{id}/itemPopularity`` (global, not
    lineup-aware).
  * ``situational_items(hero, allies, enemies)`` -- a rules layer that fires on
    the *specific* heroes in the game: heavy magic on the enemy team -> Pipe /
    Glimmer, hard lockdown -> BKB / Lotus, invis -> sentries, and so on. Grounded
    in Dota fundamentals; the trigger is always which heroes were actually
    entered.
"""
from __future__ import annotations

from functools import lru_cache

from . import heroprops as P
from . import opendota
from .heroes import resolve

_SKIP = {"tango", "flask", "clarity", "faerie_fire", "enchanted_mango", "tpscroll",
         "ward_observer", "ward_sentry", "ward_dispenser", "dust", "smoke_of_deceit",
         "branches", "recipe"}


@lru_cache(maxsize=1)
def _items() -> dict[int, dict]:
    out = {}
    for key, v in opendota.item_constants().items():
        if isinstance(v, dict) and "id" in v:
            out[v["id"]] = {"key": key, "name": v.get("dname", key),
                            "cost": v.get("cost") or 0, "qual": v.get("qual")}
    return out


_KEEP_COMPONENT = {"blink", "hand_of_midas", "helm_of_the_dominator", "aghanims_shard",
                   "power_treads", "phase_boots", "arcane_boots", "tranquil_boots",
                   "magic_wand", "wind_lace", "orb_of_venom", "bracer", "wraith_band", "null_talisman"}


def _name(item_id, *, stage: str = "mid") -> str | None:
    it = _items().get(int(item_id))
    if not it or it["key"] in _SKIP or it["key"].startswith("recipe"):
        return None
    if stage == "start":
        return it["name"]           # starting items are cheap by nature -- keep them
    if it["qual"] in ("consumable",):
        return None
    if it["qual"] == "component" and it["key"] not in _KEEP_COMPONENT and (it["cost"] or 0) < 2000:
        return None                 # drop build-up pieces, keep Blink / boots / wand
    return it["name"]


def core_build(hero, per_stage: int = 6) -> dict:
    h = resolve(hero)
    pop = opendota.item_popularity(h.id)
    stages = {"start": "start_game_items", "early": "early_game_items",
              "mid": "mid_game_items", "late": "late_game_items"}
    out = {}
    for label, key in stages.items():
        ranked = sorted((pop.get(key) or {}).items(), key=lambda kv: -kv[1])
        picked, seen = [], set()
        for iid, cnt in ranked:
            nm = _name(iid, stage=label)
            if nm and nm not in seen:
                picked.append({"item": nm, "picks": cnt})
                seen.add(nm)
            if len(picked) >= per_stage:
                break
        out[label] = picked
    return out


# --- situational rules --------------------------------------------------------
def _my_profile(hero) -> dict:
    h = resolve(hero)
    roles = set(h.roles)
    tags = P.tags_for(h)
    core = bool(roles & {"Carry", "Nuker"}) or "Pusher" in roles
    frontline = bool(roles & {"Durable", "Initiator"}) or P.TANKY in tags
    caster = "Nuker" in roles or (P.MAGIC in tags and "Carry" not in roles)
    support = "Support" in roles and "Carry" not in roles
    squishy = not frontline and (support or (caster and not core))   # wants a get-out button
    return {"core": core, "support": support, "squishy": squishy,
            "frontline": frontline, "physical": P.PHYSICAL in tags, "caster": caster,
            "can_initiate": bool(tags & {P.INITIATION}) or bool(roles & {"Initiator"})}


def situational_items(hero, allies=(), enemies=()) -> list[dict]:
    me = _my_profile(hero)
    et = P.team_tags(enemies)
    at = P.team_tags(allies) if allies else {}
    recs: list[dict] = []

    def add(item, why, urgency="consider"):
        recs.append({"item": item, "reason": why, "urgency": urgency})

    magic_heroes = list({resolve(h).localized_name for h in P.has(enemies, P.MAGIC)}
                        | {resolve(h).localized_name for h in P.has(enemies, P.AOE_DMG)})
    magic = len(magic_heroes)
    if magic >= 2:
        who = ", ".join(sorted(magic_heroes)[:3])
        if me["frontline"] or me["core"]:
            add("Pipe of Insight", f"{magic} magic-damage threats ({who}) -- Pipe shields your team", "high" if magic >= 3 else "consider")
        if me["squishy"]:
            add("Glimmer Cape", f"heavy magic burst ({who}) -- Glimmer to dodge combos", "high" if magic >= 3 else "consider")
        add("Hood of Defiance / Eternal Shroud", "cheap early magic resistance vs the nuke lineup")

    if et.get(P.LOCKDOWN, 0) + et.get(P.AOE_DISABLE, 0) >= 1 and (me["core"] or me["caster"]):
        who = ", ".join(h if isinstance(h, str) else resolve(h).localized_name for h in P.has(enemies, P.LOCKDOWN)[:3])
        add("Black King Bar", f"hard disables in the lineup ({who}) -- BKB to fight through them",
            "high" if et.get(P.LOCKDOWN, 0) >= 2 else "consider")
        if me["squishy"]:
            add("Lotus Orb / Aeon Disk", "single-target lockdown on you -- reactive dispel / survival")

    if et.get(P.INVIS, 0) >= 1:
        who = ", ".join(h if isinstance(h, str) else resolve(h).localized_name for h in P.has(enemies, P.INVIS)[:3])
        add("Sentry Wards + Dust", f"enemy invisibility ({who}) -- carry detection", "high")
        if me["core"]:
            add("Gem of True Sight", f"{who} -- a Gem shuts the invis hero out of the game", "consider")

    if et.get(P.PHYSICAL, 0) >= 2 and (me["frontline"] or me["support"]):
        add("Assault Cuirass / Shiva's Guard" if me["frontline"] else "Ghost Scepter / Solar Crest",
            f"{et[P.PHYSICAL]} right-click cores -- stack armor / kite them")

    if et.get(P.EVASION, 0) >= 1 and me["physical"]:
        add("Monkey King Bar", "enemy evasion -- MKB to guarantee your hits", "high")

    if et.get(P.HEAL, 0) + et.get(P.LIFESTEAL, 0) >= 2:
        who = ", ".join(h if isinstance(h, str) else resolve(h).localized_name
                        for h in (P.has(enemies, P.HEAL) + P.has(enemies, P.LIFESTEAL))[:3])
        add("Spirit Vessel" if me["support"] else "any Break / heal-reduction",
            f"strong sustain on the enemy team ({who}) -- cut their healing")

    if et.get(P.SUMMONS, 0) >= 2 and me["frontline"]:
        add("Crimson Guard", f"{et[P.SUMMONS]} summon/illusion heroes -- Crimson eats the chip damage")

    if et.get(P.SILENCE, 0) + et.get(P.MANA_BURN, 0) >= 1 and me["caster"]:
        add("Eul's Scepter / Manta Style", "silence or mana burn against your spells -- a dispel keeps you casting")

    if et.get(P.BLINK_PICKOFF, 0) >= 1 and (me["squishy"]):
        who = ", ".join(h if isinstance(h, str) else resolve(h).localized_name for h in P.has(enemies, P.BLINK_PICKOFF)[:2])
        add("Force Staff / Aeon Disk", f"blink pickoff threat ({who}) -- a get-out button")

    if et.get(P.BREAK, 0) >= 1 and (P.EVASION in P.tags_for(hero) or P.LIFESTEAL in P.tags_for(hero) or P.TANKY in P.tags_for(hero)):
        add("(note) don't itemise around your passive", f"enemy has Break -- value active items instead", "note")

    # ally-side gaps
    if allies and not (at.get(P.INITIATION, 0)) and me["can_initiate"]:
        add("Blink Dagger", "your lineup has no reliable initiation -- you're the one who starts fights", "high")
    if allies and not (at.get(P.SAVE, 0)) and (me["support"]):
        add("Force Staff / Glimmer Cape", "no save in your lineup -- bring one for your cores")
    if allies and not (at.get(P.DISPEL, 0)) and et.get(P.SILENCE, 0) + et.get(P.LOCKDOWN, 0) >= 2:
        add("Lotus Orb / Guardian Greaves", "enemy control + your team has no dispel")

    # de-dupe by item, keep the strongest urgency
    order = {"high": 0, "consider": 1, "note": 2}
    best: dict[str, dict] = {}
    for r in recs:
        if r["item"] not in best or order[r["urgency"]] < order[best[r["item"]]["urgency"]]:
            best[r["item"]] = r
    return sorted(best.values(), key=lambda r: order[r["urgency"]])[:7]
