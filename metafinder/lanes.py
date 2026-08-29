"""Lane advice.

  * ``lane_roles(hero)``  -- win rate per lane role from OpenDota
    ``/scenarios/laneRoles`` (parsed matches, ~30 min mark). Global.
  * ``lane_advice(hero, my_role, allies, enemies)`` -- the lineup-aware part:
    the direct 1v1 reads from the counter matrix (who you dread laning against),
    plus warnings from the enemy support / rotation profile.
"""
from __future__ import annotations

from . import heroprops as P
from . import opendota
from .heroes import resolve

ROLE_NAME = {1: "safelane", 2: "mid", 3: "offlane", 4: "jungle"}
ROLE_KEY = {"carry": 1, "safelane": 1, "pos1": 1, "mid": 2, "pos2": 2,
            "offlane": 3, "pos3": 3, "support": 1, "hard_support": 1, "pos4": 3, "pos5": 1}


def lane_roles(hero, at_time: int = 1800) -> list[dict]:
    h = resolve(hero)
    agg: dict[int, list[int]] = {}
    for row in opendota.scenarios_lane_roles(h.id):
        if int(row.get("time", 0)) != at_time:
            continue
        r = int(row["lane_role"])
        g, w = int(row["games"]), int(row["wins"])
        agg.setdefault(r, [0, 0])
        agg[r][0] += g
        agg[r][1] += w
    out = [{"role": ROLE_NAME.get(r, str(r)), "games": g, "win_rate": round(w / g, 4) if g else 0.0}
           for r, (g, w) in agg.items()]
    out.sort(key=lambda x: -x["games"])
    return out


def lane_advice(hero, my_role: str | None, allies, enemies, counters=None) -> dict:
    h = resolve(hero)
    roles = lane_roles(h)
    main = roles[0]["role"] if roles else None

    tough, favourable = [], []
    if counters is not None:
        for e in enemies:
            en = resolve(e)
            adv = counters.advantage(h, en)
            if adv <= -0.03:
                tough.append((en.localized_name, round(adv, 3)))
            elif adv >= 0.03:
                favourable.append((en.localized_name, round(adv, 3)))
        tough.sort(key=lambda x: x[1])
        favourable.sort(key=lambda x: -x[1])

    notes: list[str] = []
    if main:
        wr = next((r["win_rate"] for r in roles if r["role"] == main), None)
        notes.append(f"Usually goes **{main}**"
                     + (f" ({wr*100:.0f}% win rate there)" if wr else "") + ".")
    if my_role and ROLE_KEY.get(my_role.lower().replace(" ", "_")) and main:
        want = ROLE_NAME.get(ROLE_KEY[my_role.lower().replace(" ", "_")])
        if want and want != main and main != "jungle":
            notes.append(f"You asked for **{my_role}** but this hero's data leans **{main}** -- "
                         f"playable, just off-meta.")

    et = P.team_tags(enemies)
    if et.get(P.LOCKDOWN, 0) + et.get(P.AOE_DISABLE, 0) >= 2:
        notes.append("Enemy has multiple disables -- expect aggressive lanes / a possible trilane; "
                     "ward deep and don't greed for last hits at low HP.")
    roamers = P.has(enemies, P.MOBILE)
    roam_names = [x if isinstance(x, str) else resolve(x).localized_name for x in roamers]
    if len(roamers) >= 2:
        notes.append(f"Mobile enemies ({', '.join(roam_names[:3])}) roam early -- "
                     f"keep vision on the rune / your jungle entrance.")
    if tough:
        notes.append("Rough matchup" + (f"s" if len(tough) > 1 else "") + " for you: "
                     + ", ".join(f"{n} ({a:+.2f})" for n, a in tough[:3])
                     + " -- try to lane away from " + ("them" if len(tough) > 1 else tough[0][0]) + ".")
    if favourable:
        notes.append("You beat " + ", ".join(f"{n} ({a:+.2f})" for n, a in favourable[:3])
                     + " in a direct lane.")

    return {
        "hero": h.localized_name,
        "primary_lane": main,
        "lane_win_rates": roles,
        "tough_matchups": tough,
        "favourable_matchups": favourable,
        "notes": notes,
    }
