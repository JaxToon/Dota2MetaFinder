"""Hero metadata + patch tier list from OpenDota ``/heroStats``."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import opendota

# OpenDota /heroStats rank brackets: 1..7 = Herald..Divine have data;
# bracket 8 (Immortal) is not populated -> "immortal"/"high" alias to Divine.
BRACKETS = {
    "herald": "1", "guardian": "2", "crusader": "3", "archon": "4",
    "legend": "5", "ancient": "6", "divine": "7",
    "immortal": "7", "high": "7", "low": "2",
}
LANE_ROLES = ("Carry", "Mid", "Offlane", "Support", "Hard Support")


@dataclass(frozen=True)
class Hero:
    id: int
    name: str            # npc_dota_hero_*
    localized_name: str
    primary_attr: str
    attack_type: str     # Melee | Ranged
    roles: tuple[str, ...]

    @property
    def short(self) -> str:
        return self.name.replace("npc_dota_hero_", "")


@lru_cache(maxsize=1)
def all_heroes() -> dict[int, Hero]:
    out = {}
    for h in opendota.heroes():
        out[h["id"]] = Hero(h["id"], h["name"], h["localized_name"],
                            h["primary_attr"], h["attack_type"], tuple(h.get("roles", [])))
    return out


def by_name(q: str) -> Hero:
    q = q.strip().lower()
    heroes = all_heroes()
    for h in heroes.values():
        if h.localized_name.lower() == q or h.short == q.replace(" ", "_"):
            return h
    # fuzzy contains
    hits = [h for h in heroes.values() if q in h.localized_name.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"no hero matching {q!r}")
    raise KeyError(f"ambiguous {q!r}: {[h.localized_name for h in hits]}")


def resolve(x) -> Hero:
    if isinstance(x, Hero):
        return x
    if isinstance(x, int):
        return all_heroes()[x]
    return by_name(str(x))


def _wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - m)


def tier_list(bracket: str = "immortal", *, min_pick_share: float = 0.005,
              shrink: float = 250.0) -> list[dict]:
    """Rank heroes for a bracket. ``bracket`` is a key of BRACKETS, or 'pub'/'pro'.

    Score = shrinkage-adjusted win rate (pulls low-sample heroes toward 50%),
    with pick share reported alongside. Pro also carries contest rate (pick+ban).
    """
    stats = opendota.hero_stats()
    heroes = all_heroes()

    if bracket == "pub":
        pk, wk = "pub_pick", "pub_win"
    elif bracket == "pro":
        pk, wk = "pro_pick", "pro_win"
    else:
        b = BRACKETS[bracket]
        pk, wk = f"{b}_pick", f"{b}_win"

    total = sum(s.get(pk, 0) for s in stats) or 1
    rows = []
    for s in stats:
        picks = s.get(pk, 0)
        wins = s.get(wk, 0)
        share = picks / total
        if share < min_pick_share:
            continue
        wr = wins / picks if picks else 0.0
        wr_adj = (wins + shrink * 0.5) / (picks + shrink)
        row = {
            "hero_id": s["id"],
            "hero": heroes[s["id"]].localized_name,
            "win_rate": round(wr, 4),
            "win_rate_adj": round(wr_adj, 4),
            "win_rate_ci_low": round(_wilson_lower(wins, picks), 4),
            "pick_share": round(share, 5),
            "games": picks,
            "roles": list(heroes[s["id"]].roles),
        }
        if bracket == "pro":
            row["ban_rate"] = round(s.get("pro_ban", 0) / (total / len(stats) + 1e-9), 4)
            row["contest_rate"] = round((picks + s.get("pro_ban", 0)) / (total + 1e-9), 5)
        rows.append(row)

    rows.sort(key=lambda r: r["win_rate_adj"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["tier"] = _tier_letter(i, len(rows))
    return rows


def _tier_letter(rank: int, n: int) -> str:
    q = rank / n
    return ("S" if q <= 0.08 else "A" if q <= 0.25 else "B" if q <= 0.50
            else "C" if q <= 0.78 else "D")
