"""Assemble a meta report from the data modules + render it as text."""
from __future__ import annotations

from . import counters as counters_mod
from . import heroes as heroes_mod
from . import store as store_mod
from . import synergy as synergy_mod
from .draft import score_comp


def build(bracket: str = "divine", *, min_avg_rank: int | None = 50,
          with_counters: bool = True, with_synergy: bool = True) -> dict:
    tl = heroes_mod.tier_list(bracket)
    rep: dict = {"bracket": bracket, "n_heroes": len(tl), "tier_list": tl}

    if with_counters:
        cm = counters_mod.from_opendota()
        rep["_counters"] = cm
        rep["biggest_counters"] = _biggest_counters(cm, tl)

    if with_synergy:
        con = store_mod.connect()
        if store_mod.count(con) >= 500:
            sm = synergy_mod.build(con, min_avg_rank=min_avg_rank)
            rep["_synergy"] = sm
            rep["matches_analysed"] = store_mod.count(con)
            rep["best_pairs"] = _best_pairs(sm, tl)
    return rep


def _biggest_counters(cm: counters_mod.CounterMatrix, tl: list[dict], k: int = 12):
    top = [r["hero_id"] for r in tl[: max(40, len(tl) // 3)]]
    rows = []
    for a in cm.adv:
        for b, v in cm.adv[a].items():
            if a in top and b in top and v > 0:
                rows.append((heroes_mod.all_heroes()[a].localized_name,
                             heroes_mod.all_heroes()[b].localized_name,
                             round(v, 3), cm.games[a][b]))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:k]


def _best_pairs(sm: synergy_mod.SynergyMatrix, tl: list[dict], k: int = 12):
    meta_ids = {r["hero_id"] for r in tl[: max(50, len(tl) // 2)]}
    seen = set()
    rows = []
    for (a, b), g in sm.pair_games.items():
        if g < 40 or a not in meta_ids or b not in meta_ids or (a, b) in seen:
            continue
        seen.add((a, b))
        s = sm.synergy(a, b)
        rows.append((heroes_mod.all_heroes()[a].localized_name,
                     heroes_mod.all_heroes()[b].localized_name, s, g))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:k]


def render_text(rep: dict) -> str:
    L = [f"DOTA 2 META  --  bracket: {rep['bracket']}  ({rep['n_heroes']} heroes ranked)",
         "=" * 68, "", "TIER LIST (shrinkage-adjusted win rate)"]
    cur = None
    for r in rep["tier_list"][:40]:
        if r["tier"] != cur:
            cur = r["tier"]
            L.append(f"\n  [{cur}]")
        L.append(f"    {r['hero']:<24} {r['win_rate']*100:5.1f}%   pick {r['pick_share']*100:4.1f}%"
                 f"   {'/'.join(r['roles'][:2])}")

    if rep.get("biggest_counters"):
        L += ["", "-" * 68, "", "HARDEST COUNTERS among meta heroes  (A beats B)"]
        for a, b, v, g in rep["biggest_counters"]:
            L.append(f"    {a:<20} > {b:<20}  +{v:.3f}   (n={g:,})")

    if rep.get("best_pairs"):
        L += ["", "-" * 68, "",
              f"BEST SYNERGY PAIRS among meta heroes  ({rep['matches_analysed']:,} matches)"]
        for a, b, s, g in rep["best_pairs"]:
            L.append(f"    {a:<20} + {b:<20}  synergy {s:+.3f}   (n={g})")
    return "\n".join(L)
