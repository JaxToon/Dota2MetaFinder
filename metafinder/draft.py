"""Draft recommender + comp scorer.

score(H) = w_meta * (metaWinAdj[H] - 0.5)                 # is H strong this patch
         + w_syn  * mean synergy(H, ally)                 # fits our picks
         + w_ctr  * mean counterAdv(H, enemy)             # beats their picks
         - w_vuln * mean counterAdv(enemy, H)             # how hard they beat H
         + w_role * roleFit(H)                            # covers a missing role

All terms are on a comparable ~[-0.5, 0.5] scale so the weights read directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .counters import CounterMatrix
from .heroes import Hero, all_heroes, resolve
from .synergy import SynergyMatrix

ROLE_TAGS = {
    "carry": {"Carry"}, "mid": {"Nuker", "Carry"}, "offlane": {"Initiator", "Durable"},
    "support": {"Support"}, "hard_support": {"Support", "Disabler"},
}


@dataclass
class DraftWeights:
    meta: float = 1.0
    synergy: float = 1.4
    counter: float = 1.2
    vulnerability: float = 1.0
    role: float = 0.6


@dataclass
class Candidate:
    hero: str
    hero_id: int
    score: float
    meta: float
    synergy: float
    counter: float
    vulnerability: float
    role: float
    notes: list[str] = field(default_factory=list)


def _meta_win(tier_rows: list[dict]) -> dict[int, float]:
    return {r["hero_id"]: r["win_rate_adj"] for r in tier_rows}


def _role_fit(h: Hero, my_role: str | None, ally_roles: list[set[str]]) -> float:
    if not my_role:
        return 0.0
    want = ROLE_TAGS.get(my_role.lower().replace(" ", "_"), set())
    return 0.5 if want & set(h.roles) else -0.15


def recommend(allies, enemies, bans, *, tier_rows: list[dict],
              synergy: SynergyMatrix, counters: CounterMatrix,
              my_role: str | None = None, weights: DraftWeights | None = None,
              n: int = 10) -> list[Candidate]:
    w = weights or DraftWeights()
    allies = [resolve(a).id for a in allies]
    enemies = [resolve(e).id for e in enemies]
    taken = set(allies) | set(enemies) | {resolve(b).id for b in bans}
    meta = _meta_win(tier_rows)
    heroes = all_heroes()
    ally_roles = [set(heroes[a].roles) for a in allies]

    out: list[Candidate] = []
    for hid, h in heroes.items():
        if hid in taken or hid not in meta:
            continue
        m = meta[hid] - 0.5
        syn = (sum(synergy.synergy(hid, a) for a in allies) / len(allies)) if allies else 0.0
        ctr = (sum(counters.advantage(hid, e) for e in enemies) / len(enemies)) if enemies else 0.0
        vul = (sum(counters.advantage(e, hid) for e in enemies) / len(enemies)) if enemies else 0.0
        rf = _role_fit(h, my_role, ally_roles)
        score = w.meta * m + w.synergy * syn + w.counter * ctr - w.vulnerability * vul + w.role * rf

        notes = []
        if enemies:
            worst = min(enemies, key=lambda e: counters.advantage(hid, e))
            best = max(enemies, key=lambda e: counters.advantage(hid, e))
            if counters.advantage(hid, best) > 0.03:
                notes.append(f"strong into {heroes[best].localized_name} (+{counters.advantage(hid, best):.2f})")
            if counters.advantage(hid, worst) < -0.03:
                notes.append(f"loses to {heroes[worst].localized_name} ({counters.advantage(hid, worst):.2f})")
        if allies:
            bp = max(allies, key=lambda a: synergy.synergy(hid, a))
            if synergy.synergy(hid, bp) > 0.015:
                notes.append(f"pairs with {heroes[bp].localized_name} (+{synergy.synergy(hid, bp):.2f})")

        out.append(Candidate(h.localized_name, hid, round(score, 4), round(m, 4),
                             round(syn, 4), round(ctr, 4), round(vul, 4), round(rf, 4), notes))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:n]


def score_comp(heroes, *, tier_rows: list[dict], synergy: SynergyMatrix) -> dict:
    ids = [resolve(h).id for h in heroes]
    meta = _meta_win(tier_rows)
    hs = all_heroes()

    meta_sum = sum(meta.get(i, 0.5) - 0.5 for i in ids)
    pair_syn = [(hs[a].localized_name, hs[b].localized_name, synergy.synergy(a, b))
                for k, a in enumerate(ids) for b in ids[k + 1:]]
    syn_sum = sum(s for _, _, s in pair_syn)

    attrs = [hs[i].primary_attr for i in ids]
    dmg = [hs[i].attack_type for i in ids]
    role_pool = set().union(*[set(hs[i].roles) for i in ids])
    coverage = {r: (r in role_pool) for r in ("Carry", "Support", "Initiator", "Disabler", "Nuker", "Durable")}

    return {
        "heroes": [hs[i].localized_name for i in ids],
        "comp_score": round(meta_sum + 1.6 * syn_sum, 4),
        "meta_strength": round(meta_sum, 4),
        "total_synergy": round(syn_sum, 4),
        "top_pairs": sorted(pair_syn, key=lambda p: p[2], reverse=True)[:3],
        "weak_pairs": sorted(pair_syn, key=lambda p: p[2])[:2],
        "attr_mix": {a: attrs.count(a) for a in set(attrs)},
        "melee_ranged": {"melee": dmg.count("Melee"), "ranged": dmg.count("Ranged")},
        "role_coverage": coverage,
        "gaps": [r for r, ok in coverage.items() if not ok],
    }
