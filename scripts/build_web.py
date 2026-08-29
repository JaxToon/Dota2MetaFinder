#!/usr/bin/env python
"""Bake the full data blob the interactive web tool needs into web/data.json.

    python scripts/fetch.py            # first: warm heroStats/matchups/matches
    python scripts/build_web.py        # then: + itemPopularity + laneRoles per hero

Re-run both to refresh the meta, then re-publish web/index.html.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metafinder import builds, counters, heroprops, heroes, lanes, opendota, store, synergy  # noqa: E402

BRACKET = "divine"
MIN_ADV = 0.012           # prune tiny counter/synergy entries to keep the blob small


def main() -> None:
    t0 = time.time()
    hero_map = heroes.all_heroes()
    tl = {r["hero_id"]: r for r in heroes.tier_list(BRACKET)}

    print(f"counter matrix ({len(hero_map)} heroes) ...", flush=True)
    cm = counters.from_opendota()

    con = store.connect()
    n_matches = store.count(con)
    sm = synergy.build(con) if n_matches >= 500 else None
    print(f"synergy from {n_matches:,} matches", flush=True)

    print("item popularity + lane roles per hero ...", flush=True)
    build_by, lanes_by = {}, {}
    for i, hid in enumerate(hero_map, 1):
        try:
            build_by[hid] = builds.core_build(hid, per_stage=6)
        except Exception as e:
            build_by[hid] = {}; print(f"  ! build {hid}: {e}")
        try:
            lanes_by[hid] = lanes.lane_roles(hid)
        except Exception as e:
            lanes_by[hid] = []; print(f"  ! lanes {hid}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(hero_map)}", flush=True)

    def sparse(mat):
        out = {}
        for a, row in mat.items():
            r = {b: round(v, 3) for b, v in row.items() if abs(v) >= MIN_ADV}
            if r:
                out[a] = r
        return out

    blob = {
        "generated": date.today().isoformat(),
        "bracket": BRACKET,
        "matches": n_matches,
        "heroes": [
            {"id": h.id, "name": h.localized_name, "img": h.short,
             "attr": h.primary_attr, "attack": h.attack_type,
             "roles": list(h.roles), "tags": sorted(heroprops.tags_for(h))}
            for h in sorted(hero_map.values(), key=lambda x: x.localized_name)
        ],
        "tier": {str(hid): {"tier": r["tier"], "rank": r["rank"],
                            "win_rate": r["win_rate"], "pick": r["pick_share"]}
                 for hid, r in tl.items()},
        "counter": {str(a): {str(b): v for b, v in row.items()}
                    for a, row in sparse(cm.adv).items()},
        "synergy": ({str(a): {str(b): v for b, v in row.items()}
                     for a, row in sparse({k: dict(vv) for k, vv in _syn_matrix(sm).items()}).items()}
                    if sm else {}),
        "solo_wr": ({str(k): round(v, 3) for k, v in sm.solo_wr.items()} if sm else {}),
        "builds": {str(k): v for k, v in build_by.items()},
        "lanes": {str(k): v for k, v in lanes_by.items()},
        "tag_labels": _TAG_LABELS,
    }

    web = ROOT / "web"
    web.mkdir(exist_ok=True)
    payload = json.dumps(blob, separators=(",", ":"))
    (web / "data.json").write_text(payload)

    tpl = (web / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", payload)
    (web / "index.html").write_text(html, encoding="utf-8")

    kb = len(payload) / 1024
    print(f"\n-> web/data.json + web/index.html  (data {kb:.0f} KB)  in {time.time()-t0:.0f}s")


def _syn_matrix(sm):
    """Materialise the pair synergy into a dense-ish dict for pruning."""
    out: dict[int, dict[int, float]] = {}
    for (a, b) in sm.pair_games:
        if sm.pair_games[(a, b)] < 25:
            continue
        s = sm.synergy(a, b)
        out.setdefault(a, {})[b] = s
        out.setdefault(b, {})[a] = s
    return out


_TAG_LABELS = {
    "magic_burst": "magic damage", "physical_carry": "right-click carry",
    "pure_damage": "pure damage", "hard_lockdown": "hard disable",
    "aoe_disable": "AoE disable", "invisibility": "invisibility", "silence": "silence",
    "break": "break", "team_heal": "healing", "lifesteal_sustain": "lifesteal",
    "grants_evasion": "evasion", "summons_push": "summons / illusions",
    "blink_pickoff": "blink pickoff", "mana_burn": "mana burn", "tanky": "tanky",
    "illusion_based": "illusions", "very_mobile": "mobile",
    "brings_initiation": "initiation", "brings_save": "save", "brings_dispel": "dispel",
    "brings_aoe": "AoE damage",
}

if __name__ == "__main__":
    main()
