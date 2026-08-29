#!/usr/bin/env python
"""Draft coach in the terminal -- picks + build + lane, lineup-aware.

    python scripts/coach.py --allies "Crystal Maiden,Tidehunter" \\
        --enemies "Lion,Lina,Zeus,Sniper,Bane" --hero Juggernaut --role carry

    # no --hero: pick suggestions only
    python scripts/coach.py --enemies "Sniper,Drow Ranger"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metafinder import builds, counters, heroes, lanes, store, synergy  # noqa: E402
from metafinder.draft import DraftWeights, recommend  # noqa: E402


def _split(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--allies", default="")
    p.add_argument("--enemies", default="")
    p.add_argument("--bans", default="")
    p.add_argument("--hero", default=None)
    p.add_argument("--role", default=None)
    p.add_argument("--bracket", default="divine")
    args = p.parse_args()

    allies, enemies = _split(args.allies), _split(args.enemies)
    tl = heroes.tier_list(args.bracket)
    cm = counters.from_opendota()
    con = store.connect()
    sm = synergy.build(con) if store.count(con) >= 500 else None

    if not args.hero:
        if sm is None:
            class _Z:
                def synergy(self, *_): return 0.0
            sm = _Z()
        recs = recommend(allies, enemies, _split(args.bans), tier_rows=tl,
                         synergy=sm, counters=cm, my_role=args.role, weights=DraftWeights(), n=12)
        print(f"\nPICK SUGGESTIONS  (allies: {args.allies or '-'} | enemies: {args.enemies or '-'})")
        print("-" * 66)
        for c in recs:
            print(f"  {c.hero:<20}{c.score:>+7.3f}   " + "; ".join(c.notes[:2]))
        return

    h = heroes.resolve(args.hero)
    print(f"\n=== {h.localized_name}"
          + (f"  ({args.role})" if args.role else "") + " ===")

    cb = builds.core_build(h, per_stage=6)
    print("\nCORE BUILD")
    for stage, lbl in [("start", "start"), ("early", "early"), ("mid", "mid"), ("late", "late")]:
        its = ", ".join(x["item"] for x in cb.get(stage, []))
        if its:
            print(f"  {lbl:6}: {its}")

    sit = builds.situational_items(h, allies, enemies)
    if sit:
        print("\nSITUATIONAL  (vs this lineup)")
        for s in sit:
            print(f"  [{s['urgency']:8}] {s['item']:<30} {s['reason']}")

    la = lanes.lane_advice(h, args.role, allies, enemies, counters=cm)
    print("\nLANE")
    if la["lane_win_rates"]:
        print("  " + "   ".join(f"{r['role']} {r['win_rate']*100:.0f}%/{r['games']}g"
                                for r in la["lane_win_rates"] if r["games"] >= 8))
    for n in la["notes"]:
        print("  - " + n.replace("**", ""))


if __name__ == "__main__":
    main()
