#!/usr/bin/env python
"""Draft helper -- next-pick recommendation + comp scoring.

    # what should I pick? (my role = mid; we have Crystal Maiden, they have AM + Sniper)
    python scripts/draft.py --allies "Crystal Maiden" --enemies "Anti-Mage,Sniper" --role mid

    # score a full 5-stack
    python scripts/draft.py --comp "Mars,Zeus,Juggernaut,Crystal Maiden,Hoodwink"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metafinder import counters, heroes, store, synergy  # noqa: E402
from metafinder.draft import DraftWeights, recommend, score_comp  # noqa: E402


def _heroes(s: str | None) -> list[str]:
    return [h.strip() for h in (s or "").split(",") if h.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--allies", default="")
    p.add_argument("--enemies", default="")
    p.add_argument("--bans", default="")
    p.add_argument("--role", default=None, help="carry|mid|offlane|support|hard_support")
    p.add_argument("--bracket", default="divine")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--comp", default=None, help="score a comma-separated 5-hero comp")
    args = p.parse_args()

    tl = heroes.tier_list(args.bracket)
    cm = counters.from_opendota()
    con = store.connect()
    sm = synergy.build(con) if store.count(con) >= 500 else None
    if sm is None:
        print("! synergy unavailable (run scripts/fetch.py to fill the match store)\n")

    if args.comp:
        comp = _heroes(args.comp)
        if sm is None:
            print("comp scoring needs the match store."); return
        r = score_comp(comp, tier_rows=tl, synergy=sm)
        print(f"\nCOMP: {', '.join(r['heroes'])}")
        print("-" * 60)
        print(f"  comp score      {r['comp_score']:+.3f}   "
              f"(meta {r['meta_strength']:+.3f}  synergy {r['total_synergy']:+.3f})")
        print(f"  attribute mix   {r['attr_mix']}   melee/ranged {r['melee_ranged']}")
        print(f"  role coverage   {[k for k,v in r['role_coverage'].items() if v]}")
        if r["gaps"]:
            print(f"  GAPS            {r['gaps']}")
        print("  strongest pairs " + "; ".join(f"{a}+{b} {s:+.2f}" for a, b, s in r["top_pairs"]))
        print("  weakest pairs   " + "; ".join(f"{a}+{b} {s:+.2f}" for a, b, s in r["weak_pairs"]))
        return

    if sm is None:
        class _Z:
            def synergy(self, *_): return 0.0
        sm = _Z()

    recs = recommend(_heroes(args.allies), _heroes(args.enemies), _heroes(args.bans),
                     tier_rows=tl, synergy=sm, counters=cm, my_role=args.role,
                     weights=DraftWeights(), n=args.top)
    print(f"\nNEXT-PICK RECOMMENDATIONS  ({args.bracket})")
    if args.allies:  print(f"  allies:  {args.allies}")
    if args.enemies: print(f"  enemies: {args.enemies}")
    if args.role:    print(f"  role:    {args.role}")
    print("-" * 68)
    print(f"  {'hero':<22}{'score':>7} {'meta':>7} {'syn':>7} {'ctr':>7} {'vuln':>7}")
    for c in recs:
        print(f"  {c.hero:<22}{c.score:>+7.3f} {c.meta:>+7.3f} {c.synergy:>+7.3f} "
              f"{c.counter:>+7.3f} {c.vulnerability:>+7.3f}")
        for note in c.notes[:2]:
            print(f"      - {note}")


if __name__ == "__main__":
    main()
