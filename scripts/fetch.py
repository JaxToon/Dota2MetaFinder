#!/usr/bin/env python
"""Warm the local cache from OpenDota (run this first, and to refresh).

    python scripts/fetch.py                 # heroStats + all hero matchups + 20k matches
    python scripts/fetch.py --matches 60000 --min-rank 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metafinder import heroes, opendota, store  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=20000, help="target public matches in the store")
    p.add_argument("--min-rank", type=int, default=None, help="OpenDota rank tier, e.g. 60 = Divine")
    p.add_argument("--skip-matchups", action="store_true")
    args = p.parse_args()
    t0 = time.time()

    print("heroStats ...", flush=True)
    hs = opendota.hero_stats()
    hero_map = heroes.all_heroes()
    print(f"  {len(hs)} heroes, {len(hero_map)} in metadata")

    if not args.skip_matchups:
        print(f"hero matchups ({len(hero_map)} heroes, ~{len(hero_map)*1.1:.0f}s if uncached) ...", flush=True)
        for i, hid in enumerate(hero_map, 1):
            opendota.hero_matchups(hid)
            if i % 20 == 0:
                print(f"  {i}/{len(hero_map)}", flush=True)

    print(f"public matches -> store (target {args.matches:,}) ...", flush=True)
    n = store.ingest(args.matches, min_rank=args.min_rank)

    print(f"\ndone in {time.time()-t0:.0f}s. store holds {n:,} matches. "
          f"cache: {ROOT / '.cache'}")


if __name__ == "__main__":
    main()
