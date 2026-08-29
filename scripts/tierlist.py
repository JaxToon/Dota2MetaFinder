#!/usr/bin/env python
"""Current-patch hero tier list.

    python scripts/tierlist.py                    # divine bracket
    python scripts/tierlist.py --bracket archon --role support --top 20
    python scripts/tierlist.py --bracket pro --json tierlist.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metafinder import heroes  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bracket", default="divine",
                   help="herald|guardian|crusader|archon|legend|ancient|divine|pub|pro")
    p.add_argument("--role", default=None, help="filter by hero role tag, e.g. Carry / Support")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    rows = heroes.tier_list(args.bracket)
    if args.role:
        rows = [r for r in rows if args.role.lower() in [x.lower() for x in r["roles"]]]

    print(f"\nDOTA 2 TIER LIST  --  {args.bracket}"
          + (f"  ·  role: {args.role}" if args.role else "") + "\n" + "-" * 60)
    cur = None
    for r in rows[: args.top]:
        if r["tier"] != cur:
            cur = r["tier"]
            print(f"\n [{cur}]")
        print(f"   {r['rank']:>3}. {r['hero']:<24} {r['win_rate']*100:5.1f}%  "
              f"pick {r['pick_share']*100:4.1f}%  {'/'.join(r['roles'][:2])}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
