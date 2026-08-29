#!/usr/bin/env python
"""Full meta report: tier list + hardest counters + best synergy pairs.

    python scripts/meta.py --bracket divine --json meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metafinder import report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bracket", default="divine")
    p.add_argument("--min-rank", type=int, default=50, help="OpenDota avg_rank_tier floor for synergy")
    p.add_argument("--no-counters", action="store_true")
    p.add_argument("--no-synergy", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    rep = report.build(args.bracket, min_avg_rank=args.min_rank,
                       with_counters=not args.no_counters, with_synergy=not args.no_synergy)
    print(report.render_text(rep))

    if args.json:
        dump = {k: v for k, v in rep.items() if not k.startswith("_")}
        Path(args.json).write_text(json.dumps(dump, indent=2))
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
