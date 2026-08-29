#!/usr/bin/env python
"""Event-aligned meta / power-spike report from a checkpoint (spec 15, M8).

    python scripts/meta_report.py runs/dota_rl_f2_build/checkpoints/ckpt_000040.pt
    python scripts/meta_report.py <ckpt> --episodes 60 --json report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dota_rl.analytics.report import build_report, load_policy, render_text  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--phase", default=None)
    p.add_argument("--json", default=None, help="also write the full report as JSON here")
    args = p.parse_args()

    policy, phase = load_policy(args.checkpoint)
    report = build_report(policy, phase=args.phase or phase, episodes=args.episodes)
    print(render_text(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nfull report -> {args.json}")


if __name__ == "__main__":
    main()
