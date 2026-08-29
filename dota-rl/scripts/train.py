#!/usr/bin/env python
"""Run a curriculum-phase training job.

    python scripts/train.py configs/ppo_f0.yaml
    python scripts/train.py configs/ppo_f1.yaml --set ppo.lr=1e-4 train.iterations=80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dota_rl.training.train import train  # noqa: E402
from dota_rl.utils.config import apply_dotted, load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    cfg = apply_dotted(load_config(args.config), args.set)
    name = cfg.get("experiment", {}).get("name", "run")
    run_dir = Path(args.run_dir or ROOT / "runs" / name)

    result = train(cfg, run_dir)
    print("\n=== summary ===")
    s = result["final_eval"]["vs_scripted_bot"]
    print(f"winrate vs scripted bot : {s['winrate']:.3f}  (ci {s['winrate_ci_low']:.3f}-{s['winrate_ci_high']:.3f})")
    print(f"random baseline winrate : {result['final_eval']['random_vs_scripted_bot']['winrate']:.3f}")
    print(f"invalid action rate     : {result['invalid_action_rate']:.5f}   (AC-03 needs < 0.01)")
    print(f"promotion gate passed   : {result['promotion_passed']}")
    print(f"checkpoint              : {result['checkpoint']['path'] if result['checkpoint'] else None}")


if __name__ == "__main__":
    main()
