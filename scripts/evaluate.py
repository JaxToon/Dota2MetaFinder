#!/usr/bin/env python
"""Evaluate a checkpoint against the frozen + rotating benchmark (spec 9).

    python scripts/evaluate.py runs/dota_rl_f1_baseline/checkpoints/ckpt_000060.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dota_rl.agents.policy import ActorCritic  # noqa: E402
from dota_rl.evaluation.harness import EvalConfig, evaluate_policy  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--phase", default=None)
    p.add_argument("--games", type=int, default=40)
    args = p.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = blob.get("config", {})
    mc = cfg.get("model", {})
    policy = ActorCritic(encoder_dim=int(mc.get("encoder_dim", 256)),
                         lstm_hidden=int(mc.get("lstm_hidden", 256)))
    policy.load_state_dict(blob["policy"])

    phase = args.phase or blob.get("meta", {}).get("curriculum_phase", "F1")
    ecfg = EvalConfig(phase=phase, games_per_opponent=args.games)
    frozen = evaluate_policy(policy, ecfg)
    rotating = evaluate_policy(policy, ecfg, rotating=True)
    print(json.dumps({"frozen": frozen, "rotating": rotating}, indent=2))


if __name__ == "__main__":
    main()
