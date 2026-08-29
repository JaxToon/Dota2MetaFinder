#!/usr/bin/env python
"""Event-aligned meta / power-spike report from a checkpoint (spec 15, M8).

    python scripts/meta_report.py runs/dota_rl_f1_baseline/checkpoints/ckpt_000060.pt
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
from dota_rl.analytics.value_events import analyze_events, rollout_trace  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--phase", default=None)
    args = p.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = blob.get("config", {}).get("model", {})
    policy = ActorCritic(encoder_dim=int(mc.get("encoder_dim", 256)),
                         lstm_hidden=int(mc.get("lstm_hidden", 256)))
    policy.load_state_dict(blob["policy"])
    phase = args.phase or blob.get("meta", {}).get("curriculum_phase", "F1")

    traces = [rollout_trace(policy, phase=phase, seed=30000 + i) for i in range(args.episodes)]
    report = analyze_events(traces)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
