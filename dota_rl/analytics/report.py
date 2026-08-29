"""Assemble a meta-discovery report from episode traces (spec 15).

The output is deliberately about *distributions and matched comparisons*, not a
single "best item = X" claim (spec 15 warning).
"""
from __future__ import annotations

import torch

from ..agents.policy import ActorCritic
from . import build_analysis, heatmaps
from .value_events import analyze_events, rollout_trace


def build_report(policy: ActorCritic, phase: str = "F2", episodes: int = 40,
                 seed_base: int = 30000, device: str = "cpu") -> dict:
    traces = [rollout_trace(policy, phase=phase, seed=seed_base + i, device=device)
              for i in range(episodes)]
    return {
        "phase": phase,
        "episodes": episodes,
        "summary": build_analysis.summarize(traces),
        "value_events": analyze_events(traces),
        "heatmaps": heatmaps.grids(traces),
    }


def load_policy(checkpoint: str, device: str = "cpu") -> tuple[ActorCritic, str]:
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    mc = blob.get("config", {}).get("model", {})
    pol = ActorCritic(encoder_dim=int(mc.get("encoder_dim", 256)),
                      lstm_hidden=int(mc.get("lstm_hidden", 256))).to(device)
    pol.load_state_dict(blob["policy"])
    pol.eval()
    phase = blob.get("meta", {}).get("curriculum_phase", "F2")
    return pol, phase


def render_text(report: dict) -> str:
    s = report["summary"]
    L = [
        f"META-DISCOVERY REPORT  --  phase {report['phase']}, {report['episodes']} episodes",
        "=" * 66,
        f"winrate {s['winrate']:.2f} | mean episode length {s['mean_episode_len']:.0f}",
        "",
        "LANE CONTROL",
        f"  {report['heatmaps']['lane_control']}",
        f"  occupancy centre of mass: {report['heatmaps']['occupancy_centre_of_mass']}",
        "  occupancy heatmap (own tower bottom-left, enemy tower top-right):",
    ]
    L += ["    " + ln for ln in heatmaps.ascii_heatmap(report["heatmaps"]["occupancy"]).splitlines()]
    L += ["", "ABILITY USAGE (learned timing)"]
    for name, d in s["ability_timing"].items():
        t = d["at_game_time"]; eh = d["at_enemy_hp"]
        L.append(f"  {name}: {d['uses_per_episode']}/ep, "
                 f"t~{t.get('p50','?')} (p10-p90 {t.get('p10','?')}-{t.get('p90','?')}), "
                 f"enemy HP {eh.get('p50','n/a')}, self mana {d['at_mana'].get('p50','?')}")
    L += ["", "ITEM USAGE"]
    for name, d in s["item_timing"].items():
        L.append(f"  {name}: {d['uses_per_episode']}/ep, first use t~{d['first_use_game_time'].get('p50','?')}, "
                 f"at self HP {d['at_self_hp'].get('p50','?')}")
    L += ["", "POWER SPIKES (dV after event)"]
    for tag, d in s["power_spikes"].items():
        L.append(f"  {tag}: dV {d.get('mean',0):+.3f} (n={d.get('n',0)}) around level {d['at']['level']}, t~{d['at']['t']:.0f}")
    L += ["", "EVENT-ALIGNED VALUE (vs matched baseline)"]
    for tag, d in list(report["value_events"]["events"].items())[:8]:
        L.append(f"  {tag}: dV_vs_baseline {d['delta_vs_baseline']:+.3f} (n={d['n']}), "
                 f"covariates {d['covariates']}")
    L += ["", "TRADE DISCIPLINE"]
    td = s["trade_discipline"]
    L.append(f"  {td['reads_as']}")
    L.append(f"  HP advantage when engaging: {td['hp_advantage_when_engaging'].get('mean','?')} "
             f"vs not engaging: {td['hp_advantage_when_not_engaging'].get('mean','?')}")
    return "\n".join(L)
