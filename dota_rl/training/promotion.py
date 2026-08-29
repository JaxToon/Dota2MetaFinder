"""Multidimensional curriculum promotion gates (spec 8).

Replaces the single "winrate > 70%" idea with a per-phase checklist. Returns a
report; the operator (or scripts/train.py) decides to advance.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateReport:
    phase: str
    passed: bool
    checks: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"[promotion gate {self.phase}] {'PASS' if self.passed else 'FAIL'}"]
        for k, (ok, detail) in self.checks.items():
            lines.append(f"  {'ok ' if ok else 'NO '} {k}: {detail}")
        return "\n".join(lines)


def check_f0(env_health: dict) -> GateReport:
    checks = {}
    vt = env_health.get("valid_transition_rate", 0.0)
    checks["valid_transitions>0.99"] = (vt >= 0.99, f"{vt:.4f}")
    dl = env_health.get("protocol_deadlocks", 1)
    checks["zero_deadlocks"] = (dl == 0, str(dl))
    ro = env_health.get("reward_observed", False)
    checks["reward_signal_observable"] = (bool(ro), str(ro))
    inv = env_health.get("invalid_action_rate", 1.0)
    checks["invalid_action_rate<0.01"] = (inv < 0.01, f"{inv:.4f}")
    passed = all(ok for ok, _ in checks.values())
    return GateReport("F0", passed, checks)


def check_f2(f2_eval: dict, f1_eval: dict, baseline_winrate: float,
             f1_prev_winrate: float | None = None) -> GateReport:
    s = f2_eval["vs_scripted_bot"]
    f1 = f1_eval["vs_scripted_bot"]
    f1_floor = (f1_prev_winrate - 0.15) if f1_prev_winrate is not None else 0.5
    checks = {
        "beats_baseline": (s["winrate"] > baseline_winrate + 0.1,
                           f"{s['winrate']:.3f} vs baseline {baseline_winrate:.3f}"),
        "winrate_ci_low>0.5": (s["winrate_ci_low"] > 0.5, f"{s['winrate_ci_low']:.3f}"),
        "no_regression_on_F1": (f1["winrate"] >= f1_floor,
                                f"F1 winrate {f1['winrate']:.3f} (floor {f1_floor:.3f})"),
        "build_progression": (s["net_worth_mean"] > f1["net_worth_mean"] * 0.6,
                              f"F2 nw {s['net_worth_mean']:.0f}"),
    }
    passed = all(ok for ok, _ in checks.values())
    return GateReport("F2", passed, checks)


def check_f1(eval_out: dict, baseline_winrate: float, seeds_stable: bool) -> GateReport:
    s = eval_out["vs_scripted_bot"]
    checks = {
        "beats_baseline": (s["winrate"] > baseline_winrate + 0.1,
                           f"{s['winrate']:.3f} vs baseline {baseline_winrate:.3f}"),
        "winrate_ci_low>0.5": (s["winrate_ci_low"] > 0.5, f"{s['winrate_ci_low']:.3f}"),
        "stable_across_seeds": (seeds_stable, str(seeds_stable)),
        "death_rate_sane": (s["deaths_mean"] < s["kills_mean"] + 3,
                            f"K {s['kills_mean']:.1f} / D {s['deaths_mean']:.1f}"),
    }
    passed = all(ok for ok, _ in checks.values())
    return GateReport("F1", passed, checks)
