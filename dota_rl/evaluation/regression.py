"""Champion-vs-challenger regression gate (spec 9.3, 10 step 12).

A challenger is only promoted if it does not *significantly* regress against the
champion on the frozen suite: require both a point-estimate margin and a
confidence-lower-bound that does not fall below (champion - threshold).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegressionResult:
    passed: bool
    champion_winrate: float
    challenger_winrate: float
    challenger_ci_low: float
    reason: str


def check_regression(champion_eval: dict, challenger_eval: dict,
                     threshold: float = 0.05) -> RegressionResult:
    c = champion_eval["vs_scripted_bot"]
    n = challenger_eval["vs_scripted_bot"]
    champ_wr = c["winrate"]
    champ_lo = c["winrate_ci_low"]
    chal_wr = n["winrate"]
    chal_lo = n["winrate_ci_low"]

    # Regression = the point estimate drops materially AND the challenger's
    # confidence-lower-bound falls below the champion's own lower bound
    # (spec 9.3: require both a point-estimate and a CI-lower-bound criterion).
    point_ok = chal_wr + 1e-9 >= champ_wr - threshold
    ci_ok = chal_lo + 1e-9 >= champ_lo - threshold
    if point_ok and ci_ok:
        return RegressionResult(True, champ_wr, chal_wr, chal_lo, "no significant regression")
    return RegressionResult(
        False, champ_wr, chal_wr, chal_lo,
        f"challenger winrate {chal_wr:.3f} (ci_low {chal_lo:.3f}) regressed vs "
        f"champion {champ_wr:.3f} (ci_low {champ_lo:.3f}) beyond threshold {threshold}",
    )
