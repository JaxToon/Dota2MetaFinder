"""Positioning / value heatmaps (spec 15, repo §11).

Two grids over the normalised map:
  * occupancy  -- where the agent spends time (lane control),
  * mean V(s)  -- where the policy thinks it is winning.
"""
from __future__ import annotations

import numpy as np

from .value_events import Trace


def grids(traces: list[Trace], bins: int = 12) -> dict:
    occ = np.zeros((bins, bins), dtype=np.float64)
    val_sum = np.zeros((bins, bins), dtype=np.float64)
    val_cnt = np.zeros((bins, bins), dtype=np.float64)

    for tr in traces:
        for s in tr.steps:
            x, y = s["pos"]
            ix = min(bins - 1, max(0, int(x * bins)))
            iy = min(bins - 1, max(0, int(y * bins)))
            occ[iy, ix] += 1.0
            val_sum[iy, ix] += s["value"]
            val_cnt[iy, ix] += 1.0

    total = occ.sum() or 1.0
    occ_norm = occ / total
    with np.errstate(invalid="ignore", divide="ignore"):
        val_mean = np.where(val_cnt > 0, val_sum / np.maximum(val_cnt, 1), np.nan)

    # centre of mass of occupancy = the agent's preferred lane position
    ys, xs = np.mgrid[0:bins, 0:bins]
    com_x = float((occ * (xs + 0.5) / bins).sum() / total)
    com_y = float((occ * (ys + 0.5) / bins).sum() / total)

    return {
        "bins": bins,
        "occupancy": occ_norm.round(4).tolist(),
        "value_mean": np.where(np.isnan(val_mean), None, val_mean.round(3)).tolist(),
        "occupancy_centre_of_mass": [round(com_x, 3), round(com_y, 3)],
        "lane_control": _lane_read(com_x, com_y),
    }


def _lane_read(cx: float, cy: float) -> str:
    # agent tower ~(0.10,0.10), enemy tower ~(0.90,0.90), creeps mid ~(0.5,0.5)
    d_mid = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    if cx + cy < 0.7:
        return "defensive: holds near own tower"
    if cx + cy > 1.3:
        return "aggressive: pushes toward the enemy tower"
    if d_mid < 0.12:
        return "balanced: contests the creep line at mid"
    return "roams the lane"


def ascii_heatmap(grid: list[list[float]], width: int = 24) -> str:
    a = np.asarray([[0.0 if v is None else v for v in row] for row in grid], dtype=float)
    if a.max() > a.min():
        a = (a - a.min()) / (a.max() - a.min())
    ramp = " .:-=+*#%@"
    lines = []
    for row in a[::-1]:  # flip so y increases upward
        lines.append("".join(ramp[min(len(ramp) - 1, int(v * (len(ramp) - 1)))] for v in row))
    return "\n".join(lines)
