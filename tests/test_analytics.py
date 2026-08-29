"""Spec 15 / AC-08: event-aligned meta analytics produce sane, matched outputs."""
import numpy as np
import torch

from dota_rl.agents.policy import ActorCritic
from dota_rl.analytics import build_analysis, heatmaps
from dota_rl.analytics.report import build_report, render_text
from dota_rl.analytics.value_events import analyze_events, rollout_trace


def _traces(n=6, phase="F1"):
    torch.manual_seed(0)
    pol = ActorCritic(encoder_dim=32, lstm_hidden=32)
    return pol, [rollout_trace(pol, phase=phase, seed=500 + i) for i in range(n)]


def test_trace_records_are_well_formed():
    _, traces = _traces(4)
    for tr in traces:
        assert len(tr.steps) > 0
        for s in tr.steps:
            assert set(s) >= {"value", "pos", "hp_ratio", "action_type", "events", "t"}
            assert 0.0 <= s["pos"][0] <= 1.0 and 0.0 <= s["pos"][1] <= 1.0
            assert np.isfinite(s["value"])
        assert tr.outcome in (-1, 0, 1)


def test_event_analysis_has_baseline_and_matched_deltas():
    _, traces = _traces(8)
    out = analyze_events(traces, k=3, m=4)
    assert "baseline_delta_v" in out and out["baseline_delta_v"]["n"] > 0
    for tag, d in out["events"].items():
        assert "delta_vs_baseline" in d
        assert set(d["covariates"]) == {"level", "net_worth", "self_hp", "enemy_hp", "enemy_dist"}


def test_heatmap_grids_normalised_and_shaped():
    _, traces = _traces(5)
    g = heatmaps.grids(traces, bins=10)
    occ = np.asarray(g["occupancy"])
    assert occ.shape == (10, 10)
    assert abs(occ.sum() - 1.0) < 1e-6
    assert len(g["occupancy_centre_of_mass"]) == 2
    assert isinstance(heatmaps.ascii_heatmap(g["occupancy"]), str)


def test_build_analysis_summarize_keys():
    _, traces = _traces(6)
    s = build_analysis.summarize(traces)
    for key in ("winrate", "ability_timing", "item_timing", "power_spikes", "trade_discipline"):
        assert key in s


def test_full_report_renders():
    pol, _ = _traces(2)
    report = build_report(pol, phase="F1", episodes=4, seed_base=700)
    text = render_text(report)
    assert "META-DISCOVERY REPORT" in text and "LANE CONTROL" in text
