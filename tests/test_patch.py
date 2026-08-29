"""Spec 16 / AC-09: patch-diff classification + migration decision flow."""
from dataclasses import replace

import torch

from dota_rl.agents.policy import ActorCritic
from dota_rl.env.patch import HeroStats, default_manifest
from dota_rl.training.patch_pipeline import diff_manifests, migrate


def test_manifest_hash_and_structural_key():
    m = default_manifest()
    assert m.content_hash() == default_manifest().content_hash()
    m2 = m.bumped(hero=replace(m.hero, attack_damage=99.0))
    assert m2.content_hash() != m.content_hash()
    assert m2.structural_key == m.structural_key          # numeric only


def test_numeric_change_is_auto():
    old = default_manifest()
    new = old.bumped(patch_id="p2", hero=replace(old.hero, attack_damage=26.0))
    d = diff_manifests(old, new)
    assert not d.structural and d.approval_state == "auto"
    assert any("attack_damage" in c for c in d.changed_entities)


def test_ability_kind_change_is_structural():
    old = default_manifest()
    abts = [dict(a) for a in old.abilities]
    abts[1]["kind"] = "nuke"          # was "heal"
    new = old.bumped(patch_id="p2", abilities=tuple(abts))
    d = diff_manifests(old, new)
    assert d.structural and d.approval_state == "needs_human"


def test_added_item_slot_is_structural():
    old = default_manifest()
    new = old.bumped(patch_id="p2", items=old.items + ({"cd": 10, "kind": "nuke", "dmg": 40.0, "range": 0.2},))
    d = diff_manifests(old, new)
    assert d.structural


def _tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    pol = ActorCritic(encoder_dim=32, lstm_hidden=32)
    p = tmp_path / "ckpt.pt"
    torch.save({"policy": pol.state_dict(),
                "config": {"model": {"encoder_dim": 32, "lstm_hidden": 32}},
                "meta": {"patch_id": "sandbox_mock_v1", "curriculum_phase": "F1"}}, p)
    return str(p)


def test_structural_patch_is_blocked_without_approval(tmp_path):
    ckpt = _tiny_ckpt(tmp_path)
    old = default_manifest()
    abts = [dict(a) for a in old.abilities]
    abts[0]["kind"] = "heal"
    new = old.bumped(patch_id="p2", abilities=tuple(abts))
    report = migrate(ckpt, new, phase="F1", finetune_iters=0, run_dir=tmp_path)
    assert report["promoted"] is False
    assert "BLOCKED" in report["decision"]
    assert (tmp_path / "patch_diff.json").exists()
