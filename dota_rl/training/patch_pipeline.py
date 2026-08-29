"""Patch-resilience pipeline (spec 16, milestone M9, AC-09).

    patch detected
      -> extract old/new hero + item metadata
      -> schema diff + semantic diff
      -> numeric-only  : auto fine-tune -> regression gate -> promote/rollback
      -> structural    : human approval gate -> (approved) fine-tune -> gate

Fine-tuning follows spec section 4: retrain 1-3% of the original schedule from
the pre-patch checkpoint (attribute embeddings absorb *numeric* stat changes;
new ability/item *kinds* are the structural case that needs review).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..agents.policy import ActorCritic
from ..env.mock_env import MockDotaEnv
from ..env.patch import PatchManifest, default_manifest
from ..env.reward import RewardConfig
from ..evaluation.harness import EvalConfig, evaluate_policy
from ..evaluation.regression import check_regression
from .ppo import PPOConfig, PPOTrainer
from .rollout import RolloutCollector

_NUMERIC_TOL = 1e-9


@dataclass
class PatchDiff:
    old_patch: str
    new_patch: str
    changed_entities: list[str] = field(default_factory=list)
    structural_changes: list[str] = field(default_factory=list)
    approval_state: str = "auto"        # auto | needs_human | approved | rejected

    @property
    def structural(self) -> bool:
        return bool(self.structural_changes)

    def to_record(self) -> dict:
        return asdict(self) | {"structural": self.structural}


def _num_changes(prefix: str, a: dict, b: dict) -> tuple[list[str], list[str]]:
    numeric, structural = [], []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            structural.append(f"{prefix}.{key} added/removed")
        elif key == "kind" and a[key] != b[key]:
            structural.append(f"{prefix}.kind {a[key]}->{b[key]}")
        elif isinstance(a[key], (int, float)) and abs(a[key] - b[key]) > _NUMERIC_TOL:
            numeric.append(f"{prefix}.{key} {a[key]}->{b[key]}")
        elif a[key] != b[key]:
            structural.append(f"{prefix}.{key} {a[key]}->{b[key]}")
    return numeric, structural


def diff_manifests(old: PatchManifest, new: PatchManifest) -> PatchDiff:
    changed, structural = [], []

    if old.structural_key != new.structural_key:
        structural.append(f"structural_key {old.structural_key} -> {new.structural_key}")

    n, s = _num_changes("hero", old.hero.__dict__, new.hero.__dict__)
    changed += n; structural += s

    for i in range(max(len(old.abilities), len(new.abilities))):
        oa = old.abilities[i] if i < len(old.abilities) else {}
        na = new.abilities[i] if i < len(new.abilities) else {}
        n, s = _num_changes(f"ability[{i}]", oa or {"_": 0}, na or {"_": 0})
        changed += n; structural += s
    for i in range(max(len(old.items), len(new.items))):
        oi = old.items[i] if i < len(old.items) else {}
        ni = new.items[i] if i < len(new.items) else {}
        n, s = _num_changes(f"item[{i}]", oi or {"_": 0}, ni or {"_": 0})
        changed += n; structural += s

    d = PatchDiff(old.patch_id, new.patch_id, changed, structural)
    d.approval_state = "needs_human" if d.structural else "auto"
    return d


def _make_env_factory(manifest: PatchManifest, phase: str, reward_cfg: RewardConfig):
    def make(i: int) -> MockDotaEnv:
        return MockDotaEnv(phase=phase, reward_cfg=reward_cfg, seed=4242 + i,
                           patch=manifest, max_steps=400)
    return make


def migrate(pre_patch_ckpt: str, new_manifest: PatchManifest, *,
            phase: str = "F2", finetune_iters: int = 20, num_envs: int = 16,
            horizon: int = 128, device: str = "cpu",
            approve_structural: bool = False, run_dir: str | Path | None = None) -> dict:
    """Run the patch pipeline for one detected patch. Returns a report dict and,
    if ``run_dir`` is given, writes ``patch_diff.json`` (spec 12 data contract)."""
    blob = torch.load(pre_patch_ckpt, map_location=device, weights_only=False)
    mc = blob.get("config", {}).get("model", {})
    old_manifest = default_manifest().bumped(
        patch_id=blob.get("meta", {}).get("patch_id", "pre_patch"))

    diff = diff_manifests(old_manifest, new_manifest)
    report: dict = {"diff": diff.to_record(), "phase": phase}

    if diff.structural and not approve_structural:
        diff.approval_state = "needs_human"
        report["decision"] = "BLOCKED: structural change requires human approval"
        report["promoted"] = False
        _dump(run_dir, diff, report)
        return report
    if diff.structural:
        diff.approval_state = "approved"

    reward_cfg = RewardConfig()
    policy = ActorCritic(encoder_dim=int(mc.get("encoder_dim", 256)),
                         lstm_hidden=int(mc.get("lstm_hidden", 256))).to(device)
    policy.load_state_dict(blob["policy"], strict=False)

    ecfg = EvalConfig(phase=phase, games_per_opponent=24)
    # baseline: how the *unadapted* pre-patch policy does on the new patch
    baseline_eval = _eval_on(policy, ecfg, new_manifest, device, reward_cfg)
    base_s = baseline_eval["vs_scripted_bot"]
    report["pre_patch_policy_on_new_patch"] = base_s

    # spec 5.3: attribute embeddings absorb *numeric* stat changes. If the
    # pre-patch policy is still competent on the new patch, no retrain needed.
    if not diff.structural and base_s["winrate"] >= 0.6 and base_s["winrate_ci_low"] >= 0.4:
        report["decision"] = "NO-OP: embeddings absorbed the numeric change; pre-patch checkpoint kept"
        report["promoted"] = False
        report["adaptation"] = "none needed"
        _dump(run_dir, diff, report)
        return report

    # fine-tune 1-3% of a nominal schedule, gently (resume optimiser, low LR)
    trainer = PPOTrainer(policy, PPOConfig(epochs=3, num_minibatches=4, lr=8e-5,
                                           entropy_coef=0.008, entropy_anneal=False), device)
    if blob.get("optimizer"):
        try:
            trainer.opt.load_state_dict(blob["optimizer"])
            for g in trainer.opt.param_groups:
                g["lr"] = 8e-5
        except Exception:
            pass
    collector = RolloutCollector(_make_env_factory(new_manifest, phase, reward_cfg),
                                 num_envs, horizon, device)
    for _ in range(finetune_iters):
        trainer.update(collector.collect(policy))

    adapted_eval = _eval_on(policy, ecfg, new_manifest, device, reward_cfg)
    adapt_s = adapted_eval["vs_scripted_bot"]
    report["adapted_policy_on_new_patch"] = adapt_s

    reg = check_regression(baseline_eval, adapted_eval, threshold=0.05)
    promoted = (adapt_s["winrate"] >= base_s["winrate"] - 0.02
                and adapt_s["winrate_ci_low"] >= 0.3)
    report["regression_check"] = {"passed": reg.passed, "reason": reg.reason}
    report["decision"] = ("PROMOTE adapted checkpoint" if promoted
                          else "ROLLBACK: fine-tune did not recover performance")
    report["promoted"] = bool(promoted)

    if run_dir and promoted:
        out = Path(run_dir) / f"checkpoint_{new_manifest.content_hash()}.pt"
        torch.save({"policy": policy.state_dict(), "config": blob.get("config", {}),
                    "meta": {"patch_id": new_manifest.patch_id,
                             "curriculum_phase": phase, "adapted_from": pre_patch_ckpt}}, out)
        report["adapted_checkpoint"] = str(out)

    _dump(run_dir, diff, report)
    return report


def _eval_on(policy, ecfg, manifest, device, reward_cfg):
    return evaluate_policy(policy, ecfg, device=device, reward_cfg=reward_cfg, manifest=manifest)


def _dump(run_dir, diff: PatchDiff, report: dict) -> None:
    if not run_dir:
        return
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "patch_diff.json").write_text(json.dumps(report, indent=2, default=str))
