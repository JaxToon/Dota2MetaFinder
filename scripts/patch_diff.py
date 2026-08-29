#!/usr/bin/env python
"""Patch-diff detection + migration pipeline (spec 16, M9).

    # just classify a hypothetical patch
    python scripts/patch_diff.py --mutate "hero.attack_damage=26" "ability.0.dmg=130"

    # structural change -> blocked pending approval
    python scripts/patch_diff.py --mutate "ability.1.kind=nuke"

    # detect + auto fine-tune + regression gate a checkpoint
    python scripts/patch_diff.py --mutate "hero.attack_damage=26" \\
        --migrate runs/dota_rl_f2_build/checkpoints/ckpt_000040.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dota_rl.env.patch import HeroStats, PatchManifest, default_manifest  # noqa: E402
from dota_rl.training.patch_pipeline import diff_manifests, migrate  # noqa: E402


def _apply_mutations(m: PatchManifest, muts: list[str]) -> PatchManifest:
    hero = dict(m.hero.__dict__)
    abilities = [dict(a) for a in m.abilities]
    items = [dict(i) for i in m.items]
    for mut in muts:
        path, _, raw = mut.partition("=")
        val = json.loads(raw) if raw[:1] in "0123456789-[{\"" else raw
        parts = path.split(".")
        if parts[0] == "hero":
            hero[parts[1]] = val
        elif parts[0] == "ability":
            abilities[int(parts[1])][parts[2]] = val
        elif parts[0] == "item":
            items[int(parts[1])][parts[2]] = val
    return replace(m, patch_id=m.patch_id + "+patched", hero=HeroStats(**hero),
                   abilities=tuple(abilities), items=tuple(items))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mutate", nargs="*", default=[], help="e.g. hero.attack_damage=26 ability.0.dmg=130")
    p.add_argument("--new-json", default=None, help="load the new manifest hero/abilities/items from JSON")
    p.add_argument("--migrate", default=None, help="pre-patch checkpoint to detect + adapt")
    p.add_argument("--phase", default="F2")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--approve-structural", action="store_true")
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    old = default_manifest()
    if args.new_json:
        d = json.loads(Path(args.new_json).read_text())
        new = replace(old, patch_id=d.get("patch_id", old.patch_id + "+patched"),
                      hero=HeroStats(**d.get("hero", old.hero.__dict__)),
                      abilities=tuple(d.get("abilities", old.abilities)),
                      items=tuple(d.get("items", old.items)))
    else:
        new = _apply_mutations(old, args.mutate)

    diff = diff_manifests(old, new)
    print(json.dumps(diff.to_record(), indent=2))
    print(f"\n-> {'STRUCTURAL (needs human approval)' if diff.structural else 'numeric-only (auto)'}")

    if args.migrate:
        print(f"\nrunning migration from {args.migrate} ...")
        report = migrate(args.migrate, new, phase=args.phase, finetune_iters=args.iters,
                         approve_structural=args.approve_structural,
                         run_dir=args.run_dir or ROOT / "runs" / "patch_migration")
        print(json.dumps({k: v for k, v in report.items() if k != "diff"}, indent=2, default=str))


if __name__ == "__main__":
    main()
