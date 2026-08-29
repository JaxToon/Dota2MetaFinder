"""Checkpoint registry with the metadata required by spec 12.

Every checkpoint records model_hash, config_hash, git_commit, patch_id,
curriculum_phase and metrics so evaluation / rollback / regression can be
reproduced.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch


def _hash_state(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


class CheckpointRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else []

    def save(self, policy, optimizer, *, iteration: int, config: dict,
             patch_id: str, curriculum_phase: str, metrics: dict) -> dict:
        sd = policy.state_dict()
        entry = {
            "iteration": iteration,
            "time": time.time(),
            "model_hash": _hash_state(sd),
            "config_hash": hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16],
            "git_commit": _git_commit(),
            "patch_id": patch_id,
            "curriculum_phase": curriculum_phase,
            "metrics": metrics,
            "path": str(self.root / f"ckpt_{iteration:06d}.pt"),
        }
        torch.save(
            {"policy": sd,
             "optimizer": optimizer.state_dict() if optimizer else None,
             "meta": entry, "config": config},
            entry["path"],
        )
        self.index.append(entry)
        self.index_path.write_text(json.dumps(self.index, indent=2, default=str))
        return entry

    def latest(self) -> dict | None:
        return self.index[-1] if self.index else None

    def load(self, path: str, policy, optimizer=None, map_location="cpu"):
        blob = torch.load(path, map_location=map_location, weights_only=False)
        policy.load_state_dict(blob["policy"])
        if optimizer and blob.get("optimizer"):
            optimizer.load_state_dict(blob["optimizer"])
        return blob["meta"]
