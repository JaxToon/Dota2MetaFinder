"""Versioned experiment config loading (spec 6/10.1: weights & hyperparameters
live in config, never hard-coded). Thin wrapper over YAML with dotted overrides.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def apply_dotted(cfg: dict, dotted: list[str]) -> dict:
    """Apply CLI overrides like ``ppo.lr=1e-3`` ``train.iterations=5``."""
    cfg = copy.deepcopy(cfg)
    for item in dotted:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def pick(cfg: dict, section: str, dataclass_type) -> Any:
    """Build a dataclass from cfg[section], ignoring unknown keys."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(dataclass_type)}
    kwargs = {k: v for k, v in (cfg.get(section) or {}).items() if k in fields}
    return dataclass_type(**kwargs)
