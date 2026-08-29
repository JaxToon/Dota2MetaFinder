"""Convert lists of numpy observation dicts into batched torch tensors."""
from __future__ import annotations

import numpy as np
import torch

_FLOAT_KEYS = ("agent_scalars", "abilities", "items", "entities")
_INT_KEYS = ("ability_effect_ids", "entity_hero_ids")
_MASK_KEYS = ("action_type", "entity", "ability", "item")


def obs_to_tensors(obs_list: list[dict], device="cpu") -> dict:
    out = {}
    for k in _FLOAT_KEYS:
        out[k] = torch.as_tensor(
            np.stack([o[k] for o in obs_list]), dtype=torch.float32, device=device
        )
    for k in _INT_KEYS:
        out[k] = torch.as_tensor(
            np.stack([o[k] for o in obs_list]), dtype=torch.long, device=device
        )
    out["masks"] = {
        k: torch.as_tensor(
            np.stack([np.asarray(o["masks"][k]) for o in obs_list]),
            dtype=torch.bool, device=device,
        )
        for k in _MASK_KEYS
    }
    return out


def stack_time(step_tensors: list[dict]) -> dict:
    """Stack a list of per-step batched obs dicts into (T, B, ...) tensors."""
    out = {}
    for k in _FLOAT_KEYS + _INT_KEYS:
        out[k] = torch.stack([s[k] for s in step_tensors], dim=0)
    out["masks"] = {
        k: torch.stack([s["masks"][k] for s in step_tensors], dim=0)
        for k in _MASK_KEYS
    }
    return out
