"""Real Dota 2 sandbox adapter -- stub (spec ticket #10, milestone M0).

Implements the SAME reset()/step() contract as ``MockDotaEnv`` so the learner is
unchanged when swapped in. It is intentionally not wired up: per blueprint
section 0 the first real deliverable is the Lua API probe
(``scripts/api_probe.lua``) that verifies which observation fields the *public*
Bot Scripting API actually exposes. Until that probe passes, training runs on
``MockDotaEnv``.
"""
from __future__ import annotations


class DotaAdapterNotAvailable(RuntimeError):
    pass


class DotaSandboxEnv:
    def __init__(self, *args, **kwargs):
        raise DotaAdapterNotAvailable(
            "Real Dota adapter is not implemented. Run scripts/api_probe.lua in a "
            "custom lobby with `-dev -vconsole +dota_force_gamemode 1` first, fill "
            "in the observation access matrix, then implement the ZMQ bridge "
            "(transport/zmq_bridge.py) against this contract."
        )

    def reset(self, seed=None):  # pragma: no cover - stub
        raise DotaAdapterNotAvailable

    def step(self, action):  # pragma: no cover - stub
        raise DotaAdapterNotAvailable
