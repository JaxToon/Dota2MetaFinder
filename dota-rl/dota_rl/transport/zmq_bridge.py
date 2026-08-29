"""Lua <-> Python transport (spec 3.1 Bridge, ticket #4).

Stub. Python-first per spec 3.2 (defer C++/Rust until profiling proves IPC is the
bottleneck). The bridge must implement timeout / retry / message-order checks
against the observation & action schema versions. Not active until the real
Dota adapter is wired (M0).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass


@dataclass
class BridgeConfig:
    endpoint: str = "tcp://127.0.0.1:5555"
    recv_timeout_ms: int = 200
    max_retries: int = 3


class ZmqBridge:
    def __init__(self, cfg: BridgeConfig | None = None):
        self.cfg = cfg or BridgeConfig()
        self._last_step_id = -1
        try:
            import zmq  # noqa: F401
            self._have_zmq = True
        except Exception:
            self._have_zmq = False

    def available(self) -> bool:
        return self._have_zmq

    def check_order(self, step_id: int) -> bool:
        ok = step_id == self._last_step_id + 1
        self._last_step_id = step_id
        return ok

    def send_action(self, action: dict) -> None:  # pragma: no cover - stub
        raise NotImplementedError("wire pyzmq REQ/REP here once M0 passes")

    def recv_observation(self) -> dict:  # pragma: no cover - stub
        raise NotImplementedError

    @staticmethod
    def encode(obj: dict) -> bytes:
        return json.dumps(obj).encode()

    @staticmethod
    def decode(buf: bytes) -> dict:
        return json.loads(buf.decode())
