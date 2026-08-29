"""Lua <-> Python transport (spec 3.1 Bridge, ticket #4).

Protocol (JSON lines over a ZeroMQ REP socket, matching ``schemas.proto``):
  * the in-game Lua bot is the REQ client: each ``Think()`` it sends
    ``{"type": "obs", "step_id", "schema_version", "observation": {...},
       "reward", "terminated", "truncated"}`` and blocks for a reply,
  * this Python side is the REP server: it validates schema_version + step_id
    order, hands the observation to the learner, and replies with
    ``{"type": "action", "action": {...}}``.

Python-first per spec 3.2 (defer C++/Rust until profiling proves IPC is the
bottleneck). Not on the training path until the real adapter is wired (M0);
``InProcessBridge`` lets the adapter + protocol be tested without pyzmq / Dota.
"""
from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass

from ..env.protocol import ACTION_SCHEMA_VERSION, OBS_SCHEMA_VERSION


@dataclass
class BridgeConfig:
    endpoint: str = "tcp://127.0.0.1:5555"
    recv_timeout_ms: int = 500
    max_retries: int = 3
    strict_order: bool = True


class BridgeError(RuntimeError):
    pass


class _BaseBridge:
    def __init__(self, cfg: BridgeConfig | None = None):
        self.cfg = cfg or BridgeConfig()
        self._last_step_id = -1
        self.dropped = 0
        self.out_of_order = 0

    # ---- framing --------------------------------------------------------
    @staticmethod
    def encode(obj: dict) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode()

    @staticmethod
    def decode(buf: bytes) -> dict:
        return json.loads(buf.decode())

    def _check(self, msg: dict) -> dict:
        if msg.get("schema_version") not in (None, OBS_SCHEMA_VERSION):
            raise BridgeError(f"obs schema_version {msg.get('schema_version')} "
                              f"!= {OBS_SCHEMA_VERSION}")
        sid = int(msg.get("step_id", self._last_step_id + 1))
        if self.cfg.strict_order and sid != self._last_step_id + 1:
            self.out_of_order += 1
            if sid <= self._last_step_id:
                self.dropped += 1
        self._last_step_id = sid
        return msg

    def stats(self) -> dict:
        return {"last_step_id": self._last_step_id, "dropped": self.dropped,
                "out_of_order": self.out_of_order}


class ZmqBridge(_BaseBridge):
    """Real REP server. Requires ``pyzmq``."""

    def __init__(self, cfg: BridgeConfig | None = None):
        super().__init__(cfg)
        try:
            import zmq
        except Exception as e:  # pragma: no cover - env without pyzmq
            raise BridgeError("pyzmq not installed; use InProcessBridge for tests") from e
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.RCVTIMEO, self.cfg.recv_timeout_ms)
        self._sock.bind(self.cfg.endpoint)

    def recv_observation(self) -> dict:
        for attempt in range(self.cfg.max_retries):
            try:
                return self._check(self.decode(self._sock.recv()))
            except self._zmq.Again:
                if attempt == self.cfg.max_retries - 1:
                    raise BridgeError("timeout waiting for observation from Lua")
                time.sleep(0.01)
        raise BridgeError("unreachable")

    def send_action(self, action: dict) -> None:
        self._sock.send(self.encode({"type": "action",
                                     "schema_version": ACTION_SCHEMA_VERSION,
                                     "action": action}))

    def close(self) -> None:  # pragma: no cover
        self._sock.close(0)


class InProcessBridge(_BaseBridge):
    """Same contract, backed by two queues -- for tests and a simulator game loop.

    The 'game' side calls ``game_send_obs`` / ``game_recv_action``; the learner
    side calls ``recv_observation`` / ``send_action``.
    """

    def __init__(self, cfg: BridgeConfig | None = None):
        super().__init__(cfg)
        self._to_learner: queue.Queue = queue.Queue()
        self._to_game: queue.Queue = queue.Queue()

    # learner side
    def recv_observation(self) -> dict:
        try:
            return self._check(self._to_learner.get(timeout=self.cfg.recv_timeout_ms / 1000))
        except queue.Empty:
            raise BridgeError("timeout waiting for observation")

    def send_action(self, action: dict) -> None:
        self._to_game.put({"type": "action", "action": action})

    # game side
    def game_send_obs(self, msg: dict) -> None:
        self._to_learner.put(msg)

    def game_recv_action(self, timeout: float = 1.0) -> dict:
        return self._to_game.get(timeout=timeout)
