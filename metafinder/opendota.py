"""OpenDota REST client -- disk-cached, rate-limited, stdlib only.

OpenDota serves aggregated public-match data under a permissive API (read-only,
official). https://docs.opendota.com/ . A free API key raises the limit; set
``OPENDOTA_API_KEY`` in the environment to use it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.opendota.com/api"
CACHE_DIR = Path(os.environ.get("METAFINDER_CACHE", ".cache/opendota"))
USER_AGENT = "metafinder/0.1 (Dota 2 meta research; https://github.com/JaxToon/Dota2MetaFinder)"
_MIN_INTERVAL = 1.1          # seconds between live calls (stay under 60/min free tier)
_last_call = 0.0


class OpenDotaError(RuntimeError):
    pass


def _key() -> str | None:
    return os.environ.get("OPENDOTA_API_KEY") or None


def _cache_path(path: str, params: dict | None) -> Path:
    raw = path + "?" + urllib.parse.urlencode(sorted((params or {}).items()))
    return CACHE_DIR / (hashlib.sha1(raw.encode()).hexdigest() + ".json")


def get(path: str, params: dict | None = None, *, max_age: float = 86400.0,
        allow_live: bool = True):
    """GET /api{path}. Serves from cache if younger than ``max_age`` seconds."""
    cp = _cache_path(path, params)
    if cp.exists() and (time.time() - cp.stat().st_mtime) < max_age:
        return json.loads(cp.read_text(encoding="utf-8"))
    if not allow_live:
        if cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))
        raise OpenDotaError(f"no cached response for {path} and allow_live=False")

    q = dict(params or {})
    if _key():
        q["api_key"] = _key()
    url = f"{BASE}{path}"
    if q:
        url += "?" + urllib.parse.urlencode(q)

    global _last_call
    wait = _MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            _last_call = time.time()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:                      # rate limited
                time.sleep(2 + attempt * 3)
                continue
            if e.code in (502, 503, 504):
                time.sleep(1 + attempt)
                continue
            raise OpenDotaError(f"{path} -> HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                raise OpenDotaError(f"{path} -> {e}") from e
            time.sleep(1 + attempt)
    raise OpenDotaError("unreachable")


# --- typed endpoint helpers ------------------------------------------------
def heroes() -> list[dict]:
    return get("/heroes", max_age=30 * 86400)


def hero_stats() -> list[dict]:
    """Per-hero pick/win by rank bracket, pub, pro, turbo (rolling window)."""
    return get("/heroStats", max_age=6 * 3600)


def hero_matchups(hero_id: int) -> list[dict]:
    """{hero_id, games_played, wins} vs every other hero (opposite team)."""
    return get(f"/heroes/{hero_id}/matchups", max_age=3 * 86400)


def public_matches(less_than_match_id: int | None = None,
                   min_rank: int | None = None) -> list[dict]:
    p: dict = {}
    if less_than_match_id:
        p["less_than_match_id"] = less_than_match_id
    if min_rank:
        p["min_rank"] = min_rank
    return get("/publicMatches", p, max_age=365 * 86400)   # immutable once fetched


def scenarios_lane_roles(hero_id: int | None = None) -> list[dict]:
    p = {"hero_id": hero_id} if hero_id else None
    return get("/scenarios/laneRoles", p, max_age=3 * 86400)


def item_popularity(hero_id: int) -> dict:
    """{start_game_items, early_game_items, mid_game_items, late_game_items} ->
    {item_id: count}."""
    return get(f"/heroes/{hero_id}/itemPopularity", max_age=3 * 86400)


def item_constants() -> dict:
    """item key -> {id, dname, cost, qual, ...}."""
    return get("/constants/items", max_age=30 * 86400)
