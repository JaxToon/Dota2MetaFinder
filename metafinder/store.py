"""Local SQLite store of public matches -- the raw material for synergy and a
second counter signal. Immutable rows; fetch once, query forever.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from . import opendota

DB_PATH = Path(os.environ.get("METAFINDER_DB", ".cache/matches.sqlite"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
  match_id       INTEGER PRIMARY KEY,
  radiant_win    INTEGER NOT NULL,
  avg_rank_tier  INTEGER,
  duration       INTEGER,
  start_time     INTEGER,
  radiant        TEXT NOT NULL,
  dire           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_rank ON matches(avg_rank_tier);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(_SCHEMA)
    return con


def count(con: sqlite3.Connection) -> int:
    return con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]


def ingest(target: int = 20000, *, min_rank: int | None = None,
           con: sqlite3.Connection | None = None, verbose: bool = True) -> int:
    """Fetch public matches back from now until the store holds >= ``target``
    usable rows. ``min_rank`` is an OpenDota rank tier (e.g. 60 = Divine)."""
    own = con is None
    con = con or connect()
    have = count(con)
    less_than = None
    added = 0
    empty_batches = 0
    while have + added < target and empty_batches < 5:
        batch = opendota.public_matches(less_than_match_id=less_than, min_rank=min_rank)
        if not batch:
            break
        less_than = batch[-1]["match_id"]
        rows = []
        for m in batch:
            r, d = m.get("radiant_team") or [], m.get("dire_team") or []
            if len(r) != 5 or len(d) != 5 or not any(r) or m.get("duration", 0) < 600:
                continue
            rows.append((m["match_id"], int(m["radiant_win"]), m.get("avg_rank_tier"),
                         m.get("duration"), m.get("start_time"),
                         json.dumps(r), json.dumps(d)))
        if not rows:
            empty_batches += 1
            continue
        empty_batches = 0
        con.executemany("INSERT OR IGNORE INTO matches VALUES (?,?,?,?,?,?,?)", rows)
        con.commit()
        added += con.total_changes and len(rows)
        added = count(con) - have
        if verbose:
            print(f"  stored {have + added} matches (last id {less_than})", flush=True)
    if own:
        con.close()
    return count(connect()) if own else count(con)


def iter_teams(con: sqlite3.Connection, *, min_avg_rank: int | None = None):
    """Yield (heroes: list[int], won: bool) -- one row per team per match."""
    q = "SELECT radiant_win, radiant, dire FROM matches"
    args: tuple = ()
    if min_avg_rank is not None:
        q += " WHERE avg_rank_tier >= ?"
        args = (min_avg_rank,)
    for radiant_win, r, d in con.execute(q, args):
        yield json.loads(r), bool(radiant_win)
        yield json.loads(d), not bool(radiant_win)


def iter_matchups(con: sqlite3.Connection, *, min_avg_rank: int | None = None):
    """Yield (team_a, team_b, a_won) for the opposing-pair / counter signal."""
    q = "SELECT radiant_win, radiant, dire FROM matches"
    args: tuple = ()
    if min_avg_rank is not None:
        q += " WHERE avg_rank_tier >= ?"
        args = (min_avg_rank,)
    for radiant_win, r, d in con.execute(q, args):
        yield json.loads(r), json.loads(d), bool(radiant_win)
