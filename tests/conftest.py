import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# a tiny synthetic hero pool: 6 heroes, ids 1..6
_HEROES = [
    {"id": 1, "name": "npc_dota_hero_a", "localized_name": "Alpha",
     "primary_attr": "str", "attack_type": "Melee", "roles": ["Carry", "Durable"]},
    {"id": 2, "name": "npc_dota_hero_b", "localized_name": "Bravo",
     "primary_attr": "agi", "attack_type": "Ranged", "roles": ["Carry", "Escape"]},
    {"id": 3, "name": "npc_dota_hero_c", "localized_name": "Charlie",
     "primary_attr": "int", "attack_type": "Ranged", "roles": ["Support", "Disabler"]},
    {"id": 4, "name": "npc_dota_hero_d", "localized_name": "Delta",
     "primary_attr": "int", "attack_type": "Ranged", "roles": ["Nuker", "Support"]},
    {"id": 5, "name": "npc_dota_hero_e", "localized_name": "Echo",
     "primary_attr": "str", "attack_type": "Melee", "roles": ["Initiator", "Disabler"]},
    {"id": 6, "name": "npc_dota_hero_f", "localized_name": "Foxtrot",
     "primary_attr": "agi", "attack_type": "Melee", "roles": ["Carry", "Pusher"]},
]


@pytest.fixture
def fake_opendota(monkeypatch):
    from metafinder import opendota

    stats = []
    for h in _HEROES:
        hid = h["id"]
        pick = 1000 * hid
        win = int(pick * (0.44 + 0.03 * hid))         # Foxtrot strongest
        row = {"id": hid, "pub_pick": pick, "pub_win": win, "pro_pick": hid, "pro_win": hid // 2,
               "pro_ban": 0}
        for b in range(1, 8):
            row[f"{b}_pick"] = pick
            row[f"{b}_win"] = win
        stats.append(row)

    def matchups(hid):
        # hero 5 (Echo) beats hero 2 (Bravo) hard; otherwise ~even
        out = []
        for o in _HEROES:
            if o["id"] == hid:
                continue
            g = 200
            w = 100
            if hid == 5 and o["id"] == 2:
                w = 140
            if hid == 2 and o["id"] == 5:
                w = 60
            out.append({"hero_id": o["id"], "games_played": g, "wins": w})
        return out

    monkeypatch.setattr(opendota, "heroes", lambda: _HEROES)
    monkeypatch.setattr(opendota, "hero_stats", lambda: stats)
    monkeypatch.setattr(opendota, "hero_matchups", matchups)
    from metafinder import heroes as H
    H.all_heroes.cache_clear()
    yield
    H.all_heroes.cache_clear()


@pytest.fixture
def store_con():
    from metafinder import store
    con = store.connect(":memory:")
    yield con
    con.close()
