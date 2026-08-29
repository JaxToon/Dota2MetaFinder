"""Core metafinder logic on a synthetic 6-hero dataset (no network)."""
import json

import pytest


def test_tier_list_ranks_and_tiers(fake_opendota):
    from metafinder import heroes
    tl = heroes.tier_list("divine", min_pick_share=0.0)
    assert [r["hero"] for r in tl][:1] == ["Foxtrot"]          # highest win rate
    assert tl[0]["rank"] == 1 and tl[0]["tier"] in ("S", "A")
    assert tl[-1]["tier"] in ("C", "D")
    assert all(a["win_rate_adj"] >= b["win_rate_adj"] for a, b in zip(tl, tl[1:]))


def test_hero_name_resolution(fake_opendota):
    from metafinder import heroes
    assert heroes.resolve("Echo").id == 5
    assert heroes.resolve(2).localized_name == "Bravo"
    with pytest.raises(KeyError):
        heroes.resolve("Nonexistent")


def test_counter_matrix_direction(fake_opendota):
    from metafinder import counters
    cm = counters.from_opendota(shrink=10.0)
    assert cm.advantage("Echo", "Bravo") > 0.10          # Echo beats Bravo
    assert cm.advantage("Bravo", "Echo") < -0.10          # and vice versa
    worst = cm.worst_against("Bravo", 1)[0]
    assert worst[0] == "Echo"


def test_store_ingest_and_iter(store_con):
    from metafinder import store
    rows = [(100 + i, i % 2, 55, 1800, 0,
             json.dumps([1, 2, 3, 4, 5]), json.dumps([6, 1, 2, 3, 4])) for i in range(20)]
    store_con.executemany("INSERT INTO matches VALUES (?,?,?,?,?,?,?)", rows)
    store_con.commit()
    assert store.count(store_con) == 20
    teams = list(store.iter_teams(store_con))
    assert len(teams) == 40 and all(len(t[0]) == 5 for t in teams)


def test_synergy_detects_a_winning_pair(store_con):
    from metafinder import synergy
    # 10-hero pool, disjoint 5v5. A team containing BOTH 1 and 2 wins 72% of the
    # time; otherwise it's a coin flip. Heroes 1 and 2 stay ~individually average.
    import random
    rng = random.Random(0)
    pool = list(range(1, 11))
    rows = []
    for i in range(1500):
        rng.shuffle(pool)
        radiant, dire = pool[:5], pool[5:]
        r_pair = {1, 2} <= set(radiant)
        d_pair = {1, 2} <= set(dire)
        p = 0.72 if r_pair else (0.28 if d_pair else 0.5)
        radiant_win = int(rng.random() < p)
        rows.append((i, radiant_win, 55, 1800, 0, json.dumps(radiant), json.dumps(dire)))
    store_con.executemany("INSERT INTO matches VALUES (?,?,?,?,?,?,?)", rows)
    store_con.commit()
    sm = synergy.build(store_con, shrink=15.0)
    others = [sm.synergy(a, b) for a in range(3, 11) for b in range(a + 1, 11)]
    assert sm.synergy(1, 2) > 0.015                       # positive residual synergy
    assert sm.synergy(1, 2) > max(others) - 1e-9          # the strongest pair in the pool
    assert sm.pair_winrate(1, 2)[0] > 0.62                # raw pair win rate is high


def test_draft_recommends_a_counter(fake_opendota, store_con):
    from metafinder import counters, heroes, synergy
    from metafinder.draft import recommend
    tl = heroes.tier_list("divine", min_pick_share=0.0)
    cm = counters.from_opendota(shrink=10.0)
    sm = synergy.build(store_con)          # empty -> zero synergy
    recs = recommend(allies=[], enemies=["Bravo"], bans=[], tier_rows=tl,
                     synergy=sm, counters=cm, n=5)
    names = [c.hero for c in recs]
    assert "Echo" in names[:2]             # Echo counters Bravo -> ranked high
    assert "Bravo" not in names            # already taken by the enemy


def test_score_comp_shape(fake_opendota, store_con):
    from metafinder import heroes, synergy
    from metafinder.draft import score_comp
    tl = heroes.tier_list("divine", min_pick_share=0.0)
    sm = synergy.build(store_con)
    r = score_comp(["Alpha", "Bravo", "Charlie", "Delta", "Echo"], tier_rows=tl, synergy=sm)
    assert len(r["heroes"]) == 5
    assert "comp_score" in r and "role_coverage" in r and "gaps" in r
