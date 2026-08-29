"""Build & lane advice: core build parsing + the lineup-aware rules layer."""


def test_core_build_filters_consumables_and_components(fake_opendota):
    from metafinder import builds
    b = builds.core_build("Alpha", per_stage=5)
    assert "Quelling Blade" in [x["item"] for x in b["start"]]     # start keeps cheap items
    assert "Tango" not in [x["item"] for x in b["mid"]]            # consumables dropped elsewhere
    assert "Broadsword" not in [x["item"] for x in b["mid"]]       # cheap component dropped
    assert "Black King Bar" in [x["item"] for x in b["mid"]]
    assert "Blink Dagger" in [x["item"] for x in b["early"]]       # component whitelist


def test_situational_reacts_to_magic_lineup(fake_opendota):
    from metafinder import builds
    # Delta + Charlie carry magic_burst via role default; Alpha is a physical carry
    recs = builds.situational_items("Alpha", allies=["Echo"],
                                    enemies=["Delta", "Charlie", "Echo"])
    items = {r["item"] for r in recs}
    reasons = " ".join(r["reason"] for r in recs)
    assert any("Pipe" in i for i in items)                        # heavy magic -> Pipe
    assert "magic" in reasons.lower()


def test_situational_reacts_to_lockdown(fake_opendota):
    from metafinder import builds
    recs = builds.situational_items("Delta", allies=[], enemies=["Charlie", "Echo"])
    assert any("Black King Bar" in r["item"] for r in recs)       # Charlie+Echo lockdown


def test_situational_flags_missing_initiation(fake_opendota):
    from metafinder import builds
    # Echo can initiate; allies (Alpha, Bravo carries) bring none
    recs = builds.situational_items("Echo", allies=["Alpha", "Bravo"], enemies=["Delta"])
    assert any("Blink" in r["item"] and "initiation" in r["reason"] for r in recs)


def test_situational_capped(fake_opendota):
    from metafinder import builds
    recs = builds.situational_items("Alpha", allies=["Bravo"],
                                    enemies=["Charlie", "Delta", "Echo", "Foxtrot", "Bravo"])
    assert len(recs) <= 7


def test_lane_advice_uses_counter_matrix(fake_opendota):
    from metafinder import counters
    from metafinder.lanes import lane_advice
    cm = counters.from_opendota(shrink=10.0)
    adv = lane_advice("Bravo", "carry", allies=[], enemies=["Echo"], counters=cm)
    assert adv["primary_lane"] == "safelane"
    assert adv["tough_matchups"] and adv["tough_matchups"][0][0] == "Echo"   # Echo beats Bravo
    assert any("Echo" in n for n in adv["notes"])
