"""Spec 9: Wilson CI, Elo, and the champion/challenger regression gate."""
from dota_rl.evaluation.elo import EloBook, expected_score
from dota_rl.evaluation.metrics import wilson_interval
from dota_rl.evaluation.regression import check_regression


def test_wilson_interval_brackets_point_estimate():
    lo, hi = wilson_interval(15, 20)
    assert lo < 0.75 < hi
    assert 0.0 <= lo and hi <= 1.0
    lo0, hi0 = wilson_interval(0, 0)
    assert (lo0, hi0) == (0.0, 1.0)


def test_elo_moves_toward_winner():
    book = EloBook(base=1000.0)
    book.record_series("a", "b", wins_a=10, draws=0, losses_a=0)
    assert book.get("a") > 1000.0 > book.get("b")
    assert abs(expected_score(1000, 1000) - 0.5) < 1e-9


def _eval(wr, lo, hi):
    return {"vs_scripted_bot": {"winrate": wr, "winrate_ci_low": lo, "winrate_ci_high": hi}}


def test_equal_performance_is_not_a_regression():
    champ = _eval(0.75, 0.55, 0.88)
    chal = _eval(0.75, 0.55, 0.88)
    assert check_regression(champ, chal, threshold=0.05).passed


def test_real_collapse_is_flagged():
    champ = _eval(0.80, 0.62, 0.91)
    chal = _eval(0.40, 0.24, 0.58)
    assert not check_regression(champ, chal, threshold=0.05).passed


def test_small_improvement_passes():
    champ = _eval(0.70, 0.50, 0.85)
    chal = _eval(0.78, 0.58, 0.90)
    assert check_regression(champ, chal, threshold=0.05).passed
