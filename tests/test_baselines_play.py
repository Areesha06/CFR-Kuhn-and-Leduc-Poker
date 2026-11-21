import pytest
from games.kuhn import Kuhn
from games.leduc import Leduc
from games.baselines import *
from games.base import uniform_policy
from games.play import exact_value, exact_matchup, simulate


def test_kuhn_nash_value_is_minus_one_eighteenth():
    g = Kuhn()
    for alpha in (0.0, 0.1, 1 / 3):
        nash = kuhn_nash_policy(g, alpha)
        assert exact_value(g, [nash, nash]) == pytest.approx(-1 / 18)


def test_zero_sum_symmetry_and_uniform_self_play():
    g = Leduc()
    u = uniform_policy(g)
    r = leduc_rule_policy(g)
    m1, m2 = exact_matchup(g, r, u), exact_matchup(g, u, r)
    assert m1["seat0"] == pytest.approx(-m2["seat1"])
    assert m1["seat1"] == pytest.approx(-m2["seat0"])


def test_simulation_close_to_exact_kuhn():
    g = Kuhn()
    A, B = kuhn_nash_policy(g, 0.0), kuhn_rule_policy(g)
    ex = exact_matchup(g, A, B)["avg"]
    sim = simulate(g, A, B, hands_per_seat=20_000, seed=1)
    assert abs(sim["mean"] - ex) < 4 * sim["ci95"] / 1.96
