import pytest
from open_spiel.python import policy as ospolicy
from open_spiel.python.algorithms import exploitability
from games.kuhn import Kuhn
from games.leduc import Leduc
from games.openspiel_wrapper import kuhn_os, leduc_os
from games.os_bridge import build_key_mapping, to_openspiel_policy
from games.base import uniform_policy
from games.baselines import kuhn_nash_policy


@pytest.mark.parametrize("mine,theirs", [(Kuhn, kuhn_os), (Leduc, leduc_os)])
def test_uniform_exploitability_matches_openspiel(mine, theirs):
    g, o = mine(), theirs()
    tab = to_openspiel_policy(g, o, uniform_policy(g), build_key_mapping(g, o))
    ref = exploitability.exploitability(o.g, ospolicy.UniformRandomPolicy(o.g))
    assert exploitability.exploitability(o.g, tab) == pytest.approx(ref)


def test_kuhn_nash_has_zero_exploitability():
    g, o = Kuhn(), kuhn_os()
    for alpha in (0.0, 0.1, 1 / 3):
        tab = to_openspiel_policy(g, o, kuhn_nash_policy(g, alpha), build_key_mapping(g, o))
        assert exploitability.exploitability(o.g, tab) == pytest.approx(0.0, abs=1e-9)
