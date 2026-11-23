
import numpy as np
import pyspiel

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score

from games.openspiel_wrapper import kuhn_os
from solvers.tree import build_tree
from solvers.cfr import cfr_traverse, UniformTable


def test_uniform_cfr_matches_openspiel():
    # Our traversal using OpenSpiel's Kuhn game.
    game = kuhn_os()
    tree = build_tree(game)
    table = UniformTable()
    our_value, nodes = cfr_traverse(tree, table, traverser=0)
    # OpenSpiel's own uniform strategy.
    os_game = pyspiel.load_game("kuhn_poker")
    uniform = policy.UniformRandomPolicy(os_game)
    expected = expected_game_score.policy_value(os_game.new_initial_state(),[uniform, uniform])
    print("Our root value:", our_value)
    print("OpenSpiel value:", expected[0])
    print("Nodes visited:", nodes)
    assert np.isclose(our_value, expected[0])
