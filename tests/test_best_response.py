
from games.kuhn import Kuhn
from games.base import uniform_policy
from solvers.tree import build_tree
from eval.best_response import (best_response_policy,best_response_value)


def test_best_response():
    game = Kuhn()
    tree = build_tree(game)
    opponent_policy = uniform_policy(game)
    response = best_response_policy(tree, opponent_policy, player=0)
    value = best_response_value(tree, opponent_policy, player=0)
    print("Best response policy:", response)
    print("Best response value:", value)
    assert len(response) == len(tree.infoset_keys)
    assert value >= 0.0
