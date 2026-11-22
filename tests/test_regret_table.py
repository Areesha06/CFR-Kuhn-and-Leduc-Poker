import numpy as np
from solvers.regret_table import RegretTable, regret_matching

def test_two_positive_regrets():
    # The scores add to 4, so the moves get 1/4 and 3/4.
    np.testing.assert_allclose(regret_matching([1, 3]), [0.25, 0.75])

def test_one_negative_regret():
    # Only the second move has a positive score, so it gets all the probability.
    np.testing.assert_allclose(regret_matching([-2, 3]), [0.0, 1.0])

def test_no_positive_regrets():
    # Neither score is positive, so both moves get an equal chance.
    np.testing.assert_allclose(regret_matching([-2, 0]), [0.5, 0.5])

def test_strategy_stays_fixed_until_iteration_ends():
    table = RegretTable()
    key = "jack_at_start"

    first_strategy = table.strategy(key, 2)
    np.testing.assert_allclose(first_strategy, [0.5, 0.5])

    # Checking did worse and betting did better on this visit.
    table.add(key, [-1.0, 1.0], own_reach=1.0, sigma=first_strategy)

    # We're still in the same iteration, so the probabilities stay at 50/50.
    np.testing.assert_allclose(table.strategy(key, 2), [0.5, 0.5])

    table.end_iteration()

    # The new regrets make betting the only move in the current strategy.
    np.testing.assert_allclose(table.policy("current")[key], [0.0, 1.0])

    # The first iteration was played at 50/50, so that's the average so far.
    np.testing.assert_allclose(table.policy("avg_uniform")[key], [0.5, 0.5])

def test_repeated_visits_and_both_averages():
    table = RegretTable()
    key = "jack_at_start"

    first_strategy = table.strategy(key, 2)

    # If the opponent has a Queen on one visit and a King on the other, we can't see their card, so both visits use the same decision key.
    table.add(key, [-0.5, 0.5], own_reach=1.0, sigma=first_strategy)
    table.add(key, [-0.5, 0.5], own_reach=1.0, sigma=first_strategy)
    table.end_iteration()
    # Both visits contribute to the regret total.
    np.testing.assert_allclose(table.regrets[key], [-1.0, 1.0])

    second_strategy = table.strategy(key, 2)
    table.add(key, [0.0, 0.0], own_reach=0.5, sigma=second_strategy)
    table.end_iteration()

    # The first iteration counts once. The second counts half as much because our earlier moves reach this decision with probability 0.5.
    np.testing.assert_allclose(table.policy("avg_uniform")[key], [1/3, 2/3])

    # Linear averaging doubles the second iteration's weight. Its weight is now 2*0.5 = 1, so both strategies count equally.
    np.testing.assert_allclose(table.policy("avg_linear")[key], [0.25, 0.75])