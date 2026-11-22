import numpy as np


def regret_matching(regrets):
    """Turn regret scores into move probabilities."""
    # A positive score means we wish we'd played that move more often.
    # For this calculation, treat negative scores as zero.
    positive_regrets = np.maximum(regrets, 0.0)
    total = positive_regrets.sum()

    if total == 0:
        # No move has a positive score, so split the chances equally.
        number_of_moves = len(regrets)
        return np.full(number_of_moves, 1.0 / number_of_moves)

    return positive_regrets / total


class RegretTable:
    def __init__(self):
        # Keep the totals from finished iterations and the changes from this one.
        self.regrets = {}
        self.pending_regrets = {}

        self.uniform_sum = {}
        self.linear_sum = {}
        self.pending_strategy = {}

        self.iteration = 0

    def strategy(self, key, n):
        """Give the current probabilities for this decision's n legal moves."""
        if key not in self.regrets:
            # First time making this decision. Every move starts at zero regret.
            self.regrets[key] = np.zeros(n)
            self.uniform_sum[key] = np.zeros(n)
            self.linear_sum[key] = np.zeros(n)

        # Read the totals from the last finished iteration. That way, the
        # probabilities stay fixed while we're still exploring this one.
        return regret_matching(self.regrets[key])

    def add(self, key, regret_delta, own_reach, sigma):
        """Save the regret change from one visit to this decision."""
        if key not in self.pending_regrets:
            self.pending_regrets[key] = np.zeros(len(sigma))

            # Different hidden cards can lead to the same decision.
            # Save the strategy once, weighted by how likely our own
            # earlier moves were to bring us here.
            self.pending_strategy[key] = own_reach * np.asarray(sigma)

        # Each visit gives us another regret change to add.
        # The traversal has already accounted for the opponent and chance.
        self.pending_regrets[key] += regret_delta

    def end_iteration(self):
        """Apply the changes once the traversal is finished."""
        self.iteration += 1

        for key, regret_delta in self.pending_regrets.items():
            # Vanilla CFR keeps the full total, even if it goes below zero.
            self.regrets[key] += regret_delta

            # Use the probabilities we actually played this iteration.
            played_strategy = self.pending_strategy[key]
            self.uniform_sum[key] += played_strategy

            # Give later iterations more weight in the linear average.
            self.linear_sum[key] += self.iteration * played_strategy

        # Clear this iteration's changes so they can't be counted twice.
        self.pending_regrets.clear()
        self.pending_strategy.clear()

    def policy(self, kind):
        """Return the move probabilities for each decision."""
        if kind == "current":
            return {
                key: regret_matching(regrets)
                for key, regrets in self.regrets.items()
            }

        if kind == "avg_uniform":
            strategy_sums = self.uniform_sum
        elif kind == "avg_linear":
            strategy_sums = self.linear_sum
        else:
            raise ValueError("Use 'current', 'avg_uniform', or 'avg_linear'.")

        result = {}
        for key, scores in strategy_sums.items():
            # Divide each move's total by the sum so the probabilities add to 1.
            # If there's no weight yet, split the chances equally.
            result[key] = regret_matching(scores)

        return result