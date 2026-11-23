
import numpy as np


class UniformTable:
    """Temporary table for testing CFR with a uniform strategy."""

    def strategy(self, key, n):
        return np.ones(n) / n
    def add(self, key, regret_delta, own_reach, sigma):
        pass
    def end_iteration(self):
        pass


def cfr_traverse(tree, table, traverser=0):
    """
    Traverse the game tree for one traversing player.

    Returns:
        root_value: Expected utility for the traversing player.
        nodes_visited: Number of nodes visited.
    """

    nodes_visited = 0

    def walk(node_id, reach0, reach1, chance_reach):
        nonlocal nodes_visited
        nodes_visited += 1
        node = tree.nodes[node_id]
        # Terminal node: return the traverser's payoff.
        if node.utility is not None:
            return node.utility[traverser]
        # Chance node: calculate the probability-weighted value.
        if node.player == -1:
            value = 0.0
            for child_id, prob in zip(node.children, node.chance_probs):
                value += prob * walk(child_id,reach0,reach1,chance_reach * prob)
            return value

        # Decision node.
        player = node.player
        key = tree.infoset_keys[node.infoset]
        n = len(node.children)
        sigma = np.asarray(table.strategy(key, n), dtype=float)
        if sigma.shape != (n,):
            raise ValueError(f"Invalid strategy size at {key}")
        action_values = []

        for i, child_id in enumerate(node.children):
            prob = sigma[i]

            if player == 0:
                value = walk(
                    child_id,
                    reach0 * prob,
                    reach1,
                    chance_reach
                )
            else:
                value = walk(
                    child_id,
                    reach0,
                    reach1 * prob,
                    chance_reach
                )

            action_values.append(value)

        action_values = np.asarray(action_values)

        # Expected value under the current strategy.
        node_value = np.dot(sigma, action_values)

        # Update regrets only at the traverser's information sets.
        if player == traverser:
            if player == 0:
                own_reach = reach0
                opponent_reach = reach1
            else:
                own_reach = reach1
                opponent_reach = reach0

            regret_delta = (
                opponent_reach
                * chance_reach
                * (action_values - node_value)
            )

            table.add(
                key,
                regret_delta,
                own_reach,
                sigma
            )

        return node_value

    root_value = walk(tree.root, 1.0, 1.0, 1.0)

    return root_value, nodes_visited


def cfr_iteration(tree, table):
    """Run one iteration for both players."""

    value0, nodes0 = cfr_traverse(tree, table, traverser=0)
    value1, nodes1 = cfr_traverse(tree, table, traverser=1)

    table.end_iteration()

    return {
        "value0": value0,
        "value1": value1,
        "nodes_visited": nodes0 + nodes1
    }
