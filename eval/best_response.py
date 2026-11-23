
import numpy as np


def best_response_policy(tree, opponent_policy, player):
    """
    Find the best response for one player against a fixed opponent policy.

    Returns a policy dictionary: infoset_key -> probability array.
    """

    # Find every node's depth and group decision nodes by infoset.
    depths = {}
    infoset_nodes = {}

    def get_depth(node_id, depth):
        depths[node_id] = depth
        node = tree.nodes[node_id]

        if node.player != -1:
            infoset_nodes.setdefault(node.infoset, []).append(node_id)

        for child_id in node.children:
            get_depth(child_id, depth + 1)

    get_depth(tree.root, 0)

    # These will store the chosen action at each infoset.
    response = {}

    # Calculate the counterfactual reach of every node:
    # opponent's reach multiplied by chance reach.
    reaches = {}

    def calculate_reaches(node_id, opponent_reach, chance_reach):
        reaches[node_id] = opponent_reach * chance_reach
        node = tree.nodes[node_id]

        if node.utility is not None:
            return

        if node.player == -1:
            for child_id, prob in zip(
                node.children, node.chance_probs
            ):
                calculate_reaches(
                    child_id,
                    opponent_reach,
                    chance_reach * prob
                )

        elif node.player != player:
            key = tree.infoset_keys[node.infoset]
            sigma = np.asarray(opponent_policy[key])

            for i, child_id in enumerate(node.children):
                calculate_reaches(
                    child_id,
                    opponent_reach * sigma[i],
                    chance_reach
                )

        else:
            # Do not multiply by our own strategy.
            for child_id in node.children:
                calculate_reaches(
                    child_id,
                    opponent_reach,
                    chance_reach
                )

    calculate_reaches(tree.root, 1.0, 1.0)

    # Evaluate a node using already-selected best responses
    # at the player's later information sets.
    def value(node_id):
        node = tree.nodes[node_id]

        if node.utility is not None:
            return node.utility[player]

        if node.player == -1:
            return sum(
                prob * value(child_id)
                for child_id, prob in zip(
                    node.children, node.chance_probs
                )
            )

        key = tree.infoset_keys[node.infoset]

        if node.player == player:
            sigma = response[key]
        else:
            sigma = np.asarray(opponent_policy[key])

        return sum(
            sigma[i] * value(child_id)
            for i, child_id in enumerate(node.children)
        )

    # Process information sets from deepest to shallowest.
    ordered_infosets = sorted(
        infoset_nodes,
        key=lambda i: max(depths[n] for n in infoset_nodes[i]),
        reverse=True
    )

    for infoset_id in ordered_infosets:
        nodes = infoset_nodes[infoset_id]
        key = tree.infoset_keys[infoset_id]
        num_actions = len(tree.nodes[nodes[0]].children)

        action_totals = np.zeros(num_actions)

        # Sum action values across all hidden states in this infoset.
        for node_id in nodes:
            node = tree.nodes[node_id]
            reach = reaches[node_id]

            for i, child_id in enumerate(node.children):
                action_totals[i] += reach * value(child_id)

        # Choose the action with the highest total value.
        best_action = int(np.argmax(action_totals))

        probabilities = np.zeros(num_actions)
        probabilities[best_action] = 1.0

        response[key] = probabilities
    return response


def best_response_value(tree, opponent_policy, player):
    """Calculate the expected payoff of the best response."""

    response = best_response_policy(
        tree, opponent_policy, player
    )

    def evaluate(node_id):
        node = tree.nodes[node_id]

        if node.utility is not None:
            return node.utility[player]

        if node.player == -1:
            return sum(
                prob * evaluate(child_id)
                for child_id, prob in zip(
                    node.children, node.chance_probs
                )
            )

        key = tree.infoset_keys[node.infoset]

        if node.player == player:
            sigma = response[key]
        else:
            sigma = np.asarray(opponent_policy[key])

        return sum(
            sigma[i] * evaluate(child_id)
            for i, child_id in enumerate(node.children)
        )

    return evaluate(tree.root)
