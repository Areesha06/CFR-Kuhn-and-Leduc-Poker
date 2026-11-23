
from games.base import Game


class Node:
    def __init__(self):
        self.player = -1
        self.children = []
        self.chance_probs = []
        self.infoset = None
        self.utility = None
        self.actions = []


class Tree:
    def __init__(self):
        self.nodes = []
        self.root = None
        self.infoset_keys = []
        self.infoset_actions = []


def build_tree(game: Game) -> Tree:
    tree = Tree()

    # Maps each information-set key to an integer ID
    infoset_map = {}
    def add_node(state):
        node = Node()
        # Give this node an ID and store it
        node_id = len(tree.nodes)
        tree.nodes.append(node)
        # Case 1: Game has ended
        if state.is_terminal():
            node.utility = state.returns()
            return node_id
        # Case 2: Cards are being dealt (chance node)
        if state.is_chance():
            outcomes = state.chance_outcomes()

            for action, prob in outcomes:
                child_state = state.child(action)
                child_id = add_node(child_state)

                node.children.append(child_id)
                node.chance_probs.append(prob)

        # Case 3: A player must make a decision
        else:
            player = state.current_player()
            key = state.infoset_key(player)
            actions = state.legal_actions()

            node.player = player
            node.actions = actions

            # Assign an integer ID to this information set
            if key not in infoset_map:
                infoset_id = len(tree.infoset_keys)
                infoset_map[key] = infoset_id

                tree.infoset_keys.append(key)
                tree.infoset_actions.append(actions)

            node.infoset = infoset_map[key]

            # Explore every possible action
            for action in actions:
                child_state = state.child(action)
                child_id = add_node(child_state)

                node.children.append(child_id)

        return node_id

    # Start building from the initial game state
    tree.root = add_node(game.new_initial_state())

    return tree
