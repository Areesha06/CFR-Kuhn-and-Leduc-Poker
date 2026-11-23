import numpy as np


def kuhn_nash_policy():
    return {
        # Player 0's first decision
        "0": np.array([2/3, 1/3]),  # J: sometimes bluff
        "1": np.array([1.0, 0.0]),  # Q: check
        "2": np.array([0.0, 1.0]),  # K: bet

        # Player 1 facing Player 0's bet
        "0b": np.array([1.0, 0.0]),  # J: fold
        "1b": np.array([2/3, 1/3]),  # Q: sometimes call
        "2b": np.array([0.0, 1.0]),  # K: call

        # Player 1 after Player 0 checks
        "0p": np.array([2/3, 1/3]),  # J: sometimes bluff
        "1p": np.array([1.0, 0.0]),  # Q: check
        "2p": np.array([0.0, 1.0]),  # K: bet

        # Player 0 facing Player 1's bet
        "0pb": np.array([1.0, 0.0]),  # J: fold
        "1pb": np.array([1/3, 2/3]),   # Q: sometimes call
        "2pb": np.array([0.0, 1.0]),  # K: call
    }