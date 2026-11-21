"""Baseline bots, all expressed as policies (dict infoset_key -> prob array)."""
import numpy as np
from .base import build_policy


def random_policy(game):
    return build_policy(game, lambda s, p, acts: np.ones(len(acts)) / len(acts))


def _onehot(acts, a):
    v = np.zeros(len(acts)); v[acts.index(a)] = 1.0
    return v


# ---------------- Kuhn: 0=J 1=Q 2=K, actions 0=pass 1=bet
def kuhn_rule_policy(game):
    def fn(s, p, acts):
        card = s.cards[p]                       # ONLY own card is used
        facing_bet = s.hist.endswith("b")
        if card == 2:   return _onehot(acts, 1)                         # K: bet / call
        if card == 1:   return _onehot(acts, 1 if facing_bet else 0)    # Q: check, then call
        return _onehot(acts, 0)                                         # J: check, then fold
    return build_policy(game, fn)


def kuhn_nash_policy(game, alpha=0.0):
    """One member of Kuhn's equilibrium family (0 <= alpha <= 1/3). Game value for P0 = -1/18."""
    a = alpha
    table = {
        "0": a, "1": 0.0, "2": 3 * a,                 # P0 first action: P(bet)
        "0pb": 0.0, "1pb": a + 1 / 3, "2pb": 1.0,     # P0 facing a bet after checking: P(call)
        "0b": 0.0, "1b": 1 / 3, "2b": 1.0,            # P1 facing a bet: P(call)
        "0p": 1 / 3, "1p": 0.0, "2p": 1.0,            # P1 after a check: P(bet)
    }
    return build_policy(game, lambda s, p, acts: [1 - table[s.infoset_key(p)], table[s.infoset_key(p)]])


# ---------------- Leduc: rank = card//2 (0=J 1=Q 2=K), actions 0=fold 1=call 2=raise
def leduc_rule_policy(game):
    def fn(s, p, acts):
        rank = s.cards[p] // 2                                   # own card
        pair = s.board is not None and s.board // 2 == rank      # public board
        facing = s.to_call() > 0
        if rank == 2 or pair:
            return _onehot(acts, 2 if 2 in acts else 1)          # K or pair: raise
        if rank == 1:
            return _onehot(acts, 1)                              # Q: call / check
        return _onehot(acts, 0 if facing else 1)                 # J: fold to a bet, else check
    return build_policy(game, fn)
