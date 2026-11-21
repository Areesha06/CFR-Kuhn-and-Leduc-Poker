"""Head-to-head evaluation: exact (tree walk) and simulated (sampled hands)."""
import numpy as np


def exact_value(game, pols):
    """Exact expected payoff of player 0 when player i follows pols[i]. No sampling."""
    def rec(s):
        if s.is_terminal():
            return s.returns()[0]
        if s.is_chance():
            return sum(pr * rec(s.child(a)) for a, pr in s.chance_outcomes())
        p = s.current_player()
        probs = pols[p][s.infoset_key(p)]
        return sum(probs[i] * rec(s.child(a)) for i, a in enumerate(s.legal_actions()) if probs[i] > 0)
    return rec(game.new_initial_state())


def exact_matchup(game, polA, polB):
    """Mean payoff to A (per hand) when A sits in seat 0, seat 1, and averaged over both seats."""
    seat0 = exact_value(game, [polA, polB])
    seat1 = -exact_value(game, [polB, polA])
    return {"seat0": seat0, "seat1": seat1, "avg": (seat0 + seat1) / 2}


def play_hand(game, pols, rng):
    s = game.new_initial_state()
    while not s.is_terminal():
        if s.is_chance():
            outs = s.chance_outcomes()
            a = outs[rng.choice(len(outs), p=[pr for _, pr in outs])][0]
        else:
            p = s.current_player()
            acts = s.legal_actions()
            a = acts[rng.choice(len(acts), p=pols[p][s.infoset_key(p)])]
        s = s.child(a)
    return s.returns()


def simulate(game, polA, polB, hands_per_seat=10_000, seed=0):
    """Play hands_per_seat hands in each seat. Returns per-hand payoffs to A and summary stats."""
    rng = np.random.default_rng(seed)
    pay = {0: [], 1: []}
    for seat in (0, 1):
        pols = [polA, polB] if seat == 0 else [polB, polA]
        for _ in range(hands_per_seat):
            pay[seat].append(play_hand(game, pols, rng)[seat])
    allp = np.array(pay[0] + pay[1], dtype=float)
    return {"seat0": np.array(pay[0]), "seat1": np.array(pay[1]), "all": allp,
            "mean": allp.mean(), "ci95": ci95(allp)}


def ci95(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))
