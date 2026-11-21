"""Leduc poker. 6 cards: id 0..5, rank = id // 2 (0=J,1=Q,2=K), suit = id % 2.
Actions: 0=fold, 1=call (=check when nothing to call), 2=raise.
Ante 1 each. Raise size 2 in round 1, 4 in round 2. Max 2 raises per round.
Player 0 acts first in both rounds.
"""
import copy
from .base import Game, State

FOLD_WHEN_NO_BET = False   # set to match OpenSpiel (run games/probe_openspiel.py)
RAISE_SIZE = {1: 2, 2: 4}
MAX_RAISES = 2


class LeducState(State):
    def __init__(self):
        self.cards = []          # private cards dealt so far
        self.board = None
        self.rnd = 1
        self.hist = ["", ""]     # action string per round
        self.contrib = [1, 1]
        self.raises = 0
        self.acted = 0
        self.player = 0
        self.need_board = False
        self.done = False
        self.fold_winner = None

    # ---- node type
    def is_terminal(self):
        return self.done

    def is_chance(self):
        return (not self.done) and (len(self.cards) < 2 or self.need_board)

    def current_player(self):
        return self.player

    def to_call(self, p=None):
        p = self.player if p is None else p
        return max(self.contrib) - self.contrib[p]

    def legal_actions(self):
        acts = []
        if self.to_call() > 0 or FOLD_WHEN_NO_BET:
            acts.append(0)
        acts.append(1)
        if self.raises < MAX_RAISES:
            acts.append(2)
        return acts

    def chance_outcomes(self):
        rem = [c for c in range(6) if c not in self.cards and c != self.board]
        return [(c, 1.0 / len(rem)) for c in rem]

    def child(self, a):
        s = copy.copy(self)
        s.cards = list(self.cards); s.hist = list(self.hist); s.contrib = list(self.contrib)
        if self.is_chance():
            if len(self.cards) < 2:
                s.cards.append(a)
            else:
                s.board = a
                s.need_board = False
                s.rnd = 2
                s.raises = 0
                s.acted = 0
                s.player = 0
            return s
        p = self.player
        tc = self.to_call()
        r = self.rnd - 1
        if a == 0:
            s.hist[r] += "f"
            s.fold_winner = 1 - p
            s.done = True
        elif a == 1:
            s.hist[r] += "c"
            s.contrib[p] += tc
            s.acted += 1
            if self.raises > 0 or s.acted >= 2:          # betting round over
                if self.rnd == 1:
                    s.need_board = True
                else:
                    s.done = True
            else:
                s.player = 1 - p
        else:
            s.hist[r] += "r"
            s.contrib[p] += tc + RAISE_SIZE[self.rnd]
            s.raises += 1
            s.acted += 1
            s.player = 1 - p
        return s

    def returns(self):
        if self.fold_winner is not None:
            w = self.fold_winner
            r = [0.0, 0.0]; r[w] = float(self.contrib[1 - w]); r[1 - w] = -float(self.contrib[1 - w])
            return r
        def strength(p):
            rank = self.cards[p] // 2
            return (1 if rank == self.board // 2 else 0, rank)
        s0, s1 = strength(0), strength(1)
        if s0 == s1:
            return [0.0, 0.0]
        w = 0 if s0 > s1 else 1
        r = [0.0, 0.0]; r[w] = float(self.contrib[1 - w]); r[1 - w] = -float(self.contrib[1 - w])
        return r

    def infoset_key(self, player):
        b = "-" if self.board is None else str(self.board)
        return f"{self.cards[player]}|{b}|{self.hist[0]}|{self.hist[1]}"

    def history_str(self):
        return f"{self.cards}|{self.board}|{self.hist}"

    def with_private(self, player, card):               # used by the leakage test
        s = copy.copy(self); s.cards = list(self.cards); s.cards[player] = card
        return s


class Leduc(Game):
    name = "leduc"

    def new_initial_state(self):
        return LeducState()
