"""Kuhn poker. Cards: 0=J, 1=Q, 2=K. Actions: 0=pass (check/fold), 1=bet (bet/call)."""
from .base import Game, State

TERMINAL = {"pp", "bp", "pbp", "pbb", "bb"}


class KuhnState(State):
    def __init__(self, cards=(), hist=""):
        self.cards = tuple(cards)   # dealt so far: (card_p0, card_p1)
        self.hist = hist            # string of 'p' / 'b'

    def is_chance(self):
        return len(self.cards) < 2

    def is_terminal(self):
        return self.hist in TERMINAL

    def current_player(self):
        return len(self.hist) % 2

    def legal_actions(self):
        return [0, 1]

    def chance_outcomes(self):
        rem = [c for c in range(3) if c not in self.cards]
        return [(c, 1.0 / len(rem)) for c in rem]

    def child(self, a):
        if self.is_chance():
            return KuhnState(self.cards + (a,), self.hist)
        return KuhnState(self.cards, self.hist + ("p" if a == 0 else "b"))

    def returns(self):
        h = self.hist
        if h == "bp":
            return [1.0, -1.0]
        if h == "pbp":
            return [-1.0, 1.0]
        stake = 2.0 if h in ("bb", "pbb") else 1.0      # pp -> 1, bb/pbb -> 2
        return [stake, -stake] if self.cards[0] > self.cards[1] else [-stake, stake]

    def infoset_key(self, player):
        return f"{self.cards[player]}{self.hist}"       # own card + public history only

    def history_str(self):
        return f"{self.cards}|{self.hist}"

    def with_private(self, player, card):               # used by the leakage test
        c = list(self.cards); c[player] = card
        return KuhnState(c, self.hist)


class Kuhn(Game):
    name = "kuhn"

    def new_initial_state(self):
        return KuhnState()
