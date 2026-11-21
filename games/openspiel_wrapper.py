"""Wraps an OpenSpiel game behind our interface (B and C can start with this)."""
import pyspiel
from .base import Game, State


class OSState(State):
    def __init__(self, s):
        self.s = s

    def is_terminal(self): return self.s.is_terminal()
    def is_chance(self): return self.s.is_chance_node()
    def current_player(self): return self.s.current_player()
    def legal_actions(self): return list(self.s.legal_actions())
    def chance_outcomes(self): return list(self.s.chance_outcomes())

    def child(self, a):
        c = self.s.clone(); c.apply_action(a)
        return OSState(c)

    def returns(self): return list(self.s.returns())
    def infoset_key(self, player): return self.s.information_state_string(player)
    def history_str(self): return str(self.s.history())


class OSGame(Game):
    def __init__(self, name):
        self.name = "os_" + name
        self.g = pyspiel.load_game(name)

    def new_initial_state(self):
        return OSState(self.g.new_initial_state())


def kuhn_os(): return OSGame("kuhn_poker")
def leduc_os(): return OSGame("leduc_poker")
