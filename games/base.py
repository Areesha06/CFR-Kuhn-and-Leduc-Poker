"""Shared interface (write this TOGETHER on Day 0).

A policy is a dict:  infoset_key (str) -> np.array of probabilities,
aligned with state.legal_actions() at that infoset.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import json
import numpy as np


class State(ABC):
    @abstractmethod
    def is_terminal(self) -> bool: ...
    @abstractmethod
    def is_chance(self) -> bool: ...
    @abstractmethod
    def current_player(self) -> int: ...            # 0 or 1 (decision nodes only)
    @abstractmethod
    def legal_actions(self) -> list: ...            # sorted ints (decision nodes)
    @abstractmethod
    def chance_outcomes(self) -> list: ...          # [(action, prob), ...]
    @abstractmethod
    def child(self, action) -> "State": ...         # returns NEW state, never mutates
    @abstractmethod
    def returns(self) -> list: ...                  # [r0, r1] at terminal nodes
    @abstractmethod
    def infoset_key(self, player: int) -> str: ...  # must NOT depend on opponent's card
    def history_str(self) -> str:
        return ""


class Game(ABC):
    name = "game"
    num_players = 2

    @abstractmethod
    def new_initial_state(self) -> State: ...


# ---------------------------------------------------------------- helpers
def walk(game: Game):
    """Yield every state (history) of the game tree."""
    stack = [game.new_initial_state()]
    while stack:
        s = stack.pop()
        yield s
        if s.is_terminal():
            continue
        acts = [a for a, _ in s.chance_outcomes()] if s.is_chance() else s.legal_actions()
        for a in acts:
            stack.append(s.child(a))


def collect_infosets(game: Game) -> dict:
    """key -> {'player': p, 'actions': [...], 'rep': a representative state}."""
    out = {}
    for s in walk(game):
        if s.is_terminal() or s.is_chance():
            continue
        p = s.current_player()
        k = s.infoset_key(p)
        acts = s.legal_actions()
        if k in out:
            assert out[k]["player"] == p and out[k]["actions"] == acts, f"inconsistent infoset {k}"
        else:
            out[k] = {"player": p, "actions": acts, "rep": s}
    return out


def uniform_policy(game: Game) -> dict:
    return {k: np.ones(len(v["actions"])) / len(v["actions"])
            for k, v in collect_infosets(game).items()}


def build_policy(game: Game, fn) -> dict:
    """fn(state, player, legal_actions) -> list/array of probs. Called once per infoset."""
    pol = {}
    for k, v in collect_infosets(game).items():
        p = np.asarray(fn(v["rep"], v["player"], v["actions"]), dtype=float)
        assert abs(p.sum() - 1) < 1e-9 and len(p) == len(v["actions"])
        pol[k] = p
    return pol


def save_policy(policy: dict, path: str):
    with open(path, "w") as f:
        json.dump({k: list(map(float, v)) for k, v in policy.items()}, f)


def load_policy(path: str) -> dict:
    with open(path) as f:
        return {k: np.array(v) for k, v in json.load(f).items()}
