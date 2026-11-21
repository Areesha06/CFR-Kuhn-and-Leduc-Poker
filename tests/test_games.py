import random
import pytest
from games.base import walk, collect_infosets
from games.kuhn import Kuhn
from games.leduc import Leduc
from games.openspiel_wrapper import kuhn_os, leduc_os


def parallel_compare(mine, theirs, mapping):
    """Traverse both trees in lock-step (same actions) and compare everything."""
    stack = [(mine.new_initial_state(), theirs.new_initial_state())]
    n = 0
    while stack:
        a, b = stack.pop()
        n += 1
        assert a.is_terminal() == b.is_terminal()
        if a.is_terminal():
            assert a.returns() == pytest.approx(b.returns())
            continue
        assert a.is_chance() == b.is_chance()
        if a.is_chance():
            oa, ob = dict(a.chance_outcomes()), dict(b.chance_outcomes())
            assert oa.keys() == ob.keys()
            for k in oa:
                assert oa[k] == pytest.approx(ob[k])
            acts = list(oa)
        else:
            p = a.current_player()
            assert p == b.current_player()
            assert a.legal_actions() == b.legal_actions()
            ka, kb = a.infoset_key(p), b.infoset_key(p)
            assert mapping.setdefault(ka, kb) == kb, "same infoset of mine maps to 2 OpenSpiel infosets"
            acts = a.legal_actions()
        for x in acts:
            stack.append((a.child(x), b.child(x)))
    return n


@pytest.mark.parametrize("mine,theirs,expected", [(Kuhn, kuhn_os, 12), (Leduc, leduc_os, 936)])
def test_infoset_counts_and_exhaustive_traversal(mine, theirs, expected):
    mapping = {}
    parallel_compare(mine(), theirs(), mapping)
    assert len(collect_infosets(mine())) == expected
    assert len(collect_infosets(theirs())) == expected
    assert len(mapping) == expected
    assert len(set(mapping.values())) == expected      # bijection => same information structure


def test_leduc_random_playouts_vs_openspiel():
    rng = random.Random(0)
    me, os_ = Leduc(), leduc_os()
    for _ in range(10_000):
        a, b = me.new_initial_state(), os_.new_initial_state()
        while not a.is_terminal():
            assert a.is_chance() == b.is_chance()
            if a.is_chance():
                oa, ob = dict(a.chance_outcomes()), dict(b.chance_outcomes())
                assert oa == pytest.approx(ob)
                x = rng.choice(list(oa))
            else:
                assert a.legal_actions() == b.legal_actions()
                x = rng.choice(a.legal_actions())
            a, b = a.child(x), b.child(x)
        assert b.is_terminal()
        assert a.returns() == pytest.approx(b.returns())


@pytest.mark.parametrize("game,ncards", [(Kuhn(), 3), (Leduc(), 6)])
def test_no_leakage_of_opponent_card(game, ncards):
    """Swap ONLY the opponent's private card: acting player's infoset key must not change."""
    checked = 0
    for s in walk(game):
        if s.is_terminal() or s.is_chance():
            continue
        p = s.current_player()
        opp = 1 - p
        used = set(s.cards) | ({s.board} if getattr(s, "board", None) is not None else set())
        for c in range(ncards):
            if c in used:
                continue
            t = s.with_private(opp, c)
            assert t.infoset_key(p) == s.infoset_key(p)
            checked += 1
    assert checked > 0


def test_leduc_rules_spot_checks():
    s = Leduc().new_initial_state()
    s = s.child(0).child(2)                       # P0 gets card 0 (J), P1 gets card 2 (Q)
    s = s.child(2)                                # P0 raises (+2): contrib [3,1]
    assert s.legal_actions() == [0, 1, 2]
    s = s.child(2)                                # P1 re-raises: contrib [3,5]
    assert s.legal_actions() == [0, 1]            # max 2 raises reached
    s = s.child(1)                                # P0 calls -> round ends -> board dealt
    assert s.is_chance()
    s = s.child(4)                                # board = card 4 (rank 2, K)
    assert s.current_player() == 0 and s.contrib == [5, 5]
    s = s.child(2)                                # raise 4 in round 2
    assert s.contrib == [9, 5]
    assert s.child(0).returns() == [5.0, -5.0]    # P1 folds: P0 wins P1's 5 chips
