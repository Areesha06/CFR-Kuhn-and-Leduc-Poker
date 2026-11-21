"""Bridge between our policies and OpenSpiel (for cross-checking exploitability / CFR).

    mapping = build_key_mapping(Leduc(), leduc_os())          # my infoset key -> OpenSpiel key
    tab = to_openspiel_policy(Leduc(), leduc_os(), policy, mapping)
    from open_spiel.python.algorithms import exploitability
    exploitability.exploitability(leduc_os().g, tab)           # = (BR0 + BR1) / 2
"""
from .base import collect_infosets


def build_key_mapping(mine, theirs):
    """Walk both trees in lock-step; return {my_infoset_key: openspiel_info_state_string}."""
    mapping = {}
    stack = [(mine.new_initial_state(), theirs.new_initial_state())]
    while stack:
        a, b = stack.pop()
        if a.is_terminal():
            continue
        if a.is_chance():
            acts = [x for x, _ in a.chance_outcomes()]
        else:
            p = a.current_player()
            ka, kb = a.infoset_key(p), b.infoset_key(p)
            assert mapping.setdefault(ka, kb) == kb
            acts = a.legal_actions()
        for x in acts:
            stack.append((a.child(x), b.child(x)))
    return mapping


def to_openspiel_policy(mine, theirs, policy, mapping):
    """Expand our per-legal-action arrays into an OpenSpiel TabularPolicy (indexed by action id)."""
    from open_spiel.python import policy as ospolicy
    tab = ospolicy.TabularPolicy(theirs.g)
    legal = collect_infosets(mine)
    for mykey, oskey in mapping.items():
        arr = tab.policy_for_key(oskey)
        arr[:] = 0.0
        for a, pr in zip(legal[mykey]["actions"], policy[mykey]):
            arr[a] = pr
    return tab
