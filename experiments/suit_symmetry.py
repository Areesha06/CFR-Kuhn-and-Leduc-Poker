"""Leduc: 936 infosets collapse to 288 when suits are ignored.
Optional: pass a solved policy to check it is the same across suit-swapped infosets.
    python -m experiments.suit_symmetry [--policy results/policies/leduc_cfr.json]
"""
import argparse
from collections import defaultdict
import numpy as np
from games.leduc import Leduc
from games.base import collect_infosets, load_policy

def collapse(key):
    c, b, h1, h2 = key.split("|")
    return f"{int(c)//2}|{'-' if b == '-' else int(b)//2}|{h1}|{h2}"

ap = argparse.ArgumentParser(); ap.add_argument("--policy"); args = ap.parse_args()
info = collect_infosets(Leduc())
groups = defaultdict(list)
for k in info:
    groups[collapse(k)].append(k)
print("infosets with suits:", len(info), "| without suits:", len(groups))
if args.policy:
    pol = load_policy(args.policy)
    worst = max(np.abs(pol[a] - pol[b]).max() for ks in groups.values() for a in ks for b in ks)
    print("max |difference| in action probabilities across suit-swapped infosets:", worst)
