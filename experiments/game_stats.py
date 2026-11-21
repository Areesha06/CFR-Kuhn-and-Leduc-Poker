"""Table of histories / terminals / infosets for the report. Writes results/game_stats.csv"""
import csv, os
from games.base import walk, collect_infosets
from games.kuhn import Kuhn
from games.leduc import Leduc

rows = []
for g in (Kuhn(), Leduc()):
    hist = term = chance = dec = 0
    for s in walk(g):
        hist += 1
        if s.is_terminal(): term += 1
        elif s.is_chance(): chance += 1
        else: dec += 1
    info = collect_infosets(g)
    rows.append({"game": g.name, "histories": hist, "terminal": term, "chance_nodes": chance,
                 "decision_nodes": dec, "infosets": len(info),
                 "infosets_p0": sum(v["player"] == 0 for v in info.values()),
                 "infosets_p1": sum(v["player"] == 1 for v in info.values())})
os.makedirs("results", exist_ok=True)
with open("results/game_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
