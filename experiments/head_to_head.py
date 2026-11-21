"""Exact + simulated head-to-head. Usage:
    python -m experiments.head_to_head                       # bots only (works today)
    python -m experiments.head_to_head --kuhn_cfr results/policies/kuhn_cfr.json \
                                       --leduc_cfr results/policies/leduc_cfr.json
(Policy files = json dict infoset_key -> list of probs, saved by B/C with games.base.save_policy.
 For CFR+ later: pass --kuhn_cfr results/policies/kuhn_cfrplus.json etc. and --tag cfrplus.)
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from games.kuhn import Kuhn
from games.leduc import Leduc
from games.base import load_policy
from games.baselines import random_policy, kuhn_rule_policy, leduc_rule_policy
from games.play import exact_matchup, simulate, ci95

ap = argparse.ArgumentParser()
ap.add_argument("--kuhn_cfr"); ap.add_argument("--leduc_cfr")
ap.add_argument("--tag", default="cfr")
ap.add_argument("--hands", type=int, default=10_000)
args = ap.parse_args()
os.makedirs("results", exist_ok=True)

setups = [
    (Kuhn(), {"random": random_policy, "rule": kuhn_rule_policy}, args.kuhn_cfr),
    (Leduc(), {"random": random_policy, "rule": leduc_rule_policy}, args.leduc_cfr),
]
rows = []
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
ax_i = 0
for game, bots, cfr_path in setups:
    if not cfr_path:
        print(f"[skip] no CFR policy for {game.name}; pass --{game.name}_cfr"); continue
    cfr = load_policy(cfr_path)
    for bname, mk in bots.items():
        bot = mk(game)
        ex = exact_matchup(game, cfr, bot)
        sim = simulate(game, cfr, bot, hands_per_seat=args.hands, seed=0)
        s0, s1 = sim["seat0"], sim["seat1"]
        rows.append({"game": game.name, "solver": args.tag, "opponent": bname,
                     "exact_seat0": ex["seat0"], "exact_seat1": ex["seat1"], "exact_avg": ex["avg"],
                     "sim_seat0": s0.mean(), "ci_seat0": ci95(s0),
                     "sim_seat1": s1.mean(), "ci_seat1": ci95(s1),
                     "sim_avg": sim["mean"], "ci_avg": sim["ci95"]})
        print(rows[-1])
        # running mean converging to the exact value
        ax = axes.flat[ax_i]; ax_i += 1
        x = sim["all"]; run = np.cumsum(x) / np.arange(1, len(x) + 1)
        ax.plot(run, lw=1, label="simulated running mean")
        ax.axhline(ex["avg"], color="r", ls="--", label=f"exact = {ex['avg']:.4f}")
        ax.set_title(f"{game.name}: {args.tag} vs {bname}"); ax.set_xlabel("hands"); ax.legend()
if rows:
    with open(f"results/head_to_head_{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    plt.tight_layout(); plt.savefig(f"results/head_to_head_{args.tag}.png", dpi=150)
