"""Human vs solved policy in Leduc. Prints the bot's action probabilities at each decision.
    python -m experiments.cli_demo --policy results/policies/leduc_cfrplus.json
Without --policy the bot plays uniformly at random (for testing).
"""
import argparse
import numpy as np
from games.leduc import Leduc
from games.base import load_policy, uniform_policy

RANKS = "JQK"; SUITS = "sh"; ACT = {0: "fold", 1: "call/check", 2: "raise"}
name = lambda c: f"{RANKS[c // 2]}{SUITS[c % 2]}"

ap = argparse.ArgumentParser(); ap.add_argument("--policy"); ap.add_argument("--seed", type=int)
args = ap.parse_args()
game = Leduc()
bot = load_policy(args.policy) if args.policy else uniform_policy(game)
rng = np.random.default_rng(args.seed)
total, hands, seat = 0.0, 0, 0
print("Leduc: ante 1, raise +2 (round 1) / +4 (round 2), max 2 raises per round. J < Q < K, pair with board wins.")
print("Keys: f=fold, c=call/check, r=raise, q=quit\n")

while True:
    me = seat                      # alternate seats every hand
    s = game.new_initial_state()
    print(f"--- Hand {hands + 1} (you are player {me}) ---")
    while not s.is_terminal():
        if s.is_chance():
            outs = s.chance_outcomes()
            a = outs[rng.choice(len(outs), p=[p for _, p in outs])][0]
            s = s.child(a)
            if s.board is not None and not s.is_chance() and s.rnd == 2 and s.hist[1] == "":
                print(f"Board card: {name(s.board)}")
            continue
        p = s.current_player(); acts = s.legal_actions()
        if p == me:
            print(f"Your card: {name(s.cards[me])} | board: {name(s.board) if s.board is not None else '-'} | "
                  f"pot: {sum(s.contrib)} | to call: {s.to_call()} | history: {s.hist}")
            while True:
                ch = input(f"Your move {[ {0:'f',1:'c',2:'r'}[x] for x in acts ]}: ").strip().lower()
                if ch == "q":
                    print(f"Final: {total:+.1f} over {hands} hands"); raise SystemExit
                m = {"f": 0, "c": 1, "r": 2}.get(ch)
                if m in acts: break
                print("Illegal move.")
            s = s.child(m)
        else:
            probs = bot[s.infoset_key(p)]
            print("  Bot's action probabilities: " + ", ".join(f"{ACT[a]}={pr:.2f}" for a, pr in zip(acts, probs)))
            a = acts[rng.choice(len(acts), p=probs)]
            print(f"  Bot chooses: {ACT[a]}")
            s = s.child(a)
    r = s.returns()[me]; total += r; hands += 1; seat = 1 - seat
    print(f"Bot's card was {name(s.cards[1 - me])}. You {'won' if r > 0 else 'lost' if r < 0 else 'tied'} {abs(r):.0f}. Total: {total:+.1f}\n")
