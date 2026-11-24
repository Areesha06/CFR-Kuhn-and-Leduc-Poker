"""Human vs a policy, in the terminal. Prints the bot's action probabilities at each decision.
    python -m experiments.cli_demo --game kuhn  [--policy results/policies/kuhn_cfr.json]
    python -m experiments.cli_demo --game leduc [--policy results/policies/leduc_cfr.json]
Without --policy the bot plays uniformly at random (for testing).
"""
import argparse, csv, os, time
import numpy as np
from games.kuhn import Kuhn
from games.leduc import Leduc
from games.base import load_policy, uniform_policy

ap = argparse.ArgumentParser()
ap.add_argument("--game", choices=["kuhn", "leduc"], default="leduc")
ap.add_argument("--policy"); ap.add_argument("--seed", type=int)
ap.add_argument("--log", help="CSV path (default: results/demo_logs/<game>_<time>.csv)")
ap.add_argument("--no-log", action="store_true")
args = ap.parse_args()

if args.game == "kuhn":
    game = Kuhn()
    card_name = lambda c: "JQK"[c]
    ACT = {0: "pass (check/fold)", 1: "bet/call"}
    KEYS = {"p": 0, "b": 1}
    intro = "Kuhn: cards J < Q < K, ante 1, one betting round, bet = 1. Highest card wins at showdown."
    def describe(s, me):
        return f"Your card: {card_name(s.cards[me])} | history: '{s.hist or '-'}' (p=pass, b=bet)"
else:
    game = Leduc()
    card_name = lambda c: f"{'JQK'[c // 2]}{'sh'[c % 2]}"
    ACT = {0: "fold", 1: "call/check", 2: "raise"}
    KEYS = {"f": 0, "c": 1, "r": 2}
    intro = ("Leduc: ante 1, raise +2 (round 1) / +4 (round 2), max 2 raises per round. "
             "J < Q < K, a pair with the board card wins.")
    def describe(s, me):
        b = card_name(s.board) if s.board is not None else "-"
        return (f"Your card: {card_name(s.cards[me])} | board: {b} | pot: {sum(s.contrib)} | "
                f"to call: {s.to_call()} | history: {s.hist}")

bot = load_policy(args.policy) if args.policy else uniform_policy(game)
rng = np.random.default_rng(args.seed)
total, hands, seat = 0.0, 0, 0
key_of = {v: k for k, v in KEYS.items()}
bot_name = os.path.basename(args.policy) if args.policy else "uniform_random"
log_path = None
if not args.no_log:
    log_path = args.log or f"results/demo_logs/{args.game}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["hand", "game", "bot", "your_seat", "your_card", "bot_card",
                                "moves", "your_payoff", "cumulative"])
    print(f"Saving results to {log_path}")
print(intro)
print("Moves: " + ", ".join(f"{k}={ACT[v]}" for k, v in KEYS.items()) + ", q=quit. Seats alternate each hand.\n")

while True:
    me = seat
    s = game.new_initial_state()
    print(f"--- Hand {hands + 1} (you are player {me}) ---")
    announced_board = False
    while not s.is_terminal():
        if s.is_chance():
            outs = s.chance_outcomes()
            s = s.child(outs[rng.choice(len(outs), p=[p for _, p in outs])][0])
            continue
        if args.game == "leduc" and s.board is not None and not announced_board:
            print(f"Board card: {card_name(s.board)}"); announced_board = True
        p = s.current_player(); acts = s.legal_actions()
        if p == me:
            print(describe(s, me))
            while True:
                ch = input(f"Your move {[key_of[a] for a in acts]}: ").strip().lower()
                if ch == "q":
                    print(f"Final: {total:+.1f} over {hands} hands" + (f" (saved to {log_path})" if log_path else "")); raise SystemExit
                m = KEYS.get(ch)
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
    if log_path:
        moves = s.hist if args.game == "kuhn" else "|".join(s.hist)
        with open(log_path, "a", newline="") as f:      # appended every hand, so nothing is lost on quit
            csv.writer(f).writerow([hands, args.game, bot_name, me, card_name(s.cards[me]),
                                    card_name(s.cards[1 - me]), moves, r, total])
    print(f"Bot's card was {card_name(s.cards[1 - me])}. You {'won' if r > 0 else 'lost' if r < 0 else 'tied'} {abs(r):.0f}. Total: {total:+.1f}\n")