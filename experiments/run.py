"""Train vanilla CFR on Kuhn or Leduc and save checkpoint results."""

import argparse
import csv
from pathlib import Path
from time import perf_counter
from games.base import save_policy

from eval.exploitability import exploitability
from games.kuhn import Kuhn
from games.leduc import Leduc
from solvers.cfr import cfr_iteration
from solvers.regret_table import RegretTable
from solvers.tree import build_tree


# Turn the name from the command line into the game we want to train on.
GAMES = {"kuhn": Kuhn, "leduc": Leduc}

# At each checkpoint, compare the latest strategy with both averages.
POLICY_KINDS = ("current", "avg_uniform", "avg_linear")
# Keep the CSV columns in the same order for every run.
COLUMNS = [
    "game",
    "iteration",
    "training_seconds",
    "nodes_visited",
    "exploitability_current",
    "exploitability_avg_uniform",
    "exploitability_avg_linear",
]


def main():
    # Choose the game, when to save results, and where to put the CSV.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=GAMES, default="kuhn")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--output", type=Path, help="Path for the results CSV.")
    args = parser.parse_args()

    # Use a set so repeated checkpoint numbers only give one result.
    checkpoints = set(args.checkpoints)
    if min(checkpoints) < 1:
        parser.error("checkpoints must be positive iteration numbers")

    # Use a separate file for each game unless the user gives us a path.
    path = args.output or Path("results") / f"{args.game}_cfr.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build the game tree once and keep the same regret table as we train.
    tree = build_tree(GAMES[args.game]())
    table = RegretTable()
    training_seconds = 0.0
    nodes_visited = 0

    # Start the CSV with a header so the plot script can look up columns by name.
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()

        for iteration in range(1, max(checkpoints) + 1):
            # Stop the timer before measuring exploitability.
            start = perf_counter()
            step = cfr_iteration(tree, table)
            training_seconds += perf_counter() - start
            nodes_visited += step["nodes_visited"]

            # Train every iteration and measure exploitability at checkpoints.
            if iteration not in checkpoints:
                continue

            row = {
                "game": args.game,
                "iteration": iteration,
                "training_seconds": training_seconds,
                "nodes_visited": nodes_visited,
            }
            # Fill in the score for each of the three strategies.
            for kind in POLICY_KINDS:
                policy = table.policy(kind)
                row[f"exploitability_{kind}"] = exploitability(tree, policy)

            # Write the row now so we can see it even while training continues.
            writer.writerow(row)
            file.flush()
            print(f"{args.game}: saved checkpoint at iteration {iteration}", flush=True)

    print(f"Results saved to {path}")
    # Save the trained bot's move probabilities for the playable demo.
    policy_path = path.with_name(f"{path.stem}_policy.json")
    save_policy(table.policy("avg_uniform"), str(policy_path))
    print(f"Trained policy saved to {policy_path}")


if __name__ == "__main__":
    main()
