"""Plot the three CFR strategies from one game's results CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# The CSV column names go on the left and their legend labels go on the right.
SERIES = {
    "exploitability_current": "Current strategy",
    "exploitability_avg_uniform": "Uniform average",
    "exploitability_avg_linear": "Linear average",
}
# Each graph uses a different way to measure training progress.
X_AXES = {
    "iteration": "Iterations",
    "nodes_visited": "Cumulative nodes visited",
    "training_seconds": "Cumulative training time (seconds)",
}


def main():
    # Take the CSV path from the command line and choose where to save the plots.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "plots")
    args = parser.parse_args()

    # Read each CSV row as a dictionary so we can use the column names.
    with args.csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        parser.error("the CSV has no checkpoint rows")

    # Keep Kuhn and Leduc on their own graphs.
    game = rows[0]["game"]
    if any(row["game"] != game for row in rows):
        parser.error("use a CSV containing one game at a time")

    # Sort first so the lines connect the checkpoints in training order.
    rows.sort(key=lambda row: int(row["iteration"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Make one figure per x-axis, with a line for each strategy.
    for x_column, x_label in X_AXES.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        x_values = [float(row[x_column]) for row in rows]
        all_scores = []

        for column, label in SERIES.items():
            # CSV values come in as text. Convert them to numbers for the plot.
            scores = [float(row[column]) for row in rows]
            all_scores.extend(scores)
            ax.plot(x_values, scores, marker="o", label=label)

        # A log scale gives checkpoints like 10, 100, and 1,000 equal spacing.
        ax.set_xscale("log")
        # Use a scale that can show zero if any exploitability score reaches it.
        if min(all_scores) > 0:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-8)

        # Label the graph so you can tell what the axes and lines mean.
        ax.set_xlabel(x_label)
        ax.set_ylabel("Exploitability")
        ax.set_title(f"{game.capitalize()} Poker: Vanilla CFR")
        ax.grid(True, alpha=0.25)
        ax.legend()

        # Include the game and x-axis in the filename to tell the plots apart.
        path = args.output_dir / f"{game}_exploitability_vs_{x_column}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
