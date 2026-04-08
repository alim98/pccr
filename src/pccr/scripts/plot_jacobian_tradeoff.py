#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Dice-vs-SDlogJ trade-off from summary.json files.")
    parser.add_argument("summary_paths", nargs="+", help="Paths to summary.json files.")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    return parser.parse_args()


def load_record(summary_path: str, label: str) -> dict[str, float | str]:
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return {
        "label": label,
        "summary_path": summary_path,
        "registered_dice_mean_fg_mean": summary["registered_dice_mean_fg_mean"],
        "registered_hd95_mean_fg_mean": summary["registered_hd95_mean_fg_mean"],
        "registered_sdlogj_mean": summary["registered_sdlogj_mean"],
        "registered_jacobian_nonpositive_fraction_mean": summary["registered_jacobian_nonpositive_fraction_mean"],
    }


def main():
    args = parse_args()
    if args.labels and len(args.labels) != len(args.summary_paths):
        raise ValueError("--labels must match the number of summary paths.")

    records = []
    for index, summary_path in enumerate(args.summary_paths):
        label = args.labels[index] if args.labels else Path(summary_path).parent.name
        records.append(load_record(summary_path, label))

    frame = pd.DataFrame(records).sort_values("registered_sdlogj_mean")
    output_csv = Path(args.output_csv)
    output_png = Path(args.output_png)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(
        frame["registered_sdlogj_mean"],
        frame["registered_dice_mean_fg_mean"],
        s=90,
        c="#d8572a",
    )
    for row in frame.itertuples(index=False):
        ax.annotate(
            row.label,
            (row.registered_sdlogj_mean, row.registered_dice_mean_fg_mean),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    ax.set_xlabel("SDlogJ (lower is better)")
    ax.set_ylabel("Dice FG mean (higher is better)")
    ax.set_title("Accuracy–Topology Trade-off")
    ax.grid(alpha=0.25)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")

    print(f"tradeoff_csv: {output_csv}")
    print(f"tradeoff_png: {output_png}")


if __name__ == "__main__":
    main()
