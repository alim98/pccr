#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_KEYS = [
    "identity_dice_mean_fg_mean",
    "registered_dice_mean_fg_mean",
    "improvement_dice_mean_fg_mean",
    "identity_hd95_mean_fg_mean",
    "registered_hd95_mean_fg_mean",
    "improvement_hd95_mean_fg_mean",
    "registered_sdlogj_mean",
    "registered_jacobian_nonpositive_fraction_mean",
    "registered_runtime_seconds_mean",
    "num_pairs",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare registration summary JSON files.")
    parser.add_argument("summaries", nargs="+", help="One or more summary.json paths.")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = []
    for path_str in args.summaries:
        path = Path(path_str)
        with open(path, "r", encoding="utf-8") as handle:
            summaries.append((path.parent.name, json.load(handle)))

    header = ["metric"] + [name for name, _ in summaries]
    widths = [max(len(col), 18) for col in header]

    for idx, (_, summary) in enumerate(summaries, start=1):
        widths[idx] = max(widths[idx], 16)
        for key in DEFAULT_KEYS:
            widths[0] = max(widths[0], len(key))
            widths[idx] = max(widths[idx], len(f"{summary.get(key, 'n/a')}"))

    def print_row(values):
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(values)))

    print_row(header)
    print_row(["-" * widths[i] for i in range(len(widths))])
    for key in DEFAULT_KEYS:
        row = [key]
        for _, summary in summaries:
            row.append(summary.get(key, "n/a"))
        print_row(row)


if __name__ == "__main__":
    main()
