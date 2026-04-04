#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pccr.eval_utils import aggregate_metrics, save_metrics_report


def parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded registration evaluation reports.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input shard directories or per_pair_metrics.csv files.",
    )
    parser.add_argument("--output_dir", required=True, help="Merged output directory.")
    return parser.parse_args()


def resolve_csv(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        csv_path = path / "per_pair_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find {csv_path}")
        return csv_path
    if path.name.endswith(".csv"):
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")
        return path
    raise ValueError(f"Unsupported input {path}; expected a directory or per_pair_metrics.csv file.")


def main():
    args = parse_args()

    frames = [pd.read_csv(resolve_csv(path_str)) for path_str in args.inputs]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("pair_index").drop_duplicates(subset=["pair_index"], keep="first")
    records = merged.to_dict(orient="records")
    summary = aggregate_metrics(records)
    summary_path, pairs_path = save_metrics_report(args.output_dir, summary, records)

    print(f"merged_pairs: {len(records)}")
    print(f"summary_json: {summary_path}")
    print(f"per_pair_csv: {pairs_path}")


if __name__ == "__main__":
    main()
