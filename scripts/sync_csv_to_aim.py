#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aim.sdk.run import Run


def parse_args():
    parser = argparse.ArgumentParser(description="Sync a Lightning CSV metrics file into an Aim run.")
    parser.add_argument("--csv-path", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--run-hash", type=str, default=None)
    return parser.parse_args()


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(float(value))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def main():
    args = parse_args()
    csv_path = args.csv_path.resolve()
    repo = args.repo.resolve()
    state_path = args.state_path.resolve() if args.state_path else csv_path.with_suffix(".aim-sync.json")

    state = load_state(state_path)
    run_hash = args.run_hash or state.get("run_hash")
    start_row = int(state.get("last_row", -1)) + 1

    run = Run(
        run_hash=run_hash,
        repo=repo,
        experiment=args.experiment,
        force_resume=bool(run_hash),
        capture_terminal_logs=False,
        log_system_params=False,
    )
    run["experiment_name"] = args.experiment
    run["sync_source"] = "csv"
    run["sync_csv_path"] = str(csv_path)

    rows_processed = 0
    metrics_logged = 0
    current_epoch = None
    current_step = None

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index < start_row:
                continue

            parsed_epoch = parse_int(row.get("epoch"), current_epoch)
            parsed_step = parse_int(row.get("step"), current_step)
            current_epoch = parsed_epoch
            current_step = parsed_step

            for name, raw_value in row.items():
                if name in {"epoch", "step"}:
                    continue
                value = parse_float(raw_value)
                if value is None:
                    continue
                kwargs = {"name": name, "step": parsed_step, "context": {"source": "csv_sync"}}
                if parsed_epoch is not None:
                    kwargs["epoch"] = parsed_epoch
                run.track(value, **kwargs)
                metrics_logged += 1

            rows_processed += 1
            state["last_row"] = row_index

    state["run_hash"] = run.hash
    save_state(state_path, state)
    run.close()
    print(
        f"Synced {rows_processed} row(s) and {metrics_logged} metric point(s) "
        f"from {csv_path} into Aim run {run.hash}"
    )


if __name__ == "__main__":
    main()
