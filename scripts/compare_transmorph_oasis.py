#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare our Learn2Reg OASIS dataset layout against TransMorph OASIS .pkl preprocessing."
    )
    parser.add_argument(
        "--l2r_root",
        type=Path,
        default=Path("/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis_l2r"),
        help="Path to the Learn2Reg OASIS root with OASIS_dataset.json/imagesTr/labelsTr.",
    )
    parser.add_argument(
        "--transmorph_train_dir",
        type=Path,
        default=None,
        help="Optional directory containing TransMorph OASIS training .pkl files.",
    )
    parser.add_argument(
        "--transmorph_val_dir",
        type=Path,
        default=None,
        help="Optional directory containing TransMorph OASIS validation/test .pkl files.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=8,
        help="Maximum number of .pkl files to inspect per split.",
    )
    return parser.parse_args()


def summarize_l2r(root: Path) -> dict[str, Any]:
    metadata_path = root / "OASIS_dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing Learn2Reg metadata: {metadata_path}")

    payload = json.loads(metadata_path.read_text())
    train_entries = payload.get("training", [])
    val_entries = payload.get("registration_val", [])
    train_subject_ids = [Path(item["image"]).name.replace("_0000.nii.gz", "") for item in train_entries]
    val_pairs = [
        (
            Path(item["fixed"]).name.replace("_0000.nii.gz", ""),
            Path(item["moving"]).name.replace("_0000.nii.gz", ""),
        )
        for item in val_entries
    ]
    val_subject_ids = sorted({subject_id for pair in val_pairs for subject_id in pair})

    return {
        "native_shape": tuple(int(dim) for dim in payload["tensorImageShape"]["0"]),
        "json_training_entries": len(train_subject_ids),
        "effective_train_subjects": len([sid for sid in train_subject_ids if sid not in set(val_subject_ids)]),
        "val_subjects": len(val_subject_ids),
        "val_pairs": len(val_pairs),
        "imagesTr_files": len(list((root / "imagesTr").glob("*.nii.gz"))),
        "labelsTr_files": len(list((root / "labelsTr").glob("*.nii.gz"))),
        "eval_label_ids": list(range(1, 36)),
    }


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    summary = {
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": str(array.dtype),
    }
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size:
            summary.update(
                {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                }
            )
    else:
        unique = np.unique(array)
        summary.update(
            {
                "min": int(unique.min()) if unique.size else 0,
                "max": int(unique.max()) if unique.size else 0,
                "num_unique": int(unique.size),
            }
        )
    return summary


def inspect_pkl_file(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, tuple):
        raise TypeError(f"Expected tuple payload in {path}, got {type(payload).__name__}")

    summary: dict[str, Any] = {"file": path.name, "tuple_len": len(payload)}
    if len(payload) == 2:
        image, label = payload
        summary["kind"] = "subject"
        summary["image"] = _array_summary(np.asarray(image))
        summary["label"] = _array_summary(np.asarray(label))
    elif len(payload) == 4:
        moving, fixed, moving_label, fixed_label = payload
        summary["kind"] = "pair"
        summary["moving"] = _array_summary(np.asarray(moving))
        summary["fixed"] = _array_summary(np.asarray(fixed))
        summary["moving_label"] = _array_summary(np.asarray(moving_label))
        summary["fixed_label"] = _array_summary(np.asarray(fixed_label))
    else:
        summary["kind"] = "unknown"
    return summary


def summarize_transmorph_split(split_dir: Path, max_files: int) -> dict[str, Any]:
    pkl_files = sorted(split_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found under {split_dir}")

    inspected = [inspect_pkl_file(path) for path in pkl_files[: max(1, max_files)]]
    tuple_lengths = Counter(item["tuple_len"] for item in inspected)
    kinds = Counter(item.get("kind", "unknown") for item in inspected)

    reference = inspected[0]
    shape_summary: dict[str, Any] = {}
    for key in ("image", "label", "moving", "fixed", "moving_label", "fixed_label"):
        if key in reference:
            shape_summary[key] = reference[key]["shape"]

    return {
        "split_dir": str(split_dir),
        "num_pkl_files": len(pkl_files),
        "tuple_len_counts": dict(tuple_lengths),
        "kind_counts": dict(kinds),
        "reference_shapes": shape_summary,
        "samples": inspected,
    }


def print_block(title: str, payload: dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()

    l2r_summary = summarize_l2r(args.l2r_root)
    print_block("Learn2Reg OASIS", l2r_summary)

    if args.transmorph_train_dir is not None:
        print_block(
            "TransMorph train .pkl",
            summarize_transmorph_split(args.transmorph_train_dir, args.max_files),
        )

    if args.transmorph_val_dir is not None:
        print_block(
            "TransMorph val/test .pkl",
            summarize_transmorph_split(args.transmorph_val_dir, args.max_files),
        )

    if args.transmorph_train_dir is None and args.transmorph_val_dir is None:
        print(
            "\nNo TransMorph .pkl directories were provided. "
            "Pass --transmorph_train_dir and/or --transmorph_val_dir after downloading their OASIS .pkl data."
        )


if __name__ == "__main__":
    main()
