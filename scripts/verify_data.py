#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from src.data.datasets import OASISL2RDataset
from src.pccr.config import PCCRConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Quick Learn2Reg OASIS data sanity check.")
    parser.add_argument("--config", type=str, default="config/pairwise_oasis_fullres.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--pair_index", type=int, default=0)
    parser.add_argument("--output_png", type=str, default="logs/pccr/verify_data/middle_slice.png")
    return parser.parse_args()


def normalize_slice(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def center_slice(volume: np.ndarray) -> np.ndarray:
    z = volume.shape[-1] // 2
    return np.rot90(normalize_slice(volume[:, :, z]))


def center_label_slice(volume: np.ndarray) -> np.ndarray:
    z = volume.shape[-1] // 2
    return np.rot90(volume[:, :, z].astype(np.int32))


def label_histogram(label: np.ndarray) -> list[tuple[int, int]]:
    values, counts = np.unique(label.astype(np.int64), return_counts=True)
    return [(int(value), int(count)) for value, count in zip(values, counts)]


def print_label_summary(prefix: str, label: np.ndarray) -> None:
    histogram = label_histogram(label)
    unique_labels = [value for value, _ in histogram]
    print(f"{prefix}_unique_labels ({len(unique_labels)}): {unique_labels}")
    print(f"{prefix}_voxels_per_label:")
    for value, count in histogram:
        print(f"  label={value:2d} voxels={count}")


def save_preview(
    output_path: Path,
    source: np.ndarray,
    target: np.ndarray,
    source_label: np.ndarray,
    target_label: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    src_slice = center_slice(source)
    tgt_slice = center_slice(target)
    src_lbl_slice = center_label_slice(source_label)
    tgt_lbl_slice = center_label_slice(target_label)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axes[0, 0].imshow(src_slice, cmap="gray", interpolation="lanczos")
    axes[0, 0].set_title("Source")
    axes[0, 1].imshow(tgt_slice, cmap="gray", interpolation="lanczos")
    axes[0, 1].set_title("Target")

    vmax = max(int(src_lbl_slice.max()), int(tgt_lbl_slice.max()), 1)
    for axis, image, title in [
        (axes[1, 0], src_lbl_slice, "Source Labels"),
        (axes[1, 1], tgt_lbl_slice, "Target Labels"),
    ]:
        axis.imshow(image, cmap="nipy_spectral", norm=colors.Normalize(vmin=0, vmax=vmax), interpolation="nearest")
        axis.set_title(title)
    for axis in axes.ravel():
        axis.axis("off")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = PCCRConfig.from_yaml(args.config)
    data_path = args.data_path or str(Path(config.oasis_l2r_data_root).expanduser())

    dataset = OASISL2RDataset(
        input_dim=config.data_size,
        data_path=data_path,
        split=args.split,
        is_pair=True,
    )

    if not dataset.pairs:
        raise ValueError(f"No pairs available for split '{args.split}'.")
    pair_index = min(max(args.pair_index, 0), len(dataset.pairs) - 1)
    source, target, source_label, target_label = dataset[pair_index]

    source_np = source.squeeze(0).numpy()
    target_np = target.squeeze(0).numpy()
    source_label_np = source_label.squeeze(0).numpy()
    target_label_np = target_label.squeeze(0).numpy()

    print(f"data_path: {data_path}")
    print(f"split: {args.split}")
    print(f"pair_index: {pair_index}")
    print(f"num_pairs: {len(dataset)}")
    print(f"native_shape: {dataset.native_shape}")
    print(f"eval_label_ids: {dataset.eval_label_ids}")
    print(f"source_shape: {tuple(source.shape)}")
    print(f"target_shape: {tuple(target.shape)}")
    print(f"source_label_shape: {tuple(source_label.shape)}")
    print(f"target_label_shape: {tuple(target_label.shape)}")
    print_label_summary("source", source_label_np)
    print_label_summary("target", target_label_np)

    output_path = Path(args.output_png)
    save_preview(output_path, source_np, target_np, source_label_np, target_label_np)
    print(f"saved_png: {output_path}")


if __name__ == "__main__":
    main()
