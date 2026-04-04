#!/usr/bin/env python3
"""Render a quick-look visualization for an OASIS subject volume."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import nibabel as nib
import numpy as np


class SegmentationBundle(NamedTuple):
    t1_path: Path
    aseg_path: Path
    stats_path: Path | None


def resolve_volume_path(subject_path: Path) -> Path:
    """Accept a subject dir, RAW dir, or direct Analyze/NIfTI file path."""
    if subject_path.is_file():
        return subject_path

    raw_dir = subject_path / "RAW"
    search_dir = raw_dir if raw_dir.is_dir() else subject_path

    candidates = sorted(search_dir.glob("*mpr-1_anon.hdr"))
    if not candidates:
        candidates = sorted(search_dir.glob("*.hdr"))
    if not candidates:
        candidates = sorted(search_dir.glob("*.nii*"))
    if not candidates:
        raise FileNotFoundError(f"No volume file found under {subject_path}")

    return candidates[0]


def load_volume(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    volume = np.asarray(image.get_fdata())
    if volume.ndim == 4:
        volume = volume[..., 0]
    return volume


def resolve_segmentation_bundle(subject_path: Path) -> SegmentationBundle:
    """Resolve the standard FreeSurfer files inside a subject directory."""
    t1_path = subject_path / "mri" / "T1.mgz"
    aseg_path = subject_path / "mri" / "aseg.mgz"
    stats_path = subject_path / "stats" / "aseg.stats"

    missing = [path for path in (t1_path, aseg_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing FreeSurfer files: {missing_text}")

    return SegmentationBundle(
        t1_path=t1_path,
        aseg_path=aseg_path,
        stats_path=stats_path if stats_path.exists() else None,
    )


def normalize_slice(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)

    scaled = np.clip((data - low) / (high - low), 0.0, 1.0)
    return scaled.astype(np.float32)


def make_montage(volume: np.ndarray, slices: int = 9) -> np.ndarray:
    z_positions = np.linspace(0, volume.shape[2] - 1, slices, dtype=int)
    images = [np.rot90(normalize_slice(volume[:, :, z])) for z in z_positions]
    return np.concatenate(images, axis=1)


def make_overlay(base_slice: np.ndarray, label_slice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.rot90(normalize_slice(base_slice))
    labels = np.rot90(label_slice.astype(np.int32))

    overlay = np.full(labels.shape, np.nan, dtype=np.float32)
    unique_labels = sorted(int(label) for label in np.unique(labels) if label > 0)
    if unique_labels:
        label_to_index = {label: index + 1 for index, label in enumerate(unique_labels)}
        for label, index in label_to_index.items():
            overlay[labels == label] = index

    return base, overlay


def make_overlay_montage(
    base_volume: np.ndarray, label_volume: np.ndarray, slices: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    z_positions = np.linspace(0, base_volume.shape[2] - 1, slices, dtype=int)
    base_images = []
    overlay_images = []

    for z_index in z_positions:
        base, overlay = make_overlay(base_volume[:, :, z_index], label_volume[:, :, z_index])
        base_images.append(base)
        overlay_images.append(overlay)

    return np.concatenate(base_images, axis=1), np.concatenate(overlay_images, axis=1)


def parse_aseg_stats(stats_path: Path | None, limit: int = 8) -> list[str]:
    if stats_path is None:
        return ["No aseg.stats file found."]

    regions: list[tuple[float, str]] = []
    for line in stats_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        volume_mm3 = float(parts[3])
        if volume_mm3 <= 0:
            continue
        structure_name = parts[4]
        regions.append((volume_mm3, structure_name))

    regions.sort(reverse=True)
    return [f"{name}: {volume_mm3:,.0f} mm^3" for volume_mm3, name in regions[:limit]]


def save_preview(
    volume: np.ndarray,
    output_path: Path,
    title: str,
    segmentation_bundle: SegmentationBundle | None = None,
) -> None:
    center = tuple(dim // 2 for dim in volume.shape)

    sagittal = np.rot90(normalize_slice(volume[center[0], :, :]))
    coronal = np.rot90(normalize_slice(volume[:, center[1], :]))
    axial = np.rot90(normalize_slice(volume[:, :, center[2]]))
    montage = make_montage(volume)

    if segmentation_bundle is None:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 8),
            gridspec_kw={"height_ratios": [1.0, 1.15]},
            constrained_layout=True,
        )

        panels = [
            (axes[0, 0], sagittal, "Sagittal"),
            (axes[0, 1], coronal, "Coronal"),
            (axes[1, 0], axial, "Axial"),
            (axes[1, 1], montage, "Axial Montage"),
        ]

        for axis, image, label in panels:
            axis.imshow(image, cmap="gray", interpolation="nearest")
            axis.set_title(label)
            axis.axis("off")
    else:
        t1_volume = load_volume(segmentation_bundle.t1_path)
        aseg_volume = load_volume(segmentation_bundle.aseg_path).astype(np.int32)
        seg_center = tuple(dim // 2 for dim in t1_volume.shape)

        axial_base, axial_overlay = make_overlay(
            t1_volume[:, :, seg_center[2]], aseg_volume[:, :, seg_center[2]]
        )
        coronal_base, coronal_overlay = make_overlay(
            t1_volume[:, seg_center[1], :], aseg_volume[:, seg_center[1], :]
        )
        overlay_base_montage, overlay_montage = make_overlay_montage(t1_volume, aseg_volume)
        summary_lines = parse_aseg_stats(segmentation_bundle.stats_path)

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 10),
            gridspec_kw={"height_ratios": [1.0, 1.15]},
            constrained_layout=True,
        )

        base_panels = [
            (axes[0, 0], sagittal, "Raw Sagittal"),
            (axes[0, 1], coronal, "Raw Coronal"),
            (axes[0, 2], axial, "Raw Axial"),
            (axes[1, 0], montage, "Raw Axial Montage"),
        ]

        for axis, image, label in base_panels:
            axis.imshow(image, cmap="gray", interpolation="nearest")
            axis.set_title(label)
            axis.axis("off")

        overlay_norm = colors.Normalize(vmin=1, vmax=max(2, np.nanmax(overlay_montage)))

        axes[1, 1].imshow(overlay_base_montage, cmap="gray", interpolation="nearest")
        axes[1, 1].imshow(
            overlay_montage,
            cmap="nipy_spectral",
            norm=overlay_norm,
            interpolation="nearest",
            alpha=0.45,
        )
        axes[1, 1].set_title("FreeSurfer aseg Overlay Montage")
        axes[1, 1].axis("off")

        axes[0, 2].imshow(axial_base, cmap="gray", interpolation="nearest")
        axes[0, 2].imshow(
            axial_overlay,
            cmap="nipy_spectral",
            norm=overlay_norm,
            interpolation="nearest",
            alpha=0.45,
        )
        axes[0, 2].set_title("Axial aseg Overlay")
        axes[0, 2].axis("off")

        axes[0, 1].imshow(coronal_base, cmap="gray", interpolation="nearest")
        axes[0, 1].imshow(
            coronal_overlay,
            cmap="nipy_spectral",
            norm=overlay_norm,
            interpolation="nearest",
            alpha=0.45,
        )
        axes[0, 1].set_title("Coronal aseg Overlay")
        axes[0, 1].axis("off")

        axes[0, 0].imshow(sagittal, cmap="gray", interpolation="nearest")
        axes[0, 0].set_title("Raw Sagittal")
        axes[0, 0].axis("off")

        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.0,
            1.0,
            "Largest aseg regions\n\n" + "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
        axes[1, 2].set_title("aseg.stats Summary")

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a quick-look PNG preview for an OASIS MRI subject."
    )
    parser.add_argument(
        "subject_path",
        type=Path,
        help="Path to an OASIS subject directory, RAW directory, or volume file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/test/oasis_preview.png"),
        help="Where to save the rendered PNG.",
    )
    parser.add_argument(
        "--seg-subject-path",
        type=Path,
        help="Optional FreeSurfer subject directory for segmentation overlay.",
    )
    args = parser.parse_args()

    volume_path = resolve_volume_path(args.subject_path)
    volume = load_volume(volume_path)
    segmentation_bundle = None
    if args.seg_subject_path is not None:
        segmentation_bundle = resolve_segmentation_bundle(args.seg_subject_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_preview(
        volume,
        args.output,
        title=volume_path.stem,
        segmentation_bundle=segmentation_bundle,
    )
    print(args.output)


if __name__ == "__main__":
    main()
