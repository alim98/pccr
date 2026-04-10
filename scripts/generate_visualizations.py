#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.results_utils import (
    RESULTS_ROOT,
    VIS_ROOT,
    build_eval_args,
    discover_best_hvit_checkpoint,
    discover_best_pccr_checkpoint,
    ensure_dir,
    load_runtime_for_visualization,
    run_single_pair,
    select_pair_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate side-by-side visualizations for PCCR and H-ViT.")
    parser.add_argument("--pccr_checkpoint", type=str, default=None)
    parser.add_argument("--hvit_checkpoint", type=str, default=None)
    parser.add_argument("--pccr_config", type=str, default=None)
    parser.add_argument("--hvit_config", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs", "oasis_l2r"], default="auto")
    parser.add_argument("--pair_indices", nargs="+", type=int, default=None)
    parser.add_argument("--evaluation_json", type=str, default=str(RESULTS_ROOT / "full_evaluation" / "full_evaluation.json"))
    parser.add_argument("--focus_labels", nargs="+", type=int, default=None)
    parser.add_argument("--top_pairs_per_label", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=str(VIS_ROOT))
    return parser.parse_args()


def normalize_slice(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def mid_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    z = volume.shape[2] // 2
    y = volume.shape[1] // 2
    x = volume.shape[0] // 2
    return {
        "axial": np.rot90(volume[:, :, z]),
        "coronal": np.rot90(volume[:, y, :]),
        "sagittal": np.rot90(volume[x, :, :]),
    }


def overlay_seg(axis, base: np.ndarray, label: np.ndarray, title: str) -> None:
    axis.imshow(normalize_slice(base), cmap="gray", interpolation="nearest")
    masked = np.where(label > 0, label, np.nan)
    vmax = max(int(np.nanmax(masked)) if np.isfinite(masked).any() else 1, 1)
    axis.imshow(masked, cmap="nipy_spectral", norm=colors.Normalize(vmin=1, vmax=vmax), alpha=0.45, interpolation="nearest")
    axis.set_title(title)
    axis.axis("off")


def overlay_binary(axis, base: np.ndarray, mask: np.ndarray, title: str, color_name: str = "lime") -> None:
    axis.imshow(normalize_slice(base), cmap="gray", interpolation="nearest")
    mask = mask.astype(bool)
    if mask.any():
        color = colors.to_rgba(color_name)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[..., :3] = color[:3]
        overlay[..., 3] = mask.astype(np.float32) * 0.45
        axis.imshow(overlay, interpolation="nearest")
        axis.contour(mask.astype(np.float32), levels=[0.5], colors=[color_name], linewidths=1.25)
    axis.set_title(title)
    axis.axis("off")


def quiver_grid_for_base(base: np.ndarray, field_u: np.ndarray, field_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = base.shape
    field_u = np.asarray(field_u)
    field_v = np.asarray(field_v)
    xs = np.linspace(0, width - 1, field_u.shape[1], dtype=np.float32)
    ys = np.linspace(0, height - 1, field_u.shape[0], dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, field_u, field_v


def extract_plane(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return np.rot90(volume[:, :, index])
    if plane == "coronal":
        return np.rot90(volume[:, index, :])
    if plane == "sagittal":
        return np.rot90(volume[index, :, :])
    raise ValueError(f"Unsupported plane: {plane}")


def best_plane_for_label(mask_volume: np.ndarray) -> tuple[str, int]:
    if mask_volume.any():
        plane_scores = {
            "axial": mask_volume.sum(axis=(0, 1)),
            "coronal": mask_volume.sum(axis=(0, 2)),
            "sagittal": mask_volume.sum(axis=(1, 2)),
        }
        best_plane = None
        best_score = -1.0
        best_index = 0
        for plane, scores in plane_scores.items():
            index = int(np.argmax(scores))
            score = float(scores[index])
            if score > best_score:
                best_plane = plane
                best_score = score
                best_index = index
        if best_plane is not None:
            return best_plane, best_index
    return "axial", mask_volume.shape[2] // 2


def crop_bounds_from_mask(mask_slice: np.ndarray, padding: int = 8) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask_slice > 0)
    if coords.size == 0:
        return None
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    row_min = max(int(row_min) - padding, 0)
    col_min = max(int(col_min) - padding, 0)
    row_max = min(int(row_max) + padding + 1, mask_slice.shape[0])
    col_max = min(int(col_max) + padding + 1, mask_slice.shape[1])
    return row_min, row_max, col_min, col_max


def crop_slice(slice_: np.ndarray, bounds: tuple[int, int, int, int] | None) -> np.ndarray:
    if bounds is None:
        return slice_
    row_min, row_max, col_min, col_max = bounds
    return slice_[row_min:row_max, col_min:col_max]


def focused_plane_payload(payload: dict, label_id: int) -> dict[str, np.ndarray | str | int]:
    target_mask = payload["target_label"] == label_id
    warped_mask = payload["warped_label"] == label_id
    moving_mask = payload["source_label"] == label_id
    union_mask = target_mask | warped_mask | moving_mask
    plane, index = best_plane_for_label(union_mask)

    target_mask_slice = extract_plane(target_mask.astype(np.uint8), plane, index)
    warped_mask_slice = extract_plane(warped_mask.astype(np.uint8), plane, index)
    moving_mask_slice = extract_plane(moving_mask.astype(np.uint8), plane, index)
    union_mask_slice = target_mask_slice | warped_mask_slice | moving_mask_slice
    bounds = crop_bounds_from_mask(union_mask_slice)

    return {
        "plane": plane,
        "index": index,
        "moving": crop_slice(extract_plane(payload["source"], plane, index), bounds),
        "fixed": crop_slice(extract_plane(payload["target"], plane, index), bounds),
        "warped": crop_slice(extract_plane(payload["moved"], plane, index), bounds),
        "moving_mask": crop_slice(moving_mask_slice.astype(np.uint8), bounds),
        "fixed_mask": crop_slice(target_mask_slice.astype(np.uint8), bounds),
        "warped_mask": crop_slice(warped_mask_slice.astype(np.uint8), bounds),
        "jacobian": crop_slice(extract_plane(payload["jacobian"], plane, index), bounds),
        "bounds": bounds,
    }


def focused_quiver_components(payload: dict, plane: str, index: int, bounds: tuple[int, int, int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    disp = payload["displacement"]
    if plane == "axial":
        u = np.rot90(disp[2, :, :, index])
        v = np.rot90(disp[1, :, :, index])
    elif plane == "coronal":
        u = np.rot90(disp[2, :, index, :])
        v = np.rot90(disp[0, :, index, :])
    else:
        u = np.rot90(disp[1, index, :, :])
        v = np.rot90(disp[0, index, :, :])
    return crop_slice(u, bounds), crop_slice(v, bounds)


def label_gain_lookup(evaluation_json: Path, label_ids: list[int]) -> dict[tuple[int, int], float]:
    if not evaluation_json.exists():
        return {}
    payload = json.loads(evaluation_json.read_text(encoding="utf-8"))
    pccr_pairs = payload.get("models", {}).get("pccr", {}).get("per_pair", [])
    hvit_pairs = payload.get("models", {}).get("hvit", {}).get("per_pair", [])
    gains = {}
    for pccr_pair, hvit_pair in zip(pccr_pairs, hvit_pairs):
        pair_index = int(pccr_pair["pair_index"])
        for label_id in label_ids:
            label_key = str(label_id)
            gains[(label_id, pair_index)] = (
                float(pccr_pair["registered_dice_per_structure"][label_key])
                - float(hvit_pair["registered_dice_per_structure"][label_key])
            )
    return gains


def choose_pairs_from_results(evaluation_json: Path, focus_labels: list[int], top_pairs_per_label: int) -> list[int]:
    if not evaluation_json.exists():
        return []
    payload = json.loads(evaluation_json.read_text(encoding="utf-8"))
    pccr_pairs = payload.get("models", {}).get("pccr", {}).get("per_pair", [])
    hvit_pairs = payload.get("models", {}).get("hvit", {}).get("per_pair", [])
    ordered_pairs: list[int] = []
    for label_id in focus_labels:
        label_key = str(label_id)
        candidates = []
        for pccr_pair, hvit_pair in zip(pccr_pairs, hvit_pairs):
            diff = float(pccr_pair["registered_dice_per_structure"][label_key]) - float(hvit_pair["registered_dice_per_structure"][label_key])
            candidates.append((diff, int(pccr_pair["pair_index"])))
        candidates.sort(reverse=True)
        for _, pair_index in candidates[: max(1, top_pairs_per_label)]:
            if pair_index not in ordered_pairs:
                ordered_pairs.append(pair_index)
    return ordered_pairs


def save_slices_figure(pair_dir: Path, pair_index: int, pccr: dict, hvit: dict) -> None:
    fig, axes = plt.subplots(2, 9, figsize=(28, 8), constrained_layout=True)
    for row, (name, payload) in enumerate([("PCCR", pccr), ("H-ViT", hvit)]):
        slices = {
            "moving": mid_slices(payload["source"]),
            "fixed": mid_slices(payload["target"]),
            "warped": mid_slices(payload["moved"]),
        }
        panels = [
            ("moving", "axial"),
            ("fixed", "axial"),
            ("warped", "axial"),
            ("moving", "coronal"),
            ("fixed", "coronal"),
            ("warped", "coronal"),
            ("moving", "sagittal"),
            ("fixed", "sagittal"),
            ("warped", "sagittal"),
        ]
        for col, (image_name, plane) in enumerate(panels):
            axes[row, col].imshow(normalize_slice(slices[image_name][plane]), cmap="gray")
            axes[row, col].set_title(f"{name} {image_name} {plane}")
            axes[row, col].axis("off")
    fig.suptitle(f"Pair {pair_index:03d}: moving/fixed/warped mid-slices")
    fig.savefig(pair_dir / "slices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_quiver_figure(pair_dir: Path, pair_index: int, pccr: dict, hvit: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for row, (name, payload) in enumerate([("PCCR", pccr), ("H-ViT", hvit)]):
        source_slices = mid_slices(payload["source"])
        disp = payload["displacement"]
        step = max(4, disp.shape[1] // 16)
        z = disp.shape[3] // 2
        y = disp.shape[2] // 2
        x = disp.shape[1] // 2

        planes = [
            ("axial", source_slices["axial"], disp[2, ::step, ::step, z], disp[1, ::step, ::step, z]),
            ("coronal", source_slices["coronal"], disp[2, ::step, y, ::step], disp[0, ::step, y, ::step]),
            ("sagittal", source_slices["sagittal"], disp[1, x, ::step, ::step], disp[0, x, ::step, ::step]),
        ]
        for col, (title, base, u, v) in enumerate(planes):
            axes[row, col].imshow(normalize_slice(base), cmap="gray")
            u_rot = np.rot90(u)
            v_rot = np.rot90(v)
            xx, yy, u_plot, v_plot = quiver_grid_for_base(base, u_rot, v_rot)
            axes[row, col].quiver(xx, yy, u_plot, v_plot, color="cyan", scale=20)
            axes[row, col].set_title(f"{name} {title}")
            axes[row, col].axis("off")
    fig.suptitle(f"Pair {pair_index:03d}: deformation quiver overlays")
    fig.savefig(pair_dir / "quiver.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_jacobian_figure(pair_dir: Path, pair_index: int, pccr: dict, hvit: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for row, (name, payload) in enumerate([("PCCR", pccr), ("H-ViT", hvit)]):
        jac_slices = mid_slices(payload["jacobian"])
        for col, plane in enumerate(["axial", "coronal", "sagittal"]):
            jac = jac_slices[plane]
            axes[row, col].imshow(jac, cmap="coolwarm", vmin=np.nanpercentile(jac, 2), vmax=np.nanpercentile(jac, 98))
            negative = np.where(jac <= 0, 1.0, np.nan)
            axes[row, col].imshow(negative, cmap=colors.ListedColormap(["red"]), alpha=0.55)
            axes[row, col].set_title(f"{name} {plane}")
            axes[row, col].axis("off")
    fig.suptitle(f"Pair {pair_index:03d}: Jacobian determinant maps")
    fig.savefig(pair_dir / "jacobian.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_segmentation_figure(pair_dir: Path, pair_index: int, pccr: dict, hvit: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for row, (name, payload) in enumerate([("PCCR", pccr), ("H-ViT", hvit)]):
        source_slices = mid_slices(payload["source"])
        target_slices = mid_slices(payload["target"])
        moved_slices = mid_slices(payload["moved"])
        source_seg = mid_slices(payload["source_label"])
        target_seg = mid_slices(payload["target_label"])
        warped_seg = mid_slices(payload["warped_label"])
        overlay_seg(axes[row, 0], source_slices["axial"], source_seg["axial"], f"{name} moving seg")
        overlay_seg(axes[row, 1], target_slices["axial"], target_seg["axial"], f"{name} fixed seg")
        overlay_seg(axes[row, 2], moved_slices["axial"], warped_seg["axial"], f"{name} warped seg")
    fig.suptitle(f"Pair {pair_index:03d}: segmentation overlays")
    fig.savefig(pair_dir / "segmentation_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_label_focus_figure(
    output_dir: Path,
    pair_index: int,
    label_id: int,
    pccr: dict,
    hvit: dict,
    dice_gain: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    for row, (name, payload) in enumerate([("PCCR", pccr), ("H-ViT", hvit)]):
        focus = focused_plane_payload(payload, label_id)
        overlay_binary(axes[row, 0], focus["moving"], focus["moving_mask"], f"{name} moving")
        overlay_binary(axes[row, 1], focus["fixed"], focus["fixed_mask"], f"{name} fixed")
        overlay_binary(axes[row, 2], focus["warped"], focus["warped_mask"], f"{name} warped")

        axes[row, 3].imshow(normalize_slice(focus["fixed"]), cmap="gray", interpolation="nearest")
        u, v = focused_quiver_components(payload, focus["plane"], focus["index"], focus["bounds"])
        step = max(1, min(u.shape) // 16)
        u_sampled = u[::step, ::step]
        v_sampled = v[::step, ::step]
        xx, yy, u_plot, v_plot = quiver_grid_for_base(focus["fixed"], u_sampled, v_sampled)
        axes[row, 3].quiver(xx, yy, u_plot, v_plot, color="cyan", scale=12)
        if np.any(focus["fixed_mask"]):
            axes[row, 3].contour(focus["fixed_mask"].astype(np.float32), levels=[0.5], colors=["yellow"], linewidths=1.0)
        axes[row, 3].set_title(f"{name} deformation")
        axes[row, 3].axis("off")

        jac = focus["jacobian"]
        axes[row, 4].imshow(jac, cmap="coolwarm", vmin=np.nanpercentile(jac, 2), vmax=np.nanpercentile(jac, 98))
        negative = np.where(jac <= 0, 1.0, np.nan)
        axes[row, 4].imshow(negative, cmap=colors.ListedColormap(["red"]), alpha=0.55, interpolation="nearest")
        if np.any(focus["fixed_mask"]):
            axes[row, 4].contour(focus["fixed_mask"].astype(np.float32), levels=[0.5], colors=["yellow"], linewidths=1.0)
        axes[row, 4].set_title(f"{name} jacobian")
        axes[row, 4].axis("off")

    title = f"Label {label_id} focus, pair {pair_index:03d}"
    if dice_gain is not None:
        title += f", PCCR-HViT Dice +{dice_gain:.3f}"
    fig.suptitle(title)
    fig.savefig(output_dir / f"label_{label_id:03d}_pair_{pair_index:03d}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    pccr_eval_args = build_eval_args(
        model="pccr",
        checkpoint_path=args.pccr_checkpoint or discover_best_pccr_checkpoint(),
        config=args.pccr_config,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        num_gpus=args.num_gpus,
    )
    hvit_eval_args = build_eval_args(
        model="hvit",
        checkpoint_path=args.hvit_checkpoint or discover_best_hvit_checkpoint(),
        config=args.hvit_config,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        num_gpus=args.num_gpus,
    )

    pccr_runtime, pccr_device = load_runtime_for_visualization(pccr_eval_args)
    hvit_runtime, hvit_device = load_runtime_for_visualization(hvit_eval_args)
    num_pairs = min(len(pccr_runtime["val_loader"].dataset), len(hvit_runtime["val_loader"].dataset))
    evaluation_json = Path(args.evaluation_json)
    if args.pair_indices is not None:
        pair_indices = select_pair_indices(num_pairs, args.pair_indices, count=len(args.pair_indices))
    elif args.focus_labels:
        chosen = choose_pairs_from_results(evaluation_json, args.focus_labels, args.top_pairs_per_label)
        pair_indices = select_pair_indices(num_pairs, chosen, count=len(chosen) or 3)
    else:
        pair_indices = select_pair_indices(num_pairs, None, count=3)

    dice_gains = label_gain_lookup(evaluation_json, args.focus_labels or [])

    for pair_index in pair_indices:
        pair_dir = ensure_dir(output_dir / f"pair_{pair_index:03d}")
        pccr_payload = run_single_pair(pccr_runtime, pccr_eval_args, pair_index, pccr_device)
        hvit_payload = run_single_pair(hvit_runtime, hvit_eval_args, pair_index, hvit_device)
        save_slices_figure(pair_dir, pair_index, pccr_payload, hvit_payload)
        save_quiver_figure(pair_dir, pair_index, pccr_payload, hvit_payload)
        save_jacobian_figure(pair_dir, pair_index, pccr_payload, hvit_payload)
        save_segmentation_figure(pair_dir, pair_index, pccr_payload, hvit_payload)
        print(f"visualization_pair_dir: {pair_dir}")
        if args.focus_labels:
            focus_dir = ensure_dir(output_dir / "label_focus")
            for label_id in args.focus_labels:
                save_label_focus_figure(
                    focus_dir,
                    pair_index,
                    label_id,
                    pccr_payload,
                    hvit_payload,
                    dice_gain=dice_gains.get((label_id, pair_index)),
                )
                print(f"label_focus_figure: {focus_dir / f'label_{label_id:03d}_pair_{pair_index:03d}.png'}")


if __name__ == "__main__":
    main()
