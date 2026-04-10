#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch

from src.data.datasets import load_volume, normalize_image
from src.pccr.config import PCCRConfig
from src.pccr.data import create_real_pair_dataloaders
from src.pccr.eval_utils import jacobian_determinant, measure_inference_time, pair_metrics, warp_segmentation
from src.pccr.trainer import LiTPCCR
from src.pccr.utils import resize_displacement
from src.model.transformation import SpatialTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize representative PCCR registration pairs.")
    parser.add_argument("--config", type=str, default="src/pccr/configs/pairwise_oasis.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--val_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs", "oasis_l2r"], default="auto")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_num_steps", type=int, default=200)
    parser.add_argument("--max_train_subjects", type=int, default=0)
    parser.add_argument("--max_val_subjects", type=int, default=0)
    parser.add_argument("--max_val_pairs", type=int, default=20)
    parser.add_argument("--num_pairs", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="pccr_viz")
    return parser.parse_args()


def resolve_device(args) -> torch.device:
    if args.accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if args.accelerator == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_precision(args, config: PCCRConfig, device: torch.device) -> str:
    if device.type != "cuda":
        return "32-true"
    if args.precision is not None:
        return args.precision
    return "bf16-mixed" if config.use_amp else "32-true"


def normalize_slice(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def center_axial(volume: np.ndarray) -> np.ndarray:
    z = volume.shape[-1] // 2
    return np.rot90(normalize_slice(volume[:, :, z]))


def center_axial_label(volume: np.ndarray) -> np.ndarray:
    z = volume.shape[-1] // 2
    return np.rot90(volume[:, :, z].astype(np.int32))


def masked_center_axial(volume: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
    if label is not None and np.any(label > 0):
        z_indices = np.where(label > 0)[2]
        z = int(np.median(z_indices))
    else:
        z = volume.shape[-1] // 2
    return np.rot90(normalize_slice(volume[:, :, z]))


def masked_center_axial_label(volume: np.ndarray, label: np.ndarray) -> np.ndarray:
    if np.any(label > 0):
        z_indices = np.where(label > 0)[2]
        z = int(np.median(z_indices))
    else:
        z = volume.shape[-1] // 2
    return np.rot90(volume[:, :, z].astype(np.int32))


def overlay_segmentation(axis, base: np.ndarray, label: np.ndarray, title: str):
    axis.imshow(base, cmap="gray", interpolation="lanczos")
    masked = np.where(label > 0, label, np.nan)
    vmax = max(2, int(np.nanmax(masked)) if np.isfinite(masked).any() else 2)
    axis.imshow(masked, cmap="nipy_spectral", norm=colors.Normalize(vmin=1, vmax=vmax), alpha=0.45, interpolation="nearest")
    axis.set_title(title)
    axis.axis("off")


def save_pair_visualization(
    output_path: Path,
    source: np.ndarray,
    target: np.ndarray,
    moved: np.ndarray,
    source_label: np.ndarray,
    target_label: np.ndarray,
    moved_label: np.ndarray,
    jacobian_map: np.ndarray,
    metrics: dict[str, float],
):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    src_slice = masked_center_axial(source, source_label)
    tgt_slice = masked_center_axial(target, target_label)
    moved_slice = masked_center_axial(moved, moved_label)
    diff_slice = np.abs(moved_slice - tgt_slice)

    axes[0, 0].imshow(src_slice, cmap="gray", interpolation="lanczos")
    axes[0, 0].set_title("Source")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tgt_slice, cmap="gray", interpolation="lanczos")
    axes[0, 1].set_title("Target")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(moved_slice, cmap="gray", interpolation="lanczos")
    axes[0, 2].set_title("Warped Source")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(diff_slice, cmap="magma", interpolation="lanczos")
    axes[0, 3].set_title("|Warped - Target|")
    axes[0, 3].axis("off")

    overlay_segmentation(axes[1, 0], src_slice, masked_center_axial_label(source_label, source_label), "Source Seg")
    overlay_segmentation(axes[1, 1], tgt_slice, masked_center_axial_label(target_label, target_label), "Target Seg")
    overlay_segmentation(axes[1, 2], moved_slice, masked_center_axial_label(moved_label, moved_label), "Warped Seg")

    jac_slice = np.rot90(jacobian_map[:, :, jacobian_map.shape[-1] // 2])
    im = axes[1, 3].imshow(jac_slice, cmap="coolwarm", interpolation="lanczos")
    axes[1, 3].set_title("Jacobian Determinant")
    axes[1, 3].axis("off")
    fig.colorbar(im, ax=axes[1, 3], shrink=0.8)

    fig.suptitle(
        f"Dice_fg={metrics['dice_mean_fg']:.4f} | HD95={metrics['hd95_mean_fg']:.4f} | "
        f"SDlogJ={metrics['sdlogj']:.4f} | NonPosJ={metrics['jacobian_nonpositive_fraction']:.4f}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_native_pair(dataset, pair_idx: int):
    if not hasattr(dataset, "pairs"):
        return None

    src_subject, tgt_subject = dataset.pairs[pair_idx]
    source = normalize_image(load_volume(src_subject.image_path))
    target = normalize_image(load_volume(tgt_subject.image_path))
    source_label = load_volume(src_subject.label_path).astype(np.int64)
    target_label = load_volume(tgt_subject.label_path).astype(np.int64)
    return source, target, source_label, target_label


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    config = PCCRConfig.from_yaml(args.config)
    config.phase = "real"
    if args.batch_size is None:
        args.batch_size = config.batch_size

    _, val_loader = create_real_pair_dataloaders(args, config)
    dataset_num_labels = getattr(val_loader.dataset, "num_labels", 0) or 0
    if dataset_num_labels:
        config.num_labels = max(config.num_labels, dataset_num_labels)

    model = LiTPCCR.load_from_checkpoint(
        args.checkpoint_path,
        args=args,
        config=config,
        experiment_logger=None,
    )
    model.eval()

    device = resolve_device(args)
    resolved_precision = resolve_precision(args, config, device)
    model = model.to(device)
    use_cuda = device.type == "cuda"
    autocast_enabled = use_cuda and resolved_precision.startswith("bf16")

    output_dir = Path(args.output_dir or f"logs/pccr/{args.experiment_name}_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    with torch.no_grad():
        for pair_idx, batch in enumerate(val_loader):
            if pair_idx >= args.num_pairs:
                break
            source, target, source_label, target_label = [tensor.to(device) for tensor in batch]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs, elapsed = measure_inference_time(model, source, target, use_cuda)
            _, warped_label_lowres = warp_segmentation(
                source_label,
                outputs["phi_s2t"],
                config.num_labels,
                model.model.decoder.final_transformer,
            )
            metrics = pair_metrics(
                outputs=outputs,
                source_label=source_label,
                target_label=target_label,
                num_labels=config.num_labels,
                label_ids=config.eval_label_ids,
                transformer=model.model.decoder.final_transformer,
                inference_seconds=elapsed,
            )
            native_pair = load_native_pair(val_loader.dataset, pair_idx)
            if native_pair is not None:
                native_source, native_target, native_source_label, native_target_label = native_pair
                native_shape = tuple(native_source.shape)
                native_source_tensor = torch.from_numpy(native_source).unsqueeze(0).unsqueeze(0).to(device)
                native_disp = resize_displacement(outputs["phi_s2t"].float(), native_shape)
                native_img_transformer = SpatialTransformer(native_shape).to(device)
                moved_native = native_img_transformer(native_source_tensor.float(), native_disp.float())
                jacobian_map = jacobian_determinant(native_disp.float()).squeeze(0).detach().cpu().numpy()
                warped_native_label = torch.nn.functional.interpolate(
                    warped_label_lowres.float(),
                    size=native_shape,
                    mode="nearest",
                ).long()
                source_np = native_source
                target_np = native_target
                moved_np = moved_native.squeeze().detach().cpu().numpy()
                source_label_np = native_source_label
                target_label_np = native_target_label
                warped_label_np = warped_native_label.squeeze().detach().cpu().numpy()
            else:
                jacobian_map = jacobian_determinant(outputs["phi_s2t"].float()).squeeze(0).detach().cpu().numpy()
                source_np = source.squeeze().detach().cpu().numpy()
                target_np = target.squeeze().detach().cpu().numpy()
                moved_np = outputs["moved_source"].squeeze().detach().cpu().numpy()
                source_label_np = source_label.squeeze().detach().cpu().numpy()
                target_label_np = target_label.squeeze().detach().cpu().numpy()
                warped_label_np = warped_label_lowres.squeeze().detach().cpu().numpy()

            out_path = output_dir / f"pair_{pair_idx:03d}.png"

            save_pair_visualization(
                out_path,
                source_np,
                target_np,
                moved_np,
                source_label_np,
                target_label_np,
                warped_label_np,
                jacobian_map,
                metrics,
            )
            manifest.append({"pair_index": pair_idx, "image_path": str(out_path), **metrics})

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"visualization_dir: {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
