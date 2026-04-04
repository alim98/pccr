#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch

from src.data.datasets import get_dataloader, load_volume, normalize_image
from src.model.transformation import SpatialTransformer
from src.pccr.config import PCCRConfig
from src.pccr.eval_utils import identity_metrics, jacobian_determinant, measure_inference_time, pair_metrics, warp_segmentation
from src.pccr.trainer import LiTPCCR
from src.pccr.utils import resize_displacement
from src.scripts.evaluate_hvit import DTYPE_MAP
from src.trainer import LiTHViT
from src.utils import read_yaml_file


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize side-by-side registration results for HViT and two PCCR checkpoints.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--pccr_config", type=str, default="src/pccr/configs/pairwise_oasis.yaml")
    parser.add_argument("--hvit_checkpoint", type=str, required=True)
    parser.add_argument("--pccr_short_checkpoint", type=str, required=True)
    parser.add_argument("--pccr_long_checkpoint", type=str, required=True)
    parser.add_argument("--hvit_name", type=str, default="HViT")
    parser.add_argument("--pccr_short_name", type=str, default="PCCR short")
    parser.add_argument("--pccr_long_name", type=str, default="PCCR long")
    parser.add_argument("--train_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--val_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs"], default="oasis_fs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--hvit_precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--pccr_precision", type=str, default="bf16-mixed")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_val_subjects", type=int, default=0)
    parser.add_argument("--max_val_pairs", type=int, default=0)
    parser.add_argument("--pair_indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output_dir", type=str, default="logs/registration_comparison_viz")
    return parser.parse_args()


def resolve_device(args) -> torch.device:
    if args.accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if args.accelerator == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_slice(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def masked_center_axial(volume: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
    if label is not None and np.any(label > 0):
        z = int(np.median(np.where(label > 0)[2]))
    else:
        z = volume.shape[-1] // 2
    return np.rot90(normalize_slice(volume[:, :, z]))


def masked_center_axial_label(volume: np.ndarray, label: np.ndarray) -> np.ndarray:
    if np.any(label > 0):
        z = int(np.median(np.where(label > 0)[2]))
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


def load_native_pair(dataset, pair_idx: int):
    src_subject, tgt_subject = dataset.pairs[pair_idx]
    source = normalize_image(load_volume(src_subject.image_path))
    target = normalize_image(load_volume(tgt_subject.image_path))
    source_label = load_volume(src_subject.label_path).astype(np.int64)
    target_label = load_volume(tgt_subject.label_path).astype(np.int64)
    return source, target, source_label, target_label


def build_hvit_model(args, config: dict, num_labels: int):
    hvit_args = argparse.Namespace(**vars(args))
    hvit_args.lr = getattr(hvit_args, "lr", 1e-4)
    hvit_args.mse_weights = getattr(hvit_args, "mse_weights", 1.0)
    hvit_args.dice_weights = getattr(hvit_args, "dice_weights", 1.0)
    hvit_args.grad_weights = getattr(hvit_args, "grad_weights", 0.02)
    hvit_args.tgt2src_reg = getattr(hvit_args, "tgt2src_reg", True)
    hvit_args.num_labels = num_labels
    hvit_args.precision = args.hvit_precision
    hvit_args.hvit_light = True
    return LiTHViT.load_from_checkpoint(args.hvit_checkpoint, args=hvit_args, experiment_logger=None)


def build_pccr_model(args, checkpoint_path: str, config: PCCRConfig):
    pccr_args = argparse.Namespace(**vars(args))
    pccr_args.phase = "real"
    pccr_args.precision = args.pccr_precision
    pccr_args.lr = 1e-4
    pccr_args.max_epochs = 1
    return LiTPCCR.load_from_checkpoint(checkpoint_path, args=pccr_args, config=config, experiment_logger=None)


def run_hvit(model: LiTHViT, source: torch.Tensor, target: torch.Tensor, source_label: torch.Tensor, target_label: torch.Tensor, num_labels: int, device: torch.device, precision: str):
    dtype_ = DTYPE_MAP.get(precision, torch.float32)
    use_cuda = device.type == "cuda"
    autocast_enabled = use_cuda and dtype_ in {torch.bfloat16, torch.float16}
    with torch.autocast(device_type=device.type, dtype=dtype_, enabled=autocast_enabled):
        outputs_tuple, elapsed = measure_inference_time(
            lambda src, tgt: model.hvit(src.to(dtype=dtype_), tgt.to(dtype=dtype_)),
            source,
            target,
            use_cuda,
        )
    moved_source, displacement = outputs_tuple
    outputs = {"moved_source": moved_source, "phi_s2t": displacement}
    metrics = pair_metrics(
        outputs=outputs,
        source_label=source_label,
        target_label=target_label,
        num_labels=num_labels,
        transformer=model.hvit.spatial_trans,
        inference_seconds=elapsed,
    )
    return outputs, metrics


def run_pccr(model: LiTPCCR, source: torch.Tensor, target: torch.Tensor, source_label: torch.Tensor, target_label: torch.Tensor, num_labels: int, device: torch.device, precision: str):
    use_cuda = device.type == "cuda"
    autocast_enabled = use_cuda and precision.startswith("bf16")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        outputs, elapsed = measure_inference_time(model, source, target, use_cuda)
    metrics = pair_metrics(
        outputs=outputs,
        source_label=source_label,
        target_label=target_label,
        num_labels=num_labels,
        transformer=model.model.decoder.final_transformer,
        inference_seconds=elapsed,
    )
    return outputs, metrics


def render_native_outputs(
    native_source: np.ndarray,
    native_target: np.ndarray,
    native_source_label: np.ndarray,
    native_target_label: np.ndarray,
    displacement: torch.Tensor,
    num_labels: int,
    device: torch.device,
):
    native_shape = tuple(native_source.shape)
    native_source_tensor = torch.from_numpy(native_source).unsqueeze(0).unsqueeze(0).to(device)
    native_source_label_tensor = torch.from_numpy(native_source_label).unsqueeze(0).unsqueeze(0).to(device)
    native_disp = resize_displacement(displacement.float(), native_shape)

    image_transformer = SpatialTransformer(native_shape).to(device)
    label_transformer = SpatialTransformer(native_shape, mode="nearest").to(device)

    moved_native = image_transformer(native_source_tensor.float(), native_disp.float()).squeeze().detach().cpu().numpy()
    _, warped_label = warp_segmentation(
        native_source_label_tensor,
        native_disp.float(),
        num_labels,
        label_transformer,
    )
    jacobian_map = jacobian_determinant(native_disp.float()).squeeze(0).detach().cpu().numpy()
    return moved_native, warped_label.squeeze().detach().cpu().numpy(), jacobian_map


def save_comparison_figure(
    output_path: Path,
    pair_index: int,
    identity: dict[str, float],
    source: np.ndarray,
    target: np.ndarray,
    source_label: np.ndarray,
    target_label: np.ndarray,
    model_rows: list[dict[str, object]],
):
    fig, axes = plt.subplots(len(model_rows) + 1, 6, figsize=(22, 4.5 * (len(model_rows) + 1)), constrained_layout=True)

    src_slice = masked_center_axial(source, source_label)
    tgt_slice = masked_center_axial(target, target_label)
    src_seg_slice = masked_center_axial_label(source_label, source_label)
    tgt_seg_slice = masked_center_axial_label(target_label, target_label)
    identity_diff = np.abs(src_slice - tgt_slice)

    axes[0, 0].imshow(src_slice, cmap="gray", interpolation="lanczos")
    axes[0, 0].set_title("Source")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tgt_slice, cmap="gray", interpolation="lanczos")
    axes[0, 1].set_title("Target")
    axes[0, 1].axis("off")

    overlay_segmentation(axes[0, 2], src_slice, src_seg_slice, "Source Seg")
    overlay_segmentation(axes[0, 3], tgt_slice, tgt_seg_slice, "Target Seg")

    axes[0, 4].imshow(identity_diff, cmap="magma", interpolation="lanczos")
    axes[0, 4].set_title("|Source - Target|")
    axes[0, 4].axis("off")

    axes[0, 5].axis("off")
    axes[0, 5].text(
        0.02,
        0.98,
        f"Identity\nDice_fg: {identity['dice_mean_fg']:.4f}\nHD95: {identity['hd95_mean_fg']:.4f}",
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    for row_idx, row in enumerate(model_rows, start=1):
        moved_slice = masked_center_axial(row["moved"], row["warped_label"])
        warped_seg_slice = masked_center_axial_label(row["warped_label"], row["warped_label"])
        target_seg_slice = masked_center_axial_label(target_label, target_label)
        diff_slice = np.abs(moved_slice - tgt_slice)
        jacobian_map = row["jacobian"]
        jac_slice = np.rot90(jacobian_map[:, :, jacobian_map.shape[-1] // 2])

        axes[row_idx, 0].imshow(moved_slice, cmap="gray", interpolation="lanczos")
        axes[row_idx, 0].set_title(f"{row['name']} warped")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_ylabel(row["name"], rotation=90, fontsize=12)

        overlay_segmentation(axes[row_idx, 1], moved_slice, warped_seg_slice, "Warped Seg")
        overlay_segmentation(axes[row_idx, 2], moved_slice, target_seg_slice, "Target Seg on Warped")

        axes[row_idx, 3].imshow(diff_slice, cmap="magma", interpolation="lanczos")
        axes[row_idx, 3].set_title("|Warped - Target|")
        axes[row_idx, 3].axis("off")

        im = axes[row_idx, 4].imshow(jac_slice, cmap="coolwarm", interpolation="lanczos")
        axes[row_idx, 4].set_title("Jacobian")
        axes[row_idx, 4].axis("off")
        fig.colorbar(im, ax=axes[row_idx, 4], shrink=0.8)

        metrics = row["metrics"]
        axes[row_idx, 5].axis("off")
        axes[row_idx, 5].text(
            0.02,
            0.98,
            (
                f"Dice_fg: {metrics['dice_mean_fg']:.4f}\n"
                f"HD95: {metrics['hd95_mean_fg']:.4f}\n"
                f"SDlogJ: {metrics['sdlogj']:.4f}\n"
                f"NonPosJ: {metrics['jacobian_nonpositive_fraction']:.6f}\n"
                f"Time: {metrics['runtime_seconds']:.4f}s"
            ),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )

    fig.suptitle(f"Registration Comparison | pair_index={pair_index}", fontsize=16)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    hvit_config = read_yaml_file(args.config)
    pccr_config = PCCRConfig.from_yaml(args.pccr_config)

    val_loader = get_dataloader(
        data_path=args.val_data_path,
        input_dim=hvit_config["data_size"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_format=args.dataset_format,
        split="val",
        is_pair=True,
        shuffle=False,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_subjects=args.max_val_subjects,
        max_pairs=args.max_val_pairs,
    )
    dataset = val_loader.dataset
    num_labels = getattr(dataset, "num_labels", 0) or 0
    if num_labels:
        pccr_config.num_labels = max(pccr_config.num_labels, num_labels)

    device = resolve_device(args)

    hvit_model = build_hvit_model(args, hvit_config, max(num_labels, 36)).to(device).eval()
    pccr_short_model = build_pccr_model(args, args.pccr_short_checkpoint, pccr_config).to(device).eval()
    pccr_long_model = build_pccr_model(args, args.pccr_long_checkpoint, pccr_config).to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    with torch.no_grad():
        for pair_index in args.pair_indices:
            if pair_index < 0 or pair_index >= len(dataset):
                raise IndexError(f"pair_index {pair_index} is out of range for validation set of size {len(dataset)}")

            batch = dataset[pair_index]
            source, target, source_label, target_label = [tensor.unsqueeze(0).to(device) for tensor in batch]
            identity = identity_metrics(source_label, target_label, max(num_labels, 36))

            native_source, native_target, native_source_label, native_target_label = load_native_pair(dataset, pair_index)

            hvit_outputs, hvit_metrics = run_hvit(
                hvit_model,
                source,
                target,
                source_label,
                target_label,
                max(num_labels, 36),
                device,
                args.hvit_precision,
            )
            pccr_short_outputs, pccr_short_metrics = run_pccr(
                pccr_short_model,
                source,
                target,
                source_label,
                target_label,
                pccr_config.num_labels,
                device,
                args.pccr_precision,
            )
            pccr_long_outputs, pccr_long_metrics = run_pccr(
                pccr_long_model,
                source,
                target,
                source_label,
                target_label,
                pccr_config.num_labels,
                device,
                args.pccr_precision,
            )

            hvit_moved, hvit_warped_label, hvit_jacobian = render_native_outputs(
                native_source,
                native_target,
                native_source_label,
                native_target_label,
                hvit_outputs["phi_s2t"],
                max(num_labels, 36),
                device,
            )
            pccr_short_moved, pccr_short_warped_label, pccr_short_jacobian = render_native_outputs(
                native_source,
                native_target,
                native_source_label,
                native_target_label,
                pccr_short_outputs["phi_s2t"],
                pccr_config.num_labels,
                device,
            )
            pccr_long_moved, pccr_long_warped_label, pccr_long_jacobian = render_native_outputs(
                native_source,
                native_target,
                native_source_label,
                native_target_label,
                pccr_long_outputs["phi_s2t"],
                pccr_config.num_labels,
                device,
            )

            model_rows = [
                {
                    "name": args.hvit_name,
                    "moved": hvit_moved,
                    "warped_label": hvit_warped_label,
                    "jacobian": hvit_jacobian,
                    "metrics": hvit_metrics,
                },
                {
                    "name": args.pccr_short_name,
                    "moved": pccr_short_moved,
                    "warped_label": pccr_short_warped_label,
                    "jacobian": pccr_short_jacobian,
                    "metrics": pccr_short_metrics,
                },
                {
                    "name": args.pccr_long_name,
                    "moved": pccr_long_moved,
                    "warped_label": pccr_long_warped_label,
                    "jacobian": pccr_long_jacobian,
                    "metrics": pccr_long_metrics,
                },
            ]

            out_path = output_dir / f"pair_{pair_index:04d}.png"
            save_comparison_figure(
                out_path,
                pair_index,
                identity,
                native_source,
                native_target,
                native_source_label,
                native_target_label,
                model_rows,
            )

            manifest.append(
                {
                    "pair_index": pair_index,
                    "image_path": str(out_path),
                    "identity": identity,
                    "models": {
                        args.hvit_name: hvit_metrics,
                        args.pccr_short_name: pccr_short_metrics,
                        args.pccr_long_name: pccr_long_metrics,
                    },
                }
            )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"visualization_dir: {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
