#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
import torch.nn.functional as F

from src.pccr.config import PCCRConfig
from src.pccr.data import create_real_pair_dataloaders
from src.pccr.eval_utils import aggregate_metrics, pair_metrics, prefix_metrics, save_metrics_report
from src.pccr.trainer import LiTPCCR


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate whether local window search can recover missed alignment.")
    parser.add_argument("--config", type=str, default="src/pccr/configs/pairwise_oasis.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--val_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs"], default="oasis_fs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_num_steps", type=int, default=200)
    parser.add_argument("--max_train_subjects", type=int, default=0)
    parser.add_argument("--max_val_subjects", type=int, default=0)
    parser.add_argument("--max_val_pairs", type=int, default=50)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--search_radius", type=int, default=1)
    parser.add_argument("--patch_window", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="pccr_local_window_search")
    return parser.parse_args()


def resolve_device(args) -> torch.device:
    if args.accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if args.accelerator == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_offset_tensor(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    offsets = [
        (dz, dy, dx)
        for dz in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    return torch.tensor(offsets, device=device, dtype=dtype)


def local_search_residual(
    moved_source: torch.Tensor,
    target: torch.Tensor,
    transformer,
    radius: int,
    patch_window: int,
) -> tuple[torch.Tensor, float]:
    if patch_window % 2 == 0:
        raise ValueError("--patch_window must be odd.")

    start = time.perf_counter()
    batch_size, _, depth, height, width = moved_source.shape
    offsets = build_offset_tensor(radius, moved_source.device, moved_source.dtype)
    padding = patch_window // 2

    def local_lncc(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        cand_mean = F.avg_pool3d(candidate, kernel_size=patch_window, stride=1, padding=padding)
        ref_mean = F.avg_pool3d(reference, kernel_size=patch_window, stride=1, padding=padding)
        cand_var = F.avg_pool3d(candidate * candidate, kernel_size=patch_window, stride=1, padding=padding) - cand_mean.square()
        ref_var = F.avg_pool3d(reference * reference, kernel_size=patch_window, stride=1, padding=padding) - ref_mean.square()
        cross = F.avg_pool3d(candidate * reference, kernel_size=patch_window, stride=1, padding=padding) - cand_mean * ref_mean
        score = cross.square() / (cand_var * ref_var + 1e-5)
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    scores = []
    for offset in offsets:
        flow = moved_source.new_zeros(batch_size, 3, depth, height, width)
        flow[:, 0] = offset[0]
        flow[:, 1] = offset[1]
        flow[:, 2] = offset[2]
        shifted = transformer(moved_source, flow)
        score = local_lncc(shifted, target)
        scores.append(score)

    score_volume = torch.cat(scores, dim=1)
    best_index = score_volume.argmax(dim=1)
    residual = offsets[best_index.reshape(-1)].view(batch_size, depth, height, width, 3)
    residual = residual.permute(0, 4, 1, 2, 3).contiguous()
    elapsed = time.perf_counter() - start
    return residual, elapsed


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    config = PCCRConfig.from_yaml(args.config)
    config.phase = "real"

    _, val_loader = create_real_pair_dataloaders(args, config)
    dataset_num_labels = getattr(val_loader.dataset, "num_labels", 0) or 0
    if dataset_num_labels:
        config.num_labels = max(config.num_labels, dataset_num_labels)

    model = LiTPCCR.load_from_checkpoint(
        args.checkpoint_path,
        args=args,
        config=config,
        experiment_logger=None,
        strict=False,
    )
    model.eval()

    device = resolve_device(args)
    model = model.to(device)
    use_cuda = device.type == "cuda"
    autocast_enabled = use_cuda and args.precision.startswith("bf16")

    records = []
    total_pairs = len(val_loader)
    print(
        f"Running local window search on {total_pairs} validation pairs "
        f"(radius={args.search_radius}, patch_window={args.patch_window})"
    )
    with torch.no_grad():
        for pair_idx, batch in enumerate(val_loader):
            source, target, source_label, target_label = [tensor.to(device) for tensor in batch]

            if use_cuda:
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs = model(source, target)
            if use_cuda:
                torch.cuda.synchronize()
            infer_elapsed = time.perf_counter() - infer_start

            base_metrics = pair_metrics(
                outputs=outputs,
                source_label=source_label,
                target_label=target_label,
                num_labels=config.num_labels,
                transformer=model.model.decoder.final_transformer,
                inference_seconds=infer_elapsed,
            )

            residual_field, search_elapsed = local_search_residual(
                moved_source=outputs["moved_source"].float(),
                target=target.float(),
                transformer=model.model.decoder.final_transformer,
                radius=args.search_radius,
                patch_window=args.patch_window,
            )
            refined_displacement = outputs["phi_s2t"].float() + residual_field
            refined_outputs = {"phi_s2t": refined_displacement}
            searched_metrics = pair_metrics(
                outputs=refined_outputs,
                source_label=source_label,
                target_label=target_label,
                num_labels=config.num_labels,
                transformer=model.model.decoder.final_transformer,
                inference_seconds=infer_elapsed + search_elapsed,
            )

            record = {"pair_index": pair_idx}
            record.update(prefix_metrics(base_metrics, "base_"))
            record.update(prefix_metrics(searched_metrics, "searched_"))
            record["improvement_dice_mean_fg"] = (
                searched_metrics["dice_mean_fg"] - base_metrics["dice_mean_fg"]
            )
            record["improvement_hd95_mean_fg"] = (
                base_metrics["hd95_mean_fg"] - searched_metrics["hd95_mean_fg"]
            )
            record["improvement_sdlogj"] = base_metrics["sdlogj"] - searched_metrics["sdlogj"]
            records.append(record)

            if (pair_idx + 1) % max(args.progress_every, 1) == 0 or (pair_idx + 1) == total_pairs:
                print(
                    f"[{pair_idx + 1}/{total_pairs}] "
                    f"base_dice_fg={base_metrics['dice_mean_fg']:.4f} "
                    f"searched_dice_fg={searched_metrics['dice_mean_fg']:.4f} "
                    f"delta={record['improvement_dice_mean_fg']:+.4f}"
                )

    summary = aggregate_metrics(records)
    output_dir = args.output_dir or f"logs/pccr/{args.experiment_name}"
    summary_path, pairs_path = save_metrics_report(output_dir, summary, records)
    print("Local-search summary:")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")
    print(f"summary_json: {summary_path}")
    print(f"per_pair_csv: {pairs_path}")


if __name__ == "__main__":
    main()
