#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

from src.pccr.config import PCCRConfig
from src.pccr.data import create_real_pair_dataloaders
from src.pccr.eval_utils import (
    aggregate_metrics,
    identity_metrics,
    measure_inference_time,
    pair_metrics,
    prefix_metrics,
    save_metrics_report,
)
from src.pccr.trainer import LiTPCCR


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PCCR checkpoint.")
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
    parser.add_argument("--max_val_pairs", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="pccr_eval")
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--skip_hd95", action="store_true")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()


def resolve_device(args) -> torch.device:
    if args.accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if args.accelerator == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    config = PCCRConfig.from_yaml(args.config)
    config.phase = "real"

    _, val_loader = create_real_pair_dataloaders(args, config)
    dataset_num_labels = getattr(val_loader.dataset, "num_labels", 0) or 0
    if dataset_num_labels:
        config.num_labels = max(config.num_labels, dataset_num_labels)

    shard_indices = list(range(len(val_loader.dataset)))[args.shard_index :: args.num_shards]
    if not shard_indices:
        raise ValueError(
            f"Shard {args.shard_index}/{args.num_shards} received no pairs from dataset of size {len(val_loader.dataset)}."
        )
    shard_dataset = Subset(val_loader.dataset, shard_indices)
    val_loader = DataLoader(
        shard_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

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

    records = []
    autocast_enabled = use_cuda and args.precision.startswith("bf16")
    total_pairs = len(val_loader)
    print(
        f"Evaluating shard {args.shard_index + 1}/{args.num_shards}: "
        f"{total_pairs} validation pairs on device={device} (skip_hd95={args.skip_hd95})"
    )
    with torch.no_grad():
        for local_idx, batch in enumerate(val_loader):
            pair_idx = shard_indices[local_idx]
            source, target, source_label, target_label = [tensor.to(device) for tensor in batch]
            before = identity_metrics(
                source_label=source_label,
                target_label=target_label,
                num_labels=config.num_labels,
                inference_seconds=0.0,
                include_hd95=not args.skip_hd95,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs, elapsed = measure_inference_time(model, source, target, use_cuda)

            after = pair_metrics(
                outputs=outputs,
                source_label=source_label,
                target_label=target_label,
                num_labels=config.num_labels,
                transformer=model.model.decoder.final_transformer,
                inference_seconds=elapsed,
                include_hd95=not args.skip_hd95,
            )
            record = {"pair_index": pair_idx}
            record.update(prefix_metrics(before, "identity_"))
            record.update(prefix_metrics(after, "registered_"))
            record.update(
                {
                    "improvement_dice_mean_all": after["dice_mean_all"] - before["dice_mean_all"],
                    "improvement_dice_mean_fg": after["dice_mean_fg"] - before["dice_mean_fg"],
                    "improvement_hd95_mean_fg": before["hd95_mean_fg"] - after["hd95_mean_fg"],
                    "improvement_hd95_median_fg": before["hd95_median_fg"] - after["hd95_median_fg"],
                }
            )
            records.append(record)
            if (local_idx + 1) % max(args.progress_every, 1) == 0 or (local_idx + 1) == total_pairs:
                print(
                    f"[{local_idx + 1}/{total_pairs}] "
                    f"global_pair={pair_idx} "
                    f"id_dice_fg={before['dice_mean_fg']:.4f} "
                    f"reg_dice_fg={after['dice_mean_fg']:.4f} "
                    f"reg_time={after['runtime_seconds']:.4f}s"
                )

    summary = aggregate_metrics(records)
    output_dir = args.output_dir or f"logs/pccr/{args.experiment_name}_eval"
    summary_path, pairs_path = save_metrics_report(output_dir, summary, records)

    print("Evaluation summary:")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")
    print(f"summary_json: {summary_path}")
    print(f"per_pair_csv: {pairs_path}")


if __name__ == "__main__":
    main()
