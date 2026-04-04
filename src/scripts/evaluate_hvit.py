#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.data.datasets import get_dataloader
from src.pccr.eval_utils import (
    aggregate_metrics,
    identity_metrics,
    measure_inference_time,
    pair_metrics,
    prefix_metrics,
    save_metrics_report,
)
from src.trainer import LiTHViT
from src.utils import read_yaml_file


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an H-ViT checkpoint with identity-vs-registered metrics.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--val_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs"], default="oasis_fs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_num_steps", type=int, default=200)
    parser.add_argument("--max_train_subjects", type=int, default=0)
    parser.add_argument("--max_val_subjects", type=int, default=0)
    parser.add_argument("--max_val_pairs", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="hvit_eval")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mse_weights", type=float, default=1.0)
    parser.add_argument("--dice_weights", type=float, default=1.0)
    parser.add_argument("--grad_weights", type=float, default=0.02)
    parser.add_argument("--tgt2src_reg", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--hvit_light", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--num_labels", type=int, default=36)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--skip_hd95", action="store_true")
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
    config = read_yaml_file(args.config)

    val_loader = get_dataloader(
        data_path=args.val_data_path,
        input_dim=config["data_size"],
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

    dataset_num_labels = getattr(val_loader.dataset, "num_labels", 0) or 0
    if dataset_num_labels:
        args.num_labels = max(args.num_labels, dataset_num_labels)

    model = LiTHViT.load_from_checkpoint(
        args.checkpoint_path,
        args=args,
        experiment_logger=None,
    )
    model.eval()

    device = resolve_device(args)
    model = model.to(device)
    use_cuda = device.type == "cuda"
    dtype_ = DTYPE_MAP.get(args.precision, torch.float32)
    autocast_enabled = use_cuda and dtype_ in {torch.bfloat16, torch.float16}

    records = []
    total_pairs = len(val_loader)
    print(f"Evaluating {total_pairs} validation pairs on device={device} (skip_hd95={args.skip_hd95})")
    with torch.no_grad():
        for pair_idx, batch in enumerate(val_loader):
            source, target, source_label, target_label = [tensor.to(device) for tensor in batch]
            before = identity_metrics(
                source_label=source_label,
                target_label=target_label,
                num_labels=args.num_labels,
                inference_seconds=0.0,
                include_hd95=not args.skip_hd95,
            )

            with torch.autocast(device_type=device.type, dtype=dtype_, enabled=autocast_enabled):
                outputs_tuple, elapsed = measure_inference_time(
                    lambda src, tgt: model.hvit(src.to(dtype=dtype_), tgt.to(dtype=dtype_)),
                    source,
                    target,
                    use_cuda,
                )
            moved_source, displacement = outputs_tuple
            after = pair_metrics(
                outputs={"moved_source": moved_source, "phi_s2t": displacement},
                source_label=source_label,
                target_label=target_label,
                num_labels=args.num_labels,
                transformer=model.hvit.spatial_trans,
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
            if (pair_idx + 1) % max(args.progress_every, 1) == 0 or (pair_idx + 1) == total_pairs:
                print(
                    f"[{pair_idx + 1}/{total_pairs}] "
                    f"id_dice_fg={before['dice_mean_fg']:.4f} "
                    f"reg_dice_fg={after['dice_mean_fg']:.4f} "
                    f"reg_time={after['runtime_seconds']:.4f}s"
                )

    summary = aggregate_metrics(records)
    output_dir = args.output_dir or f"logs/{args.experiment_name}_eval"
    summary_path, pairs_path = save_metrics_report(output_dir, summary, records)

    print("Evaluation summary:")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")
    print(f"summary_json: {summary_path}")
    print(f"per_pair_csv: {pairs_path}")


if __name__ == "__main__":
    main()
