#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from argparse import Namespace
from dataclasses import fields
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from src.data.datasets import get_dataloader, infer_dataset_format, load_oasis_l2r_metadata
from src.pccr.config import PCCRConfig
from src.pccr.data import create_real_pair_dataloaders
from src.pccr.eval_utils import (
    aggregate_metrics,
    dice_statistics,
    hd95_statistics,
    identity_metrics,
    jacobian_statistics,
    measure_inference_time,
    per_label_dice_statistics,
    warp_segmentation,
)
from src.pccr.trainer import LiTPCCR
from src.trainer import LiTHViT
from src.utils import get_one_hot, read_yaml_file


DEFAULT_CONFIGS = {
    "hvit": "config/hvit_fullres.yaml",
    "pccr": "config/pairwise_oasis_fullres.yaml",
}

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
    "32-true": torch.float32,
}


def load_checkpoint_hyperparameters(checkpoint_path: str | Path) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`",
            category=FutureWarning,
        )
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    return checkpoint.get("hyper_parameters", {})


def load_hvit_config_from_checkpoint(checkpoint_path: str | Path) -> tuple[dict, str] | tuple[None, None]:
    hyperparameters = load_checkpoint_hyperparameters(checkpoint_path)
    config = hyperparameters.get("config")
    if isinstance(config, dict):
        return dict(config), f"{checkpoint_path}::hyper_parameters.config"
    return None, None


def load_pccr_config_from_checkpoint(checkpoint_path: str | Path) -> tuple[PCCRConfig, str] | tuple[None, None]:
    hyperparameters = load_checkpoint_hyperparameters(checkpoint_path)
    config = hyperparameters.get("config")
    if not isinstance(config, dict):
        return None, None

    payload = dict(config)
    valid_fields = {field.name for field in fields(PCCRConfig)}
    payload = {key: value for key, value in payload.items() if key in valid_fields}

    # Old checkpoints do not store this flag; preserve their trained resolution rather than
    # auto-expanding to the native Learn2Reg shape during evaluation.
    payload.setdefault("align_data_size_to_native_shape", False)
    return PCCRConfig(**payload), f"{checkpoint_path}::hyper_parameters.config"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an H-ViT or PCCR baseline on Learn2Reg OASIS validation pairs.")
    parser.add_argument("--model", choices=["hvit", "pccr"], required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, default=None)
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
    parser.add_argument("--max_val_pairs", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument("--skip_hd95", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mse_weights", type=float, default=1.0)
    parser.add_argument("--dice_weights", type=float, default=1.0)
    parser.add_argument("--grad_weights", type=float, default=0.02)
    parser.add_argument("--tgt2src_reg", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--hvit_light", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=None)
    parser.add_argument("--num_labels", type=int, default=None)
    return parser.parse_args()


def resolve_device(args) -> torch.device:
    if args.accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if args.accelerator == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_hvit_data_path(cli_value: str | None, config: dict) -> str:
    if cli_value:
        return cli_value
    if config.get("dataset_format") == "oasis_l2r" and config.get("oasis_l2r_data_root"):
        return str(Path(config["oasis_l2r_data_root"]).expanduser())
    raise ValueError("H-ViT evaluation requires --val_data_path or config.oasis_l2r_data_root.")


def resolve_hvit_dataset_format(args, config: dict, data_path: str) -> str:
    if args.dataset_format != "auto":
        return args.dataset_format
    configured = config.get("dataset_format", "auto")
    if configured != "auto":
        return configured
    return infer_dataset_format(data_path)


def sync_hvit_config_with_dataset(config: dict, data_path: str, dataset_format: str) -> None:
    if dataset_format != "oasis_l2r":
        config.setdefault("eval_label_ids", [])
        return

    metadata = load_oasis_l2r_metadata(Path(data_path))
    native_shape = list(metadata.native_shape)
    current_shape = list(config.get("data_size", []))
    should_align = bool(config.get("align_data_size_to_native_shape", False))
    if current_shape and sorted(current_shape) == sorted(native_shape) and current_shape != native_shape:
        should_align = True
    if should_align and current_shape != native_shape:
        print(
            "[eval_baseline] Aligning H-ViT config.data_size "
            f"from {config.get('data_size')} to dataset native shape {native_shape}."
        )
        config["data_size"] = native_shape
    if not config.get("eval_label_ids"):
        config["eval_label_ids"] = list(metadata.eval_label_ids)
    config["num_labels"] = max(int(config.get("num_labels", 0) or 0), max(metadata.eval_label_ids) + 1)


def resolve_hvit_precision(args, config: dict, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if args.precision is not None:
        return args.precision
    if "precision" in config:
        return str(config["precision"])
    return "bf16" if config.get("use_amp", False) else "fp32"


def _jsonify_label_scores(scores: dict[int, float]) -> dict[str, float]:
    return {str(label_id): float(value) for label_id, value in scores.items()}


def _label_score_means(records: list[dict[str, object]], key: str, label_ids: list[int]) -> dict[str, float]:
    means = {}
    for label_id in label_ids:
        values = [float(record[key][str(label_id)]) for record in records]
        means[str(label_id)] = float(sum(values) / len(values))
    return means


def _build_hvit_eval_args(args, config: dict, num_labels: int, label_ids: list[int], precision: str) -> Namespace:
    return Namespace(
        lr=args.lr,
        mse_weights=args.mse_weights,
        dice_weights=args.dice_weights,
        grad_weights=args.grad_weights,
        tgt2src_reg=args.tgt2src_reg,
        hvit_light=bool(config.get("hvit_light", True) if args.hvit_light is None else args.hvit_light),
        precision=precision,
        num_labels=num_labels,
        eval_label_ids=list(label_ids),
        save_model_every_n_epochs=int(config.get("save_model_every_n_epochs", 1)),
    )


def load_hvit_runtime(args, device: torch.device):
    if args.config is not None:
        config_path = args.config
        config = read_yaml_file(config_path)
        if config is None:
            raise ValueError(f"Failed to read H-ViT config: {config_path}")
    else:
        config, config_path = load_hvit_config_from_checkpoint(args.checkpoint_path)
        if config is None:
            config_path = DEFAULT_CONFIGS["hvit"]
            config = read_yaml_file(config_path)
            if config is None:
                raise ValueError(f"Failed to read H-ViT config: {config_path}")

    val_data_path = resolve_hvit_data_path(args.val_data_path, config)
    dataset_format = resolve_hvit_dataset_format(args, config, val_data_path)
    sync_hvit_config_with_dataset(config, val_data_path, dataset_format)
    batch_size = args.batch_size or int(config.get("batch_size", 1))

    val_loader = get_dataloader(
        data_path=val_data_path,
        input_dim=config["data_size"],
        batch_size=batch_size,
        num_workers=args.num_workers,
        dataset_format=dataset_format,
        split="val",
        is_pair=True,
        shuffle=False,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_subjects=args.max_val_subjects,
        max_pairs=args.max_val_pairs,
    )

    num_labels = max(
        int(config.get("num_labels", 36)),
        int(args.num_labels or 0),
        int(getattr(val_loader.dataset, "num_labels", 0) or 0),
    )
    label_ids = [
        int(label_id)
        for label_id in (
            config.get("eval_label_ids")
            or getattr(val_loader.dataset, "eval_label_ids", [])
        )
        if 0 <= int(label_id) < num_labels
    ]
    precision = resolve_hvit_precision(args, config, device)
    hvit_args = _build_hvit_eval_args(args, config, num_labels, label_ids, precision)
    model = LiTHViT.load_from_checkpoint(
        args.checkpoint_path,
        args=hvit_args,
        experiment_logger=None,
    )
    model.eval().to(device)

    def run_model(source: torch.Tensor, target: torch.Tensor):
        return model.hvit(source, target)

    return {
        "config_path": config_path,
        "dataset_path": val_data_path,
        "dataset_format": dataset_format,
        "val_loader": val_loader,
        "num_labels": num_labels,
        "label_ids": label_ids,
        "model": model,
        "run_model": run_model,
        "transformer": model.hvit.spatial_trans,
        "precision": precision,
    }


def resolve_pccr_precision(args, config: PCCRConfig, device: torch.device) -> str:
    if device.type != "cuda":
        return "32-true"
    if args.precision is not None:
        return args.precision
    return "bf16-mixed" if config.use_amp else "32-true"


def load_pccr_runtime(args, device: torch.device):
    if args.config is not None:
        config_path = args.config
        config = PCCRConfig.from_yaml(config_path)
    else:
        config, config_path = load_pccr_config_from_checkpoint(args.checkpoint_path)
        if config is None:
            config_path = DEFAULT_CONFIGS["pccr"]
            config = PCCRConfig.from_yaml(config_path)
    config.phase = "real"
    if args.batch_size is None:
        args.batch_size = config.batch_size

    pccr_args = Namespace(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        train_num_steps=args.train_num_steps,
        max_train_subjects=args.max_train_subjects,
        max_val_subjects=args.max_val_subjects,
        max_val_pairs=args.max_val_pairs,
        phase="real",
        lr=args.lr,
        max_epochs=1,
    )

    _, val_loader = create_real_pair_dataloaders(pccr_args, config)
    num_labels = max(
        int(config.num_labels),
        int(args.num_labels or 0),
        int(getattr(val_loader.dataset, "num_labels", 0) or 0),
    )
    config.num_labels = num_labels
    label_ids = [int(label_id) for label_id in config.eval_label_ids if 0 <= int(label_id) < num_labels]
    precision = resolve_pccr_precision(args, config, device)
    model = LiTPCCR.load_from_checkpoint(
        args.checkpoint_path,
        args=pccr_args,
        config=config,
        experiment_logger=None,
        strict=False,
    )
    model.eval().to(device)

    def run_model(source: torch.Tensor, target: torch.Tensor):
        return model(source, target)

    dataset_path = args.val_data_path or args.train_data_path or getattr(config, "oasis_l2r_data_root", "")
    dataset_format = args.dataset_format
    if dataset_format == "auto" and getattr(config, "dataset_variant", "default") == "oasis_l2r":
        dataset_format = "oasis_l2r"
    return {
        "config_path": config_path,
        "dataset_path": dataset_path,
        "dataset_format": dataset_format,
        "val_loader": val_loader,
        "num_labels": num_labels,
        "label_ids": label_ids,
        "model": model,
        "run_model": run_model,
        "transformer": model.model.decoder.final_transformer,
        "precision": precision,
    }


def to_registration_outputs(model_name: str, outputs):
    if model_name == "hvit":
        moved_source, displacement = outputs
        return {"moved_source": moved_source, "phi_s2t": displacement}
    return outputs


def evaluate_runtime(args, runtime: dict, device: torch.device) -> dict[str, object]:
    val_loader = runtime["val_loader"]
    num_labels = runtime["num_labels"]
    label_ids = runtime["label_ids"]
    transformer = runtime["transformer"]
    precision = runtime["precision"]
    dtype_ = DTYPE_MAP.get(precision, torch.float32)
    autocast_enabled = device.type == "cuda" and dtype_ in {torch.bfloat16, torch.float16}
    use_cuda = device.type == "cuda"

    flat_records: list[dict[str, float]] = []
    detailed_records: list[dict[str, object]] = []

    total_pairs = len(val_loader)
    print(
        f"Evaluating {args.model} on {total_pairs} validation pairs "
        f"using label_ids={label_ids} device={device} precision={precision}"
    )
    with torch.no_grad():
        for pair_idx, batch in enumerate(val_loader):
            source, target, source_label, target_label = [tensor.to(device) for tensor in batch]

            identity_onehot = get_one_hot(source_label.long(), num_labels).float()
            before = identity_metrics(
                source_label=source_label,
                target_label=target_label,
                num_labels=num_labels,
                label_ids=label_ids,
                inference_seconds=0.0,
                include_hd95=not args.skip_hd95,
            )
            before_per_label = per_label_dice_statistics(
                identity_onehot,
                target_label,
                num_labels=num_labels,
                label_ids=label_ids,
            )

            with torch.autocast(device_type=device.type, dtype=dtype_, enabled=autocast_enabled):
                raw_outputs, elapsed = measure_inference_time(runtime["run_model"], source, target, use_cuda)
            outputs = to_registration_outputs(args.model, raw_outputs)
            warped_onehot, warped_label = warp_segmentation(
                source_label,
                outputs["phi_s2t"],
                num_labels=num_labels,
                transformer=transformer,
            )
            after = {}
            after.update(
                dice_statistics(
                    warped_onehot,
                    target_label,
                    num_labels=num_labels,
                    label_ids=label_ids,
                )
            )
            if args.skip_hd95:
                after.update({"hd95_mean_fg": float("nan"), "hd95_median_fg": float("nan")})
            else:
                after.update(
                    hd95_statistics(
                        warped_label,
                        target_label,
                        num_labels=num_labels,
                        label_ids=label_ids,
                    )
                )
            after.update(jacobian_statistics(outputs["phi_s2t"]))
            after["runtime_seconds"] = float(elapsed)
            after_per_label = per_label_dice_statistics(
                warped_onehot,
                target_label,
                num_labels=num_labels,
                label_ids=label_ids,
            )

            before["jacobian_nonpositive_percent"] = before["jacobian_nonpositive_fraction"] * 100.0
            after["jacobian_nonpositive_percent"] = after["jacobian_nonpositive_fraction"] * 100.0

            flat_record = {
                "identity_dice_mean_all": before["dice_mean_all"],
                "identity_dice_mean_fg": before["dice_mean_fg"],
                "identity_hd95_mean_fg": before["hd95_mean_fg"],
                "identity_hd95_median_fg": before["hd95_median_fg"],
                "registered_dice_mean_all": after["dice_mean_all"],
                "registered_dice_mean_fg": after["dice_mean_fg"],
                "registered_hd95_mean_fg": after["hd95_mean_fg"],
                "registered_hd95_median_fg": after["hd95_median_fg"],
                "registered_sdlogj": after["sdlogj"],
                "registered_jacobian_nonpositive_fraction": after["jacobian_nonpositive_fraction"],
                "registered_jacobian_nonpositive_percent": after["jacobian_nonpositive_percent"],
                "runtime_seconds": after["runtime_seconds"],
                "improvement_dice_mean_fg": after["dice_mean_fg"] - before["dice_mean_fg"],
            }
            flat_records.append(flat_record)
            detailed_records.append(
                {
                    "pair_index": pair_idx,
                    "identity": before,
                    "registered": after,
                    "identity_dice_per_structure": _jsonify_label_scores(before_per_label),
                    "registered_dice_per_structure": _jsonify_label_scores(after_per_label),
                }
            )

            if (pair_idx + 1) % max(args.progress_every, 1) == 0 or (pair_idx + 1) == total_pairs:
                print(
                    f"[{pair_idx + 1}/{total_pairs}] "
                    f"id_dice_fg={before['dice_mean_fg']:.4f} "
                    f"reg_dice_fg={after['dice_mean_fg']:.4f} "
                    f"reg_hd95={after['hd95_mean_fg']:.4f} "
                    f"reg_sdlogj={after['sdlogj']:.4f}"
                )

    summary = aggregate_metrics(flat_records)
    summary["identity_dice_per_structure_mean"] = _label_score_means(
        detailed_records,
        "identity_dice_per_structure",
        label_ids,
    )
    summary["registered_dice_per_structure_mean"] = _label_score_means(
        detailed_records,
        "registered_dice_per_structure",
        label_ids,
    )
    summary["label_ids"] = list(label_ids)

    return {
        "summary": summary,
        "per_pair": detailed_records,
    }


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args)

    if args.model == "hvit":
        runtime = load_hvit_runtime(args, device)
    else:
        runtime = load_pccr_runtime(args, device)

    report = evaluate_runtime(args, runtime, device)
    report.update(
        {
            "model": args.model,
            "config_path": runtime["config_path"],
            "checkpoint_path": args.checkpoint_path,
            "dataset_path": runtime["dataset_path"],
            "dataset_format": runtime["dataset_format"],
            "num_pairs": len(runtime["val_loader"].dataset),
            "num_labels": runtime["num_labels"],
        }
    )

    output_json = args.output_json
    if output_json is None:
        experiment_name = args.experiment_name or f"{args.model}_baseline_eval"
        output_json = str(Path("logs") / experiment_name / "baseline_eval.json")
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print(f"saved_json: {output_path}")
    print(f"registered_dice_mean_fg_mean: {report['summary']['registered_dice_mean_fg_mean']}")
    print(f"registered_hd95_mean_fg_mean: {report['summary']['registered_hd95_mean_fg_mean']}")
    print(f"registered_sdlogj_mean: {report['summary']['registered_sdlogj_mean']}")
    print(
        "registered_jacobian_nonpositive_percent_mean: "
        f"{report['summary']['registered_jacobian_nonpositive_percent_mean']}"
    )


if __name__ == "__main__":
    main()
