import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import argparse
import importlib

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.pccr.data import (
    create_overfit_pair_dataloaders,
    create_real_pair_dataloaders,
    create_synthetic_dataloader,
    create_synthetic_pair_dataloaders,
)
from src.pccr.periodic_eval import IterativeEvalCallback
from src.pccr_v6.config import PCCRV6Config
from src.pccr_v6.trainer import LiTPCCRV6


def load_aim_logger_class():
    for module_name, attr_name in [
        ("aim.pytorch_lightning", "AimLogger"),
        ("aimstack.experiment_tracker.pytorch_lightning", "Logger"),
    ]:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError):
            continue
    raise ImportError("Aim logger is not available in the current environment.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the PCCR v6 research codebase.")
    parser.add_argument("--config", type=str, default="src/pccr_v6/configs/pairwise_oasis_v6a.yaml")
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--phase", choices=["real", "synthetic"], default="real")
    parser.add_argument("--data_source", choices=["auto", "real", "synthetic"], default="auto")
    parser.add_argument("--train_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--val_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs", "oasis_l2r"], default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_num_steps", type=int, default=200)
    parser.add_argument("--max_train_subjects", type=int, default=0)
    parser.add_argument("--max_val_subjects", type=int, default=0)
    parser.add_argument("--max_val_pairs", type=int, default=0)
    parser.add_argument("--logger_backend", choices=["aim", "csv", "none"], default="aim")
    parser.add_argument("--aim_repo", type=str, default="/u/almik/others/hvit/aim")
    parser.add_argument("--experiment_name", type=str, default="pccr_v6_oasis")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--iter_eval_every_n_epochs", type=int, default=0)
    parser.add_argument("--iter_eval_num_pairs", type=int, default=100)
    parser.add_argument("--iter_eval_skip_hd95", action="store_true")
    parser.add_argument("--iter_viz_every_n_epochs", type=int, default=0)
    parser.add_argument("--iter_viz_pair_index", type=int, default=0)
    parser.add_argument("--overfit_num_pairs", type=int, default=0)
    parser.add_argument("--overfit_split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--freeze_mode",
        choices=[
            "full",
            "final_refinement",
            "coarse_decoder",
            "matcher",
            "decoder_and_refinement",
        ],
        default="full",
    )
    parser.add_argument("--oracle_correspondence", choices=["none", "synthetic_gt"], default="none")
    parser.add_argument("--oracle_handoff", action="store_true")
    return parser.parse_args()


def build_logger(args):
    if args.logger_backend == "none":
        return None
    if args.logger_backend == "csv":
        return CSVLogger(save_dir="logs", name=args.experiment_name)
    AimLogger = load_aim_logger_class()
    return AimLogger(repo=args.aim_repo, experiment=args.experiment_name)


def resolve_devices(args):
    if args.accelerator == "cpu":
        return "cpu", 1
    if args.accelerator == "gpu":
        return "gpu", args.num_gpus
    if torch.cuda.is_available():
        return "gpu", args.num_gpus
    return "cpu", 1


def resolve_strategy(accelerator: str, devices: int) -> str:
    if accelerator == "gpu" and devices > 1:
        return "ddp_find_unused_parameters_true"
    return "auto"


def apply_config_overrides(config: PCCRV6Config, override_items: list[str]) -> PCCRV6Config:
    if not override_items:
        return config

    overrides = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(f"Invalid --config_override '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw_value)
    return config.apply_overrides(overrides)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    config = apply_config_overrides(PCCRV6Config.from_yaml(args.config), args.config_override)
    config.phase = args.phase
    if args.oracle_correspondence != "none":
        config.diagnostic_oracle_correspondence = True
    if args.oracle_handoff:
        config.diagnostic_oracle_handoff = True
    experiment_logger = build_logger(args)
    checkpoint_dir = Path("checkpoints") / "pccr_v6" / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resolved_data_source = args.data_source
    if resolved_data_source == "auto":
        resolved_data_source = "synthetic" if args.phase == "synthetic" else "real"

    if args.phase == "synthetic":
        train_loader = create_synthetic_dataloader(args, config)
        val_loader = None
    else:
        if resolved_data_source == "synthetic":
            train_loader, val_loader = create_synthetic_pair_dataloaders(args, config)
        elif args.overfit_num_pairs > 0:
            train_loader, val_loader = create_overfit_pair_dataloaders(args, config)
        else:
            train_loader, val_loader = create_real_pair_dataloaders(args, config)
        dataset_num_labels = max(
            getattr(train_loader.dataset, "num_labels", 0) or 0,
            getattr(val_loader.dataset, "num_labels", 0) or 0,
        )
        if dataset_num_labels:
            config.num_labels = max(config.num_labels, dataset_num_labels)

    accelerator, devices = resolve_devices(args)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch{epoch:03d}-val{val_avg_loss:.4f}" if val_loader is not None else "epoch{epoch:03d}",
        auto_insert_metric_name=False,
        monitor="val_avg_loss" if val_loader is not None else None,
        mode="min",
        save_top_k=2 if val_loader is not None else -1,
        save_last=True,
        every_n_epochs=1,
    )
    callbacks = [checkpoint_callback]
    if args.phase == "real" and args.iter_eval_every_n_epochs > 0:
        callbacks.append(
            IterativeEvalCallback(
                dataset=val_loader.dataset,
                num_labels=config.num_labels,
                label_ids=config.eval_label_ids,
                every_n_epochs=args.iter_eval_every_n_epochs,
                num_pairs=args.iter_eval_num_pairs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                precision=args.precision if accelerator == "gpu" else "32-true",
                output_dir=Path("logs") / "pccr_v6" / args.experiment_name / "iter_eval",
                include_hd95=not args.iter_eval_skip_hd95,
                visualization_every_n_epochs=args.iter_viz_every_n_epochs,
                visualization_pair_index=args.iter_viz_pair_index,
                visualization_dir=Path("logs") / "pccr_v6" / args.experiment_name / "iter_viz",
            )
        )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.max_epochs,
        logger=experiment_logger,
        callbacks=callbacks,
        default_root_dir=str(Path("logs") / "pccr_v6" / args.experiment_name),
        precision=args.precision if accelerator == "gpu" else "32-true",
        strategy=resolve_strategy(accelerator, devices),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
    )

    if args.mode == "train":
        if args.resume_from_checkpoint:
            model = LiTPCCRV6(args=args, config=config, experiment_logger=experiment_logger)
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=args.resume_from_checkpoint,
            )
        else:
            if args.checkpoint_path:
                model = LiTPCCRV6.load_from_checkpoint(
                    args.checkpoint_path,
                    args=args,
                    config=config,
                    experiment_logger=experiment_logger,
                    strict=False,
                )
            else:
                model = LiTPCCRV6(args=args, config=config, experiment_logger=experiment_logger)
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        if val_loader is None:
            raise ValueError("Test mode requires a real validation/test dataloader.")
        if not args.checkpoint_path:
            raise ValueError("Test mode requires --checkpoint_path.")
        model = LiTPCCRV6.load_from_checkpoint(
            args.checkpoint_path,
            args=args,
            config=config,
            experiment_logger=experiment_logger,
        )
        trainer.test(model, dataloaders=val_loader)


if __name__ == "__main__":
    main()
