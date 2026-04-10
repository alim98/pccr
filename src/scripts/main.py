import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


import argparse
import importlib
import torch
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import CSVLogger

# Add this line after the imports
torch.set_float32_matmul_precision('medium')

from src import logger
from src.trainer import LiTHViT
from src.utils import read_yaml_file
from src.data.datasets import get_dataloader, infer_dataset_format, load_oasis_l2r_metadata


LIGHTNING_PRECISION_MAP = {
    "bf16": "bf16-mixed",
    "fp16": "16-mixed",
    "fp32": "32-true",
}

DEFAULT_TRAIN_DATA_PATH = "/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/Mori/DATA/OASIS/OASIS_L2R_2021_task03/train"
DEFAULT_VAL_DATA_PATH = "/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/Mori/DATA/OASIS/OASIS_L2R_2021_task03/test"
DEFAULT_TEST_DATA_PATH = "/home/mori/HViT/OASIS_small/test"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run training or inference")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the YAML config")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use. Use '-1' for all available GPUs.")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto", help="Execution accelerator")
    parser.add_argument("--experiment_name", type=str, default="OASIS", help="Experiment name")
    parser.add_argument("--mode", choices=["train", "inference"], default="train", help="Mode to run: train or inference")
    parser.add_argument("--train_data_path", type=str, default=None, help="Path to the train set")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to the validation set")
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to the test set")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model/checkpoint_path to load")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the best model")
    parser.add_argument("--mse_weights",type=float, default=1, help="MSE Loss weights")
    parser.add_argument("--dice_weights", type=float, default=1, help="Dice Loss weights")
    parser.add_argument("--grad_weights", type=float, default=0.02, help="Grad Loss weights")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--tgt2src_reg", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True, help="target to source registration during training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labels")
    parser.add_argument("--precision", type=str, default=None, help="Precision")
    parser.add_argument("--hvit_light", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=None, help="Use HViT-Light")
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs", "oasis_l2r"], default="auto", help="Dataset layout")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation fraction when using a shared OASIS root")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed used for train/val split")
    parser.add_argument("--train_num_steps", type=int, default=1000, help="Training samples per epoch for random-pair datasets")
    parser.add_argument("--max_train_subjects", type=int, default=0, help="Optional cap on train subjects for smoke tests")
    parser.add_argument("--max_val_subjects", type=int, default=0, help="Optional cap on validation subjects for smoke tests")
    parser.add_argument("--max_val_pairs", type=int, default=0, help="Optional cap on validation pairs for smoke tests")
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Lightning limit for train batches")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Lightning limit for validation batches")
    parser.add_argument("--limit_test_batches", type=float, default=1.0, help="Lightning limit for test batches")
    parser.add_argument("--save_model_every_n_epochs", type=int, default=None, help="Checkpoint save frequency in epochs")
    parser.add_argument("--logger_backend", choices=["aim", "csv", "none"], default="aim", help="Experiment logger backend")
    parser.add_argument("--aim_repo", type=str, default="/u/almik/others/hvit/aim", help="Path to the Aim repo")
    parser.add_argument("--aim_experiment", type=str, default=None, help="Aim experiment name")
    args = parser.parse_args()
    return args


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
    raise ImportError(
        "Aim logger could not be imported. Install `aim` and ensure either "
        "`aim.pytorch_lightning.AimLogger` or "
        "`aimstack.experiment_tracker.pytorch_lightning.Logger` is available."
    )


def build_logger(args):
    if args.logger_backend == "none":
        return None
    if args.logger_backend == "aim":
        AimLogger = load_aim_logger_class()
        return AimLogger(
            repo=args.aim_repo,
            experiment=args.aim_experiment or args.experiment_name,
            train_metric_prefix="train_",
            val_metric_prefix="val_",
            test_metric_prefix="test_",
        )
    if args.logger_backend == "csv":
        return CSVLogger(save_dir="logs", name=args.experiment_name)
    return None


def resolve_trainer_device(args):
    if args.accelerator == "cpu":
        return "cpu", 1

    if args.accelerator == "gpu":
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise RuntimeError("Requested GPU training, but CUDA devices are not available.")
        devices = available_gpus if args.num_gpus == -1 else min(args.num_gpus, available_gpus)
        return "gpu", devices

    if torch.cuda.is_available() and torch.cuda.device_count() > 0 and args.num_gpus != 0:
        available_gpus = torch.cuda.device_count()
        devices = available_gpus if args.num_gpus == -1 else min(args.num_gpus, available_gpus)
        return "gpu", devices

    return "cpu", 1


def resolve_trainer_precision(args, accelerator):
    if accelerator == "cpu":
        return LIGHTNING_PRECISION_MAP["fp32"]
    return LIGHTNING_PRECISION_MAP.get(args.precision, LIGHTNING_PRECISION_MAP["fp32"])


def resolve_data_path(args, config, split: str) -> str:
    cli_value = getattr(args, f"{split}_data_path")
    if cli_value:
        return cli_value
    if config.get("dataset_format") == "oasis_l2r" and config.get("oasis_l2r_data_root"):
        return str(Path(config["oasis_l2r_data_root"]).expanduser())
    fallback = {
        "train": DEFAULT_TRAIN_DATA_PATH,
        "val": DEFAULT_VAL_DATA_PATH,
        "test": DEFAULT_TEST_DATA_PATH,
    }
    return fallback[split]


def resolve_dataset_format(args, config, data_path: str) -> str:
    if args.dataset_format != "auto":
        return args.dataset_format
    configured = config.get("dataset_format", "auto")
    if configured != "auto":
        return configured
    return infer_dataset_format(data_path)


def sync_config_with_dataset(config: dict, data_path: str, dataset_format: str) -> None:
    if dataset_format != "oasis_l2r":
        if "eval_label_ids" not in config:
            config["eval_label_ids"] = []
        return

    metadata = load_oasis_l2r_metadata(Path(data_path))
    native_shape = list(metadata.native_shape)
    if list(config.get("data_size", [])) != native_shape:
        print(
            "[hvit.main] Aligning config.data_size "
            f"from {config.get('data_size')} to dataset native shape {native_shape}."
        )
        config["data_size"] = native_shape

    if not config.get("eval_label_ids"):
        config["eval_label_ids"] = list(metadata.eval_label_ids)
    config["num_labels"] = max(int(config.get("num_labels", 0) or 0), max(metadata.eval_label_ids) + 1)


def apply_runtime_defaults(args, config: dict) -> None:
    if args.batch_size is None:
        args.batch_size = int(config.get("batch_size", 2))
    if args.max_epochs is None:
        args.max_epochs = int(config.get("max_epochs", 1000))
    if args.num_labels is None:
        args.num_labels = int(config.get("num_labels", 36))
    if args.precision is None:
        if "precision" in config:
            args.precision = str(config["precision"])
        elif config.get("use_amp", False):
            args.precision = "bf16"
        else:
            args.precision = "fp32"
    if args.hvit_light is None:
        args.hvit_light = bool(config.get("hvit_light", True))
    if args.save_model_every_n_epochs is None:
        args.save_model_every_n_epochs = int(config.get("save_model_every_n_epochs", 1))
    args.eval_label_ids = [int(label_id) for label_id in config.get("eval_label_ids", [])]

def main():
    args = parse_arguments()
    config = read_yaml_file(args.config)
    if config is None:
        raise ValueError(f"Failed to read config file: {args.config}")

    train_data_path = resolve_data_path(args, config, "train")
    val_data_path = resolve_data_path(args, config, "val")
    test_data_path = resolve_data_path(args, config, "test")
    resolved_dataset_format = resolve_dataset_format(args, config, train_data_path)
    sync_config_with_dataset(config, train_data_path, resolved_dataset_format)
    apply_runtime_defaults(args, config)

    experiment_logger = build_logger(args)
    if experiment_logger is not None and hasattr(experiment_logger, "log_hyperparams"):
        experiment_logger.log_hyperparams(
            {
                "experiment_name": args.experiment_name,
                "dataset_format": resolved_dataset_format,
                "train_data_path": train_data_path,
                "val_data_path": val_data_path,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "precision": args.precision,
                "hvit_light": args.hvit_light,
                "save_model_every_n_epochs": args.save_model_every_n_epochs,
                "config": config,
            }
        )

    # get dataloaders
    train_dataloader = get_dataloader(
        data_path=train_data_path,
        input_dim=config["data_size"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_format=resolved_dataset_format,
        split="train",
        is_pair=False,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        num_steps=args.train_num_steps,
        max_subjects=args.max_train_subjects,
    )

    val_dataloader = get_dataloader(
        data_path=val_data_path,
        input_dim=config["data_size"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_format=resolved_dataset_format,
        split="val",
        is_pair=True,
        shuffle=False,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_subjects=args.max_val_subjects,
        max_pairs=args.max_val_pairs,
    )

    dataset_num_labels = max(
        getattr(train_dataloader.dataset, "num_labels", 0) or 0,
        getattr(val_dataloader.dataset, "num_labels", 0) or 0,
    )
    if dataset_num_labels:
        args.num_labels = max(args.num_labels, dataset_num_labels)
        logger.info(f"Using num_labels={args.num_labels}")
    for dataset in (getattr(train_dataloader, "dataset", None), getattr(val_dataloader, "dataset", None)):
        if dataset is not None and getattr(dataset, "eval_label_ids", None) and not args.eval_label_ids:
            args.eval_label_ids = [int(label_id) for label_id in dataset.eval_label_ids]
            break

    accelerator, devices = resolve_trainer_device(args)
    if accelerator == "cpu":
        args.precision = "fp32"
    trainer_precision = resolve_trainer_precision(args, accelerator)
    print(f"Using accelerator={accelerator}, devices={devices}, precision={trainer_precision} ...")

    if accelerator == "gpu" and devices > 1:
        trainer_strategy = DDPStrategy(find_unused_parameters=True)
    else:
        trainer_strategy = "auto"

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=experiment_logger,
        precision=trainer_precision,
        accelerator=accelerator,
        devices=devices,
        strategy=trainer_strategy,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        log_every_n_steps=1,
    )

    # train/test
    if args.mode == "train":
        if args.resume_from_checkpoint:
            model = LiTHViT.load_from_checkpoint(args.resume_from_checkpoint, args=args, experiment_logger=experiment_logger)
            print(f"Resuming training from epoch {model.last_epoch + 1}")
        else:
            model = LiTHViT(
                args,
                config,
                experiment_logger=experiment_logger,
                save_model_every_n_epochs=args.save_model_every_n_epochs,
            )
            print("Starting new training run")
        logger.info("Starting training")
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            datamodule=None,
            ckpt_path=args.resume_from_checkpoint,
        )

    elif args.mode == "inference":
        logger.info("Starting inference")

        test_dataloader = get_dataloader(
            data_path=test_data_path,
            input_dim=config["data_size"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dataset_format=resolved_dataset_format,
            split="val",
            is_pair=True,
            shuffle=False,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            max_subjects=args.max_val_subjects,
            max_pairs=args.max_val_pairs,
        )


        # # Get the latest checkpoint folder
        # checkpoints_dir = Path("checkpoints")
        # checkpoints = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()], key=os.path.getctime, reverse=True)
        # latest_checkpoint = checkpoints[1] if checkpoints else None

        # if os.path.exists(latest_checkpoint):
        #     logger.info(f"Using latest checkpoint: {latest_checkpoint}")

        #     if args.checkpoint_path:
        #         best_model_path = f"{args.checkpoint_path}/best_model.ckpt"
        #     else:
        #         best_model_path = f"{latest_checkpoint}/best_model.ckpt"

        if args.checkpoint_path:
            model = LiTHViT.load_from_checkpoint(args.checkpoint_path, args=args, experiment_logger=experiment_logger)
            print(f"Checkpoint loaded. Resuming from epoch {model.last_epoch + 1}")
        else:
            raise Exception("No checkpoint found")
        trainer.test(model, dataloaders=test_dataloader)

if __name__ == "__main__":
    main()
