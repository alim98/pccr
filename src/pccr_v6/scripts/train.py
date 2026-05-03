import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import argparse
import importlib
import os
import re

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from src.pccr.data import (
    create_overfit_pair_dataloaders,
    create_real_pair_dataloaders,
    create_synthetic_dataloader,
    create_synthetic_pair_dataloaders,
    _loader_runtime_kwargs,
)
from src.pccr.periodic_eval import IterativeEvalCallback
from src.pccr_v6.config import PCCRV6Config
from src.pccr_v6.data import AugmentedRealPairDataset
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
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--ddp_find_unused_parameters", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=1)
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_num_steps", type=int, default=200)
    parser.add_argument("--max_train_subjects", type=int, default=0)
    parser.add_argument("--max_val_subjects", type=int, default=0)
    parser.add_argument("--max_val_pairs", type=int, default=0)
    parser.add_argument("--logger_backend", type=str, default="aim")
    parser.add_argument("--aim_repo", type=str, default="/u/almik/others/hvit/aim")
    parser.add_argument("--aim_run_hash", type=str, default=None)
    parser.add_argument("--aim_run_hash_file", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="pccr_v6_oasis")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
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
    parser.add_argument("--no_augment", action="store_true", help="Disable real-phase train augmentation")
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


def parse_logger_backends(raw_value: str) -> list[str]:
    valid = {"aim", "csv", "none"}
    aliases = {"both": ["aim", "csv"], "dual": ["aim", "csv"]}
    parts = [part for part in re.split(r"[\s,+]+", (raw_value or "aim").strip().lower()) if part]
    if not parts:
        parts = ["aim"]

    resolved = []
    for part in parts:
        resolved.extend(aliases.get(part, [part]))

    if "none" in resolved:
        if len(resolved) != 1:
            raise ValueError("--logger_backend=none cannot be combined with other loggers.")
        return ["none"]

    invalid = [part for part in resolved if part not in valid]
    if invalid:
        raise ValueError(
            f"Unsupported logger backend(s): {', '.join(sorted(set(invalid)))}. "
            "Use aim, csv, none, or combinations like aim,csv."
        )

    deduped = []
    for part in resolved:
        if part not in deduped:
            deduped.append(part)
    return deduped


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _read_first_line(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return _clean_optional(Path(path).read_text(encoding="utf-8").splitlines()[0])
    except (FileNotFoundError, IndexError):
        return None


def resolve_aim_run_hash(args) -> str | None:
    return (
        _clean_optional(args.aim_run_hash)
        or _clean_optional(os.environ.get("AIM_RUN_HASH"))
        or _read_first_line(args.aim_run_hash_file or os.environ.get("AIM_RUN_HASH_FILE"))
    )


def resolve_run_name(args) -> str:
    return (
        _clean_optional(args.run_name)
        or _clean_optional(os.environ.get("AIM_RUN_NAME"))
        or args.experiment_name
    )


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class AimRunStateCallback(Callback):
    def __init__(self, args, config: PCCRV6Config):
        self.args = args
        self.config = config

    def _iter_loggers(self, trainer):
        loggers = getattr(trainer, "loggers", None)
        if loggers is None:
            logger = getattr(trainer, "logger", None)
            return [] if logger is None else [logger]
        return list(loggers)

    def _write_hash_file(self, run_hash: str) -> None:
        hash_file = _clean_optional(self.args.aim_run_hash_file) or _clean_optional(os.environ.get("AIM_RUN_HASH_FILE"))
        if not hash_file:
            return
        path = Path(hash_file)
        existing = _read_first_line(str(path))
        if existing and existing != run_hash:
            print(
                f"[pccr_v6.train] WARNING: Aim hash file {path} contains {existing}, "
                f"but active run is {run_hash}."
            )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(f"{run_hash}\n", encoding="utf-8")
        tmp_path.replace(path)

    def _record_metadata(self, run) -> None:
        run_name = resolve_run_name(self.args)
        if run_name:
            run.name = run_name
        metadata = {
            "experiment_name": self.args.experiment_name,
            "run_name": run_name,
            "phase": self.args.phase,
            "mode": self.args.mode,
            "checkpoint_path": self.args.checkpoint_path,
            "resume_from_checkpoint": self.args.resume_from_checkpoint,
            "checkpoint_dir": self.args.checkpoint_dir,
            "aim_run_hash_file": self.args.aim_run_hash_file or os.environ.get("AIM_RUN_HASH_FILE"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_nnodes": os.environ.get("SLURM_NNODES"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "command": " ".join(sys.argv),
        }
        run["pccr_run"] = _jsonable(metadata)
        run["args"] = _jsonable(vars(self.args))
        run["config"] = _jsonable(self.config.to_dict())

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        for logger in self._iter_loggers(trainer):
            experiment = getattr(logger, "experiment", None)
            if experiment is None or not hasattr(experiment, "track"):
                continue
            try:
                run_hash = getattr(experiment, "hash", None)
                if run_hash:
                    self._write_hash_file(run_hash)
                self._record_metadata(experiment)
            except Exception as exc:
                print(f"[pccr_v6.train] WARNING: failed to record Aim run metadata: {exc}")
            break


def build_logger(args):
    backends = parse_logger_backends(args.logger_backend)
    if backends == ["none"]:
        return None
    loggers = []
    for backend in backends:
        if backend == "csv":
            loggers.append(CSVLogger(save_dir="logs", name=args.experiment_name))
        elif backend == "aim":
            AimLogger = load_aim_logger_class()
            aim_kwargs = {
                "repo": args.aim_repo,
                "experiment": args.experiment_name,
                "run_name": resolve_run_name(args),
            }
            run_hash = resolve_aim_run_hash(args)
            if run_hash:
                aim_kwargs["run_hash"] = run_hash
            try:
                loggers.append(AimLogger(**aim_kwargs))
            except TypeError:
                aim_kwargs.pop("run_hash", None)
                aim_kwargs.pop("run_name", None)
                loggers.append(AimLogger(**aim_kwargs))
    if not loggers:
        return None
    return loggers if len(loggers) > 1 else loggers[0]


def resolve_devices(args):
    if args.accelerator == "cpu":
        return "cpu", 1
    if args.accelerator == "gpu":
        return "gpu", args.num_gpus
    if torch.cuda.is_available():
        return "gpu", args.num_gpus
    return "cpu", 1


def resolve_strategy(args, accelerator: str, devices: int) -> str:
    num_nodes = getattr(args, "num_nodes", None) or 1
    find_unused = getattr(args, "ddp_find_unused_parameters", "auto")
    multi_gpu = accelerator == "gpu" and (devices > 1 or num_nodes > 1)
    if not multi_gpu:
        return "auto"
    if find_unused == "true":
        return "ddp_find_unused_parameters_true"
    if find_unused == "false":
        return "ddp"
    return "ddp"


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
    checkpoint_dir = Path(
        args.checkpoint_dir
        or os.environ.get("PCCR_CHECKPOINT_DIR", "")
        or (Path("checkpoints") / "pccr" / args.experiment_name)
    )
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
            if args.phase == "real" and not getattr(args, "no_augment", False):
                aug_dataset = AugmentedRealPairDataset(train_loader.dataset, augment=True)
                train_loader = DataLoader(
                    aug_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    **_loader_runtime_kwargs(args.num_workers),
                )
        dataset_num_labels = max(
            getattr(train_loader.dataset, "num_labels", 0) or 0,
            getattr(val_loader.dataset, "num_labels", 0) or 0,
        )
        if dataset_num_labels:
            config.num_labels = max(config.num_labels, dataset_num_labels)

    accelerator, devices = resolve_devices(args)
    checkpoint_every_n_epochs = max(int(getattr(args, "checkpoint_every_n_epochs", 1)), 1)
    checkpoint_every_n_steps = int(getattr(args, "checkpoint_every_n_train_steps", 0) or 0)
    callbacks = [
        AimRunStateCallback(args=args, config=config),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch{epoch:03d}",
            auto_insert_metric_name=False,
            monitor=None,
            save_top_k=-1,
            save_last=True,
            every_n_epochs=checkpoint_every_n_epochs,
        )
    ]
    if checkpoint_every_n_steps > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="step{step:08d}",
                auto_insert_metric_name=False,
                monitor=None,
                save_top_k=1,
                save_last=True,
                every_n_train_steps=checkpoint_every_n_steps,
            )
        )
    if val_loader is not None and args.phase != "synthetic":
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="best-dice-epoch{epoch:03d}-dice{val_dice:.4f}",
                auto_insert_metric_name=False,
                monitor="val_dice",
                mode="max",
                save_top_k=3,
                save_last=False,
                every_n_epochs=1,
            )
        )
    elif val_loader is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="best-loss-epoch{epoch:03d}-val{val_avg_loss:.4f}",
                auto_insert_metric_name=False,
                monitor="val_avg_loss",
                mode="min",
                save_top_k=3,
                save_last=False,
                every_n_epochs=1,
            )
        )
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

    num_nodes = getattr(args, "num_nodes", None) or 1
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        max_epochs=args.max_epochs,
        logger=experiment_logger,
        callbacks=callbacks,
        default_root_dir=str(Path("logs") / "pccr_v6" / args.experiment_name),
        precision=args.precision if accelerator == "gpu" else "32-true",
        strategy=resolve_strategy(args, accelerator, devices),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=max(int(config.gradient_accumulation_steps), 1),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=max(int(getattr(args, "check_val_every_n_epoch", 1)), 1),
        log_every_n_steps=max(int(getattr(args, "log_every_n_steps", 10)), 1),
        num_sanity_val_steps=0,
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
