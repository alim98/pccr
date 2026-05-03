import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import argparse
import importlib
import re

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import TensorBoardLogger

from src.pccr.config import PCCRConfig
from src.pccr.data import (
    create_overfit_pair_dataloaders,
    create_real_pair_dataloaders,
    create_synthetic_dataloader,
    create_synthetic_pair_dataloaders,
)
from src.pccr.periodic_eval import IterativeEvalCallback
from src.pccr.trainer import LiTPCCR


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
    parser = argparse.ArgumentParser(description="Train or evaluate the PCCR research codebase.")
    parser.add_argument("--config", type=str, default="src/pccr/configs/pairwise_oasis.yaml")
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--phase", choices=["real", "synthetic"], default="real")
    parser.add_argument("--data_source", choices=["auto", "real", "synthetic"], default="auto")
    parser.add_argument("--train_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--val_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis")
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs", "oasis_l2r"], default="auto")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=None)
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
    parser.add_argument("--experiment_name", type=str, default="pccr_oasis")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--ddp_find_unused_parameters", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=1)
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--iter_eval_every_n_epochs", type=int, default=0)
    parser.add_argument("--iter_eval_num_pairs", type=int, default=100)
    parser.add_argument("--iter_eval_skip_hd95", action="store_true")
    parser.add_argument("--iter_viz_every_n_epochs", type=int, default=0)
    parser.add_argument("--iter_viz_pair_index", type=int, default=0)
    parser.add_argument("--memory_probe_only", action="store_true")
    parser.add_argument("--memory_warning_ratio", type=float, default=0.9)
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


def parse_logger_backends(raw_value: str) -> list[str]:
    valid = {"aim", "csv", "tensorboard", "none"}
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
            "Use aim, csv, tensorboard, none, or combinations like aim,csv."
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
    def __init__(self, args, config: PCCRConfig):
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
                f"[pccr.train] WARNING: Aim hash file {path} contains {existing}, "
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
                print(f"[pccr.train] WARNING: failed to record Aim run metadata: {exc}")
            break


def build_logger(args):
    backends = parse_logger_backends(args.logger_backend)
    if backends == ["none"]:
        return None

    loggers = []
    csv_logger_added = False
    aim_logger_class = None

    for backend in backends:
        if backend == "csv":
            loggers.append(CSVLogger(save_dir="logs", name=args.experiment_name))
            csv_logger_added = True
            continue

        if backend == "tensorboard":
            try:
                loggers.append(TensorBoardLogger(save_dir="logs", name=args.experiment_name))
            except ModuleNotFoundError:
                print("[pccr.train] TensorBoard is unavailable in this environment; falling back to CSV logging.")
                if not csv_logger_added:
                    loggers.append(CSVLogger(save_dir="logs", name=args.experiment_name))
                    csv_logger_added = True
            continue

        if backend == "aim":
            if aim_logger_class is None:
                aim_logger_class = load_aim_logger_class()
            aim_kwargs = {
                "repo": args.aim_repo,
                "experiment": args.experiment_name,
                "run_name": resolve_run_name(args),
            }
            run_hash = resolve_aim_run_hash(args)
            if run_hash:
                aim_kwargs["run_hash"] = run_hash
            try:
                loggers.append(aim_logger_class(**aim_kwargs))
            except TypeError:
                aim_kwargs.pop("run_hash", None)
                aim_kwargs.pop("run_name", None)
                loggers.append(aim_logger_class(**aim_kwargs))
            continue

    if not loggers:
        return None
    if len(loggers) == 1:
        return loggers[0]
    return loggers


def resolve_devices(args):
    if args.accelerator == "cpu":
        return "cpu", 1
    if args.accelerator == "gpu":
        return "gpu", args.num_gpus
    if torch.cuda.is_available():
        return "gpu", args.num_gpus
    return "cpu", 1


def resolve_strategy(args, accelerator: str, devices: int) -> str:
    if accelerator == "gpu" and devices > 1:
        find_unused = args.ddp_find_unused_parameters
        if find_unused == "true" or (find_unused == "auto" and args.phase == "synthetic"):
            return "ddp_find_unused_parameters_true"
        return "ddp"
    return "auto"


def resolve_precision(args, config: PCCRConfig, accelerator: str) -> str:
    if accelerator != "gpu":
        return "32-true"
    if args.precision is not None:
        return args.precision
    return "bf16-mixed" if config.use_amp else "32-true"


def resolve_autocast_dtype(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision.startswith("bf16"):
        return torch.bfloat16
    if precision.startswith("16"):
        return torch.float16
    return torch.float32


def _move_batch_to_device(batch, device: torch.device):
    return tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)


def run_memory_probe(
    model: LiTPCCR,
    train_loader,
    precision: str,
    warning_ratio: float,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        print("[pccr.train] Memory probe skipped on non-CUDA device.")
        return

    model = model.to(device)
    iterator = iter(train_loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Training dataloader is empty; cannot run memory probe.") from exc

    batch = _move_batch_to_device(batch, device)
    dtype = resolve_autocast_dtype(precision)
    autocast_enabled = dtype in {torch.bfloat16, torch.float16}
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    optimizer = model.configure_optimizers()["optimizer"]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        try:
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
                _, losses, _, _ = model._compute_loss(batch)
                loss = losses["avg_loss"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print("[pccr.train] Memory probe hit CUDA OOM.")
            raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    utilization = peak_bytes / max(total_bytes, 1)
    peak_gib = peak_bytes / float(1024 ** 3)
    total_gib = total_bytes / float(1024 ** 3)
    print(
        f"[pccr.train] Memory probe peak allocation: {peak_gib:.2f} GiB / "
        f"{total_gib:.2f} GiB ({utilization * 100.0:.1f}%)."
    )
    if utilization >= warning_ratio:
        print(
            "[pccr.train] WARNING: peak allocation exceeds "
            f"{warning_ratio * 100.0:.0f}% of GPU memory. "
            "Consider trying data_size=[80, 96, 112]."
        )


def apply_config_overrides(config: PCCRConfig, override_items: list[str]) -> PCCRConfig:
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
    config = apply_config_overrides(PCCRConfig.from_yaml(args.config), args.config_override)
    config.phase = args.phase
    if args.batch_size is None:
        args.batch_size = config.batch_size
    if args.oracle_correspondence != "none":
        config.diagnostic_oracle_correspondence = True
    if args.oracle_handoff:
        config.diagnostic_oracle_handoff = True
    experiment_logger = build_logger(args)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints") / "pccr" / args.experiment_name
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
    resolved_precision = resolve_precision(args, config, accelerator)
    callbacks = [AimRunStateCallback(args=args, config=config)]
    periodic_checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch{epoch:03d}",
        auto_insert_metric_name=False,
        monitor=None,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=max(int(args.checkpoint_every_n_epochs), 1),
    )
    callbacks.append(periodic_checkpoint)
    if int(args.checkpoint_every_n_train_steps) > 0:
        step_checkpoint = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="step{step:08d}",
            auto_insert_metric_name=False,
            monitor=None,
            save_top_k=1,
            save_last=True,
            every_n_train_steps=int(args.checkpoint_every_n_train_steps),
        )
        callbacks.append(step_checkpoint)
    if val_loader is not None and args.phase != "synthetic":
        best_checkpoint = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-dice-epoch{epoch:03d}-dice{val_dice:.4f}",
            auto_insert_metric_name=False,
            monitor="val_dice",
            mode="max",
            save_top_k=1,
            save_last=False,
            every_n_epochs=1,
        )
        callbacks.append(best_checkpoint)
    elif val_loader is not None:
        val_loss_checkpoint = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-loss-epoch{epoch:03d}-val{val_avg_loss:.4f}",
            auto_insert_metric_name=False,
            monitor="val_avg_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            every_n_epochs=1,
        )
        callbacks.append(val_loss_checkpoint)
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
                precision=resolved_precision,
                output_dir=Path("logs") / "pccr" / args.experiment_name / "iter_eval",
                include_hd95=not args.iter_eval_skip_hd95,
                visualization_every_n_epochs=args.iter_viz_every_n_epochs,
                visualization_pair_index=args.iter_viz_pair_index,
                visualization_dir=Path("logs") / "pccr" / args.experiment_name / "iter_viz",
            )
        )

    num_nodes = args.num_nodes if args.num_nodes is not None else int(os.environ.get("SLURM_NNODES", 1))

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        max_epochs=args.max_epochs,
        logger=experiment_logger,
        callbacks=callbacks,
        default_root_dir=str(Path("logs") / "pccr" / args.experiment_name),
        precision=resolved_precision,
        strategy=resolve_strategy(args, accelerator, devices),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=max(int(config.gradient_accumulation_steps), 1),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=max(int(args.check_val_every_n_epoch), 1),
        log_every_n_steps=max(int(args.log_every_n_steps), 1),
        num_sanity_val_steps=0,
    )

    if args.mode == "train":
        model = LiTPCCR(args=args, config=config, experiment_logger=experiment_logger)
        probe_device = torch.device("cuda" if accelerator == "gpu" else "cpu")
        if accelerator == "gpu" and devices == 1:
            run_memory_probe(
                model=model,
                train_loader=train_loader,
                precision=resolved_precision,
                warning_ratio=args.memory_warning_ratio,
                device=probe_device,
            )
        elif accelerator == "gpu" and devices > 1:
            print("[pccr.train] Skipping memory probe for multi-GPU DDP run.")
        if args.memory_probe_only:
            return
        if args.resume_from_checkpoint:
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=args.resume_from_checkpoint,
            )
        else:
            if args.checkpoint_path:
                model = LiTPCCR.load_from_checkpoint(
                    args.checkpoint_path,
                    args=args,
                    config=config,
                    experiment_logger=experiment_logger,
                    strict=False,
                )
            else:
                model = LiTPCCR(args=args, config=config, experiment_logger=experiment_logger)
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        if val_loader is None:
            raise ValueError("Test mode requires a real validation/test dataloader.")
        if not args.checkpoint_path:
            raise ValueError("Test mode requires --checkpoint_path.")
        model = LiTPCCR.load_from_checkpoint(
            args.checkpoint_path,
            args=args,
            config=config,
            experiment_logger=experiment_logger,
        )
        trainer.test(model, dataloaders=val_loader)


if __name__ == "__main__":
    main()
