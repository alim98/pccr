import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from pathlib import Path as _Path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from src.pccr.scripts.train import (
    apply_config_overrides,
    build_logger,
    parse_args,
    resolve_devices,
    resolve_precision,
    resolve_strategy,
    run_memory_probe,
)
from src.pccr.config import PCCRConfig
from src.pccr.data import (
    create_overfit_pair_dataloaders,
    create_real_pair_dataloaders,
    create_synthetic_dataloader,
    create_synthetic_pair_dataloaders,
)
from src.pccr.periodic_eval import IterativeEvalCallback
from src.pccr.trainer_halfres_final_cv import LiTPCCRHalfResFinalCV


def main():
    args = parse_args()
    import torch

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
    checkpoint_dir = _Path("checkpoints") / "pccr" / args.experiment_name
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
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch{epoch:03d}",
            auto_insert_metric_name=False,
            monitor=None,
            save_top_k=-1,
            save_last=True,
            every_n_epochs=max(int(args.checkpoint_every_n_epochs), 1),
        )
    )
    if int(args.checkpoint_every_n_train_steps) > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="step{step:08d}",
                auto_insert_metric_name=False,
                monitor=None,
                save_top_k=1,
                save_last=True,
                every_n_train_steps=int(args.checkpoint_every_n_train_steps),
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
                save_top_k=1,
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
                save_top_k=1,
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
                precision=resolved_precision,
                output_dir=_Path("logs") / "pccr" / args.experiment_name / "iter_eval",
                include_hd95=not args.iter_eval_skip_hd95,
                visualization_every_n_epochs=args.iter_viz_every_n_epochs,
                visualization_pair_index=args.iter_viz_pair_index,
                visualization_dir=_Path("logs") / "pccr" / args.experiment_name / "iter_viz",
            )
        )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.max_epochs,
        logger=experiment_logger,
        callbacks=callbacks,
        default_root_dir=str(_Path("logs") / "pccr" / args.experiment_name),
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
        model = LiTPCCRHalfResFinalCV(args=args, config=config, experiment_logger=experiment_logger)
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
                model = LiTPCCRHalfResFinalCV.load_from_checkpoint(
                    args.checkpoint_path,
                    args=args,
                    config=config,
                    experiment_logger=experiment_logger,
                    strict=False,
                )
            else:
                model = LiTPCCRHalfResFinalCV(args=args, config=config, experiment_logger=experiment_logger)
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        if val_loader is None:
            raise ValueError("Test mode requires a real validation/test dataloader.")
        if not args.checkpoint_path:
            raise ValueError("Test mode requires --checkpoint_path.")
        model = LiTPCCRHalfResFinalCV.load_from_checkpoint(
            args.checkpoint_path,
            args=args,
            config=config,
            experiment_logger=experiment_logger,
        )
        trainer.test(model, dataloaders=val_loader)


if __name__ == "__main__":
    main()
