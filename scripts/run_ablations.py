#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.results_utils import RESULTS_ROOT, ensure_dir, save_json, utc_timestamp


ABLATIONS = {
    "no_synth_pretrain": {
        "description": "Train directly on real pairs without synthetic pretraining.",
        "phase1": False,
        "overrides": [],
    },
    "no_matchability_prediction": {
        "description": "Disable matchability supervision and collapse matchability logits to one class.",
        "phase1": True,
        "overrides": [
            "synthetic_matchability_weight=0.0",
            "num_matchability_classes=1",
            "matchability_score_power=0.0",
        ],
    },
    "no_diffeomorphic_constraint": {
        "description": "Use direct per-stage flows without scaling-and-squaring integration.",
        "phase1": True,
        "overrides": [
            "svf_integration_steps=0",
        ],
    },
    "global_only_no_local_refinement": {
        "description": "Keep only the stage-3 global matcher and disable local/final refinement stages.",
        "phase1": True,
        "overrides": [
            "pointmap_stage_ids=[3]",
            "use_fine_local_refinement=false",
            "use_stage2_local_refinement=false",
            "stage2_local_refinement_radius=0",
            "use_stage1_local_refinement=false",
            "stage1_local_refinement_radius=0",
            "use_final_residual_refinement=false",
            "final_refinement_use_local_cost_volume=false",
            "final_refinement_use_local_residual_matcher=false",
        ],
    },
    "resolution_80x96x112": {
        "description": "Run the same pipeline at a lower spatial resolution for the scaling curve with proportionally reduced fine-scale search radii.",
        "phase1": True,
        "overrides": [
            "data_size=[80, 96, 112]",
            "align_data_size_to_native_shape=false",
            "stage2_local_refinement_radius=3",
            "final_refinement_cost_volume_radius=4",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the Task-5 ablation suite.")
    parser.add_argument("--variants", nargs="+", choices=sorted(ABLATIONS), default=list(ABLATIONS))
    parser.add_argument("--config", type=str, default="config/pairwise_oasis_fullres.yaml")
    parser.add_argument("--train_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis_l2r")
    parser.add_argument("--val_data_path", type=str, default="/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis_l2r")
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs", "oasis_l2r"], default="oasis_l2r")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_num_steps", type=int, default=200)
    parser.add_argument("--phase1_epochs", type=int, default=50)
    parser.add_argument("--phase2_epochs", type=int, default=200)
    parser.add_argument("--phase1_lr", type=float, default=1e-4)
    parser.add_argument("--phase2_lr", type=float, default=2e-5)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--logger_backend", choices=["aim", "csv", "tensorboard", "none"], default="csv")
    parser.add_argument("--aim_repo", type=str, default=str(Path("aim")))
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT / "ablations"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def command_to_string(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def build_train_command(
    args,
    experiment_name: str,
    phase: str,
    max_epochs: int,
    lr: float,
    checkpoint_path: str | None,
    overrides: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "src/pccr/scripts/train.py",
        "--config",
        args.config,
        "--mode",
        "train",
        "--phase",
        phase,
        "--train_data_path",
        args.train_data_path,
        "--val_data_path",
        args.val_data_path,
        "--dataset_format",
        args.dataset_format,
        "--accelerator",
        args.accelerator,
        "--num_gpus",
        str(args.num_gpus),
        "--num_workers",
        str(args.num_workers),
        "--batch_size",
        str(args.batch_size),
        "--train_num_steps",
        str(args.train_num_steps),
        "--max_epochs",
        str(max_epochs),
        "--lr",
        str(lr),
        "--precision",
        args.precision,
        "--experiment_name",
        experiment_name,
        "--logger_backend",
        args.logger_backend,
        "--aim_repo",
        args.aim_repo,
        "--checkpoint_every_n_epochs",
        "10" if phase == "synthetic" else "25",
        "--check_val_every_n_epoch",
        "10" if phase == "real" else "1",
        "--max_val_pairs",
        "5" if phase == "real" else "0",
        "--config_override",
        "lr_scheduler=cosine",
        "--config_override",
        f"lr_warmup_epochs={0 if phase == 'synthetic' else 5}",
    ]
    if checkpoint_path:
        command += ["--checkpoint_path", checkpoint_path]
    for override in overrides:
        command += ["--config_override", override]
    return command


def build_eval_command(args, experiment_name: str, pccr_experiment: str) -> list[str]:
    return [
        sys.executable,
        "scripts/full_evaluation.py",
        "--models",
        "pccr",
        "--pccr_experiment",
        pccr_experiment,
        "--pccr_config",
        args.config,
        "--train_data_path",
        args.train_data_path,
        "--val_data_path",
        args.val_data_path,
        "--dataset_format",
        args.dataset_format,
        "--batch_size",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--accelerator",
        args.accelerator,
        "--num_gpus",
        str(args.num_gpus),
        "--output_dir",
        str(Path(args.results_dir) / experiment_name),
    ]


def run_command(command: list[str], execute: bool) -> None:
    print(command_to_string(command))
    if execute:
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        subprocess.run(command, check=True, env=env)


def main():
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    manifest = {
        "generated_at": utc_timestamp(),
        "execute": args.execute,
        "variants": {},
    }

    for variant_name in args.variants:
        spec = ABLATIONS[variant_name]
        phase1_experiment = f"ablation_{variant_name}_phase1"
        phase2_experiment = f"ablation_{variant_name}_phase2"
        phase1_checkpoint = str(Path("checkpoints") / "pccr" / phase1_experiment / "last.ckpt")
        variant_manifest = {
            "description": spec["description"],
            "commands": {},
            "results_dir": str(Path(results_dir) / variant_name),
        }

        if spec["phase1"]:
            phase1_command = build_train_command(
                args=args,
                experiment_name=phase1_experiment,
                phase="synthetic",
                max_epochs=args.phase1_epochs,
                lr=args.phase1_lr,
                checkpoint_path=None,
                overrides=spec["overrides"],
            )
            variant_manifest["commands"]["phase1_train"] = phase1_command
            if not args.skip_train:
                run_command(phase1_command, execute=args.execute)

        phase2_command = build_train_command(
            args=args,
            experiment_name=phase2_experiment,
            phase="real",
            max_epochs=args.phase2_epochs,
            lr=args.phase2_lr,
            checkpoint_path=None if not spec["phase1"] else phase1_checkpoint,
            overrides=spec["overrides"],
        )
        variant_manifest["commands"]["phase2_train"] = phase2_command
        if not args.skip_train:
            run_command(phase2_command, execute=args.execute)

        eval_command = build_eval_command(args, variant_name, phase2_experiment)
        variant_manifest["commands"]["evaluation"] = eval_command
        if not args.skip_eval:
            run_command(eval_command, execute=args.execute)

        manifest["variants"][variant_name] = variant_manifest

    manifest_path = save_json(Path(results_dir) / "ablation_manifest.json", manifest)
    print(f"ablation_manifest: {manifest_path}")


if __name__ == "__main__":
    main()
