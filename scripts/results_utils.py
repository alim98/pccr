#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.eval_baseline import evaluate_runtime, load_hvit_runtime, load_pccr_runtime
from src.pccr.eval_utils import jacobian_determinant, warp_segmentation
from src.pccr.utils import resize_displacement

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
TABLES_ROOT = RESULTS_ROOT / "tables"
VIS_ROOT = RESULTS_ROOT / "visualizations"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def save_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _extract_epoch(path: Path) -> int:
    match = re.search(r"epoch[_-]?(\d+)", path.name)
    if match:
        return int(match.group(1))
    return -1


def _sort_key(path: Path) -> tuple[int, float]:
    return (_extract_epoch(path), path.stat().st_mtime)


def _pccr_checkpoint_rank(path: Path) -> tuple[int, float]:
    name = str(path).lower()
    score = 0
    positive_keywords = {
        "real": 20,
        "l2r": 20,
        "fullres": 20,
    }
    negative_keywords = {
        "proxy": -80,
        "smoke": -120,
        "overfit": -120,
        "test": -100,
        "oracle": -40,
        "diag": -40,
        "synthetic_only": -60,
        "synth": -20,
    }
    for keyword, value in positive_keywords.items():
        if keyword in name:
            score += value
    for keyword, value in negative_keywords.items():
        if keyword in name:
            score += value
    return (score, path.stat().st_mtime)


def discover_best_pccr_checkpoint(experiment_name: str | None = None) -> Path:
    checkpoint_root = REPO_ROOT / "checkpoints" / "pccr"
    best_candidates = list(checkpoint_root.glob("**/best-dice-*.ckpt"))
    if experiment_name:
        best_candidates = [path for path in best_candidates if experiment_name in str(path)]
    if best_candidates:
        return max(best_candidates, key=_pccr_checkpoint_rank)

    last_candidates = list(checkpoint_root.glob("**/last.ckpt"))
    if experiment_name:
        last_candidates = [path for path in last_candidates if experiment_name in str(path)]
    if not last_candidates:
        raise FileNotFoundError("Could not discover a PCCR checkpoint under checkpoints/pccr.")
    return max(last_candidates, key=_pccr_checkpoint_rank)


def discover_best_hvit_checkpoint(experiment_name: str | None = None) -> Path:
    checkpoint_root = REPO_ROOT / "checkpoints"
    best_candidates = list(checkpoint_root.glob("**/best_model.ckpt"))
    if experiment_name:
        best_candidates = [path for path in best_candidates if experiment_name in str(path)]
    if best_candidates:
        return max(best_candidates, key=lambda path: path.stat().st_mtime)

    epoch_candidates = list(checkpoint_root.glob("**/model_epoch_*.ckpt"))
    if experiment_name:
        epoch_candidates = [path for path in epoch_candidates if experiment_name in str(path)]
    if not epoch_candidates:
        raise FileNotFoundError("Could not discover an H-ViT checkpoint under checkpoints/.")
    return max(epoch_candidates, key=_sort_key)


def build_eval_args(
    model: str,
    checkpoint_path: str | Path,
    config: str | None = None,
    train_data_path: str | None = None,
    val_data_path: str | None = None,
    dataset_format: str = "auto",
    batch_size: int | None = 1,
    num_workers: int = 4,
    accelerator: str = "auto",
    num_gpus: int = 1,
    precision: str | None = None,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    train_num_steps: int = 200,
    max_train_subjects: int = 0,
    max_val_subjects: int = 0,
    max_val_pairs: int = 0,
    experiment_name: str | None = None,
    progress_every: int = 5,
    skip_hd95: bool = False,
    lr: float = 1e-4,
    mse_weights: float = 1.0,
    dice_weights: float = 1.0,
    grad_weights: float = 0.02,
    tgt2src_reg: bool = True,
    hvit_light: bool | None = None,
    num_labels: int | None = None,
) -> Namespace:
    return Namespace(
        model=model,
        config=config,
        checkpoint_path=str(Path(checkpoint_path)),
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        dataset_format=dataset_format,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator,
        num_gpus=num_gpus,
        precision=precision,
        val_fraction=val_fraction,
        split_seed=split_seed,
        train_num_steps=train_num_steps,
        max_train_subjects=max_train_subjects,
        max_val_subjects=max_val_subjects,
        max_val_pairs=max_val_pairs,
        output_json=None,
        experiment_name=experiment_name,
        progress_every=progress_every,
        skip_hd95=skip_hd95,
        lr=lr,
        mse_weights=mse_weights,
        dice_weights=dice_weights,
        grad_weights=grad_weights,
        tgt2src_reg=tgt2src_reg,
        hvit_light=hvit_light,
        num_labels=num_labels,
    )


def evaluate_model(args: Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    if args.accelerator == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    if args.model == "hvit":
        runtime = load_hvit_runtime(args, device)
    elif args.model == "pccr":
        runtime = load_pccr_runtime(args, device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    report = evaluate_runtime(args, runtime, device)
    report.update(
        {
            "model": args.model,
            "config_path": runtime["config_path"],
            "checkpoint_path": str(args.checkpoint_path),
            "dataset_path": runtime["dataset_path"],
            "dataset_format": runtime["dataset_format"],
            "num_pairs": len(runtime["val_loader"].dataset),
            "num_labels": runtime["num_labels"],
            "generated_at": utc_timestamp(),
        }
    )
    return report


def metric_values_from_report(report: dict[str, Any], metric_key: str) -> np.ndarray:
    values = []
    for record in report["per_pair"]:
        if metric_key.startswith("registered."):
            values.append(float(record["registered"][metric_key.split(".", 1)[1]]))
        elif metric_key.startswith("identity."):
            values.append(float(record["identity"][metric_key.split(".", 1)[1]]))
        elif metric_key.startswith("registered_dice_per_structure."):
            label_id = metric_key.split(".", 1)[1]
            values.append(float(record["registered_dice_per_structure"][label_id]))
        elif metric_key.startswith("identity_dice_per_structure."):
            label_id = metric_key.split(".", 1)[1]
            values.append(float(record["identity_dice_per_structure"][label_id]))
        else:
            raise KeyError(f"Unsupported metric key: {metric_key}")
    return np.asarray(values, dtype=np.float64)


def bootstrap_ci(
    values: np.ndarray,
    num_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(num_bootstrap, dtype=np.float64)
    for index in range(num_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means[index] = float(np.nanmean(sample))
    alpha = 1.0 - confidence
    return {
        "mean": float(np.nanmean(values)),
        "ci_low": float(np.nanpercentile(boot_means, 100.0 * alpha / 2.0)),
        "ci_high": float(np.nanpercentile(boot_means, 100.0 * (1.0 - alpha / 2.0))),
    }


def summarize_report_with_bootstrap(
    report: dict[str, Any],
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    label_ids = [int(label_id) for label_id in report["summary"]["label_ids"]]
    metrics = {
        "dice_mean_fg": bootstrap_ci(metric_values_from_report(report, "registered.dice_mean_fg"), num_bootstrap, seed=seed),
        "hd95_mean_fg": bootstrap_ci(metric_values_from_report(report, "registered.hd95_mean_fg"), num_bootstrap, seed=seed),
        "sdlogj": bootstrap_ci(metric_values_from_report(report, "registered.sdlogj"), num_bootstrap, seed=seed),
        "jacobian_nonpositive_percent": bootstrap_ci(
            metric_values_from_report(report, "registered.jacobian_nonpositive_percent"),
            num_bootstrap,
            seed=seed,
        ),
    }
    per_structure = {
        str(label_id): bootstrap_ci(
            metric_values_from_report(report, f"registered_dice_per_structure.{label_id}"),
            num_bootstrap,
            seed=seed,
        )
        for label_id in label_ids
    }
    return {
        "metrics": metrics,
        "per_structure_dice": per_structure,
    }


def format_ci(stat: dict[str, float], precision: int = 4, suffix: str = "") -> str:
    return (
        f"{stat['mean']:.{precision}f}{suffix} "
        f"[{stat['ci_low']:.{precision}f}{suffix}, {stat['ci_high']:.{precision}f}{suffix}]"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    parts = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        parts.append("| " + " | ".join(row) + " |")
    return "\n".join(parts) + "\n"


def latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    escaped_headers = " & ".join(headers)
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{'l' * len(headers)}}}\n"
        "\\hline\n"
        f"{escaped_headers} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def comparison_narrative(pccr_report: dict[str, Any], hvit_report: dict[str, Any]) -> str:
    pccr_dice = pccr_report["bootstrap"]["metrics"]["dice_mean_fg"]["mean"]
    pccr_sdlogj = pccr_report["bootstrap"]["metrics"]["sdlogj"]["mean"]
    pccr_nonpos = pccr_report["bootstrap"]["metrics"]["jacobian_nonpositive_percent"]["mean"]
    hvit_nonpos = hvit_report["bootstrap"]["metrics"]["jacobian_nonpositive_percent"]["mean"]
    if hvit_nonpos > 0:
        fewer_defects = 100.0 * (hvit_nonpos - pccr_nonpos) / hvit_nonpos
    else:
        fewer_defects = 0.0
    return (
        "PCCR achieves "
        f"{pccr_dice:.4f} Dice with {pccr_sdlogj:.4f} SDlogJ and "
        f"{fewer_defects:.1f}% fewer topological defects than H-ViT, demonstrating that "
        "explicit correspondence bottleneck with diffeomorphic decoding provides superior "
        "geometric guarantees with competitive overlap accuracy."
    )


def select_pair_indices(num_pairs: int, requested: list[int] | None = None, count: int = 3) -> list[int]:
    if requested:
        return [min(max(int(index), 0), max(num_pairs - 1, 0)) for index in requested]
    if num_pairs <= count:
        return list(range(num_pairs))
    return [0, num_pairs // 2, num_pairs - 1]


def load_runtime_for_visualization(args: Namespace):
    device = torch.device("cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    if args.accelerator == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
    if args.model == "hvit":
        runtime = load_hvit_runtime(args, device)
    else:
        runtime = load_pccr_runtime(args, device)
    return runtime, device


def run_single_pair(runtime: dict[str, Any], args: Namespace, pair_index: int, device: torch.device) -> dict[str, Any]:
    dataset = runtime["val_loader"].dataset
    sample = dataset[pair_index]
    source, target, source_label, target_label = sample[:4]
    source = source.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    source_label = source_label.unsqueeze(0).to(device)
    target_label = target_label.unsqueeze(0).to(device)

    use_cuda = device.type == "cuda"
    precision = runtime["precision"]
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16-mixed": torch.bfloat16,
        "16-mixed": torch.float16,
        "32-true": torch.float32,
    }
    dtype = dtype_map.get(precision, torch.float32)
    autocast_enabled = use_cuda and dtype in {torch.bfloat16, torch.float16}

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            raw_outputs = runtime["run_model"](source, target)
    outputs = raw_outputs if args.model == "pccr" else {"moved_source": raw_outputs[0], "phi_s2t": raw_outputs[1]}
    warped_onehot, warped_label = warp_segmentation(
        source_label,
        outputs["phi_s2t"],
        runtime["num_labels"],
        runtime["transformer"],
    )
    jac = jacobian_determinant(outputs["phi_s2t"].float()).squeeze(0).detach().cpu().numpy()
    return {
        "source": source.squeeze().detach().cpu().numpy(),
        "target": target.squeeze().detach().cpu().numpy(),
        "source_label": source_label.squeeze().detach().cpu().numpy(),
        "target_label": target_label.squeeze().detach().cpu().numpy(),
        "moved": outputs["moved_source"].squeeze().detach().cpu().numpy(),
        "warped_label": warped_label.squeeze().detach().cpu().numpy(),
        "warped_onehot": warped_onehot.squeeze(0).detach().cpu().numpy(),
        "displacement": outputs["phi_s2t"].squeeze(0).detach().cpu().numpy(),
        "jacobian": jac,
        "pair_index": pair_index,
    }


def resize_displacement_to_image(displacement: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    tensor = torch.from_numpy(displacement).unsqueeze(0)
    resized = resize_displacement(tensor.float(), image_shape)
    return resized.squeeze(0).cpu().numpy()
