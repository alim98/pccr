from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Subset

from src.pccr.eval_utils import aggregate_metrics, identity_metrics, jacobian_determinant, pair_metrics, prefix_metrics


def _normalize_slice(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _center_axial(volume: np.ndarray) -> np.ndarray:
    z = volume.shape[-1] // 2
    return np.rot90(_normalize_slice(volume[:, :, z]))


def _save_registration_visualization(
    output_path: Path,
    source: np.ndarray,
    target: np.ndarray,
    moved: np.ndarray,
    jacobian_map: np.ndarray,
    metrics: dict[str, float],
) -> np.ndarray:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    src_slice = _center_axial(source)
    tgt_slice = _center_axial(target)
    moved_slice = _center_axial(moved)
    diff_slice = np.abs(moved_slice - tgt_slice)
    jac_slice = np.rot90(jacobian_map[:, :, jacobian_map.shape[-1] // 2])

    for axis, image, title, cmap in [
        (axes[0], src_slice, "Source", "gray"),
        (axes[1], tgt_slice, "Target", "gray"),
        (axes[2], moved_slice, "Warped", "gray"),
        (axes[3], diff_slice, "|Warped-Target|", "magma"),
        (axes[4], jac_slice, "Jacobian", "coolwarm"),
    ]:
        axis.imshow(image, cmap=cmap, interpolation="lanczos")
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        (
            f"Dice_fg={metrics['dice_mean_fg']:.4f} | "
            f"HD95={metrics['hd95_mean_fg']:.4f} | "
            f"SDlogJ={metrics['sdlogj']:.4f} | "
            f"NonPosJ={metrics['jacobian_nonpositive_fraction']:.6f}"
        ),
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return image


class IterativeEvalCallback(Callback):
    def __init__(
        self,
        dataset,
        num_labels: int,
        every_n_epochs: int,
        num_pairs: int,
        batch_size: int,
        num_workers: int,
        precision: str,
        output_dir: str | Path,
        include_hd95: bool = False,
        visualization_every_n_epochs: int = 0,
        visualization_pair_index: int = 0,
        visualization_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_pairs = num_pairs
        self.num_labels = num_labels
        self.precision = precision
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_hd95 = include_hd95
        self.visualization_every_n_epochs = visualization_every_n_epochs
        self.visualization_pair_index = visualization_pair_index
        self.visualization_dir = Path(visualization_dir) if visualization_dir is not None else self.output_dir.parent / "iter_viz"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        dataset_length = len(dataset)
        if num_pairs > 0:
            dataset = Subset(dataset, range(min(num_pairs, dataset_length)))

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.dataset = dataset

    def _maybe_log_visualization(self, trainer, pl_module, device, autocast_enabled) -> None:
        if self.visualization_every_n_epochs <= 0:
            return
        current_epoch = trainer.current_epoch
        if current_epoch < 0 or (current_epoch + 1) % self.visualization_every_n_epochs != 0:
            return

        dataset_length = len(self.dataset)
        if dataset_length <= 0:
            return
        pair_index = min(max(self.visualization_pair_index, 0), dataset_length - 1)
        sample = self.dataset[pair_index]
        source, target, source_label, target_label = sample[:4]
        source = source.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        source_label = source_label.unsqueeze(0).to(device)
        target_label = target_label.unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs = pl_module(source, target)
            metrics = pair_metrics(
                outputs=outputs,
                source_label=source_label,
                target_label=target_label,
                num_labels=self.num_labels,
                transformer=pl_module.model.decoder.final_transformer,
                inference_seconds=0.0,
                include_hd95=self.include_hd95,
            )

        source_np = source.squeeze().detach().cpu().numpy()
        target_np = target.squeeze().detach().cpu().numpy()
        moved_np = outputs["moved_source"].squeeze().detach().cpu().numpy()
        jacobian_map = jacobian_determinant(outputs["phi_s2t"].float()).squeeze(0).detach().cpu().numpy()
        output_path = self.visualization_dir / f"epoch_{current_epoch + 1:04d}_pair_{pair_index:03d}.png"
        image = _save_registration_visualization(output_path, source_np, target_np, moved_np, jacobian_map, metrics)
        pl_module.log_aim_image(
            image,
            name="registration_preview",
            step=trainer.global_step,
            context={"subset": "val", "pair_index": pair_index, "epoch": current_epoch + 1},
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs <= 0:
            return
        current_epoch = trainer.current_epoch
        if current_epoch < 0 or (current_epoch + 1) % self.every_n_epochs != 0:
            return

        if not trainer.is_global_zero:
            trainer.strategy.barrier("iterative_eval_wait")
            return

        if getattr(pl_module.args, "phase", "real") == "synthetic":
            trainer.strategy.barrier("iterative_eval_wait")
            return

        pl_module.eval()
        device = pl_module.device
        use_cuda = device.type == "cuda"
        autocast_enabled = use_cuda and self.precision.startswith("bf16")

        records = []
        with torch.no_grad():
            for pair_idx, batch in enumerate(self.loader):
                source, target, source_label, target_label = [tensor.to(device) for tensor in batch[:4]]
                before = identity_metrics(
                    source_label=source_label,
                    target_label=target_label,
                    num_labels=self.num_labels,
                    inference_seconds=0.0,
                    include_hd95=self.include_hd95,
                )
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    outputs = pl_module(source, target)
                after = pair_metrics(
                    outputs=outputs,
                    source_label=source_label,
                    target_label=target_label,
                    num_labels=self.num_labels,
                    transformer=pl_module.model.decoder.final_transformer,
                    inference_seconds=0.0,
                    include_hd95=self.include_hd95,
                )
                record = {"pair_index": pair_idx}
                record.update(prefix_metrics(before, "identity_"))
                record.update(prefix_metrics(after, "registered_"))
                record.update(
                    {
                        "improvement_dice_mean_fg": after["dice_mean_fg"] - before["dice_mean_fg"],
                        "improvement_dice_mean_all": after["dice_mean_all"] - before["dice_mean_all"],
                    }
                )
                records.append(record)

        summary = aggregate_metrics(records)
        prefixed_summary = {f"iter_eval/{k}": v for k, v in summary.items()}
        pl_module._log_metrics(prefixed_summary, step=trainer.global_step)

        report_path = self.output_dir / f"epoch_{current_epoch + 1:04d}.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        print(
            f"[iter-eval] epoch={current_epoch + 1} "
            f"pairs={summary.get('num_pairs', len(records))} "
            f"id_dice_fg={summary.get('identity_dice_mean_fg_mean', float('nan')):.4f} "
            f"reg_dice_fg={summary.get('registered_dice_mean_fg_mean', float('nan')):.4f} "
            f"impr_dice_fg={summary.get('improvement_dice_mean_fg_mean', float('nan')):.4f} "
            f"report={report_path}"
        )
        self._maybe_log_visualization(trainer, pl_module, device, autocast_enabled)

        pl_module.train()
        trainer.strategy.barrier("iterative_eval_wait")
