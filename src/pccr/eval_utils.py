from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from src.loss import DiceScore
from src.model.transformation import SpatialTransformer
from src.utils import get_one_hot


def resolve_eval_label_ids(num_labels: int, label_ids: list[int] | None = None) -> list[int]:
    if label_ids:
        return [label_id for label_id in label_ids if 0 <= int(label_id) < num_labels]
    return list(range(1, num_labels))


def dice_score_array(
    warped_onehot: torch.Tensor,
    target_label: torch.Tensor,
    num_labels: int,
) -> np.ndarray:
    return DiceScore(warped_onehot, target_label.long(), num_labels).detach().cpu().numpy()


def per_label_dice_statistics(
    warped_onehot: torch.Tensor,
    target_label: torch.Tensor,
    num_labels: int,
    label_ids: list[int] | None = None,
) -> dict[int, float]:
    scores = dice_score_array(warped_onehot, target_label, num_labels)
    return {
        int(label_id): float(scores[:, int(label_id)].mean())
        for label_id in resolve_eval_label_ids(num_labels, label_ids)
    }


def warp_segmentation(
    label: torch.Tensor,
    displacement: torch.Tensor,
    num_labels: int,
    transformer: SpatialTransformer,
) -> tuple[torch.Tensor, torch.Tensor]:
    label_onehot = get_one_hot(label, num_labels)
    warped_channels = []
    for channel_id in range(num_labels):
        warped_channels.append(
            transformer(
                label_onehot[:, channel_id : channel_id + 1].float(),
                displacement.float(),
            )
        )
    warped_onehot = torch.cat(warped_channels, dim=1)
    warped_label = warped_onehot.argmax(dim=1, keepdim=True)
    return warped_onehot, warped_label


def dice_statistics(
    warped_onehot: torch.Tensor,
    target_label: torch.Tensor,
    num_labels: int,
    label_ids: list[int] | None = None,
) -> dict[str, float]:
    scores = dice_score_array(warped_onehot, target_label, num_labels)
    selected_label_ids = resolve_eval_label_ids(num_labels, label_ids)
    foreground = scores[:, selected_label_ids] if selected_label_ids else (scores[:, 1:] if scores.shape[1] > 1 else scores)
    return {
        "dice_mean_all": float(scores.mean()),
        "dice_mean_fg": float(foreground.mean()) if foreground.size else float(scores.mean()),
    }


def label_dice_statistics(
    predicted_label: torch.Tensor,
    target_label: torch.Tensor,
    num_labels: int,
    label_ids: list[int] | None = None,
) -> dict[str, float]:
    predicted_onehot = get_one_hot(predicted_label.long(), num_labels).float()
    return dice_statistics(predicted_onehot, target_label, num_labels, label_ids=label_ids)


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    if not mask_a.any() or not mask_b.any():
        return np.asarray([], dtype=np.float32)

    structure = generate_binary_structure(mask_a.ndim, 1)
    surface_a = np.logical_xor(mask_a, binary_erosion(mask_a, structure=structure, border_value=0))
    surface_b = np.logical_xor(mask_b, binary_erosion(mask_b, structure=structure, border_value=0))

    if not surface_a.any() or not surface_b.any():
        return np.asarray([], dtype=np.float32)

    dt_a = distance_transform_edt(~surface_a)
    dt_b = distance_transform_edt(~surface_b)
    distances_ab = dt_b[surface_a]
    distances_ba = dt_a[surface_b]
    return np.concatenate([distances_ab, distances_ba]).astype(np.float32)


def hd95_statistics(
    predicted_label: torch.Tensor,
    target_label: torch.Tensor,
    num_labels: int,
    label_ids: list[int] | None = None,
) -> dict[str, float]:
    pred = predicted_label.squeeze(0).squeeze(0).detach().cpu().numpy()
    target = target_label.squeeze(0).squeeze(0).detach().cpu().numpy()

    hd95_values = []
    for label_id in resolve_eval_label_ids(num_labels, label_ids):
        distances = _surface_distances(pred == label_id, target == label_id)
        if distances.size > 0:
            hd95_values.append(float(np.percentile(distances, 95)))

    return {
        "hd95_mean_fg": float(np.mean(hd95_values)) if hd95_values else math.nan,
        "hd95_median_fg": float(np.median(hd95_values)) if hd95_values else math.nan,
    }


def jacobian_determinant(displacement: torch.Tensor) -> torch.Tensor:
    dx = displacement[:, :, 1:, :-1, :-1] - displacement[:, :, :-1, :-1, :-1]
    dy = displacement[:, :, :-1, 1:, :-1] - displacement[:, :, :-1, :-1, :-1]
    dz = displacement[:, :, :-1, :-1, 1:] - displacement[:, :, :-1, :-1, :-1]

    jac = torch.zeros(
        displacement.shape[0],
        3,
        3,
        *dx.shape[2:],
        device=displacement.device,
        dtype=displacement.dtype,
    )
    jac[:, 0, 0] = 1 + dx[:, 0]
    jac[:, 0, 1] = dy[:, 0]
    jac[:, 0, 2] = dz[:, 0]
    jac[:, 1, 0] = dx[:, 1]
    jac[:, 1, 1] = 1 + dy[:, 1]
    jac[:, 1, 2] = dz[:, 1]
    jac[:, 2, 0] = dx[:, 2]
    jac[:, 2, 1] = dy[:, 2]
    jac[:, 2, 2] = 1 + dz[:, 2]
    return torch.det(jac.permute(0, 3, 4, 5, 1, 2))


def jacobian_statistics(displacement: torch.Tensor) -> dict[str, float]:
    det = jacobian_determinant(displacement.float()).detach().cpu()
    det_np = det.numpy()
    nonpositive_fraction = float((det_np <= 0).mean())
    clipped = np.clip(det_np, 1e-6, None)
    logj = np.log(clipped)
    return {
        "sdlogj": float(np.std(logj)),
        "jacobian_nonpositive_fraction": nonpositive_fraction,
        "jacobian_min": float(det_np.min()),
        "jacobian_mean": float(det_np.mean()),
    }


def identity_metrics(
    source_label: torch.Tensor,
    target_label: torch.Tensor,
    num_labels: int,
    label_ids: list[int] | None = None,
    inference_seconds: float = 0.0,
    include_hd95: bool = True,
) -> dict[str, float]:
    metrics = {}
    metrics.update(label_dice_statistics(source_label, target_label, num_labels, label_ids=label_ids))
    if include_hd95:
        metrics.update(hd95_statistics(source_label, target_label, num_labels, label_ids=label_ids))
    else:
        metrics.update({"hd95_mean_fg": math.nan, "hd95_median_fg": math.nan})
    metrics.update(
        {
            "sdlogj": 0.0,
            "jacobian_nonpositive_fraction": 0.0,
            "jacobian_min": 1.0,
            "jacobian_mean": 1.0,
            "runtime_seconds": float(inference_seconds),
        }
    )
    return metrics


def measure_inference_time(model, source: torch.Tensor, target: torch.Tensor, use_cuda: bool) -> tuple[dict[str, object], float]:
    if use_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(source, target)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return outputs, elapsed


def pair_metrics(
    outputs: dict[str, object],
    source_label: torch.Tensor,
    target_label: torch.Tensor,
    num_labels: int,
    transformer: SpatialTransformer,
    inference_seconds: float,
    include_hd95: bool = True,
    label_ids: list[int] | None = None,
) -> dict[str, float]:
    displacement = outputs["phi_s2t"]
    warped_onehot, warped_label = warp_segmentation(source_label, displacement, num_labels, transformer)

    metrics = {}
    metrics.update(dice_statistics(warped_onehot, target_label, num_labels, label_ids=label_ids))
    if include_hd95:
        metrics.update(hd95_statistics(warped_label, target_label, num_labels, label_ids=label_ids))
    else:
        metrics.update({"hd95_mean_fg": math.nan, "hd95_median_fg": math.nan})
    metrics.update(jacobian_statistics(displacement))
    metrics["runtime_seconds"] = float(inference_seconds)
    return metrics


def aggregate_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(records)
    summary = {}
    for column in frame.columns:
        values = frame[column].astype(float)
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_std"] = float(values.std(ddof=0))
    summary["num_pairs"] = int(len(records))
    return summary


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def save_metrics_report(
    output_dir: str | Path,
    summary: dict[str, float],
    per_pair_records: list[dict[str, float]],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    pairs_path = output_dir / "per_pair_metrics.csv"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    pd.DataFrame(per_pair_records).to_csv(pairs_path, index=False)
    return summary_path, pairs_path
