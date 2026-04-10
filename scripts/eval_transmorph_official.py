#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import warnings
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, correlate, distance_transform_edt, generate_binary_structure
from scipy.ndimage import map_coordinates, zoom

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSMORPH_ROOT = REPO_ROOT / "TransMorph_Transformer_for_Medical_Image_Registration" / "OASIS" / "TransMorph"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(TRANSMORPH_ROOT) not in sys.path:
    sys.path.append(str(TRANSMORPH_ROOT))

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated",
    category=FutureWarning,
)

try:
    import ml_collections  # type: ignore  # noqa: F401
except ImportError:
    class _ConfigDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, key):
            del self[key]

    ml_collections_module = types.ModuleType("ml_collections")
    ml_collections_module.ConfigDict = _ConfigDict
    sys.modules["ml_collections"] = ml_collections_module

from models.TransMorph import CONFIGS as CONFIGS_TM  # type: ignore  # noqa: E402
import models.TransMorph as OfficialTransMorph  # type: ignore  # noqa: E402


VOI_LABELS = list(range(1, 36))


class OfficialLikeSpatialTransformer(torch.nn.Module):
    def __init__(self, size: tuple[int, int, int], mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing="ij")
        grid = torch.stack(grids).unsqueeze(0).float()
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for dim in range(len(shape)):
            new_locs[:, dim, ...] = 2 * (new_locs[:, dim, ...] / (shape[dim] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an official TransMorph OASIS checkpoint with both direct validation Dice and challenge-style metrics."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=REPO_ROOT / "transmorph_large_checkpoint" / "TransMorphLarge_Validation_dsc0.8623.pth.tar",
        help="Path to the official TransMorph checkpoint (.pth.tar).",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=Path("/nexus/posix0/MBR-neuralsystems/alim/regdata/oasistransmorph/OASIS_L2R_2021_task03/Test"),
        help="Path to the TransMorph OASIS validation/test .pkl pairs directory.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="TransMorph-Large",
        choices=sorted(CONFIGS_TM.keys()),
        help="Official TransMorph config variant.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Inference precision. Use fp32 if you want the closest match to the published metric.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="If > 0, only evaluate the first N sorted validation pairs.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=5,
        help="Print progress every N pairs.",
    )
    parser.add_argument(
        "--skip_challenge_eval",
        action="store_true",
        help="Only compute the direct official validation Dice and skip the challenge-style metrics.",
    )
    parser.add_argument(
        "--submission_dir",
        type=Path,
        default=None,
        help="Optional directory to save the half-resolution float16 displacement fields as disp_XXXX_YYYY.npz.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults under results/transmorph_official/.",
    )
    parser.add_argument(
        "--checkpoint_info_only",
        action="store_true",
        help="Load and inspect the checkpoint metadata, then exit without running inference.",
    )
    return parser.parse_args()


def resolve_device(accelerator: str) -> torch.device:
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--accelerator gpu was requested but CUDA is not available.")
        return torch.device("cuda")
    if accelerator == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`",
            category=FutureWarning,
        )
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    return payload


def parse_expected_dice_from_checkpoint_name(checkpoint_path: Path) -> float | None:
    match = re.search(r"dsc([0-9]*\.[0-9]+)", checkpoint_path.name)
    if match:
        return float(match.group(1))
    return None


def load_pair(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as handle:
        moving, fixed, moving_label, fixed_label = pickle.load(handle)
    return (
        np.asarray(moving, dtype=np.float32),
        np.asarray(fixed, dtype=np.float32),
        np.asarray(moving_label),
        np.asarray(fixed_label),
    )


def dice_val_voi(predicted: np.ndarray, target: np.ndarray) -> float:
    scores = []
    for label_id in VOI_LABELS:
        pred_mask = predicted == label_id
        target_mask = target == label_id
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)
        scores.append((2.0 * intersection) / (union + 1e-5))
    return float(np.mean(scores))


def compute_dice_skip_empty(predicted: np.ndarray, target: np.ndarray) -> float:
    scores = []
    for label_id in VOI_LABELS:
        pred_mask = predicted == label_id
        target_mask = target == label_id
        if pred_mask.sum() == 0 or target_mask.sum() == 0:
            continue
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)
        scores.append((2.0 * intersection) / (union + 1e-5))
    return float(np.mean(scores)) if scores else math.nan


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
    return np.concatenate([dt_b[surface_a], dt_a[surface_b]]).astype(np.float32)


def robust_hd95_skip_empty(predicted: np.ndarray, target: np.ndarray) -> float:
    hd95_values = []
    for label_id in VOI_LABELS:
        pred_mask = predicted == label_id
        target_mask = target == label_id
        if pred_mask.sum() == 0 or target_mask.sum() == 0:
            continue
        distances = _surface_distances(pred_mask, target_mask)
        if distances.size:
            hd95_values.append(float(np.percentile(distances, 95)))
    return float(np.mean(hd95_values)) if hd95_values else math.nan


def official_jacobian_determinant(disp: np.ndarray) -> np.ndarray:
    disp = disp[np.newaxis, ...]
    gradx = np.array([-0.5, 0.0, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0.0, 0.5], dtype=np.float32).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0.0, 0.5], dtype=np.float32).reshape(1, 1, 1, 3)

    gradx_disp = np.stack(
        [correlate(disp[:, channel], gradx, mode="constant", cval=0.0) for channel in range(3)],
        axis=1,
    )
    grady_disp = np.stack(
        [correlate(disp[:, channel], grady, mode="constant", cval=0.0) for channel in range(3)],
        axis=1,
    )
    gradz_disp = np.stack(
        [correlate(disp[:, channel], gradz, mode="constant", cval=0.0) for channel in range(3)],
        axis=1,
    )

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], axis=0)
    jacobian = grad_disp + np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = (
        jacobian[0, 0] * (jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1])
        - jacobian[1, 0] * (jacobian[0, 1] * jacobian[2, 2] - jacobian[0, 2] * jacobian[2, 1])
        + jacobian[2, 0] * (jacobian[0, 1] * jacobian[1, 2] - jacobian[0, 2] * jacobian[1, 1])
    )
    return jacdet


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
    }


def save_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def main() -> None:
    args = parse_args()
    checkpoint_payload = load_checkpoint_payload(args.checkpoint_path)
    expected_dice = parse_expected_dice_from_checkpoint_name(args.checkpoint_path)

    checkpoint_summary = {
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "checkpoint_best_dsc": float(checkpoint_payload.get("best_dsc", math.nan)),
        "checkpoint_name_dsc": expected_dice,
        "variant": args.variant,
    }

    print(json.dumps(checkpoint_summary, indent=2))
    if args.checkpoint_info_only:
        return

    pair_paths = sorted(args.test_dir.glob("*.pkl"))
    if not pair_paths:
        raise FileNotFoundError(f"No .pkl files found under {args.test_dir}")
    if args.max_pairs > 0:
        pair_paths = pair_paths[: args.max_pairs]

    device = resolve_device(args.accelerator)
    dtype = resolve_dtype(args.precision, device)
    config = CONFIGS_TM[args.variant]
    model = OfficialTransMorph.TransMorph(config)
    model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    model.eval().to(device=device)
    warp_nearest = OfficialLikeSpatialTransformer(tuple(config.img_size), mode="nearest").to(device=device)

    autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    autocast_dtype = dtype if autocast_enabled else torch.float32

    if args.submission_dir is not None:
        args.submission_dir.mkdir(parents=True, exist_ok=True)

    direct_records: list[dict[str, float | str]] = []
    challenge_records: list[dict[str, float | str]] = []

    print(
        f"Evaluating {args.variant} on {len(pair_paths)} pairs "
        f"device={device.type} precision={args.precision} test_dir={args.test_dir}"
    )

    with torch.no_grad():
        for index, pair_path in enumerate(pair_paths, start=1):
            moving, fixed, moving_label, fixed_label = load_pair(pair_path)
            moving_tensor = torch.from_numpy(moving[None, None, ...]).to(device=device, dtype=torch.float32)
            fixed_tensor = torch.from_numpy(fixed[None, None, ...]).to(device=device, dtype=torch.float32)
            moving_label_tensor = torch.from_numpy(moving_label[None, None, ...]).to(device=device, dtype=torch.float32)

            model_input = torch.cat((moving_tensor, fixed_tensor), dim=1)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                _, flow = model(model_input)

            warped_label = warp_nearest(moving_label_tensor, flow.float()).long().cpu().numpy()[0, 0]
            direct_dice = dice_val_voi(warped_label, fixed_label)

            pair_id = pair_path.stem[2:]
            direct_record = {
                "pair_id": pair_id,
                "direct_validation_dice": float(direct_dice),
            }
            direct_records.append(direct_record)

            if not args.skip_challenge_eval:
                flow_np = flow.float().cpu().numpy()[0]
                flow_half = np.stack([zoom(flow_np[channel], 0.5, order=2) for channel in range(3)], axis=0).astype(np.float16)
                if args.submission_dir is not None:
                    np.savez(args.submission_dir / f"disp_{pair_id}.npz", flow_half)

                flow_eval = np.stack(
                    [zoom(flow_half[channel].astype(np.float32), 2.0, order=2) for channel in range(3)],
                    axis=0,
                ).astype(np.float32)
                identity = np.meshgrid(
                    np.arange(fixed_label.shape[0]),
                    np.arange(fixed_label.shape[1]),
                    np.arange(fixed_label.shape[2]),
                    indexing="ij",
                )
                warped_challenge = map_coordinates(moving_label, identity + flow_eval, order=0)
                jac_det = (official_jacobian_determinant(flow_eval) + 3.0).clip(1e-9, 1e9)
                log_jac_det = np.log(jac_det)

                challenge_record = {
                    "pair_id": pair_id,
                    "challenge_dice": float(compute_dice_skip_empty(warped_challenge, fixed_label)),
                    "challenge_hd95": float(robust_hd95_skip_empty(warped_challenge, fixed_label)),
                    "challenge_sdlogj": float(log_jac_det.std()),
                    "challenge_nonpositive_jacobian_percent": float((jac_det <= 0).mean() * 100.0),
                }
                challenge_records.append(challenge_record)

            if index % max(1, args.progress_every) == 0 or index == len(pair_paths):
                message = f"[{index}/{len(pair_paths)}] direct_dice={direct_dice:.4f}"
                if challenge_records:
                    latest = challenge_records[-1]
                    message += (
                        f" challenge_dice={latest['challenge_dice']:.4f}"
                        f" challenge_sdlogj={latest['challenge_sdlogj']:.4f}"
                    )
                print(message)

    direct_scores = [float(item["direct_validation_dice"]) for item in direct_records]
    direct_summary = summarize(direct_scores)

    payload = {
        "checkpoint": checkpoint_summary,
        "evaluation": {
            "test_dir": str(args.test_dir.resolve()),
            "num_pairs": len(pair_paths),
            "device": device.type,
            "precision": args.precision,
            "direct_validation": {
                "dice": direct_summary,
                "delta_vs_checkpoint_best_dsc": (
                    float(direct_summary["mean"] - checkpoint_summary["checkpoint_best_dsc"])
                    if math.isfinite(checkpoint_summary["checkpoint_best_dsc"])
                    else math.nan
                ),
                "delta_vs_checkpoint_name_dsc": (
                    float(direct_summary["mean"] - checkpoint_summary["checkpoint_name_dsc"])
                    if checkpoint_summary["checkpoint_name_dsc"] is not None
                    else math.nan
                ),
                "per_pair": direct_records,
            },
        },
    }

    if challenge_records:
        payload["evaluation"]["challenge_style"] = {
            "dice": summarize([float(item["challenge_dice"]) for item in challenge_records]),
            "hd95": summarize([float(item["challenge_hd95"]) for item in challenge_records]),
            "sdlogj": summarize([float(item["challenge_sdlogj"]) for item in challenge_records]),
            "nonpositive_jacobian_percent": summarize(
                [float(item["challenge_nonpositive_jacobian_percent"]) for item in challenge_records]
            ),
            "per_pair": challenge_records,
        }

    output_json = args.output_json
    if output_json is None:
        stem = args.checkpoint_path.stem.replace(".pth", "")
        output_json = REPO_ROOT / "results" / "transmorph_official" / f"{stem}.json"
    saved_path = save_json(output_json, payload)

    print(f"output_json: {saved_path}")
    print(
        "direct_validation_dice_mean: "
        f"{payload['evaluation']['direct_validation']['dice']['mean']:.4f} "
        f"(delta_vs_checkpoint_best={payload['evaluation']['direct_validation']['delta_vs_checkpoint_best_dsc']:.4f})"
    )
    if "challenge_style" in payload["evaluation"]:
        challenge = payload["evaluation"]["challenge_style"]
        print(
            "challenge_style: "
            f"dice={challenge['dice']['mean']:.4f} "
            f"sdlogj={challenge['sdlogj']['mean']:.4f} "
            f"hd95={challenge['hd95']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
