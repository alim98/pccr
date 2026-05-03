from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AugmentedRealPairDataset(Dataset):
    """Thin augmentation wrapper around any real-pair OASIS dataset.

    Applied on-the-fly per sample:
      - random gamma/gain/bias intensity perturbation (both volumes, independent)
      - Gaussian noise
      - small random affine: ±trans_vox translation, ±rot_deg rotation (isotropic)

    Labels are transformed with nearest-neighbour mode via the same affine grid
    so dice-relevant structure positions are correctly perturbed.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        gamma_range: tuple[float, float] = (0.8, 1.2),
        gain_range: tuple[float, float] = (0.9, 1.1),
        bias_std: float = 0.02,
        noise_std: float = 0.02,
        trans_vox: float = 5.0,
        rot_deg: float = 3.0,
        augment: bool = True,
    ) -> None:
        self.base = base_dataset
        self.gamma_range = gamma_range
        self.gain_range = gain_range
        self.bias_std = bias_std
        self.noise_std = noise_std
        self.trans_vox = trans_vox
        self.rot_deg = rot_deg
        self.augment = augment

        for attr in ("native_shape", "eval_label_ids", "num_labels"):
            if hasattr(base_dataset, attr):
                setattr(self, attr, getattr(base_dataset, attr))

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _rand(lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand(1).item()

    def _augment_intensity(self, image: torch.Tensor) -> torch.Tensor:
        gamma = self._rand(*self.gamma_range)
        gain = self._rand(*self.gain_range)
        bias = self.bias_std * torch.randn(1).item()
        out = image.clamp(0.0, 1.0).pow(gamma) * gain + bias
        if self.noise_std > 0.0:
            out = out + self.noise_std * torch.randn_like(out)
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _rotation_matrix_3d(rx: float, ry: float, rz: float) -> torch.Tensor:
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
        return Rz @ Ry @ Rx

    def _random_affine_theta(self, spatial_shape: Sequence[int]) -> torch.Tensor:
        deg = math.pi / 180.0
        rx = self._rand(-self.rot_deg, self.rot_deg) * deg
        ry = self._rand(-self.rot_deg, self.rot_deg) * deg
        rz = self._rand(-self.rot_deg, self.rot_deg) * deg
        R = self._rotation_matrix_3d(rx, ry, rz)
        H, W, D = spatial_shape
        tx = self._rand(-self.trans_vox, self.trans_vox) / (H / 2.0)
        ty = self._rand(-self.trans_vox, self.trans_vox) / (W / 2.0)
        tz = self._rand(-self.trans_vox, self.trans_vox) / (D / 2.0)
        t = torch.tensor([[tx], [ty], [tz]], dtype=torch.float32)
        theta = torch.cat([R, t], dim=1).unsqueeze(0)
        return theta

    def _apply_affine(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img5 = image.unsqueeze(0)
        grid = F.affine_grid(theta, img5.shape, align_corners=False)
        warped_img = F.grid_sample(img5, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        lbl5 = label.float().unsqueeze(0)
        warped_lbl = F.grid_sample(lbl5, grid, mode="nearest", align_corners=False, padding_mode="zeros")
        return warped_img.squeeze(0), warped_lbl.long().squeeze(0)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if not self.augment:
            return sample

        source, target = sample[0], sample[1]
        source_label, target_label = sample[2], sample[3]
        rest = sample[4:]

        spatial_shape = source.shape[-3:]

        theta_src = self._random_affine_theta(spatial_shape)
        theta_tgt = self._random_affine_theta(spatial_shape)

        source, source_label = self._apply_affine(source, source_label, theta_src)
        target, target_label = self._apply_affine(target, target_label, theta_tgt)

        source = self._augment_intensity(source)
        target = self._augment_intensity(target)

        return (source, target, source_label, target_label, *rest)
