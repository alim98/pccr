from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


def flatten_spatial(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=2).transpose(1, 2).contiguous()


def unflatten_spatial(x: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    batch, voxels, channels = x.shape
    return x.transpose(1, 2).contiguous().view(batch, channels, *spatial_shape)


@lru_cache(maxsize=32)
def _cached_grid(spatial_shape: tuple[int, int, int], device_type: str) -> torch.Tensor:
    z, y, x = spatial_shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, dtype=torch.float32),
        torch.arange(y, dtype=torch.float32),
        torch.arange(x, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((zz, yy, xx), dim=0).unsqueeze(0)


def voxel_grid(spatial_shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    grid = _cached_grid(spatial_shape, device.type).to(device)
    return grid


def normalize_grid(grid: torch.Tensor) -> torch.Tensor:
    spatial_shape = grid.shape[2:]
    out = grid.clone()
    for axis, size in enumerate(spatial_shape):
        if size <= 1:
            out[:, axis] = 0.0
        else:
            out[:, axis] = 2.0 * (out[:, axis] / (size - 1.0) - 0.5)
    return out


def resize_displacement(displacement: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    src_size = displacement.shape[2:]
    if tuple(src_size) == tuple(size):
        return displacement

    resized = F.interpolate(displacement, size=size, mode="trilinear", align_corners=False)
    scale = [
        (size_dim - 1) / max(src_dim - 1, 1)
        for src_dim, size_dim in zip(src_size, size)
    ]
    scale_tensor = displacement.new_tensor(scale).view(1, 3, 1, 1, 1)
    return resized * scale_tensor


def softmax_entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=dim)
