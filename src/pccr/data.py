from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.datasets import discover_oasis_subjects, get_dataloader, normalize_image
from src.model.transformation import SpatialTransformer
from src.pccr.utils import normalize_grid, voxel_grid


class OASISSingleSubjectDataset(Dataset):
    def __init__(self, data_root: str, input_dim: list[int]) -> None:
        self.subjects = list(discover_oasis_subjects(Path(data_root)).values())
        self.input_dim = tuple(input_dim)
        self.num_labels = max(
            int(np.asarray(nib.load(str(subject.label_path)).dataobj).max())
            for subject in self.subjects
        ) + 1

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int):
        subject = self.subjects[index]
        image = np.asarray(nib.load(str(subject.image_path)).dataobj)
        label = np.asarray(nib.load(str(subject.label_path)).dataobj)
        image = normalize_image(image).astype(np.float32)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        label_tensor = torch.from_numpy(label.astype(np.int64)).unsqueeze(0)
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=self.input_dim,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        label_tensor = F.interpolate(
            label_tensor.float().unsqueeze(0),
            size=self.input_dim,
            mode="nearest",
        ).squeeze(0).long()
        return image_tensor, label_tensor


class SyntheticOASISPairDataset(Dataset):
    def __init__(self, data_root: str, input_dim: list[int], warp_scale: float, control_grid: list[int]) -> None:
        self.base_dataset = OASISSingleSubjectDataset(data_root, input_dim)
        self.input_dim = tuple(input_dim)
        self.warp_scale = warp_scale
        self.control_grid = tuple(control_grid)
        self.transformer = SpatialTransformer(self.input_dim)
        self.nearest_transformer = SpatialTransformer(self.input_dim, mode="nearest")
        self.num_labels = self.base_dataset.num_labels

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _random_flow(self, image: torch.Tensor) -> torch.Tensor:
        flow = torch.randn(1, 3, *self.control_grid, dtype=image.dtype) * self.warp_scale
        flow = F.interpolate(flow, size=self.input_dim, mode="trilinear", align_corners=False)
        return flow

    def _augment_intensity(self, image: torch.Tensor) -> torch.Tensor:
        gain = torch.empty(1).uniform_(0.9, 1.1).item()
        bias = torch.empty(1).uniform_(-0.05, 0.05).item()
        gamma = torch.empty(1).uniform_(0.85, 1.15).item()

        augmented = image.clamp(0.0, 1.0).pow(gamma)
        augmented = augmented * gain + bias
        augmented = augmented + 0.02 * torch.randn_like(augmented)
        if torch.rand(1).item() < 0.3:
            augmented = F.avg_pool3d(augmented, kernel_size=3, stride=1, padding=1)
        return augmented.clamp(0.0, 1.0)

    @staticmethod
    def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
        dz = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dx = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]

        grad = image.new_zeros(image.shape)
        grad[:, :, 1:, :, :] += dz.abs()
        grad[:, :, :-1, :, :] += dz.abs()
        grad[:, :, :, 1:, :] += dy.abs()
        grad[:, :, :, :-1, :] += dy.abs()
        grad[:, :, :, :, 1:] += dx.abs()
        grad[:, :, :, :, :-1] += dx.abs()
        return grad

    def _matchability_targets(self, image: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        gradients = self._gradient_magnitude(image)
        valid_values = gradients[valid_mask > 0.5]
        threshold = torch.quantile(valid_values, 0.2) if valid_values.numel() > 0 else gradients.new_tensor(0.0)

        targets = torch.zeros_like(valid_mask, dtype=torch.long)
        targets[valid_mask <= 0.5] = 2
        ambiguous = (valid_mask > 0.5) & (gradients <= threshold)
        targets[ambiguous] = 1
        return targets

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)

        source_flow = self._random_flow(image)
        target_flow = self._random_flow(image)

        source = self._augment_intensity(self.transformer(image, source_flow)).squeeze(0)
        target = self._augment_intensity(self.transformer(image, target_flow)).squeeze(0)
        source_label = self.nearest_transformer(label.float(), source_flow).long().squeeze(0)
        target_label = self.nearest_transformer(label.float(), target_flow).long().squeeze(0)

        base_grid = voxel_grid(self.input_dim, image.device)
        canonical_source = normalize_grid(base_grid + source_flow)
        canonical_target = normalize_grid(base_grid + target_flow)

        support = torch.ones_like(image)
        valid_source = (self.nearest_transformer(support, source_flow) > 0.5).long()
        valid_target = (self.nearest_transformer(support, target_flow) > 0.5).long()
        matchability_source = self._matchability_targets(source.unsqueeze(0), valid_source)
        matchability_target = self._matchability_targets(target.unsqueeze(0), valid_target)

        return (
            source,
            target,
            source_label,
            target_label,
            canonical_source.squeeze(0),
            canonical_target.squeeze(0),
            valid_source.squeeze(0),
            valid_target.squeeze(0),
            matchability_source.squeeze(0),
            matchability_target.squeeze(0),
        )


def create_real_pair_dataloaders(args, config):
    train_loader = get_dataloader(
        data_path=args.train_data_path,
        input_dim=config.data_size,
        batch_size=args.batch_size,
        shuffle=True,
        is_pair=False,
        num_workers=args.num_workers,
        dataset_format=args.dataset_format,
        split="train",
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        num_steps=args.train_num_steps,
        max_subjects=args.max_train_subjects,
    )
    val_loader = get_dataloader(
        data_path=args.val_data_path,
        input_dim=config.data_size,
        batch_size=args.batch_size,
        shuffle=False,
        is_pair=True,
        num_workers=args.num_workers,
        dataset_format=args.dataset_format,
        split="val",
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_subjects=args.max_val_subjects,
        max_pairs=args.max_val_pairs,
    )
    return train_loader, val_loader


def create_synthetic_dataloader(args, config):
    dataset = SyntheticOASISPairDataset(
        data_root=args.train_data_path,
        input_dim=config.data_size,
        warp_scale=config.synthetic_warp_scale,
        control_grid=config.synthetic_control_grid,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
