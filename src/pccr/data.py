from __future__ import annotations

import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.datasets import (
    OASISFolderDataset,
    OASISL2RDataset,
    OASIS_PklDataset,
    discover_oasis_l2r_subjects,
    discover_oasis_subjects,
    infer_dataset_format,
    load_oasis_l2r_metadata,
    get_dataloader,
    normalize_image,
)
from src.model.transformation import SpatialTransformer
from src.pccr.config import OASIS_L2R_EVAL_LABEL_IDS
from src.pccr.utils import normalize_grid, voxel_grid


class OASISSingleSubjectDataset(Dataset):
    def __init__(self, data_root: str, input_dim: list[int], dataset_variant: str = "default") -> None:
        data_path = Path(data_root)
        self.mode = "volume"
        self.input_dim = tuple(input_dim)
        if dataset_variant == "pkl" or list(data_path.glob("*.pkl")):
            self.mode = "pkl"
            self.paths = sorted(data_path.glob("*.pkl"))
            if not self.paths:
                raise FileNotFoundError(f"No .pkl subject files found under {data_path}")
            image, label = self._load_pkl_subject(self.paths[0])
            self.native_shape = tuple(int(dim) for dim in image.shape)
            self.eval_label_ids = list(range(1, int(label.max()) + 1))
            self.num_labels = int(label.max()) + 1
            return
        if dataset_variant == "oasis_l2r" or ((data_path / "imagesTr").exists() and (data_path / "labelsTr").exists()):
            metadata = load_oasis_l2r_metadata(data_path)
            self.native_shape = tuple(metadata.native_shape)
            self.eval_label_ids = list(metadata.eval_label_ids)
            self.subjects = list(discover_oasis_l2r_subjects(data_path).values())
        else:
            self.native_shape = tuple(input_dim)
            self.eval_label_ids = []
            self.subjects = list(discover_oasis_subjects(data_path).values())
        self.num_labels = max(
            int(np.asarray(nib.load(str(subject.label_path)).dataobj).max())
            for subject in self.subjects
        ) + 1

    @staticmethod
    def _load_pkl_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise ValueError(f"Synthetic pkl base expects (image, label), got {type(sample).__name__} from {path}")
        image, label = sample
        return np.asarray(image), np.asarray(label)

    def __len__(self) -> int:
        if self.mode == "pkl":
            return len(self.paths)
        return len(self.subjects)

    def __getitem__(self, index: int):
        if self.mode == "pkl":
            image, label = self._load_pkl_subject(self.paths[index])
            image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
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
    def __init__(
        self,
        data_root: str,
        input_dim: list[int],
        warp_scale: float,
        control_grid: list[int],
        dataset_variant: str = "default",
        num_steps: int | None = None,
        deterministic: bool = False,
        base_seed: int = 0,
    ) -> None:
        self.base_dataset = OASISSingleSubjectDataset(data_root, input_dim, dataset_variant=dataset_variant)
        self.input_dim = tuple(self.base_dataset.input_dim)
        self.warp_scale = warp_scale
        self.control_grid = tuple(control_grid)
        self.transformer = SpatialTransformer(self.input_dim)
        self.nearest_transformer = SpatialTransformer(self.input_dim, mode="nearest")
        self.num_labels = self.base_dataset.num_labels
        self.num_steps = int(num_steps) if num_steps is not None and num_steps > 0 else len(self.base_dataset)
        self.deterministic = deterministic
        self.base_seed = int(base_seed)
        self.native_shape = getattr(self.base_dataset, "native_shape", self.input_dim)
        self.eval_label_ids = list(getattr(self.base_dataset, "eval_label_ids", []))

    def __len__(self) -> int:
        return self.num_steps

    def _make_generator(self, index: int) -> torch.Generator | None:
        if not self.deterministic:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + index)
        return generator

    def _random_flow(self, image: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        flow = torch.randn(1, 3, *self.control_grid, dtype=image.dtype, generator=generator) * self.warp_scale
        flow = F.interpolate(flow, size=self.input_dim, mode="trilinear", align_corners=False)
        return flow

    def _augment_intensity(self, image: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        gain = torch.empty(1).uniform_(0.9, 1.1, generator=generator).item()
        bias = torch.empty(1).uniform_(-0.05, 0.05, generator=generator).item()
        gamma = torch.empty(1).uniform_(0.85, 1.15, generator=generator).item()

        augmented = image.clamp(0.0, 1.0).pow(gamma)
        augmented = augmented * gain + bias
        augmented = augmented + 0.02 * torch.randn(
            augmented.shape,
            dtype=augmented.dtype,
            device=augmented.device,
            generator=generator,
        )
        if torch.rand(1, generator=generator).item() < 0.3:
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
        generator = self._make_generator(index)
        image, label = self.base_dataset[index % len(self.base_dataset)]
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)

        source_flow = self._random_flow(image, generator=generator)
        target_flow = self._random_flow(image, generator=generator)

        source = self._augment_intensity(self.transformer(image, source_flow), generator=generator).squeeze(0)
        target = self._augment_intensity(self.transformer(image, target_flow), generator=generator).squeeze(0)
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


def resolve_dataset_variant(args, config, data_path: str | None = None) -> str:
    if getattr(config, "dataset_variant", "default") == "oasis_l2r":
        return "oasis_l2r"
    requested_format = getattr(args, "dataset_format", "auto")
    if requested_format != "auto":
        return requested_format
    if data_path is not None:
        return infer_dataset_format(data_path)
    return "auto"


def resolve_data_root(args, config, split: str) -> str:
    attr_name = "train_data_path" if split == "train" else "val_data_path"
    requested_path = getattr(args, attr_name)
    if getattr(config, "dataset_variant", "default") == "oasis_l2r":
        default_legacy_root = "/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis"
        if requested_path == default_legacy_root:
            return str(Path(config.oasis_l2r_data_root).expanduser())
    return requested_path


def _sync_config_with_dataset_metadata(config, *datasets) -> None:
    native_shape = None
    eval_label_ids = None
    dataset_num_labels = 0
    for dataset in datasets:
        if dataset is None:
            continue
        native_shape = getattr(dataset, "native_shape", native_shape)
        eval_label_ids = getattr(dataset, "eval_label_ids", eval_label_ids)
        dataset_num_labels = max(dataset_num_labels, getattr(dataset, "num_labels", 0) or 0)

    should_align_to_native = bool(getattr(config, "align_data_size_to_native_shape", True))
    if should_align_to_native and native_shape is not None and list(config.data_size) != list(native_shape):
        print(f"[pccr.data] Aligning config.data_size from {config.data_size} to dataset native shape {list(native_shape)}.")
        config.data_size = list(native_shape)
    if eval_label_ids is not None and not config.eval_label_ids:
        config.eval_label_ids = list(eval_label_ids)
    if getattr(config, "dataset_variant", "default") == "oasis_l2r" and not config.eval_label_ids:
        config.eval_label_ids = list(OASIS_L2R_EVAL_LABEL_IDS)
    if dataset_num_labels:
        config.num_labels = dataset_num_labels


def create_real_pair_dataloaders(args, config):
    train_data_path = resolve_data_root(args, config, split="train")
    val_data_path = resolve_data_root(args, config, split="val")
    dataset_variant = resolve_dataset_variant(args, config, data_path=train_data_path)
    train_loader = get_dataloader(
        data_path=train_data_path,
        input_dim=config.data_size,
        batch_size=args.batch_size,
        shuffle=True,
        is_pair=False,
        num_workers=args.num_workers,
        dataset_format=dataset_variant,
        split="train",
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        num_steps=args.train_num_steps,
        max_subjects=args.max_train_subjects,
    )
    val_loader = get_dataloader(
        data_path=val_data_path,
        input_dim=config.data_size,
        batch_size=args.batch_size,
        shuffle=False,
        is_pair=True,
        num_workers=args.num_workers,
        dataset_format=dataset_variant,
        split="val",
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_subjects=args.max_val_subjects,
        max_pairs=args.max_val_pairs,
    )
    _sync_config_with_dataset_metadata(config, train_loader.dataset, val_loader.dataset)
    return train_loader, val_loader


def create_synthetic_dataloader(args, config):
    train_data_path = resolve_data_root(args, config, split="train")
    dataset_variant = resolve_dataset_variant(args, config, data_path=train_data_path)
    dataset = SyntheticOASISPairDataset(
        data_root=train_data_path,
        input_dim=config.data_size,
        warp_scale=config.synthetic_warp_scale,
        control_grid=config.synthetic_control_grid,
        dataset_variant=dataset_variant,
        num_steps=args.train_num_steps,
    )
    _sync_config_with_dataset_metadata(config, dataset)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )


def create_synthetic_pair_dataloaders(args, config):
    train_data_path = resolve_data_root(args, config, split="train")
    val_data_path = resolve_data_root(args, config, split="val")
    train_dataset_variant = resolve_dataset_variant(args, config, data_path=train_data_path)
    val_dataset_variant = resolve_dataset_variant(args, config, data_path=val_data_path)
    train_dataset = SyntheticOASISPairDataset(
        data_root=train_data_path,
        input_dim=config.data_size,
        warp_scale=config.synthetic_warp_scale,
        control_grid=config.synthetic_control_grid,
        dataset_variant=train_dataset_variant,
        num_steps=args.train_num_steps,
        deterministic=False,
    )
    val_steps = args.max_val_pairs if args.max_val_pairs > 0 else min(max(args.train_num_steps // 2, 20), 200)
    val_dataset = SyntheticOASISPairDataset(
        data_root=val_data_path,
        input_dim=config.data_size,
        warp_scale=config.synthetic_warp_scale,
        control_grid=config.synthetic_control_grid,
        dataset_variant=val_dataset_variant,
        num_steps=val_steps,
        deterministic=True,
        base_seed=args.split_seed,
    )
    _sync_config_with_dataset_metadata(config, train_dataset, val_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def build_real_pair_dataset(
    config,
    data_path: str,
    input_dim: list[int],
    dataset_format: str,
    split: str,
    val_fraction: float,
    split_seed: int,
    max_subjects: int = 0,
    max_pairs: int = 0,
):
    resolved_format = (
        getattr(config, "dataset_variant", "default")
        if getattr(config, "dataset_variant", "default") != "default"
        else dataset_format
    )
    resolved_format = resolved_format if resolved_format != "auto" else infer_dataset_format(data_path)
    if resolved_format == "pkl":
        dataset = OASIS_PklDataset(
            input_dim=input_dim,
            data_path=data_path,
            num_steps=max_pairs if max_pairs > 0 else 1000,
            is_pair=True,
        )
        if max_pairs > 0:
            dataset.paths = sorted(dataset.paths)[:max_pairs]
        return dataset

    if resolved_format == "oasis_fs":
        return OASISFolderDataset(
            input_dim=input_dim,
            data_path=data_path,
            split=split,
            is_pair=True,
            val_fraction=val_fraction,
            split_seed=split_seed,
            max_subjects=max_subjects,
            max_pairs=max_pairs,
        )

    if resolved_format == "oasis_l2r":
        return OASISL2RDataset(
            input_dim=input_dim,
            data_path=data_path,
            split=split,
            is_pair=True,
            val_fraction=val_fraction,
            split_seed=split_seed,
            max_subjects=max_subjects,
            max_pairs=max_pairs,
        )

    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def create_overfit_pair_dataloaders(args, config):
    val_data_path = resolve_data_root(args, config, split="val")
    dataset = build_real_pair_dataset(
        config=config,
        data_path=val_data_path,
        input_dim=config.data_size,
        dataset_format=args.dataset_format,
        split=args.overfit_split,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_subjects=args.max_val_subjects,
        max_pairs=args.overfit_num_pairs,
    )
    _sync_config_with_dataset_metadata(config, dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader
