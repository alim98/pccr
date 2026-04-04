import glob
import os
import pickle
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def resize_volume(volume: torch.Tensor, spatial_size, mode: str) -> torch.Tensor:
    volume = volume.unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(volume.float(), size=spatial_size, mode=mode)
    else:
        resized = F.interpolate(volume.float(), size=spatial_size, mode=mode, align_corners=False)
    return resized.squeeze(0)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite_mask = np.isfinite(image)
    if not np.any(finite_mask):
        return np.zeros_like(image, dtype=np.float32)

    valid = image[finite_mask]
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    image[~finite_mask] = 0.0
    return image.astype(np.float32)


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(nib.load(str(path)).dataobj)
    if volume.ndim == 4:
        volume = volume[..., 0]
    return volume


class OASIS_PklDataset(Dataset):
    def __init__(self, input_dim, data_path, num_steps=1000, is_pair: bool = False, ext="pkl"):
        self.paths = glob.glob(os.path.join(data_path, f"*.{ext}"))
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.is_pair = is_pair
        self.num_labels = None

    def _pkload(self, filename: str) -> tuple:
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"The file {filename} was not found.") from exc
        except pickle.UnpicklingError as exc:
            raise pickle.UnpicklingError(f"Error unpickling the file {filename}.") from exc

    def _prepare_tensor_pair(self, src, tgt, src_lbl, tgt_lbl):
        src = torch.from_numpy(src).float().unsqueeze(0)
        src_lbl = torch.from_numpy(src_lbl).long().unsqueeze(0)
        tgt = torch.from_numpy(tgt).float().unsqueeze(0)
        tgt_lbl = torch.from_numpy(tgt_lbl).long().unsqueeze(0)

        src = resize_volume(src, self.input_dim, mode="trilinear")
        tgt = resize_volume(tgt, self.input_dim, mode="trilinear")
        src_lbl = resize_volume(src_lbl.float(), self.input_dim, mode="nearest").long()
        tgt_lbl = resize_volume(tgt_lbl.float(), self.input_dim, mode="nearest").long()
        return src, tgt, src_lbl, tgt_lbl

    def __getitem__(self, index):
        if self.is_pair:
            src, tgt, src_lbl, tgt_lbl = self._pkload(self.paths[index])
        else:
            selected_items = random.sample(list(self.paths), 2)
            src, src_lbl = self._pkload(selected_items[0])
            tgt, tgt_lbl = self._pkload(selected_items[1])

        return self._prepare_tensor_pair(src, tgt, src_lbl, tgt_lbl)

    def __len__(self):
        return self.num_steps if not self.is_pair else len(self.paths)


@dataclass(frozen=True)
class OASISSubject:
    subject_id: str
    image_path: Path
    label_path: Path


class OASISFolderDataset(Dataset):
    def __init__(
        self,
        input_dim,
        data_path,
        split: str,
        is_pair: bool,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        num_steps: int = 1000,
        max_subjects: int = 0,
        max_pairs: int = 0,
    ):
        self.input_dim = input_dim
        self.is_pair = is_pair
        self.num_steps = num_steps
        self.split = split

        subjects = discover_oasis_subjects(Path(data_path))
        if len(subjects) < 2:
            raise ValueError(f"Need at least two usable subjects, found {len(subjects)} under {data_path}")

        subject_ids = sorted(subjects)
        rng = random.Random(split_seed)
        rng.shuffle(subject_ids)

        val_count = max(1, int(round(len(subject_ids) * val_fraction)))
        if val_count >= len(subject_ids):
            val_count = len(subject_ids) - 1

        val_ids = sorted(subject_ids[:val_count])
        train_ids = sorted(subject_ids[val_count:])
        split_ids = train_ids if split == "train" else val_ids

        if max_subjects > 0:
            split_ids = split_ids[:max_subjects]

        if len(split_ids) < 2:
            raise ValueError(f"Split '{split}' needs at least two subjects after filtering; got {len(split_ids)}")

        self.subjects = [subjects[subject_id] for subject_id in split_ids]
        self.max_label = int(max(load_volume(subject.label_path).max() for subject in self.subjects))
        self.num_labels = self.max_label + 1

        if self.is_pair:
            self.pairs = list(combinations(self.subjects, 2))
            if max_pairs > 0:
                self.pairs = self.pairs[:max_pairs]
            if not self.pairs:
                raise ValueError(f"Split '{split}' did not produce any validation pairs.")
        else:
            self.pairs = None

    def _load_subject(self, subject: OASISSubject):
        image = normalize_image(load_volume(subject.image_path))
        label = load_volume(subject.label_path).astype(np.int64)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        label_tensor = torch.from_numpy(label).unsqueeze(0)

        image_tensor = resize_volume(image_tensor, self.input_dim, mode="trilinear")
        label_tensor = resize_volume(label_tensor.float(), self.input_dim, mode="nearest").long()
        return image_tensor, label_tensor

    def __getitem__(self, index):
        if self.is_pair:
            src_subject, tgt_subject = self.pairs[index]
        else:
            src_subject, tgt_subject = random.sample(self.subjects, 2)

        src, src_lbl = self._load_subject(src_subject)
        tgt, tgt_lbl = self._load_subject(tgt_subject)
        return src, tgt, src_lbl, tgt_lbl

    def __len__(self):
        return len(self.pairs) if self.is_pair else self.num_steps


def discover_oasis_subjects(data_path: Path) -> dict[str, OASISSubject]:
    seg_base = data_path / "seg" if (data_path / "seg").exists() else data_path
    raw_base = data_path / "raw" if (data_path / "raw").exists() else None

    seg_subject_dirs: dict[str, Path] = {}
    for subject_dir in seg_base.rglob("OAS1_*_MR1"):
        if not subject_dir.is_dir():
            continue
        t1_path = subject_dir / "mri" / "T1.mgz"
        label_path = subject_dir / "mri" / "aseg.mgz"
        if t1_path.exists() and label_path.exists():
            seg_subject_dirs[subject_dir.name] = subject_dir

    if not seg_subject_dirs:
        raise FileNotFoundError(f"No OASIS FreeSurfer subjects with T1.mgz and aseg.mgz found under {data_path}")

    if raw_base is not None and raw_base.exists():
        raw_subject_names = {
            subject_dir.name
            for subject_dir in raw_base.rglob("OAS1_*_MR1")
            if subject_dir.is_dir()
        }
        subject_names = sorted(set(seg_subject_dirs).intersection(raw_subject_names))
    else:
        subject_names = sorted(seg_subject_dirs)

    return {
        subject_id: OASISSubject(
            subject_id=subject_id,
            image_path=seg_subject_dirs[subject_id] / "mri" / "T1.mgz",
            label_path=seg_subject_dirs[subject_id] / "mri" / "aseg.mgz",
        )
        for subject_id in subject_names
    }


def infer_dataset_format(data_path: str) -> str:
    path = Path(data_path)
    if list(path.glob("*.pkl")):
        return "pkl"
    if (path / "seg").exists() or (path / "mri" / "T1.mgz").exists():
        return "oasis_fs"
    raise ValueError(
        f"Could not infer dataset format for {data_path}. "
        "Use a directory with .pkl files or an OASIS root/subject directory."
    )


def get_dataloader(
    data_path,
    input_dim,
    batch_size,
    shuffle: bool = True,
    is_pair: bool = False,
    num_workers: int = 4,
    dataset_format: str = "auto",
    split: str = "train",
    val_fraction: float = 0.2,
    split_seed: int = 42,
    num_steps: int = 1000,
    max_subjects: int = 0,
    max_pairs: int = 0,
):
    resolved_format = infer_dataset_format(data_path) if dataset_format == "auto" else dataset_format

    if resolved_format == "pkl":
        ds = OASIS_PklDataset(
            input_dim=input_dim,
            data_path=data_path,
            num_steps=num_steps,
            is_pair=is_pair,
        )
    elif resolved_format == "oasis_fs":
        ds = OASISFolderDataset(
            input_dim=input_dim,
            data_path=data_path,
            split=split,
            is_pair=is_pair,
            val_fraction=val_fraction,
            split_seed=split_seed,
            num_steps=num_steps,
            max_subjects=max_subjects,
            max_pairs=max_pairs,
        )
    else:
        raise ValueError(f"Unsupported dataset format: {resolved_format}")

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and not is_pair,
        num_workers=num_workers,
    )
    return dataloader
