from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PCCRConfig:
    data_size: list[int] = field(default_factory=lambda: [40, 48, 56])
    in_channels: int = 1
    stage_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 3
    descriptor_dim: int = 32
    context_dim: int = 128
    matcher_temperature: float = 0.07
    matcher_topk: int = 24
    canonical_radius: float = 0.45
    pointmap_stage_ids: list[int] = field(default_factory=lambda: [2, 3])
    decoder_stage_ids: list[int] = field(default_factory=lambda: [3, 2, 1, 0])
    svf_integration_steps: int = 7
    decoder_channels: int = 64
    max_velocity: float = 1.0
    num_matchability_classes: int = 3
    use_fine_local_refinement: bool = True
    use_final_residual_refinement: bool = False
    final_refinement_hidden_channels: int = 32
    final_refinement_max_velocity: float = 0.2
    image_loss: str = "lncc"
    multiscale_similarity_factors: list[int] = field(default_factory=lambda: [1])
    multiscale_similarity_weights: list[float] = field(default_factory=lambda: [1.0])
    image_loss_weight: float = 1.0
    segmentation_supervision_weight: float = 0.0
    smoothness_weight: float = 0.02
    jacobian_weight: float = 0.01
    inverse_consistency_weight: float = 0.1
    correspondence_weight: float = 0.2
    residual_velocity_weight: float = 0.0
    matchability_score_mode: str = "legacy"
    matchability_score_power: float = 0.5
    refinement_warmup_epochs: int = 0
    encoder_lr_scale: float = 1.0
    canonical_head_lr_scale: float = 1.0
    coarse_decoder_lr_scale: float = 1.0
    residual_refinement_lr_scale: float = 1.0
    synthetic_warp_scale: float = 2.5
    synthetic_control_grid: list[int] = field(default_factory=lambda: [5, 6, 7])
    phase: str = "real"
    num_labels: int = 86

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PCCRConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
