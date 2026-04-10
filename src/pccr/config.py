from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

OASIS_L2R_NATIVE_SHAPE = [160, 224, 192]
OASIS_L2R_EVAL_LABEL_IDS = list(range(1, 36))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    base_entry = payload.pop("base_config", None)
    if base_entry is None:
        return payload

    base_paths = base_entry if isinstance(base_entry, list) else [base_entry]
    merged: dict[str, Any] = {}
    for base_item in base_paths:
        base_path = Path(base_item)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        merged = _deep_merge(merged, _load_yaml_with_base(base_path))
    return _deep_merge(merged, payload)


@dataclass
class PCCRConfig:
    dataset_variant: str = "default"
    oasis_l2r_data_root: str = "~/data/oasis_l2r"
    eval_label_ids: list[int] = field(default_factory=list)
    align_data_size_to_native_shape: bool = True
    data_size: list[int] = field(default_factory=lambda: [40, 48, 56])
    in_channels: int = 1
    stage_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 3
    descriptor_dim: int = 32
    context_dim: int = 128
    matcher_temperature: float = 0.07
    matcher_topk: int = 24
    canonical_radius: float = 0.45
    matcher_type: str = "canonical_soft"
    matcher_confidence_mode: str = "max_prob"
    refined_matcher_hidden_dim: int = 64
    refined_matcher_offset_scale: float = 0.5
    refined_matcher_output_mode: str = "topm_reweighted"
    refined_matcher_topm: int = 2
    global_match_voxel_limit: int = 50_000
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
    final_refinement_num_blocks: int = 2
    final_refinement_activation: str = "gelu"
    final_refinement_use_image_error_inputs: bool = False
    final_refinement_include_raw_image_error_inputs: bool = False
    final_refinement_image_error_channels: int = 16
    final_refinement_use_error_edges: bool = False
    final_refinement_use_local_cost_volume: bool = False
    final_refinement_cost_volume_radius: int = 0
    final_refinement_cost_volume_proj_channels: int = 16
    final_refinement_cost_volume_feature_channels: int = 16
    final_refinement_memory_efficient_cost_volume: bool = False
    final_refinement_cost_volume_offset_chunk_size: int = 8
    final_refinement_use_local_residual_matcher: bool = False
    final_refinement_local_matcher_radius: int = 0
    final_refinement_local_matcher_proj_channels: int = 16
    final_refinement_local_matcher_feature_channels: int = 16
    final_refinement_local_matcher_temperature: float = 1.0
    diagnostic_residual_only: bool = False
    diagnostic_oracle_correspondence: bool = False
    diagnostic_oracle_handoff: bool = False
    oracle_handoff_gaussian_sigma: float = 1.0
    use_stage1_local_refinement: bool = False
    stage1_local_refinement_radius: int = 0
    stage1_local_refinement_proj_channels: int = 16
    stage1_local_refinement_feature_channels: int = 16
    stage1_local_refinement_temperature: float = 1.0
    use_stage2_local_refinement: bool = False
    stage2_local_refinement_radius: int = 0
    stage2_local_refinement_proj_channels: int = 16
    stage2_local_refinement_feature_channels: int = 16
    stage2_local_refinement_temperature: float = 1.0
    use_amp: bool = True
    use_gradient_checkpointing: bool = False
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    image_loss: str = "lncc"
    lncc_window_size: int = 5
    multiscale_similarity_factors: list[int] = field(default_factory=lambda: [1])
    multiscale_similarity_weights: list[float] = field(default_factory=lambda: [1.0])
    image_loss_weight: float = 1.0
    segmentation_supervision_weight: float = 0.0
    smoothness_weight: float = 0.02
    jacobian_weight: float = 0.01
    inverse_consistency_weight: float = 0.1
    correspondence_weight: float = 0.2
    synthetic_matchability_weight: float = 0.25
    residual_velocity_weight: float = 0.0
    decoder_fitting_weight: float = 0.0
    decoder_fitting_detach_target: bool = True
    decoder_fitting_entropy_threshold: float = -1.0
    decoder_fitting_confidence_percentile: float = 0.0
    decoder_fitting_margin_power: float = 0.0
    decoder_fitting_margin_min: float = 0.0
    matchability_score_mode: str = "legacy"
    matchability_score_power: float = 0.5
    synthetic_refresh_required: bool = False
    refinement_warmup_epochs: int = 0
    encoder_lr_scale: float = 1.0
    canonical_head_lr_scale: float = 1.0
    coarse_decoder_lr_scale: float = 1.0
    residual_refinement_lr_scale: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_epochs: int = 0
    lr_min_ratio: float = 0.0
    synthetic_warp_scale: float = 2.5
    synthetic_control_grid: list[int] = field(default_factory=lambda: [5, 6, 7])
    phase: str = "real"
    num_labels: int = 86

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PCCRConfig":
        payload = _load_yaml_with_base(Path(path).resolve())
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_overrides(self, overrides: dict[str, Any]) -> "PCCRConfig":
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise KeyError(f"Unknown PCCRConfig field: {key}")
            setattr(self, key, value)
        return self
