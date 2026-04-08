from __future__ import annotations

from dataclasses import dataclass

from src.pccr.config import PCCRConfig


@dataclass
class PCCRV6Config(PCCRConfig):
    use_stage0_local_refinement: bool = True
    stage0_local_refinement_radius: int = 1
    stage0_local_refinement_proj_channels: int = 16
    stage0_local_refinement_feature_channels: int = 16
    stage0_local_refinement_temperature: float = 1.0
    stage_local_refinement_gate_with_prior: bool = True
    stage_local_refinement_prior_confidence_power: float = 0.5
    use_structured_match_handoff: bool = False
    structured_match_handoff_topm: int = 2
    structured_match_handoff_channels: int = 16
    structured_match_adapter_hidden_channels: int = 32
    structured_match_adapter_velocity_scale: float = 0.25
