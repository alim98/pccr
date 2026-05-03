from __future__ import annotations

from dataclasses import dataclass, field

from src.pccr.config import PCCRConfig


@dataclass
class PCCRV6Config(PCCRConfig):
    # Stage IDs that are cold (randomly initialised) in the warm-start checkpoint.
    # Correspondence and decoder_fitting losses are gated OFF for these stages
    # for the first new_stage_warmup_epochs epochs to prevent random matches from
    # contaminating the trained stages through the shared context_mlp.
    new_stage_ids: list[int] = field(default_factory=lambda: [2])
    new_stage_warmup_epochs: int = 15

    use_stage0_local_refinement: bool = True
    stage0_local_refinement_radius: int = 1
    stage0_local_refinement_proj_channels: int = 16
    stage0_local_refinement_feature_channels: int = 16
    stage0_local_refinement_temperature: float = 1.0
    stage0_local_refinement_memory_efficient: bool = True
    stage0_local_refinement_offset_chunk_size: int = 32
    stage_local_refinement_gate_with_prior: bool = True
    stage_local_refinement_prior_confidence_power: float = 0.5
    use_structured_match_handoff: bool = False
    structured_match_handoff_topm: int = 2
    structured_match_handoff_channels: int = 16
    structured_match_adapter_hidden_channels: int = 32
    structured_match_adapter_velocity_scale: float = 0.25
