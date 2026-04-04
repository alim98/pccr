from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.pccr.config import PCCRConfig
from src.pccr.modules import (
    CanonicalCorrelationMatcher,
    DiffeomorphicRegistrationDecoder,
    PairConditionedPointmapHead,
    SharedPyramidEncoder,
)


@dataclass
class DirectionalOutputs:
    displacement: torch.Tensor
    moved: torch.Tensor
    pointmaps: dict[int, object]
    matches: dict[int, object]
    stage_displacements: dict[int, torch.Tensor]
    stage_velocity_fields: dict[int, torch.Tensor]
    final_residual_velocity: torch.Tensor | None


class PCCRModel(nn.Module):
    """Pair-conditioned canonical registration model."""

    def __init__(self, config: PCCRConfig):
        super().__init__()
        self.config = config
        self.encoder = SharedPyramidEncoder(
            in_channels=config.in_channels,
            stage_channels=config.stage_channels,
            kernel_size=config.kernel_size,
        )
        self.pointmap_head = PairConditionedPointmapHead(
            stage_channels=config.stage_channels,
            pointmap_stage_ids=config.pointmap_stage_ids,
            context_dim=config.context_dim,
            descriptor_dim=config.descriptor_dim,
            num_matchability_classes=config.num_matchability_classes,
        )
        self.matcher = CanonicalCorrelationMatcher(
            temperature=config.matcher_temperature,
            topk=config.matcher_topk,
            canonical_radius=config.canonical_radius,
            matchability_score_mode=config.matchability_score_mode,
            matchability_score_power=config.matchability_score_power,
        )
        self.decoder = DiffeomorphicRegistrationDecoder(
            stage_channels=config.stage_channels,
            decoder_stage_ids=config.decoder_stage_ids,
            image_size=config.data_size,
            hidden_channels=config.decoder_channels,
            integration_steps=config.svf_integration_steps,
            max_velocity=config.max_velocity,
            use_fine_local_refinement=config.use_fine_local_refinement,
            use_final_residual_refinement=config.use_final_residual_refinement,
            final_refinement_hidden_channels=config.final_refinement_hidden_channels,
            final_refinement_max_velocity=config.final_refinement_max_velocity,
        )

    def _forward_direction(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> DirectionalOutputs:
        source_features, target_features = self.encoder(source, target)
        source_pointmaps, target_pointmaps = self.pointmap_head(source_features, target_features)
        match_outputs = {
            stage_id: self.matcher(source_pointmaps[stage_id], target_pointmaps[stage_id])
            for stage_id in source_pointmaps
        }
        decoder_outputs = self.decoder(
            source_image=source,
            source_features=source_features,
            target_features=target_features,
            match_outputs=match_outputs,
        )
        return DirectionalOutputs(
            displacement=decoder_outputs.displacement,
            moved=decoder_outputs.moved_source,
            pointmaps=source_pointmaps,
            matches=match_outputs,
            stage_displacements=decoder_outputs.stage_displacements,
            stage_velocity_fields=decoder_outputs.stage_velocity_fields,
            final_residual_velocity=decoder_outputs.final_residual_velocity,
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
        forward_outputs = self._forward_direction(source, target)
        backward_outputs = self._forward_direction(target, source)
        return {
            "forward": forward_outputs,
            "backward": backward_outputs,
            "moved_source": forward_outputs.moved,
            "moved_target": backward_outputs.moved,
            "phi_s2t": forward_outputs.displacement,
            "phi_t2s": backward_outputs.displacement,
            "forward_pointmaps": forward_outputs.pointmaps,
            "backward_pointmaps": backward_outputs.pointmaps,
            "forward_matches": forward_outputs.matches,
            "backward_matches": backward_outputs.matches,
            "forward_stage_displacements": forward_outputs.stage_displacements,
            "backward_stage_displacements": backward_outputs.stage_displacements,
            "forward_stage_velocity_fields": forward_outputs.stage_velocity_fields,
            "backward_stage_velocity_fields": backward_outputs.stage_velocity_fields,
            "forward_final_residual_velocity": forward_outputs.final_residual_velocity,
            "backward_final_residual_velocity": backward_outputs.final_residual_velocity,
        }
