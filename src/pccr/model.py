from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pccr.config import PCCRConfig
from src.pccr.modules import (
    CandidateRefinedMatcher,
    CanonicalCorrelationMatcher,
    DiffeomorphicRegistrationDecoder,
    PairConditionedPointmapHead,
    SharedPyramidEncoder,
)
from src.pccr.modules.matcher import MatchOutputs
from src.pccr.utils import flatten_spatial, voxel_grid


@dataclass
class DirectionalOutputs:
    displacement: torch.Tensor
    moved: torch.Tensor
    pointmaps: dict[int, object]
    matches: dict[int, object]
    stage_displacements: dict[int, torch.Tensor]
    stage_velocity_fields: dict[int, torch.Tensor]
    stage_target_displacements: dict[int, torch.Tensor]
    stage_target_confidences: dict[int, torch.Tensor]
    stage_target_margins: dict[int, torch.Tensor]
    stage_target_entropies: dict[int, torch.Tensor]
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
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )
        self.pointmap_head = PairConditionedPointmapHead(
            stage_channels=config.stage_channels,
            pointmap_stage_ids=config.pointmap_stage_ids,
            context_dim=config.context_dim,
            descriptor_dim=config.descriptor_dim,
            num_matchability_classes=config.num_matchability_classes,
        )
        matcher_common_kwargs = dict(
            temperature=config.matcher_temperature,
            topk=config.matcher_topk,
            canonical_radius=config.canonical_radius,
            matchability_score_mode=config.matchability_score_mode,
            matchability_score_power=config.matchability_score_power,
            confidence_mode=config.matcher_confidence_mode,
            global_match_voxel_limit=config.global_match_voxel_limit,
        )
        if config.matcher_type == "candidate_refined":
            self.matcher = CandidateRefinedMatcher(
                descriptor_dim=config.descriptor_dim,
                hidden_dim=config.refined_matcher_hidden_dim,
                offset_scale=config.refined_matcher_offset_scale,
                output_mode=config.refined_matcher_output_mode,
                topm=config.refined_matcher_topm,
                **matcher_common_kwargs,
            )
        else:
            self.matcher = CanonicalCorrelationMatcher(**matcher_common_kwargs)
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
            final_refinement_num_blocks=config.final_refinement_num_blocks,
            final_refinement_activation=config.final_refinement_activation,
            final_refinement_use_image_error_inputs=config.final_refinement_use_image_error_inputs,
            final_refinement_include_raw_image_error_inputs=config.final_refinement_include_raw_image_error_inputs,
            final_refinement_image_error_channels=config.final_refinement_image_error_channels,
            final_refinement_use_error_edges=config.final_refinement_use_error_edges,
            final_refinement_use_local_cost_volume=config.final_refinement_use_local_cost_volume,
            final_refinement_cost_volume_radius=config.final_refinement_cost_volume_radius,
            final_refinement_cost_volume_proj_channels=config.final_refinement_cost_volume_proj_channels,
            final_refinement_cost_volume_feature_channels=config.final_refinement_cost_volume_feature_channels,
            final_refinement_memory_efficient_cost_volume=config.final_refinement_memory_efficient_cost_volume,
            final_refinement_cost_volume_offset_chunk_size=config.final_refinement_cost_volume_offset_chunk_size,
            final_refinement_use_local_residual_matcher=config.final_refinement_use_local_residual_matcher,
            final_refinement_local_matcher_radius=config.final_refinement_local_matcher_radius,
            final_refinement_local_matcher_proj_channels=config.final_refinement_local_matcher_proj_channels,
            final_refinement_local_matcher_feature_channels=config.final_refinement_local_matcher_feature_channels,
            final_refinement_local_matcher_temperature=config.final_refinement_local_matcher_temperature,
            use_stage1_local_refinement=config.use_stage1_local_refinement,
            stage1_local_refinement_radius=config.stage1_local_refinement_radius,
            stage1_local_refinement_proj_channels=config.stage1_local_refinement_proj_channels,
            stage1_local_refinement_feature_channels=config.stage1_local_refinement_feature_channels,
            stage1_local_refinement_temperature=config.stage1_local_refinement_temperature,
            stage1_local_refinement_memory_efficient=config.stage1_local_refinement_memory_efficient,
            stage1_local_refinement_offset_chunk_size=config.stage1_local_refinement_offset_chunk_size,
            use_stage2_local_refinement=config.use_stage2_local_refinement,
            stage2_local_refinement_radius=config.stage2_local_refinement_radius,
            stage2_local_refinement_proj_channels=config.stage2_local_refinement_proj_channels,
            stage2_local_refinement_feature_channels=config.stage2_local_refinement_feature_channels,
            stage2_local_refinement_temperature=config.stage2_local_refinement_temperature,
            stage2_local_refinement_memory_efficient=config.stage2_local_refinement_memory_efficient,
            stage2_local_refinement_offset_chunk_size=config.stage2_local_refinement_offset_chunk_size,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            diagnostic_residual_only=config.diagnostic_residual_only,
        )

    def _build_oracle_match_outputs(
        self,
        source_features: dict[int, torch.Tensor],
        canonical_source: torch.Tensor,
        canonical_target: torch.Tensor,
    ) -> dict[int, MatchOutputs]:
        oracle_outputs: dict[int, MatchOutputs] = {}
        for stage_id in self.config.pointmap_stage_ids:
            spatial_shape = tuple(source_features[stage_id].shape[2:])
            batch_size = source_features[stage_id].shape[0]
            resized_source = F.interpolate(
                canonical_source,
                size=spatial_shape,
                mode="trilinear",
                align_corners=False,
            )
            resized_target = F.interpolate(
                canonical_target,
                size=spatial_shape,
                mode="trilinear",
                align_corners=False,
            )
            src_canonical = flatten_spatial(resized_source)
            tgt_canonical = flatten_spatial(resized_target)
            nearest = torch.cdist(src_canonical, tgt_canonical).argmin(dim=-1)

            target_grid = voxel_grid(spatial_shape, source_features[stage_id].device)
            target_positions = flatten_spatial(target_grid.expand(batch_size, -1, -1, -1, -1))
            expected_target_positions = torch.gather(
                target_positions,
                dim=1,
                index=nearest.unsqueeze(-1).expand(-1, -1, 3),
            )
            source_positions = flatten_spatial(target_grid.expand(batch_size, -1, -1, -1, -1))
            raw_displacement = (expected_target_positions - source_positions).transpose(1, 2).contiguous()
            raw_displacement = raw_displacement.view(batch_size, 3, *spatial_shape)

            num_voxels = tgt_canonical.shape[1]
            probabilities = F.one_hot(nearest, num_classes=num_voxels).float()
            confidence = source_features[stage_id].new_ones(batch_size, 1, *spatial_shape)
            margin = source_features[stage_id].new_ones(batch_size, 1, *spatial_shape)
            entropy = source_features[stage_id].new_zeros(batch_size, 1, *spatial_shape)

            oracle_outputs[stage_id] = MatchOutputs(
                expected_target_positions=expected_target_positions,
                raw_displacement=raw_displacement,
                probabilities=probabilities,
                confidence=confidence,
                margin=margin,
                entropy=entropy,
                source_positions=source_positions,
            )
        return oracle_outputs

    def _forward_direction(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        oracle_canonical_source: torch.Tensor | None = None,
        oracle_canonical_target: torch.Tensor | None = None,
        oracle_dense_displacement: torch.Tensor | None = None,
    ) -> DirectionalOutputs:
        source_features, target_features = self.encoder(source, target)
        if (
            self.config.diagnostic_oracle_correspondence
            and oracle_canonical_source is not None
            and oracle_canonical_target is not None
        ):
            source_pointmaps = {}
            target_pointmaps = {}
            match_outputs = self._build_oracle_match_outputs(
                source_features=source_features,
                canonical_source=oracle_canonical_source,
                canonical_target=oracle_canonical_target,
            )
        else:
            source_pointmaps, target_pointmaps = self.pointmap_head(source_features, target_features)
            match_outputs = {}
            for stage_id in source_pointmaps:
                stage_match = self.matcher(
                    source_pointmaps[stage_id],
                    target_pointmaps[stage_id],
                    stage_id=stage_id,
                )
                if stage_match is not None:
                    match_outputs[stage_id] = stage_match
        decoder_outputs = self.decoder(
            source_image=source,
            target_image=target,
            source_features=source_features,
            target_features=target_features,
            match_outputs=match_outputs,
            oracle_dense_displacement=oracle_dense_displacement,
        )
        return DirectionalOutputs(
            displacement=decoder_outputs.displacement,
            moved=decoder_outputs.moved_source,
            pointmaps=source_pointmaps,
            matches=match_outputs,
            stage_displacements=decoder_outputs.stage_displacements,
            stage_velocity_fields=decoder_outputs.stage_velocity_fields,
            stage_target_displacements=decoder_outputs.stage_target_displacements,
            stage_target_confidences=decoder_outputs.stage_target_confidences,
            stage_target_margins=decoder_outputs.stage_target_margins,
            stage_target_entropies=decoder_outputs.stage_target_entropies,
            final_residual_velocity=decoder_outputs.final_residual_velocity,
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        oracle_canonical_source: torch.Tensor | None = None,
        oracle_canonical_target: torch.Tensor | None = None,
        oracle_dense_s2t: torch.Tensor | None = None,
        oracle_dense_t2s: torch.Tensor | None = None,
    ) -> dict[str, object]:
        forward_outputs = self._forward_direction(
            source,
            target,
            oracle_canonical_source=oracle_canonical_source,
            oracle_canonical_target=oracle_canonical_target,
            oracle_dense_displacement=oracle_dense_s2t,
        )
        backward_outputs = self._forward_direction(
            target,
            source,
            oracle_canonical_source=oracle_canonical_target,
            oracle_canonical_target=oracle_canonical_source,
            oracle_dense_displacement=oracle_dense_t2s,
        )
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
            "forward_stage_target_displacements": forward_outputs.stage_target_displacements,
            "backward_stage_target_displacements": backward_outputs.stage_target_displacements,
            "forward_stage_target_confidences": forward_outputs.stage_target_confidences,
            "backward_stage_target_confidences": backward_outputs.stage_target_confidences,
            "forward_stage_target_margins": forward_outputs.stage_target_margins,
            "backward_stage_target_margins": backward_outputs.stage_target_margins,
            "forward_stage_target_entropies": forward_outputs.stage_target_entropies,
            "backward_stage_target_entropies": backward_outputs.stage_target_entropies,
            "forward_final_residual_velocity": forward_outputs.final_residual_velocity,
            "backward_final_residual_velocity": backward_outputs.final_residual_velocity,
        }
