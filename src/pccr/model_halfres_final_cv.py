from __future__ import annotations

import os

import torch.nn as nn

from src.pccr.config import PCCRConfig
from src.pccr.model import PCCRModel
from src.pccr.modules import (
    CandidateRefinedMatcher,
    CanonicalCorrelationMatcher,
    PairConditionedPointmapHead,
    SharedPyramidEncoder,
)
from src.pccr.modules.diffeomorphic_halfres_final_cv import (
    DiffeomorphicRegistrationDecoderHalfResFinalCV,
)


class PCCRModelHalfResFinalCV(PCCRModel):
    """Isolated model variant with half-resolution final cost volume."""

    def __init__(self, config: PCCRConfig):
        nn.Module.__init__(self)
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

        final_cv_downsample_factor = int(os.getenv("PCCR_FINAL_CV_DOWNSAMPLE_FACTOR", "2"))
        self.decoder = DiffeomorphicRegistrationDecoderHalfResFinalCV(
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
            final_cv_downsample_factor=final_cv_downsample_factor,
        )
