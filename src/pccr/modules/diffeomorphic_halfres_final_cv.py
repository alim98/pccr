from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from src.pccr.modules.diffeomorphic import (
    DecoderOutputs,
    DiffeomorphicRegistrationDecoder,
    compose_displacement_fields,
    resize_displacement,
)


class DiffeomorphicRegistrationDecoderHalfResFinalCV(DiffeomorphicRegistrationDecoder):
    """Isolated decoder variant that computes the final local cost volume on a
    downsampled feature grid and upsamples the encoded features back to full
    resolution before the final refinement head.

    This keeps the rest of the architecture unchanged and avoids modifying the
    shared decoder used by other runs.
    """

    def __init__(self, *args, final_cv_downsample_factor: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.final_cv_downsample_factor = max(1, int(final_cv_downsample_factor))

    def _downsample_for_final_cv(
        self,
        source_fine: torch.Tensor,
        target_fine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor = self.final_cv_downsample_factor
        if factor <= 1:
            return source_fine, target_fine
        kernel = (factor, factor, factor)
        return (
            F.avg_pool3d(source_fine, kernel_size=kernel, stride=kernel),
            F.avg_pool3d(target_fine, kernel_size=kernel, stride=kernel),
        )

    def _upsample_final_cv_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.final_cv_downsample_factor <= 1:
            return features
        return F.interpolate(
            features,
            size=self.image_size,
            mode="trilinear",
            align_corners=False,
        )

    def forward(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        source_features: dict[int, torch.Tensor],
        target_features: dict[int, torch.Tensor],
        match_outputs: dict[int, object],
        oracle_dense_displacement: torch.Tensor | None = None,
    ) -> DecoderOutputs:
        previous_displacement = None
        stage_displacements: dict[int, torch.Tensor] = {}
        stage_velocity_fields: dict[int, torch.Tensor] = {}
        stage_target_displacements: dict[int, torch.Tensor] = {}
        stage_target_confidences: dict[int, torch.Tensor] = {}
        stage_target_margins: dict[int, torch.Tensor] = {}
        stage_target_entropies: dict[int, torch.Tensor] = {}
        final_residual_velocity = None

        if not self.diagnostic_residual_only:
            for stage_id in self.decoder_stage_ids:
                source_stage = source_features[stage_id]
                target_stage = target_features[stage_id]
                stage_size = tuple(source_stage.shape[2:])
                transformer = self.transformers[str(stage_id)]
                stage_extra_features = None

                if previous_displacement is not None:
                    previous_displacement = resize_displacement(previous_displacement, stage_size)
                    warped_target = transformer(target_stage, previous_displacement)
                else:
                    warped_target = target_stage

                if oracle_dense_displacement is not None:
                    oracle_stage_displacement = resize_displacement(oracle_dense_displacement, stage_size)
                    if previous_displacement is not None:
                        raw_displacement = oracle_stage_displacement - previous_displacement
                    else:
                        raw_displacement = oracle_stage_displacement
                    confidence = source_stage.new_ones(source_stage.shape[0], 1, *stage_size)
                    margin = source_stage.new_ones(source_stage.shape[0], 1, *stage_size)
                    entropy = source_stage.new_zeros(source_stage.shape[0], 1, *stage_size)
                    stage_target_displacements[stage_id] = oracle_stage_displacement.detach()
                    stage_target_confidences[stage_id] = confidence.detach()
                    stage_target_margins[stage_id] = margin.detach()
                    stage_target_entropies[stage_id] = entropy.detach()
                elif stage_id in match_outputs:
                    stage_match = match_outputs[stage_id]
                    raw_displacement = stage_match.raw_displacement
                    confidence = stage_match.confidence
                    margin = stage_match.margin
                    entropy = stage_match.entropy
                    stage_target_displacements[stage_id] = stage_match.raw_displacement.detach()
                    stage_target_confidences[stage_id] = confidence.detach()
                    stage_target_margins[stage_id] = margin.detach()
                    stage_target_entropies[stage_id] = entropy.detach()
                    if previous_displacement is not None:
                        raw_displacement = raw_displacement - previous_displacement
                elif previous_displacement is not None and str(stage_id) in self.stage_local_refiners:
                    stage_local = self._run_local_refiner(
                        self.stage_local_refiners[str(stage_id)],
                        source_stage,
                        warped_target,
                    )
                    raw_displacement = stage_local.delta_displacement
                    confidence = stage_local.confidence
                    margin = stage_local.margin
                    entropy = stage_local.entropy
                    stage_extra_features = stage_local.encoded_features
                    stage_target_displacements[stage_id] = compose_displacement_fields(
                        previous_displacement,
                        stage_local.delta_displacement,
                        transformer,
                    ).detach()
                    stage_target_confidences[stage_id] = confidence.detach()
                    stage_target_margins[stage_id] = margin.detach()
                    stage_target_entropies[stage_id] = entropy.detach()
                else:
                    if previous_displacement is None:
                        raw_displacement = source_stage.new_zeros(source_stage.shape[0], 3, *stage_size)
                    else:
                        raw_displacement = previous_displacement
                    confidence = source_stage.new_zeros(source_stage.shape[0], 1, *stage_size)
                    entropy = source_stage.new_zeros(source_stage.shape[0], 1, *stage_size)

                velocity = self._run_tensor_module(
                    self.stage_decoders[str(stage_id)],
                    source_stage,
                    warped_target,
                    raw_displacement,
                    confidence,
                    entropy,
                    stage_extra_features,
                )
                stage_velocity_fields[stage_id] = velocity
                residual_displacement = self.integrators[str(stage_id)](velocity)
                residual_displacement = torch.nan_to_num(residual_displacement, nan=0.0, posinf=0.0, neginf=0.0)
                if previous_displacement is None:
                    current_displacement = residual_displacement
                else:
                    current_displacement = compose_displacement_fields(
                        previous_displacement,
                        residual_displacement,
                        transformer,
                    )
                current_displacement = torch.nan_to_num(current_displacement, nan=0.0, posinf=0.0, neginf=0.0)
                stage_displacements[stage_id] = current_displacement
                previous_displacement = current_displacement

        if previous_displacement is None:
            batch_size = source_image.shape[0]
            final_displacement = source_image.new_zeros(batch_size, 3, *self.image_size)
        else:
            final_displacement = resize_displacement(previous_displacement, self.image_size)
        final_displacement = torch.nan_to_num(final_displacement, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_final_residual_refinement:
            source_fine = source_features[0]
            target_fine = self.final_feature_transformer(target_features[0], final_displacement)
            moved_source = self.final_transformer(source_image, final_displacement)
            moved_source = torch.nan_to_num(moved_source, nan=0.0, posinf=0.0, neginf=0.0)
            extra_features = []
            if self.final_refinement_use_local_residual_matcher:
                local_match_outputs = self._run_local_refiner(self.local_residual_matcher, source_fine, target_fine)
                final_displacement = compose_displacement_fields(
                    final_displacement,
                    local_match_outputs.delta_displacement,
                    self.final_transformer,
                )
                final_displacement = torch.nan_to_num(final_displacement, nan=0.0, posinf=0.0, neginf=0.0)
                target_fine = self.final_feature_transformer(target_features[0], final_displacement)
                moved_source = self.final_transformer(source_image, final_displacement)
                moved_source = torch.nan_to_num(moved_source, nan=0.0, posinf=0.0, neginf=0.0)
                confidence = local_match_outputs.confidence
                entropy = local_match_outputs.entropy
                extra_features.append(local_match_outputs.encoded_features)
            else:
                match_stage_id = min(match_outputs) if match_outputs else None
                if oracle_dense_displacement is not None:
                    confidence = source_fine.new_ones(source_fine.shape[0], 1, *self.image_size)
                    entropy = source_fine.new_zeros(source_fine.shape[0], 1, *self.image_size)
                elif match_stage_id is not None:
                    confidence = F.interpolate(
                        match_outputs[match_stage_id].confidence,
                        size=self.image_size,
                        mode="trilinear",
                        align_corners=False,
                    )
                    entropy = F.interpolate(
                        match_outputs[match_stage_id].entropy,
                        size=self.image_size,
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    confidence = source_fine.new_zeros(source_fine.shape[0], 1, *self.image_size)
                    entropy = source_fine.new_zeros(source_fine.shape[0], 1, *self.image_size)
            if self.final_refinement_include_raw_image_error_inputs:
                extra_features.append(
                    self._raw_image_error_features(
                        moved_source,
                        target_image,
                        use_edge_inputs=self.final_refinement_use_error_edges,
                    )
                )
            if self.final_refinement_use_image_error_inputs:
                extra_features.append(self._run_tensor_module(self.image_error_encoder, moved_source, target_image))
            if self.final_refinement_use_local_cost_volume:
                source_cv, target_cv = self._downsample_for_final_cv(source_fine, target_fine)
                cv_features = self._run_tensor_module(self.local_cost_volume_encoder, source_cv, target_cv)
                cv_features = self._upsample_final_cv_features(cv_features)
                extra_features.append(cv_features)
            residual_velocity = self._run_tensor_module(
                self.final_refinement_head,
                source_fine,
                target_fine,
                final_displacement,
                confidence,
                entropy,
                torch.cat(extra_features, dim=1) if extra_features else None,
            )
            final_residual_velocity = residual_velocity
            residual_displacement = self.final_refinement_integrator(residual_velocity)
            residual_displacement = torch.nan_to_num(residual_displacement, nan=0.0, posinf=0.0, neginf=0.0)
            final_displacement = compose_displacement_fields(
                final_displacement,
                residual_displacement,
                self.final_transformer,
            )
            final_displacement = torch.nan_to_num(final_displacement, nan=0.0, posinf=0.0, neginf=0.0)
        moved_source = self.final_transformer(source_image, final_displacement)
        moved_source = torch.nan_to_num(moved_source, nan=0.0, posinf=0.0, neginf=0.0)
        return DecoderOutputs(
            displacement=final_displacement,
            moved_source=moved_source,
            stage_displacements=stage_displacements,
            stage_velocity_fields=stage_velocity_fields,
            stage_target_displacements=stage_target_displacements,
            stage_target_confidences=stage_target_confidences,
            stage_target_margins=stage_target_margins,
            stage_target_entropies=stage_target_entropies,
            final_residual_velocity=final_residual_velocity,
        )
