from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformation import SpatialTransformer
from src.pccr.utils import resize_displacement


def compose_displacement_fields(
    first: torch.Tensor, second: torch.Tensor, transformer: SpatialTransformer
) -> torch.Tensor:
    return first + transformer(second, first)


class ScalingAndSquaring(nn.Module):
    def __init__(self, size: tuple[int, int, int], steps: int = 7) -> None:
        super().__init__()
        self.steps = steps
        self.transformer = SpatialTransformer(size)

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        displacement = velocity / (2 ** self.steps)
        for _ in range(self.steps):
            displacement = displacement + self.transformer(displacement, displacement)
        return displacement


class StageVelocityDecoder(nn.Module):
    def __init__(self, feature_channels: int, hidden_channels: int, max_velocity: float) -> None:
        super().__init__()
        self.max_velocity = max_velocity
        in_channels = feature_channels * 2 + 3 + 1 + 1
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 3, kernel_size=3, padding=1),
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        raw_displacement: torch.Tensor,
        confidence: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        velocity = self.net(
            torch.cat(
                [source_features, target_features, raw_displacement, confidence, entropy],
                dim=1,
            )
        )
        velocity = torch.tanh(velocity) * self.max_velocity
        return torch.nan_to_num(velocity, nan=0.0, posinf=self.max_velocity, neginf=-self.max_velocity)


class FinalResidualRefinementHead(nn.Module):
    def __init__(self, feature_channels: int, hidden_channels: int, max_velocity: float) -> None:
        super().__init__()
        self.max_velocity = max_velocity
        in_channels = feature_channels * 2 + 3 + 1 + 1
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 3, kernel_size=3, padding=1),
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        displacement: torch.Tensor,
        confidence: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        velocity = self.net(
            torch.cat([source_features, target_features, displacement, confidence, entropy], dim=1)
        )
        velocity = torch.tanh(velocity) * self.max_velocity
        return torch.nan_to_num(velocity, nan=0.0, posinf=self.max_velocity, neginf=-self.max_velocity)


@dataclass
class DecoderOutputs:
    displacement: torch.Tensor
    moved_source: torch.Tensor
    stage_displacements: dict[int, torch.Tensor]
    stage_velocity_fields: dict[int, torch.Tensor]
    final_residual_velocity: torch.Tensor | None


class DiffeomorphicRegistrationDecoder(nn.Module):
    def __init__(
        self,
        stage_channels: list[int],
        decoder_stage_ids: list[int],
        image_size: list[int],
        hidden_channels: int,
        integration_steps: int,
        max_velocity: float,
        use_fine_local_refinement: bool = True,
        use_final_residual_refinement: bool = False,
        final_refinement_hidden_channels: int = 32,
        final_refinement_max_velocity: float = 0.2,
    ) -> None:
        super().__init__()
        self.decoder_stage_ids = decoder_stage_ids
        self.image_size = tuple(image_size)
        self.use_fine_local_refinement = use_fine_local_refinement
        self.use_final_residual_refinement = use_final_residual_refinement

        self.stage_decoders = nn.ModuleDict()
        self.integrators = nn.ModuleDict()
        self.transformers = nn.ModuleDict()

        stage_sizes = self._stage_sizes(image_size, len(stage_channels))
        for stage_id in decoder_stage_ids:
            size = tuple(stage_sizes[stage_id])
            self.stage_decoders[str(stage_id)] = StageVelocityDecoder(
                feature_channels=stage_channels[stage_id],
                hidden_channels=hidden_channels,
                max_velocity=max_velocity,
            )
            self.integrators[str(stage_id)] = ScalingAndSquaring(size=size, steps=integration_steps)
            self.transformers[str(stage_id)] = SpatialTransformer(size=size)

        self.final_transformer = SpatialTransformer(self.image_size)
        self.final_feature_transformer = SpatialTransformer(self.image_size)
        if self.use_final_residual_refinement:
            self.final_refinement_head = FinalResidualRefinementHead(
                feature_channels=stage_channels[0],
                hidden_channels=final_refinement_hidden_channels,
                max_velocity=final_refinement_max_velocity,
            )
            self.final_refinement_integrator = ScalingAndSquaring(
                size=self.image_size,
                steps=integration_steps,
            )

    @staticmethod
    def _stage_sizes(image_size: list[int], num_stages: int) -> list[list[int]]:
        sizes = [list(image_size)]
        for _ in range(1, num_stages):
            prev = sizes[-1]
            sizes.append([max(1, dim // 2) for dim in prev])
        return sizes

    def forward(
        self,
        source_image: torch.Tensor,
        source_features: dict[int, torch.Tensor],
        target_features: dict[int, torch.Tensor],
        match_outputs: dict[int, object],
    ) -> DecoderOutputs:
        previous_displacement = None
        stage_displacements: dict[int, torch.Tensor] = {}
        stage_velocity_fields: dict[int, torch.Tensor] = {}
        final_residual_velocity = None

        for stage_id in self.decoder_stage_ids:
            source_stage = source_features[stage_id]
            target_stage = target_features[stage_id]
            stage_size = tuple(source_stage.shape[2:])
            transformer = self.transformers[str(stage_id)]

            if previous_displacement is not None:
                previous_displacement = resize_displacement(previous_displacement, stage_size)
                warped_target = transformer(target_stage, previous_displacement)
            else:
                warped_target = target_stage

            if stage_id in match_outputs:
                stage_match = match_outputs[stage_id]
                raw_displacement = stage_match.raw_displacement
                confidence = stage_match.confidence
                entropy = stage_match.entropy
                if previous_displacement is not None:
                    raw_displacement = raw_displacement - previous_displacement
            else:
                if previous_displacement is None:
                    raw_displacement = source_stage.new_zeros(source_stage.shape[0], 3, *stage_size)
                else:
                    raw_displacement = previous_displacement
                confidence = source_stage.new_zeros(source_stage.shape[0], 1, *stage_size)
                entropy = source_stage.new_zeros(source_stage.shape[0], 1, *stage_size)

            velocity = self.stage_decoders[str(stage_id)](
                source_stage,
                warped_target,
                raw_displacement,
                confidence,
                entropy,
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

        final_displacement = resize_displacement(previous_displacement, self.image_size)
        final_displacement = torch.nan_to_num(final_displacement, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_final_residual_refinement:
            source_fine = source_features[0]
            target_fine = self.final_feature_transformer(target_features[0], final_displacement)
            match_stage_id = min(match_outputs) if match_outputs else None
            if match_stage_id is not None:
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
            residual_velocity = self.final_refinement_head(
                source_fine,
                target_fine,
                final_displacement,
                confidence,
                entropy,
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
            final_residual_velocity=final_residual_velocity,
        )
