from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformation import SpatialTransformer
from src.pccr.modules.diffeomorphic import (
    DecoderOutputs,
    FinalResidualRefinementHead,
    ImageErrorEncoder,
    LocalCostVolumeEncoder,
    LocalResidualMatcher,
    ScalingAndSquaring,
    StageLocalCorrelationRefiner,
    StageVelocityDecoder,
    compose_displacement_fields,
)
from src.pccr.utils import resize_displacement


@dataclass
class _LocalRefinementContext:
    raw_displacement: torch.Tensor
    confidence: torch.Tensor
    margin: torch.Tensor
    entropy: torch.Tensor
    target_displacement: torch.Tensor
    extra_features: torch.Tensor


class StructuredMatchEvidenceEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(self.net(features), nan=0.0, posinf=0.0, neginf=0.0)


class StructuredEvidenceVelocityAdapter(nn.Module):
    def __init__(self, evidence_channels: int, hidden_channels: int, max_velocity: float) -> None:
        super().__init__()
        self.max_velocity = max_velocity
        self.net = nn.Sequential(
            nn.Conv3d(evidence_channels + 3 + 1 + 1, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        evidence: torch.Tensor,
        raw_displacement: torch.Tensor,
        confidence: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        velocity = self.net(torch.cat([evidence, raw_displacement, confidence, entropy], dim=1))
        velocity = torch.tanh(velocity) * self.max_velocity
        return torch.nan_to_num(velocity, nan=0.0, posinf=self.max_velocity, neginf=-self.max_velocity)


class DiffeomorphicRegistrationDecoderV6(nn.Module):
    """
    v6 decoder: keep the existing coarse diffeomorphic backbone, but extend
    explicit correspondence handoff to stage 0 with propagated-confidence gating.
    """

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
        final_refinement_num_blocks: int = 2,
        final_refinement_activation: str = "gelu",
        final_refinement_use_image_error_inputs: bool = False,
        final_refinement_include_raw_image_error_inputs: bool = False,
        final_refinement_image_error_channels: int = 16,
        final_refinement_use_error_edges: bool = False,
        final_refinement_use_local_cost_volume: bool = False,
        final_refinement_cost_volume_radius: int = 0,
        final_refinement_cost_volume_proj_channels: int = 16,
        final_refinement_cost_volume_feature_channels: int = 16,
        final_refinement_use_local_residual_matcher: bool = False,
        final_refinement_local_matcher_radius: int = 0,
        final_refinement_local_matcher_proj_channels: int = 16,
        final_refinement_local_matcher_feature_channels: int = 16,
        final_refinement_local_matcher_temperature: float = 1.0,
        use_stage1_local_refinement: bool = False,
        stage1_local_refinement_radius: int = 0,
        stage1_local_refinement_proj_channels: int = 16,
        stage1_local_refinement_feature_channels: int = 16,
        stage1_local_refinement_temperature: float = 1.0,
        use_stage0_local_refinement: bool = False,
        stage0_local_refinement_radius: int = 0,
        stage0_local_refinement_proj_channels: int = 16,
        stage0_local_refinement_feature_channels: int = 16,
        stage0_local_refinement_temperature: float = 1.0,
        stage_local_refinement_gate_with_prior: bool = True,
        stage_local_refinement_prior_confidence_power: float = 0.5,
        use_structured_match_handoff: bool = False,
        structured_match_handoff_topm: int = 2,
        structured_match_handoff_channels: int = 16,
        structured_match_adapter_hidden_channels: int = 32,
        structured_match_adapter_velocity_scale: float = 0.25,
        diagnostic_residual_only: bool = False,
    ) -> None:
        super().__init__()
        self.decoder_stage_ids = decoder_stage_ids
        self.image_size = tuple(image_size)
        self.use_fine_local_refinement = use_fine_local_refinement
        self.use_final_residual_refinement = use_final_residual_refinement
        self.diagnostic_residual_only = diagnostic_residual_only
        self.final_refinement_use_image_error_inputs = final_refinement_use_image_error_inputs
        self.final_refinement_include_raw_image_error_inputs = final_refinement_include_raw_image_error_inputs
        self.final_refinement_use_error_edges = final_refinement_use_error_edges
        self.final_refinement_use_local_cost_volume = (
            use_final_residual_refinement
            and final_refinement_use_local_cost_volume
            and final_refinement_cost_volume_radius > 0
        )
        self.final_refinement_use_local_residual_matcher = (
            use_final_residual_refinement
            and final_refinement_use_local_residual_matcher
            and final_refinement_local_matcher_radius > 0
        )
        self.use_stage1_local_refinement = use_stage1_local_refinement and stage1_local_refinement_radius > 0
        self.use_stage0_local_refinement = use_stage0_local_refinement and stage0_local_refinement_radius > 0
        self.stage_local_refinement_gate_with_prior = stage_local_refinement_gate_with_prior
        self.stage_local_refinement_prior_confidence_power = stage_local_refinement_prior_confidence_power
        self.use_structured_match_handoff = (
            use_structured_match_handoff and structured_match_handoff_topm > 0 and structured_match_handoff_channels > 0
        )
        self.structured_match_handoff_topm = structured_match_handoff_topm
        self.structured_match_handoff_channels = structured_match_handoff_channels
        self.structured_match_handoff_input_channels = structured_match_handoff_topm * 4 + 6
        self.structured_match_adapter_hidden_channels = structured_match_adapter_hidden_channels
        self.structured_match_adapter_velocity_scale = structured_match_adapter_velocity_scale

        self.stage_decoders = nn.ModuleDict()
        self.integrators = nn.ModuleDict()
        self.transformers = nn.ModuleDict()
        self.stage_local_refiners = nn.ModuleDict()
        self.stage_match_evidence_encoders = nn.ModuleDict()
        self.stage_structured_adapters = nn.ModuleDict()
        self.stage_local_feature_channels: dict[int, int] = {}

        stage_sizes = self._stage_sizes(image_size, len(stage_channels))
        local_refinement_specs = {
            1: (
                self.use_stage1_local_refinement,
                stage1_local_refinement_radius,
                stage1_local_refinement_proj_channels,
                stage1_local_refinement_feature_channels,
                stage1_local_refinement_temperature,
            ),
            0: (
                self.use_stage0_local_refinement,
                stage0_local_refinement_radius,
                stage0_local_refinement_proj_channels,
                stage0_local_refinement_feature_channels,
                stage0_local_refinement_temperature,
            ),
        }
        for stage_id in decoder_stage_ids:
            size = tuple(stage_sizes[stage_id])
            extra_channels = 0
            if self.use_structured_match_handoff:
                self.stage_match_evidence_encoders[str(stage_id)] = StructuredMatchEvidenceEncoder(
                    in_channels=self.structured_match_handoff_input_channels,
                    out_channels=self.structured_match_handoff_channels,
                )
                self.stage_structured_adapters[str(stage_id)] = StructuredEvidenceVelocityAdapter(
                    evidence_channels=self.structured_match_handoff_channels,
                    hidden_channels=self.structured_match_adapter_hidden_channels,
                    max_velocity=max_velocity * self.structured_match_adapter_velocity_scale,
                )
            enabled, radius, proj_channels, feature_channels, temperature = local_refinement_specs.get(
                stage_id,
                (False, 0, 16, 16, 1.0),
            )
            if enabled:
                self.stage_local_refiners[str(stage_id)] = StageLocalCorrelationRefiner(
                    in_channels=stage_channels[stage_id],
                    radius=radius,
                    proj_channels=proj_channels,
                    out_channels=feature_channels,
                    temperature=temperature,
                )
                self.stage_local_feature_channels[stage_id] = feature_channels
                # Encoded local match features plus two scalar channels:
                # confidence gate and propagated prior confidence.
                extra_channels += feature_channels + 2

            self.stage_decoders[str(stage_id)] = StageVelocityDecoder(
                feature_channels=stage_channels[stage_id],
                hidden_channels=hidden_channels,
                max_velocity=max_velocity,
                extra_channels=extra_channels,
            )
            self.integrators[str(stage_id)] = ScalingAndSquaring(size=size, steps=integration_steps)
            self.transformers[str(stage_id)] = SpatialTransformer(size=size)

        self.final_transformer = SpatialTransformer(self.image_size)
        self.final_feature_transformer = SpatialTransformer(self.image_size)
        if self.use_final_residual_refinement:
            extra_channels = 0
            if self.final_refinement_include_raw_image_error_inputs:
                extra_channels += 3 + (1 if final_refinement_use_error_edges else 0)
            if self.final_refinement_use_image_error_inputs:
                self.image_error_encoder = ImageErrorEncoder(
                    out_channels=final_refinement_image_error_channels,
                    use_edge_inputs=final_refinement_use_error_edges,
                )
                extra_channels += final_refinement_image_error_channels
            if self.final_refinement_use_local_residual_matcher:
                self.local_residual_matcher = LocalResidualMatcher(
                    in_channels=stage_channels[0],
                    radius=final_refinement_local_matcher_radius,
                    proj_channels=final_refinement_local_matcher_proj_channels,
                    out_channels=final_refinement_local_matcher_feature_channels,
                    temperature=final_refinement_local_matcher_temperature,
                )
                extra_channels += final_refinement_local_matcher_feature_channels
            if self.final_refinement_use_local_cost_volume:
                self.local_cost_volume_encoder = LocalCostVolumeEncoder(
                    in_channels=stage_channels[0],
                    radius=final_refinement_cost_volume_radius,
                    proj_channels=final_refinement_cost_volume_proj_channels,
                    out_channels=final_refinement_cost_volume_feature_channels,
                )
                extra_channels += final_refinement_cost_volume_feature_channels
            self.final_refinement_head = FinalResidualRefinementHead(
                feature_channels=stage_channels[0],
                hidden_channels=final_refinement_hidden_channels,
                max_velocity=final_refinement_max_velocity,
                extra_channels=extra_channels,
                num_blocks=final_refinement_num_blocks,
                activation=final_refinement_activation,
            )
            self.final_refinement_integrator = ScalingAndSquaring(
                size=self.image_size,
                steps=integration_steps,
            )
            if self.use_structured_match_handoff:
                self.final_structured_adapter = StructuredEvidenceVelocityAdapter(
                    evidence_channels=self.structured_match_handoff_channels,
                    hidden_channels=self.structured_match_adapter_hidden_channels,
                    max_velocity=final_refinement_max_velocity * self.structured_match_adapter_velocity_scale,
                )

    @staticmethod
    def _stage_sizes(image_size: list[int], num_stages: int) -> list[list[int]]:
        sizes = [list(image_size)]
        for _ in range(1, num_stages):
            prev = sizes[-1]
            sizes.append([max(1, dim // 2) for dim in prev])
        return sizes

    @staticmethod
    def _raw_image_error_features(
        moved_source: torch.Tensor,
        target_image: torch.Tensor,
        use_edge_inputs: bool,
    ) -> torch.Tensor:
        abs_diff = (moved_source - target_image).abs()
        inputs = [moved_source, target_image, abs_diff]
        if use_edge_inputs:
            edge_diff = (
                ImageErrorEncoder._gradient_magnitude(moved_source)
                - ImageErrorEncoder._gradient_magnitude(target_image)
            ).abs()
            inputs.append(edge_diff)
        return torch.cat(inputs, dim=1)

    def _build_local_refinement_context(
        self,
        stage_id: int,
        source_stage: torch.Tensor,
        warped_target: torch.Tensor,
        previous_displacement: torch.Tensor,
        previous_confidence: torch.Tensor | None,
        transformer: SpatialTransformer,
    ) -> _LocalRefinementContext:
        local_refiner = self.stage_local_refiners[str(stage_id)]
        local_outputs = local_refiner(source_stage, warped_target)

        if previous_confidence is None:
            propagated_confidence = local_outputs.confidence.new_ones(local_outputs.confidence.shape)
        else:
            propagated_confidence = F.interpolate(
                previous_confidence,
                size=local_outputs.confidence.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        propagated_confidence = torch.nan_to_num(propagated_confidence, nan=0.0, posinf=0.0, neginf=0.0)
        if self.stage_local_refinement_gate_with_prior:
            confidence_gate = local_outputs.confidence * propagated_confidence.clamp_min(1e-6).pow(
                self.stage_local_refinement_prior_confidence_power
            )
        else:
            confidence_gate = local_outputs.confidence
        confidence_gate = confidence_gate.clamp(0.0, 1.0)

        raw_displacement = local_outputs.delta_displacement * confidence_gate
        target_displacement = compose_displacement_fields(
            previous_displacement,
            raw_displacement,
            transformer,
        )
        local_features = torch.cat(
            [
                local_outputs.encoded_features * confidence_gate,
                confidence_gate,
                propagated_confidence,
            ],
            dim=1,
        )
        return _LocalRefinementContext(
            raw_displacement=torch.nan_to_num(raw_displacement, nan=0.0, posinf=0.0, neginf=0.0),
            confidence=confidence_gate,
            margin=local_outputs.margin,
            entropy=local_outputs.entropy,
            target_displacement=torch.nan_to_num(target_displacement, nan=0.0, posinf=0.0, neginf=0.0),
            extra_features=torch.nan_to_num(local_features, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _encode_stage_match_evidence(self, stage_id: int, stage_match: object) -> torch.Tensor | None:
        if not self.use_structured_match_handoff:
            return None
        raw_features = getattr(stage_match, "structured_handoff_features", None)
        if raw_features is None:
            return None
        return self.stage_match_evidence_encoders[str(stage_id)](raw_features)

    def _select_final_confidence(
        self,
        stage_target_confidences: dict[int, torch.Tensor],
        stage_target_entropies: dict[int, torch.Tensor],
        batch_size: int,
        feature_shape: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
        oracle_dense_displacement: torch.Tensor | None,
        match_outputs: dict[int, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if oracle_dense_displacement is not None:
            confidence = torch.ones((batch_size, 1, *self.image_size), device=device, dtype=dtype)
            entropy = torch.zeros((batch_size, 1, *self.image_size), device=device, dtype=dtype)
            return confidence, entropy

        if stage_target_confidences:
            finest_stage = min(stage_target_confidences)
            confidence = F.interpolate(
                stage_target_confidences[finest_stage],
                size=self.image_size,
                mode="trilinear",
                align_corners=False,
            )
            entropy = F.interpolate(
                stage_target_entropies[finest_stage],
                size=self.image_size,
                mode="trilinear",
                align_corners=False,
            )
            return confidence, entropy

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
            return confidence, entropy

        zeros = torch.zeros((batch_size, 1, *feature_shape), device=device, dtype=dtype)
        return zeros, zeros

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
        previous_confidence = None
        previous_structured_evidence = None
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
                stage_structured_evidence = None

                if previous_displacement is not None:
                    previous_displacement = resize_displacement(previous_displacement, stage_size)
                    if previous_confidence is not None:
                        previous_confidence = F.interpolate(
                            previous_confidence,
                            size=stage_size,
                            mode="trilinear",
                            align_corners=False,
                        )
                    if previous_structured_evidence is not None:
                        previous_structured_evidence = F.interpolate(
                            previous_structured_evidence,
                            size=stage_size,
                            mode="trilinear",
                            align_corners=False,
                        )
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
                    stage_structured_evidence = self._encode_stage_match_evidence(stage_id, stage_match)
                    stage_target_displacements[stage_id] = stage_match.raw_displacement.detach()
                    stage_target_confidences[stage_id] = confidence.detach()
                    stage_target_margins[stage_id] = margin.detach()
                    stage_target_entropies[stage_id] = entropy.detach()
                    if previous_displacement is not None:
                        raw_displacement = raw_displacement - previous_displacement
                elif previous_displacement is not None and str(stage_id) in self.stage_local_refiners:
                    local_context = self._build_local_refinement_context(
                        stage_id=stage_id,
                        source_stage=source_stage,
                        warped_target=warped_target,
                        previous_displacement=previous_displacement,
                        previous_confidence=previous_confidence,
                        transformer=transformer,
                    )
                    raw_displacement = local_context.raw_displacement
                    confidence = local_context.confidence
                    margin = local_context.margin
                    entropy = local_context.entropy
                    stage_extra_features = local_context.extra_features
                    stage_target_displacements[stage_id] = local_context.target_displacement.detach()
                    stage_target_confidences[stage_id] = confidence.detach()
                    stage_target_margins[stage_id] = margin.detach()
                    stage_target_entropies[stage_id] = entropy.detach()
                else:
                    if previous_displacement is None:
                        raw_displacement = source_stage.new_zeros(source_stage.shape[0], 3, *stage_size)
                    else:
                        raw_displacement = previous_displacement
                    confidence = source_stage.new_zeros(source_stage.shape[0], 1, *stage_size)
                    margin = source_stage.new_zeros(source_stage.shape[0], 1, *stage_size)
                    entropy = source_stage.new_zeros(source_stage.shape[0], 1, *stage_size)
                    if previous_structured_evidence is not None:
                        stage_extra_features = previous_structured_evidence

                velocity = self.stage_decoders[str(stage_id)](
                    source_stage,
                    warped_target,
                    raw_displacement,
                    confidence,
                    entropy,
                    extra_features=stage_extra_features,
                )
                if self.use_structured_match_handoff:
                    if stage_structured_evidence is None and previous_structured_evidence is not None:
                        stage_structured_evidence = previous_structured_evidence
                    if stage_structured_evidence is not None:
                        velocity = velocity + self.stage_structured_adapters[str(stage_id)](
                            evidence=stage_structured_evidence,
                            raw_displacement=raw_displacement,
                            confidence=confidence,
                            entropy=entropy,
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
                previous_confidence = confidence
                if stage_structured_evidence is not None:
                    previous_structured_evidence = stage_structured_evidence

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
                local_match_outputs = self.local_residual_matcher(source_fine, target_fine)
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
                confidence, entropy = self._select_final_confidence(
                    stage_target_confidences=stage_target_confidences,
                    stage_target_entropies=stage_target_entropies,
                    batch_size=source_fine.shape[0],
                    feature_shape=source_fine.shape[2:],
                    device=source_fine.device,
                    dtype=source_fine.dtype,
                    oracle_dense_displacement=oracle_dense_displacement,
                    match_outputs=match_outputs,
                )
            if self.final_refinement_include_raw_image_error_inputs:
                extra_features.append(
                    self._raw_image_error_features(
                        moved_source,
                        target_image,
                        use_edge_inputs=self.final_refinement_use_error_edges,
                    )
                )
            if self.final_refinement_use_image_error_inputs:
                extra_features.append(self.image_error_encoder(moved_source, target_image))
            if self.final_refinement_use_local_cost_volume:
                extra_features.append(self.local_cost_volume_encoder(source_fine, target_fine))
            residual_velocity = self.final_refinement_head(
                source_fine,
                target_fine,
                final_displacement,
                confidence,
                entropy,
                extra_features=torch.cat(extra_features, dim=1) if extra_features else None,
            )
            if self.use_structured_match_handoff:
                if previous_structured_evidence is None:
                    final_structured_evidence = source_fine.new_zeros(
                        source_fine.shape[0],
                        self.structured_match_handoff_channels,
                        *source_fine.shape[2:],
                    )
                else:
                    final_structured_evidence = F.interpolate(
                        previous_structured_evidence,
                        size=source_fine.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                residual_velocity = residual_velocity + self.final_structured_adapter(
                    evidence=torch.nan_to_num(final_structured_evidence, nan=0.0, posinf=0.0, neginf=0.0),
                    raw_displacement=final_displacement,
                    confidence=confidence,
                    entropy=entropy,
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
