from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import DiceLoss, Grad3D
from src.model.transformation import SpatialTransformer
from src.pccr.utils import resize_displacement
from src.utils import get_one_hot


class LNCCLoss(nn.Module):
    def __init__(self, window_size: int = 5) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        padding = self.window_size // 2
        conv = lambda x: F.avg_pool3d(x, kernel_size=self.window_size, stride=1, padding=padding)

        src_mean = conv(source)
        tgt_mean = conv(target)
        src_var = conv(source * source) - src_mean * src_mean
        tgt_var = conv(target * target) - tgt_mean * tgt_mean
        cross = conv(source * target) - src_mean * tgt_mean
        lncc = cross * cross / (src_var * tgt_var + 1e-5)
        return torch.nan_to_num(1.0 - lncc.mean(), nan=1.0, posinf=1.0, neginf=1.0)


class MultiWindowLNCCLoss(nn.Module):
    """Average LNCC across several window sizes.

    Smaller windows respond to fine structures (e.g. hippocampus, amygdala);
    larger windows stabilise the cortical sheet. Drop-in replacement for
    LNCCLoss when ``windows`` has a single entry.
    """

    def __init__(self, windows: list[int]) -> None:
        super().__init__()
        if not windows:
            raise ValueError("MultiWindowLNCCLoss requires at least one window size.")
        self.losses = nn.ModuleList([LNCCLoss(window_size=int(w)) for w in windows])

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_window = [loss(source, target) for loss in self.losses]
        return torch.stack(per_window).mean()


class HyperelasticRegularizer(nn.Module):
    """Penalise volume distortion of the displacement field symmetrically.

    Loss = mean( (det(J) - 1)^p + (1/det(J) - 1)^p ).

    Unlike NegativeJacobianLoss which only penalises folding (det <= 0),
    this penalises both compression (det << 1) and expansion (det >> 1)
    symmetrically, allowing large but volume-preserving deformations and
    keeping SDlogJ small without sacrificing dice.
    """

    def __init__(self, power: float = 2.0, det_clamp: float = 0.05) -> None:
        super().__init__()
        self.power = float(power)
        self.det_clamp = float(det_clamp)

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        dx = displacement[:, :, 1:, :-1, :-1] - displacement[:, :, :-1, :-1, :-1]
        dy = displacement[:, :, :-1, 1:, :-1] - displacement[:, :, :-1, :-1, :-1]
        dz = displacement[:, :, :-1, :-1, 1:] - displacement[:, :, :-1, :-1, :-1]

        jac = torch.zeros(
            displacement.shape[0],
            3,
            3,
            *dx.shape[2:],
            device=displacement.device,
            dtype=displacement.dtype,
        )
        jac[:, 0, 0] = 1 + dx[:, 0]
        jac[:, 0, 1] = dy[:, 0]
        jac[:, 0, 2] = dz[:, 0]
        jac[:, 1, 0] = dx[:, 1]
        jac[:, 1, 1] = 1 + dy[:, 1]
        jac[:, 1, 2] = dz[:, 1]
        jac[:, 2, 0] = dx[:, 2]
        jac[:, 2, 1] = dy[:, 2]
        jac[:, 2, 2] = 1 + dz[:, 2]
        det = torch.det(jac.permute(0, 3, 4, 5, 1, 2))
        det = torch.nan_to_num(det, nan=1.0, posinf=1e3, neginf=-1e3)
        det = det.clamp(min=self.det_clamp)
        forward_term = (det - 1.0).abs().pow(self.power)
        inverse_term = (1.0 / det - 1.0).abs().pow(self.power)
        return (forward_term + inverse_term).mean()


class NegativeJacobianLoss(nn.Module):
    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        dx = displacement[:, :, 1:, :-1, :-1] - displacement[:, :, :-1, :-1, :-1]
        dy = displacement[:, :, :-1, 1:, :-1] - displacement[:, :, :-1, :-1, :-1]
        dz = displacement[:, :, :-1, :-1, 1:] - displacement[:, :, :-1, :-1, :-1]

        jac = torch.zeros(
            displacement.shape[0],
            3,
            3,
            *dx.shape[2:],
            device=displacement.device,
            dtype=displacement.dtype,
        )
        jac[:, 0, 0] = 1 + dx[:, 0]
        jac[:, 0, 1] = dy[:, 0]
        jac[:, 0, 2] = dz[:, 0]
        jac[:, 1, 0] = dx[:, 1]
        jac[:, 1, 1] = 1 + dy[:, 1]
        jac[:, 1, 2] = dz[:, 1]
        jac[:, 2, 0] = dx[:, 2]
        jac[:, 2, 1] = dy[:, 2]
        jac[:, 2, 2] = 1 + dz[:, 2]
        det = torch.det(jac.permute(0, 3, 4, 5, 1, 2))
        det = torch.nan_to_num(det, nan=0.0, posinf=1e3, neginf=-1e3)
        return F.relu(-det).mean()


class InverseConsistencyLoss(nn.Module):
    def __init__(self, image_size: list[int]) -> None:
        super().__init__()
        self.transformer = SpatialTransformer(image_size)

    def forward(self, forward_disp: torch.Tensor, backward_disp: torch.Tensor) -> torch.Tensor:
        composed = forward_disp + self.transformer(backward_disp, forward_disp)
        return torch.nan_to_num(composed.square().mean(), nan=0.0, posinf=1e3, neginf=0.0)


class CorrespondenceConsistencyLoss(nn.Module):
    def forward(self, forward_matches: dict[int, object], backward_matches: dict[int, object]) -> torch.Tensor:
        losses = []
        reference_tensor = None
        for stage_id in forward_matches:
            forward_disp = forward_matches[stage_id].raw_displacement
            backward_disp = backward_matches[stage_id].raw_displacement
            reference_tensor = forward_disp
            backward_disp = resize_displacement(backward_disp, forward_disp.shape[2:])
            losses.append((forward_disp + backward_disp).abs().mean())
        if losses:
            return torch.stack(losses).mean()
        if reference_tensor is None:
            raise ValueError("CorrespondenceConsistencyLoss requires at least one match stage.")
        return reference_tensor.new_tensor(0.0)


class DecoderFittingLoss(nn.Module):
    def __init__(
        self,
        detach_target: bool = True,
        entropy_threshold: float = -1.0,
        confidence_percentile: float = 0.0,
        margin_power: float = 0.0,
        margin_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.detach_target = detach_target
        self.entropy_threshold = entropy_threshold
        self.confidence_percentile = confidence_percentile
        self.margin_power = margin_power
        self.margin_min = margin_min

    def _single_stage(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
        entropy: torch.Tensor,
        margin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.detach_target:
            target = target.detach()
        weight = confidence.detach().clamp_min(0.0)
        if self.entropy_threshold >= 0.0:
            weight = weight * (entropy.detach() <= self.entropy_threshold).float()
        if self.confidence_percentile > 0.0:
            quantile = torch.quantile(
                confidence.detach().reshape(confidence.shape[0], -1),
                q=self.confidence_percentile,
                dim=1,
                keepdim=True,
            ).view(confidence.shape[0], 1, 1, 1, 1)
            weight = weight * (confidence.detach() >= quantile).float()
        if margin is not None and self.margin_power > 0.0:
            margin_weight = (margin.detach() - self.margin_min).clamp_min(0.0)
            margin_weight = margin_weight.pow(self.margin_power)
            weight = weight * margin_weight

        if weight.sum() <= 0:
            return predicted.new_tensor(0.0)

        per_voxel = F.smooth_l1_loss(predicted, target, reduction="none")
        per_voxel = per_voxel.mean(dim=1, keepdim=True)
        return (per_voxel * weight).sum() / weight.sum().clamp_min(1.0)

    def forward(
        self,
        predicted_displacements: dict[int, torch.Tensor],
        target_displacements: dict[int, torch.Tensor],
        confidences: dict[int, torch.Tensor],
        entropies: dict[int, torch.Tensor],
        margins: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        losses = []
        for stage_id, target in target_displacements.items():
            if stage_id not in predicted_displacements:
                continue
            losses.append(
                self._single_stage(
                    predicted=predicted_displacements[stage_id],
                    target=target,
                    confidence=confidences[stage_id],
                    entropy=entropies[stage_id],
                    margin=margins.get(stage_id) if margins is not None else None,
                )
            )
        if not losses:
            reference = next(iter(predicted_displacements.values()), None)
            if reference is None:
                raise ValueError("DecoderFittingLoss requires at least one predicted stage displacement.")
            return reference.new_tensor(0.0)
        return torch.stack(losses).mean()


@dataclass
class SyntheticTargets:
    canonical_source: torch.Tensor
    canonical_target: torch.Tensor
    valid_source: torch.Tensor
    valid_target: torch.Tensor
    matchability_source: torch.Tensor
    matchability_target: torch.Tensor


class SyntheticPointmapLoss(nn.Module):
    def forward(self, predicted: dict[int, object], target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        losses = []
        for _, outputs in predicted.items():
            resized_target = F.interpolate(
                target,
                size=outputs.canonical_coords.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            resized_valid = F.interpolate(
                valid_mask.float(),
                size=outputs.canonical_coords.shape[2:],
                mode="nearest",
            )
            diff = (outputs.canonical_coords - resized_target).abs() * resized_valid
            losses.append(diff.sum() / resized_valid.sum().clamp_min(1.0))
        return torch.stack(losses).mean()


class DescriptorContrastiveLoss(nn.Module):
    def forward(
        self,
        source_outputs: dict[int, object],
        target_outputs: dict[int, object],
        canonical_source: torch.Tensor,
        canonical_target: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for stage_id, source_prediction in source_outputs.items():
            target_prediction = target_outputs[stage_id]
            stage_size = source_prediction.descriptors.shape[2:]

            resized_source = F.interpolate(
                canonical_source,
                size=stage_size,
                mode="trilinear",
                align_corners=False,
            ).flatten(start_dim=2).transpose(1, 2)
            resized_target = F.interpolate(
                canonical_target,
                size=stage_size,
                mode="trilinear",
                align_corners=False,
            ).flatten(start_dim=2).transpose(1, 2)
            nearest = torch.cdist(resized_source, resized_target).argmin(dim=-1)

            source_desc = source_prediction.descriptors.flatten(start_dim=2).transpose(1, 2)
            target_desc = target_prediction.descriptors.flatten(start_dim=2).transpose(1, 2)
            gathered_target = torch.gather(
                target_desc,
                dim=1,
                index=nearest.unsqueeze(-1).expand(-1, -1, target_desc.shape[-1]),
            )
            losses.append(1.0 - F.cosine_similarity(source_desc, gathered_target, dim=-1).mean())
        return torch.stack(losses).mean()


class SyntheticMatchingLoss(nn.Module):
    def forward(
        self,
        match_outputs: dict[int, object],
        canonical_source: torch.Tensor,
        canonical_target: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for _, prediction in match_outputs.items():
            stage_size = tuple(prediction.raw_displacement.shape[2:])
            resized_source = F.interpolate(
                canonical_source,
                size=stage_size,
                mode="trilinear",
                align_corners=False,
            ).flatten(start_dim=2).transpose(1, 2)
            resized_target = F.interpolate(
                canonical_target,
                size=stage_size,
                mode="trilinear",
                align_corners=False,
            ).flatten(start_dim=2).transpose(1, 2)
            nearest = torch.cdist(resized_source, resized_target).argmin(dim=-1)

            target_positions = torch.stack(
                torch.meshgrid(
                    torch.arange(stage_size[0], device=resized_source.device, dtype=resized_source.dtype),
                    torch.arange(stage_size[1], device=resized_source.device, dtype=resized_source.dtype),
                    torch.arange(stage_size[2], device=resized_source.device, dtype=resized_source.dtype),
                    indexing="ij",
                ),
                dim=0,
            ).unsqueeze(0).flatten(start_dim=2).transpose(1, 2)
            target_positions = target_positions.expand(resized_source.shape[0], -1, -1)
            gathered_target = torch.gather(
                target_positions,
                dim=1,
                index=nearest.unsqueeze(-1).expand(-1, -1, 3),
            )
            losses.append((prediction.expected_target_positions - gathered_target).abs().mean())
        return torch.stack(losses).mean()


class MatchabilityClassificationLoss(nn.Module):
    def forward(self, outputs: dict[int, object], matchability_target: torch.Tensor) -> torch.Tensor:
        losses = []
        for _, prediction in outputs.items():
            if prediction.matchability_logits.shape[1] <= 1:
                continue
            resized_target = F.interpolate(
                matchability_target.float(),
                size=prediction.matchability_logits.shape[2:],
                mode="nearest",
            ).long().squeeze(1)
            max_label = prediction.matchability_logits.shape[1] - 1
            losses.append(F.cross_entropy(prediction.matchability_logits, resized_target.clamp(min=0, max=max_label)))
        if not losses:
            reference = next(iter(outputs.values()), None)
            if reference is None:
                raise ValueError("MatchabilityClassificationLoss requires at least one output stage.")
            return reference.matchability_logits.new_tensor(0.0)
        return torch.stack(losses).mean()


class UncertaintyCalibrationLoss(nn.Module):
    def forward(self, outputs: dict[int, object], target: torch.Tensor) -> torch.Tensor:
        losses = []
        for _, prediction in outputs.items():
            resized_target = F.interpolate(
                target,
                size=prediction.canonical_coords.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            error = (prediction.canonical_coords - resized_target).square().mean(dim=1, keepdim=True)
            log_var = prediction.uncertainty.clamp_min(1e-5).log()
            losses.append((torch.exp(-log_var) * error + log_var).mean())
        return torch.stack(losses).mean()


class RegistrationCriterion(nn.Module):
    def __init__(
        self,
        image_size: list[int],
        phase: str = "real",
        image_loss_weight: float = 1.0,
        multiscale_similarity_factors: list[int] | None = None,
        multiscale_similarity_weights: list[float] | None = None,
        lncc_window_size: int = 5,
        lncc_windows: list[int] | None = None,
        segmentation_supervision_weight: float = 0.0,
        per_stage_segmentation_weights: dict[int, float] | None = None,
        smoothness_weight: float = 0.02,
        jacobian_weight: float = 0.01,
        hyperelastic_weight: float = 0.0,
        hyperelastic_power: float = 2.0,
        inverse_consistency_weight: float = 0.1,
        correspondence_weight: float = 0.2,
        synthetic_matchability_weight: float = 0.25,
        residual_velocity_weight: float = 0.0,
        decoder_fitting_weight: float = 0.0,
        decoder_fitting_detach_target: bool = True,
        decoder_fitting_entropy_threshold: float = -1.0,
        decoder_fitting_confidence_percentile: float = 0.0,
        decoder_fitting_margin_power: float = 0.0,
        decoder_fitting_margin_min: float = 0.0,
        num_labels: int = 86,
    ) -> None:
        super().__init__()
        self.phase = phase
        self.image_loss_weight = image_loss_weight
        self.multiscale_similarity_factors = multiscale_similarity_factors or [1]
        self.multiscale_similarity_weights = multiscale_similarity_weights or [1.0]
        if len(self.multiscale_similarity_factors) != len(self.multiscale_similarity_weights):
            raise ValueError("multiscale similarity factors and weights must have the same length.")
        self.segmentation_supervision_weight = segmentation_supervision_weight
        self.smoothness_weight = smoothness_weight
        self.jacobian_weight = jacobian_weight
        self.hyperelastic_weight = float(hyperelastic_weight)
        self.inverse_consistency_weight = inverse_consistency_weight
        self.correspondence_weight = correspondence_weight
        self.synthetic_matchability_weight = synthetic_matchability_weight
        self.residual_velocity_weight = residual_velocity_weight
        self.decoder_fitting_weight = decoder_fitting_weight
        self.num_labels = num_labels
        if lncc_windows:
            self.image_loss = MultiWindowLNCCLoss(list(lncc_windows))
        else:
            self.image_loss = LNCCLoss(window_size=lncc_window_size)
        self.segmentation_loss = DiceLoss(num_class=num_labels)
        self.smoothness = Grad3D(penalty="l2")
        self.jacobian = NegativeJacobianLoss()
        self.hyperelastic = HyperelasticRegularizer(power=hyperelastic_power)
        self.inverse_consistency = InverseConsistencyLoss(image_size)
        self.inverse_consistency_half = InverseConsistencyLoss([s // 2 for s in image_size])
        self.correspondence_consistency = CorrespondenceConsistencyLoss()
        self.decoder_fitting = DecoderFittingLoss(
            detach_target=decoder_fitting_detach_target,
            entropy_threshold=decoder_fitting_entropy_threshold,
            confidence_percentile=decoder_fitting_confidence_percentile,
            margin_power=decoder_fitting_margin_power,
            margin_min=decoder_fitting_margin_min,
        )
        self.segmentation_transformer = SpatialTransformer(image_size)
        half_size = [s // 2 for s in image_size]
        self.segmentation_transformer_half = SpatialTransformer(half_size)
        self.per_stage_segmentation_weights: dict[int, float] = {
            int(stage_id): float(weight)
            for stage_id, weight in (per_stage_segmentation_weights or {}).items()
            if float(weight) > 0.0
        }
        self.per_stage_seg_transformers = nn.ModuleDict()
        for stage_id in self.per_stage_segmentation_weights:
            factor = 2 ** stage_id
            stage_size = [max(1, s // factor) for s in image_size]
            self.per_stage_seg_transformers[str(stage_id)] = SpatialTransformer(stage_size)
        self.synthetic_pointmap = SyntheticPointmapLoss()
        self.synthetic_descriptor = DescriptorContrastiveLoss()
        self.synthetic_matching = SyntheticMatchingLoss()
        self.synthetic_matchability = MatchabilityClassificationLoss()
        self.synthetic_uncertainty = UncertaintyCalibrationLoss()

    @staticmethod
    def _downsample_if_needed(tensor: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return tensor
        return F.avg_pool3d(tensor, kernel_size=factor, stride=factor, ceil_mode=False)

    def _multiscale_similarity(
        self,
        moved_source: torch.Tensor,
        moved_target: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for factor, weight in zip(self.multiscale_similarity_factors, self.multiscale_similarity_weights):
            moved_source_scaled = self._downsample_if_needed(moved_source, factor)
            moved_target_scaled = self._downsample_if_needed(moved_target, factor)
            source_scaled = self._downsample_if_needed(source, factor)
            target_scaled = self._downsample_if_needed(target, factor)
            losses.append(
                weight
                * (
                    self.image_loss(moved_source_scaled, target_scaled)
                    + self.image_loss(moved_target_scaled, source_scaled)
                )
            )
        return torch.stack(losses).sum()

    def _segmentation_supervision(
        self,
        forward_disp: torch.Tensor,
        backward_disp: torch.Tensor,
        source_label: torch.Tensor | None,
        target_label: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self.segmentation_supervision_weight <= 0.0
            or source_label is None
            or target_label is None
        ):
            return forward_disp.new_tensor(0.0)

        interp_kw = dict(scale_factor=0.5, align_corners=False)
        fwd_half = F.interpolate(forward_disp.float(), mode="trilinear", **interp_kw) * 0.5
        bwd_half = F.interpolate(backward_disp.float(), mode="trilinear", **interp_kw) * 0.5
        src_lbl_half = F.interpolate(source_label.float(), scale_factor=0.5, mode="nearest").long()
        tgt_lbl_half = F.interpolate(target_label.float(), scale_factor=0.5, mode="nearest").long()
        source_onehot = get_one_hot(src_lbl_half, self.num_labels).float()
        target_onehot = get_one_hot(tgt_lbl_half, self.num_labels).float()
        warped_source = self.segmentation_transformer_half(source_onehot, fwd_half)
        warped_target = self.segmentation_transformer_half(target_onehot, bwd_half)
        forward_loss = self.segmentation_loss(warped_source, tgt_lbl_half.long())
        backward_loss = self.segmentation_loss(warped_target, src_lbl_half.long())
        return 0.5 * (forward_loss + backward_loss)

    def _per_stage_segmentation(
        self,
        forward_stage_disps: dict[int, torch.Tensor] | None,
        backward_stage_disps: dict[int, torch.Tensor] | None,
        source_label: torch.Tensor | None,
        target_label: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply Dice supervision at each stage's intermediate displacement.

        Each stage k operates at resolution image_size / 2**k with displacements
        already in that stage's voxel units, so labels are nearest-downsampled
        to that resolution and warped with the stage's spatial transformer.
        Bidirectional, returns a *weighted-summed* loss (not averaged) so the
        per-stage weights act as direct loss multipliers.
        """
        if (
            not self.per_stage_segmentation_weights
            or source_label is None
            or target_label is None
            or forward_stage_disps is None
            or backward_stage_disps is None
        ):
            reference = next(
                iter((forward_stage_disps or {}).values()),
                source_label if source_label is not None else None,
            )
            if reference is None:
                return torch.zeros((), device=source_label.device if source_label is not None else "cpu")
            return reference.new_tensor(0.0) if torch.is_tensor(reference) else torch.zeros(())

        accumulated = None
        for stage_id, weight in self.per_stage_segmentation_weights.items():
            if stage_id not in forward_stage_disps or stage_id not in backward_stage_disps:
                continue
            transformer = self.per_stage_seg_transformers[str(stage_id)]
            fwd_disp = forward_stage_disps[stage_id].float()
            bwd_disp = backward_stage_disps[stage_id].float()
            stage_size = tuple(fwd_disp.shape[2:])
            src_lbl_stage = F.interpolate(
                source_label.float(), size=stage_size, mode="nearest"
            ).long()
            tgt_lbl_stage = F.interpolate(
                target_label.float(), size=stage_size, mode="nearest"
            ).long()
            source_onehot = get_one_hot(src_lbl_stage, self.num_labels).float()
            target_onehot = get_one_hot(tgt_lbl_stage, self.num_labels).float()
            warped_source = transformer(source_onehot, fwd_disp)
            warped_target = transformer(target_onehot, bwd_disp)
            forward_loss = self.segmentation_loss(warped_source, tgt_lbl_stage.long())
            backward_loss = self.segmentation_loss(warped_target, src_lbl_stage.long())
            stage_loss = 0.5 * (forward_loss + backward_loss) * float(weight)
            accumulated = stage_loss if accumulated is None else accumulated + stage_loss
        if accumulated is None:
            reference = next(iter(forward_stage_disps.values()))
            return reference.new_tensor(0.0)
        return accumulated

    @staticmethod
    def _residual_velocity_penalty(*velocity_fields: torch.Tensor | None) -> torch.Tensor | None:
        valid = [field.square().mean() for field in velocity_fields if field is not None]
        if not valid:
            return None
        return torch.stack(valid).mean()

    def forward(
        self,
        model_outputs: dict[str, object],
        source: torch.Tensor,
        target: torch.Tensor,
        source_label: torch.Tensor | None = None,
        target_label: torch.Tensor | None = None,
        synthetic_targets: SyntheticTargets | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.phase == "synthetic" and synthetic_targets is not None:
            forward_pointmaps = model_outputs["forward_pointmaps"]
            backward_pointmaps = model_outputs["backward_pointmaps"]
            forward_matches = model_outputs["forward_matches"]
            backward_matches = model_outputs["backward_matches"]

            pointmap_loss = self.synthetic_pointmap(
                forward_pointmaps,
                synthetic_targets.canonical_source,
                synthetic_targets.valid_source,
            )
            pointmap_loss = pointmap_loss + self.synthetic_pointmap(
                backward_pointmaps,
                synthetic_targets.canonical_target,
                synthetic_targets.valid_target,
            )
            descriptor_loss = self.synthetic_descriptor(
                forward_pointmaps,
                backward_pointmaps,
                synthetic_targets.canonical_source,
                synthetic_targets.canonical_target,
            )
            matching_loss = self.synthetic_matching(
                forward_matches,
                synthetic_targets.canonical_source,
                synthetic_targets.canonical_target,
            )
            matching_loss = matching_loss + self.synthetic_matching(
                backward_matches,
                synthetic_targets.canonical_target,
                synthetic_targets.canonical_source,
            )
            matchability_loss = self.synthetic_matchability(
                forward_pointmaps,
                synthetic_targets.matchability_source,
            )
            matchability_loss = matchability_loss + self.synthetic_matchability(
                backward_pointmaps,
                synthetic_targets.matchability_target,
            )
            uncertainty_loss = self.synthetic_uncertainty(
                forward_pointmaps,
                synthetic_targets.canonical_source,
            )
            uncertainty_loss = uncertainty_loss + self.synthetic_uncertainty(
                backward_pointmaps,
                synthetic_targets.canonical_target,
            )
            avg = (
                1.0 * pointmap_loss
                + 1.0 * matching_loss
                + 0.25 * descriptor_loss
                + self.synthetic_matchability_weight * matchability_loss
                + 0.1 * uncertainty_loss
            )
            return {
                "pointmap": pointmap_loss,
                "matching": matching_loss,
                "descriptor": descriptor_loss,
                "matchability": matchability_loss,
                "uncertainty": uncertainty_loss,
                "avg_loss": avg,
            }

        forward_disp = model_outputs["phi_s2t"]
        backward_disp = model_outputs["phi_t2s"]
        moved_source = model_outputs["moved_source"]
        moved_target = model_outputs["moved_target"]

        similarity = self._multiscale_similarity(moved_source, moved_target, source, target)
        segmentation = self._segmentation_supervision(
            forward_disp,
            backward_disp,
            source_label,
            target_label,
        )
        per_stage_segmentation = self._per_stage_segmentation(
            model_outputs.get("forward_stage_displacements"),
            model_outputs.get("backward_stage_displacements"),
            source_label,
            target_label,
        )
        _reg_kw = dict(scale_factor=0.5, mode="trilinear", align_corners=False)
        fwd_reg = F.interpolate(forward_disp.float(), **_reg_kw) * 0.5
        bwd_reg = F.interpolate(backward_disp.float(), **_reg_kw) * 0.5
        smoothness = self.smoothness(fwd_reg) + self.smoothness(bwd_reg)
        jacobian = self.jacobian(fwd_reg) + self.jacobian(bwd_reg)
        if self.hyperelastic_weight > 0.0:
            hyperelastic = self.hyperelastic(fwd_reg) + self.hyperelastic(bwd_reg)
        else:
            hyperelastic = forward_disp.new_tensor(0.0)
        inverse = self.inverse_consistency_half(fwd_reg, bwd_reg)
        correspondence = self.correspondence_consistency(
            model_outputs["forward_matches"],
            model_outputs["backward_matches"],
        )
        decoder_fitting = self.decoder_fitting(
            predicted_displacements=model_outputs["forward_stage_displacements"],
            target_displacements=model_outputs["forward_stage_target_displacements"],
            confidences=model_outputs["forward_stage_target_confidences"],
            entropies=model_outputs["forward_stage_target_entropies"],
            margins=model_outputs.get("forward_stage_target_margins", {}),
        )
        decoder_fitting = decoder_fitting + self.decoder_fitting(
            predicted_displacements=model_outputs["backward_stage_displacements"],
            target_displacements=model_outputs["backward_stage_target_displacements"],
            confidences=model_outputs["backward_stage_target_confidences"],
            entropies=model_outputs["backward_stage_target_entropies"],
            margins=model_outputs.get("backward_stage_target_margins", {}),
        )
        residual_velocity = self._residual_velocity_penalty(
            model_outputs.get("forward_final_residual_velocity"),
            model_outputs.get("backward_final_residual_velocity"),
        )
        similarity = torch.nan_to_num(similarity, nan=1.0, posinf=1e3, neginf=1e3)
        segmentation = torch.nan_to_num(segmentation, nan=0.0, posinf=1e3, neginf=0.0)
        per_stage_segmentation = torch.nan_to_num(per_stage_segmentation, nan=0.0, posinf=1e3, neginf=0.0)
        smoothness = torch.nan_to_num(smoothness, nan=0.0, posinf=1e3, neginf=0.0)
        jacobian = torch.nan_to_num(jacobian, nan=0.0, posinf=1e3, neginf=0.0)
        hyperelastic = torch.nan_to_num(hyperelastic, nan=0.0, posinf=1e3, neginf=0.0)
        inverse = torch.nan_to_num(inverse, nan=0.0, posinf=1e3, neginf=0.0)
        correspondence = torch.nan_to_num(correspondence, nan=0.0, posinf=1e3, neginf=0.0)
        decoder_fitting = torch.nan_to_num(decoder_fitting, nan=0.0, posinf=1e3, neginf=0.0)
        if residual_velocity is None:
            residual_velocity = forward_disp.new_tensor(0.0)
        residual_velocity = torch.nan_to_num(residual_velocity, nan=0.0, posinf=1e3, neginf=0.0)
        avg = (
            self.image_loss_weight * similarity
            + self.segmentation_supervision_weight * segmentation
            + per_stage_segmentation
            + self.smoothness_weight * smoothness
            + self.jacobian_weight * jacobian
            + self.hyperelastic_weight * hyperelastic
            + self.inverse_consistency_weight * inverse
            + self.correspondence_weight * correspondence
            + self.decoder_fitting_weight * decoder_fitting
            + self.residual_velocity_weight * residual_velocity
        )
        avg = torch.nan_to_num(avg, nan=1e3, posinf=1e3, neginf=1e3)
        return {
            "image": similarity,
            "segmentation": segmentation,
            "per_stage_segmentation": per_stage_segmentation,
            "smoothness": smoothness,
            "jacobian": jacobian,
            "hyperelastic": hyperelastic,
            "inverse": inverse,
            "correspondence": correspondence,
            "decoder_fitting": decoder_fitting,
            "residual_velocity": residual_velocity,
            "avg_loss": avg,
        }
