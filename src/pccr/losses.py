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
            resized_target = F.interpolate(
                matchability_target.float(),
                size=prediction.matchability_logits.shape[2:],
                mode="nearest",
            ).long().squeeze(1)
            losses.append(F.cross_entropy(prediction.matchability_logits, resized_target.clamp(min=0, max=2)))
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
        segmentation_supervision_weight: float = 0.0,
        smoothness_weight: float = 0.02,
        jacobian_weight: float = 0.01,
        inverse_consistency_weight: float = 0.1,
        correspondence_weight: float = 0.2,
        residual_velocity_weight: float = 0.0,
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
        self.inverse_consistency_weight = inverse_consistency_weight
        self.correspondence_weight = correspondence_weight
        self.residual_velocity_weight = residual_velocity_weight
        self.num_labels = num_labels
        self.image_loss = LNCCLoss()
        self.segmentation_loss = DiceLoss(num_class=num_labels)
        self.smoothness = Grad3D(penalty="l2")
        self.jacobian = NegativeJacobianLoss()
        self.inverse_consistency = InverseConsistencyLoss(image_size)
        self.correspondence_consistency = CorrespondenceConsistencyLoss()
        self.segmentation_transformer = SpatialTransformer(image_size)
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

        source_onehot = get_one_hot(source_label, self.num_labels).float()
        target_onehot = get_one_hot(target_label, self.num_labels).float()
        warped_source = self.segmentation_transformer(source_onehot, forward_disp.float())
        warped_target = self.segmentation_transformer(target_onehot, backward_disp.float())
        forward_loss = self.segmentation_loss(warped_source, target_label.long())
        backward_loss = self.segmentation_loss(warped_target, source_label.long())
        return 0.5 * (forward_loss + backward_loss)

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
                + 0.25 * matchability_loss
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
        smoothness = self.smoothness(forward_disp) + self.smoothness(backward_disp)
        jacobian = self.jacobian(forward_disp) + self.jacobian(backward_disp)
        inverse = self.inverse_consistency(forward_disp, backward_disp)
        correspondence = self.correspondence_consistency(
            model_outputs["forward_matches"],
            model_outputs["backward_matches"],
        )
        residual_velocity = self._residual_velocity_penalty(
            model_outputs.get("forward_final_residual_velocity"),
            model_outputs.get("backward_final_residual_velocity"),
        )
        similarity = torch.nan_to_num(similarity, nan=1.0, posinf=1e3, neginf=1e3)
        segmentation = torch.nan_to_num(segmentation, nan=0.0, posinf=1e3, neginf=0.0)
        smoothness = torch.nan_to_num(smoothness, nan=0.0, posinf=1e3, neginf=0.0)
        jacobian = torch.nan_to_num(jacobian, nan=0.0, posinf=1e3, neginf=0.0)
        inverse = torch.nan_to_num(inverse, nan=0.0, posinf=1e3, neginf=0.0)
        correspondence = torch.nan_to_num(correspondence, nan=0.0, posinf=1e3, neginf=0.0)
        if residual_velocity is None:
            residual_velocity = forward_disp.new_tensor(0.0)
        residual_velocity = torch.nan_to_num(residual_velocity, nan=0.0, posinf=1e3, neginf=0.0)
        avg = (
            self.image_loss_weight * similarity
            + self.segmentation_supervision_weight * segmentation
            + self.smoothness_weight * smoothness
            + self.jacobian_weight * jacobian
            + self.inverse_consistency_weight * inverse
            + self.correspondence_weight * correspondence
            + self.residual_velocity_weight * residual_velocity
        )
        avg = torch.nan_to_num(avg, nan=1e3, posinf=1e3, neginf=1e3)
        return {
            "image": similarity,
            "segmentation": segmentation,
            "smoothness": smoothness,
            "jacobian": jacobian,
            "inverse": inverse,
            "correspondence": correspondence,
            "residual_velocity": residual_velocity,
            "avg_loss": avg,
        }
