from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformation import SpatialTransformer
from src.pccr.losses import RegistrationCriterion
from src.utils import get_one_hot


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, num_class: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_class = num_class
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true_onehot: torch.Tensor) -> torch.Tensor:
        spatial_sum = y_true_onehot.sum(dim=[2, 3, 4]).clamp_min(1.0)
        weights = 1.0 / spatial_sum.pow(2)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        intersection = (y_pred * y_true_onehot).sum(dim=[2, 3, 4])
        denom = (y_pred + y_true_onehot).sum(dim=[2, 3, 4])
        numerator = (weights * intersection).sum(dim=1)
        denominator = (weights * denom).sum(dim=1)
        gdl = 1.0 - 2.0 * numerator / (denominator + self.eps)
        return gdl.mean()


class RegistrationCriterionV6(RegistrationCriterion):
    def __init__(
        self,
        image_size: list[int],
        *args,
        seg_multiscale_weights: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(image_size, *args, **kwargs)
        num_labels = kwargs.get("num_labels", 36)
        self.gdl = GeneralizedDiceLoss(num_class=num_labels)
        self.seg_multiscale_weights: list[float] = seg_multiscale_weights or [1.0, 0.5, 0.25]
        self._full_size = list(image_size)
        self._half_size = [s // 2 for s in image_size]

        # Cached transformers for each scale — avoids re-allocating the sampling grid
        # every forward pass (critical for the 0.25x scale which is neither full nor half).
        self._transformer_cache: dict[tuple, SpatialTransformer] = {}

        # Hot-stage gating controls correspondence stages.  Decoder fitting keeps
        # local-refinement stages active and only drops explicitly cold stages.
        self._hot_stages: frozenset[int] = frozenset({3})
        self._cold_stages: frozenset[int] = frozenset()

    def set_hot_stages(self, stages: frozenset[int]) -> None:
        self._hot_stages = stages

    def set_cold_stages(self, stages: frozenset[int]) -> None:
        self._cold_stages = stages

    def _get_transformer(self, size: list[int]) -> SpatialTransformer:
        key = tuple(size)
        if key == tuple(self._full_size):
            return self.segmentation_transformer
        if key == tuple(self._half_size):
            return self.segmentation_transformer_half
        if key not in self._transformer_cache:
            self._transformer_cache[key] = SpatialTransformer(list(size))
        t = self._transformer_cache[key]
        # move to the right device lazily
        target_device = next(self.segmentation_transformer.parameters(), None)
        if target_device is None:
            buf = next(iter(self.segmentation_transformer.buffers()), None)
            target_device = buf.device if buf is not None else torch.device("cpu")
        else:
            target_device = target_device.device
        return t.to(target_device)

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

        src_lbl = source_label.long()
        tgt_lbl = target_label.long()
        src_oh = get_one_hot(src_lbl, self.num_labels).float()
        tgt_oh = get_one_hot(tgt_lbl, self.num_labels).float()

        total_loss = forward_disp.new_tensor(0.0)
        total_weight = 0.0

        for scale_idx, scale_weight in enumerate(self.seg_multiscale_weights):
            if scale_weight <= 0.0:
                continue
            factor = 2 ** scale_idx
            if factor > 1:
                disp_scale = F.interpolate(
                    forward_disp.float(),
                    scale_factor=1.0 / factor,
                    mode="trilinear",
                    align_corners=False,
                ) / factor
                disp_bwd_scale = F.interpolate(
                    backward_disp.float(),
                    scale_factor=1.0 / factor,
                    mode="trilinear",
                    align_corners=False,
                ) / factor
                src_oh_s = F.avg_pool3d(src_oh, kernel_size=factor, stride=factor, ceil_mode=False)
                tgt_oh_s = F.avg_pool3d(tgt_oh, kernel_size=factor, stride=factor, ceil_mode=False)
            else:
                disp_scale = forward_disp.float()
                disp_bwd_scale = backward_disp.float()
                src_oh_s = src_oh
                tgt_oh_s = tgt_oh

            size = list(disp_scale.shape[2:])
            transformer = self._get_transformer(size)
            warped_src = transformer(src_oh_s, disp_scale)
            warped_tgt = transformer(tgt_oh_s, disp_bwd_scale)

            fwd_loss = self.gdl(warped_src, tgt_oh_s)
            bwd_loss = self.gdl(warped_tgt, src_oh_s)
            total_loss = total_loss + scale_weight * 0.5 * (fwd_loss + bwd_loss)
            total_weight += scale_weight

        return total_loss / max(total_weight, 1e-8)

    def _regularization(
        self,
        forward_disp: torch.Tensor,
        backward_disp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Smoothness and Jacobian at full-res (the v6 fix).
        smoothness = self.smoothness(forward_disp.float()) + self.smoothness(backward_disp.float())
        jacobian = self.jacobian(forward_disp.float()) + self.jacobian(backward_disp.float())
        # Inverse consistency at half-res: same proxy as v5, avoids the full-res
        # SpatialTransformer warp which is memory-expensive at 160x192x224.
        fwd_half = F.interpolate(forward_disp.float(), scale_factor=0.5, mode="trilinear", align_corners=False) * 0.5
        bwd_half = F.interpolate(backward_disp.float(), scale_factor=0.5, mode="trilinear", align_corners=False) * 0.5
        inverse = self.inverse_consistency_half(fwd_half, bwd_half)
        return smoothness, jacobian, inverse

    @staticmethod
    def _filter_by_hot_stages(d: dict, hot_stages: frozenset[int]) -> dict:
        if not hot_stages:
            return d
        return {k: v for k, v in d.items() if k in hot_stages}

    @staticmethod
    def _drop_cold_stages(d: dict, cold_stages: frozenset[int]) -> dict:
        if not cold_stages:
            return d
        return {k: v for k, v in d.items() if k not in cold_stages}

    def forward(self, model_outputs, source, target, source_label=None, target_label=None, synthetic_targets=None):
        if self.phase == "synthetic" and synthetic_targets is not None:
            return super().forward(
                model_outputs, source, target,
                source_label=source_label, target_label=target_label,
                synthetic_targets=synthetic_targets,
            )

        forward_disp = model_outputs["phi_s2t"]
        backward_disp = model_outputs["phi_t2s"]
        moved_source = model_outputs["moved_source"]
        moved_target = model_outputs["moved_target"]

        similarity = self._multiscale_similarity(moved_source, moved_target, source, target)
        segmentation = self._segmentation_supervision(forward_disp, backward_disp, source_label, target_label)
        smoothness, jacobian, inverse = self._regularization(forward_disp, backward_disp)

        # During warmup, cold correspondence stages are excluded so random matches
        # don't pull trained stages toward garbage. Decoder fitting still supervises
        # established local-refinement stages.
        hot = self._hot_stages
        cold = self._cold_stages
        fwd_matches_hot = self._filter_by_hot_stages(model_outputs["forward_matches"], hot)
        bwd_matches_hot = self._filter_by_hot_stages(model_outputs["backward_matches"], hot)
        correspondence = self.correspondence_consistency(fwd_matches_hot, bwd_matches_hot)

        decoder_fitting = self.decoder_fitting(
            predicted_displacements=model_outputs["forward_stage_displacements"],
            target_displacements=self._drop_cold_stages(model_outputs["forward_stage_target_displacements"], cold),
            confidences=self._drop_cold_stages(model_outputs["forward_stage_target_confidences"], cold),
            entropies=self._drop_cold_stages(model_outputs["forward_stage_target_entropies"], cold),
            margins=self._drop_cold_stages(model_outputs.get("forward_stage_target_margins", {}), cold),
        )
        decoder_fitting = decoder_fitting + self.decoder_fitting(
            predicted_displacements=model_outputs["backward_stage_displacements"],
            target_displacements=self._drop_cold_stages(model_outputs["backward_stage_target_displacements"], cold),
            confidences=self._drop_cold_stages(model_outputs["backward_stage_target_confidences"], cold),
            entropies=self._drop_cold_stages(model_outputs["backward_stage_target_entropies"], cold),
            margins=self._drop_cold_stages(model_outputs.get("backward_stage_target_margins", {}), cold),
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
        decoder_fitting = torch.nan_to_num(decoder_fitting, nan=0.0, posinf=1e3, neginf=0.0)
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
            + self.decoder_fitting_weight * decoder_fitting
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
            "decoder_fitting": decoder_fitting,
            "residual_velocity": residual_velocity,
            "avg_loss": avg,
        }
