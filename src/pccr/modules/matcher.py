from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pccr.modules.pointmap import PointmapOutputs
from src.pccr.utils import flatten_spatial, softmax_entropy, voxel_grid

LOGGER = logging.getLogger(__name__)


@dataclass
class MatchOutputs:
    expected_target_positions: torch.Tensor
    raw_displacement: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor
    margin: torch.Tensor
    entropy: torch.Tensor
    source_positions: torch.Tensor


def _batched_gather(target: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_items, channels = target.shape
    flat_target = target.reshape(batch_size * num_items, channels)
    batch_offsets = (torch.arange(batch_size, device=target.device).view(batch_size, 1, 1) * num_items)
    flat_indices = (indices + batch_offsets).reshape(-1)
    gathered = flat_target[flat_indices]
    return gathered.view(*indices.shape, channels)


def _batched_gather_scalar(target: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_items = target.shape
    flat_target = target.reshape(batch_size * num_items)
    batch_offsets = (torch.arange(batch_size, device=target.device).view(batch_size, 1, 1) * num_items)
    flat_indices = (indices + batch_offsets).reshape(-1)
    gathered = flat_target[flat_indices]
    return gathered.view(*indices.shape)


class CandidateRefinementModule(nn.Module):
    def __init__(self, descriptor_dim: int, hidden_dim: int, offset_scale: float) -> None:
        super().__init__()
        self.offset_scale = offset_scale
        input_dim = descriptor_dim * 2 + 12
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.logit_head = nn.Linear(hidden_dim, 1)
        self.offset_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        source_descriptor: torch.Tensor,
        target_descriptor: torch.Tensor,
        descriptor_similarity: torch.Tensor,
        canonical_delta: torch.Tensor,
        spatial_offset: torch.Tensor,
        canonical_distance: torch.Tensor,
        source_uncertainty: torch.Tensor,
        target_uncertainty: torch.Tensor,
        source_matchability: torch.Tensor,
        target_matchability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            [
                source_descriptor,
                target_descriptor,
                descriptor_similarity.unsqueeze(-1),
                canonical_delta,
                spatial_offset,
                canonical_distance.unsqueeze(-1),
                source_uncertainty.unsqueeze(-1),
                target_uncertainty.unsqueeze(-1),
                source_matchability.unsqueeze(-1),
                target_matchability.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.backbone(features)
        refined_logit = self.logit_head(hidden).squeeze(-1)
        offset = torch.tanh(self.offset_head(hidden)) * self.offset_scale
        return refined_logit, offset


class _BaseCorrelationMatcher(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        topk: int = 24,
        canonical_radius: float = 0.45,
        matchability_score_mode: str = "legacy",
        matchability_score_power: float = 0.5,
        confidence_mode: str = "max_prob",
        global_match_voxel_limit: int = 50_000,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.topk = topk
        self.canonical_radius = canonical_radius
        self.matchability_score_mode = matchability_score_mode
        self.matchability_score_power = matchability_score_power
        self.confidence_mode = confidence_mode
        self.global_match_voxel_limit = global_match_voxel_limit
        self._warned_skip_keys: set[str] = set()

    def _flatten_inputs(
        self,
        source: PointmapOutputs,
        target: PointmapOutputs,
    ) -> dict[str, torch.Tensor]:
        src_canonical = flatten_spatial(source.canonical_coords)
        tgt_canonical = flatten_spatial(target.canonical_coords)
        src_descriptor = flatten_spatial(source.descriptors)
        tgt_descriptor = flatten_spatial(target.descriptors)
        src_uncertainty = flatten_spatial(source.uncertainty).squeeze(-1)
        tgt_uncertainty = flatten_spatial(target.uncertainty).squeeze(-1)
        target_matchability = flatten_spatial(target.matchability_logits).softmax(dim=-1)[..., 0]
        source_matchability = flatten_spatial(source.matchability_logits).softmax(dim=-1)[..., 0]

        spatial_shape = tuple(source.canonical_coords.shape[2:])
        target_grid = voxel_grid(spatial_shape, source.canonical_coords.device)
        target_positions = flatten_spatial(target_grid.expand(source.canonical_coords.shape[0], -1, -1, -1, -1))
        source_positions = flatten_spatial(target_grid.expand(source.canonical_coords.shape[0], -1, -1, -1, -1))
        return {
            "src_canonical": src_canonical,
            "tgt_canonical": tgt_canonical,
            "src_descriptor": src_descriptor,
            "tgt_descriptor": tgt_descriptor,
            "src_uncertainty": src_uncertainty,
            "tgt_uncertainty": tgt_uncertainty,
            "target_matchability": target_matchability,
            "source_matchability": source_matchability,
            "target_positions": target_positions,
            "source_positions": source_positions,
            "spatial_shape": spatial_shape,
        }

    def _confidence_from_probabilities(
        self,
        probabilities: torch.Tensor,
        source_matchability: torch.Tensor,
        target_matchability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk = probabilities.topk(k=min(2, probabilities.shape[-1]), dim=-1).values
        top1 = topk[..., 0]
        if topk.shape[-1] > 1:
            margin = top1 - topk[..., 1]
        else:
            margin = top1

        if self.confidence_mode == "margin":
            confidence = margin
        else:
            confidence = top1

        if target_matchability is not None:
            best_index = probabilities.argmax(dim=-1, keepdim=True)
            if target_matchability.dim() == probabilities.dim() - 1:
                best_target_matchability = torch.gather(
                    target_matchability,
                    dim=-1,
                    index=best_index.squeeze(-1),
                )
            else:
                best_target_matchability = torch.gather(target_matchability, dim=-1, index=best_index).squeeze(-1)
            confidence = confidence * torch.sqrt((source_matchability * best_target_matchability).clamp_min(1e-6))
        else:
            confidence = confidence * source_matchability

        entropy = softmax_entropy(probabilities, dim=-1)
        margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
        return confidence, entropy, margin

    def _finalize_outputs(
        self,
        target_positions: torch.Tensor,
        source_positions: torch.Tensor,
        probabilities: torch.Tensor,
        source_matchability: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        target_matchability: torch.Tensor | None = None,
    ) -> MatchOutputs:
        if target_positions.dim() == 3:
            expected_target_positions = torch.bmm(probabilities, target_positions)
        else:
            expected_target_positions = torch.einsum("bij,bijd->bid", probabilities, target_positions)
        raw_displacement = (expected_target_positions - source_positions).transpose(1, 2).contiguous()
        raw_displacement = raw_displacement.view(source_positions.shape[0], 3, *spatial_shape)
        raw_displacement = torch.nan_to_num(raw_displacement, nan=0.0, posinf=0.0, neginf=0.0)

        confidence, entropy, margin = self._confidence_from_probabilities(
            probabilities=probabilities,
            source_matchability=source_matchability,
            target_matchability=target_matchability,
        )
        confidence = confidence.view(source_positions.shape[0], 1, *spatial_shape)
        margin = margin.view(source_positions.shape[0], 1, *spatial_shape)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0).view(
            source_positions.shape[0], 1, *spatial_shape
        )
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        return MatchOutputs(
            expected_target_positions=expected_target_positions,
            raw_displacement=raw_displacement,
            probabilities=probabilities,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            source_positions=source_positions,
        )

    def _should_skip_global_matching(
        self,
        flattened: dict[str, torch.Tensor],
        stage_id: int | None = None,
    ) -> bool:
        if self.global_match_voxel_limit <= 0:
            return False
        num_source = flattened["src_canonical"].shape[1]
        num_target = flattened["tgt_canonical"].shape[1]
        num_voxels = max(num_source, num_target)
        if num_voxels <= self.global_match_voxel_limit:
            return False

        stage_label = f"stage {stage_id}" if stage_id is not None else f"shape {flattened['spatial_shape']}"
        warning_key = f"{stage_label}:{flattened['spatial_shape']}"
        if warning_key not in self._warned_skip_keys:
            LOGGER.warning(
                "Skipping global canonical matching for %s because it has %d voxels, above limit %d. "
                "Decoder should fall back to local correlation refinement.",
                stage_label,
                num_voxels,
                self.global_match_voxel_limit,
            )
            self._warned_skip_keys.add(warning_key)
        return True


class CanonicalCorrelationMatcher(_BaseCorrelationMatcher):
    """Top-k matcher guided by canonical coordinates and descriptor affinity."""

    def forward(
        self,
        source: PointmapOutputs,
        target: PointmapOutputs,
        stage_id: int | None = None,
    ) -> MatchOutputs | None:
        flattened = self._flatten_inputs(source, target)
        if self._should_skip_global_matching(flattened, stage_id=stage_id):
            return None
        canonical_distance = torch.cdist(flattened["src_canonical"], flattened["tgt_canonical"])
        num_voxels = canonical_distance.shape[-1]
        topk = min(self.topk, num_voxels)
        candidate_distances, candidate_indices = canonical_distance.topk(topk, dim=-1, largest=False)
        candidate_valid = candidate_distances <= self.canonical_radius
        candidate_valid[..., 0] = True
        del canonical_distance

        source_descriptor = flattened["src_descriptor"].unsqueeze(2).expand(-1, -1, topk, -1)
        target_descriptor = _batched_gather(flattened["tgt_descriptor"], candidate_indices)
        target_positions = _batched_gather(flattened["target_positions"], candidate_indices)
        source_uncertainty = flattened["src_uncertainty"].unsqueeze(-1).expand(-1, -1, topk)
        target_uncertainty = _batched_gather_scalar(flattened["tgt_uncertainty"], candidate_indices)
        source_matchability = flattened["source_matchability"].unsqueeze(-1).expand(-1, -1, topk)
        target_matchability = _batched_gather_scalar(flattened["target_matchability"], candidate_indices)

        descriptor_similarity = (source_descriptor * target_descriptor).sum(dim=-1)
        uncertainty_penalty = source_uncertainty + target_uncertainty
        scores = descriptor_similarity - candidate_distances - 0.1 * uncertainty_penalty
        scores = scores.masked_fill(~candidate_valid, -1e4)
        if self.matchability_score_mode == "legacy":
            scores = scores + target_matchability.log().clamp_min(-20.0)
            probabilities = F.softmax(scores / self.temperature, dim=-1)
        else:
            probabilities = F.softmax(scores / self.temperature, dim=-1)
            pair_matchability = (source_matchability * target_matchability).clamp_min(1e-6)
            probabilities = probabilities * pair_matchability.pow(self.matchability_score_power)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        expected_target_positions = (probabilities.unsqueeze(-1) * target_positions).sum(dim=2)
        source_positions = flattened["source_positions"]
        raw_displacement = (expected_target_positions - source_positions).transpose(1, 2).contiguous()
        raw_displacement = raw_displacement.view(source.canonical_coords.shape[0], 3, *flattened["spatial_shape"])
        raw_displacement = torch.nan_to_num(raw_displacement, nan=0.0, posinf=0.0, neginf=0.0)

        confidence, entropy, margin = self._confidence_from_probabilities(
            probabilities=probabilities,
            source_matchability=flattened["source_matchability"],
            target_matchability=target_matchability,
        )
        confidence = confidence.view(source.canonical_coords.shape[0], 1, *flattened["spatial_shape"])
        margin = margin.view(source.canonical_coords.shape[0], 1, *flattened["spatial_shape"])
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0).view(
            source.canonical_coords.shape[0], 1, *flattened["spatial_shape"]
        )

        return MatchOutputs(
            expected_target_positions=expected_target_positions,
            raw_displacement=raw_displacement,
            probabilities=probabilities,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            source_positions=source_positions,
        )


class CandidateRefinedMatcher(_BaseCorrelationMatcher):
    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        offset_scale: float,
        output_mode: str = "topm_reweighted",
        topm: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.output_mode = output_mode
        self.topm = topm
        self.refinement = CandidateRefinementModule(
            descriptor_dim=descriptor_dim,
            hidden_dim=hidden_dim,
            offset_scale=offset_scale,
        )

    def forward(
        self,
        source: PointmapOutputs,
        target: PointmapOutputs,
        stage_id: int | None = None,
    ) -> MatchOutputs | None:
        flattened = self._flatten_inputs(source, target)
        if self._should_skip_global_matching(flattened, stage_id=stage_id):
            return None
        canonical_distance = torch.cdist(flattened["src_canonical"], flattened["tgt_canonical"])

        num_voxels = canonical_distance.shape[-1]
        topk = min(self.topk, num_voxels)
        candidate_distances, candidate_indices = canonical_distance.topk(topk, dim=-1, largest=False)
        candidate_valid = candidate_distances <= self.canonical_radius
        candidate_valid[..., 0] = True

        source_descriptor = flattened["src_descriptor"].unsqueeze(2).expand(-1, -1, topk, -1)
        target_descriptor = _batched_gather(flattened["tgt_descriptor"], candidate_indices)
        source_canonical = flattened["src_canonical"].unsqueeze(2).expand(-1, -1, topk, -1)
        target_canonical = _batched_gather(flattened["tgt_canonical"], candidate_indices)
        source_positions = flattened["source_positions"].unsqueeze(2).expand(-1, -1, topk, -1)
        target_positions = _batched_gather(flattened["target_positions"], candidate_indices)
        source_uncertainty = flattened["src_uncertainty"].unsqueeze(-1).expand(-1, -1, topk)
        target_uncertainty = _batched_gather_scalar(flattened["tgt_uncertainty"], candidate_indices)
        source_matchability = flattened["source_matchability"].unsqueeze(-1).expand(-1, -1, topk)
        target_matchability = _batched_gather_scalar(flattened["target_matchability"], candidate_indices)

        descriptor_similarity = (source_descriptor * target_descriptor).sum(dim=-1)
        canonical_delta = target_canonical - source_canonical
        spatial_offset = target_positions - source_positions
        uncertainty_penalty = source_uncertainty + target_uncertainty
        base_scores = descriptor_similarity - candidate_distances - 0.1 * uncertainty_penalty
        if self.matchability_score_mode == "legacy":
            base_scores = base_scores + target_matchability.log().clamp_min(-20.0)

        refined_delta, refined_offset = self.refinement(
            source_descriptor=source_descriptor,
            target_descriptor=target_descriptor,
            descriptor_similarity=descriptor_similarity,
            canonical_delta=canonical_delta,
            spatial_offset=spatial_offset,
            canonical_distance=candidate_distances,
            source_uncertainty=source_uncertainty,
            target_uncertainty=target_uncertainty,
            source_matchability=source_matchability,
            target_matchability=target_matchability,
        )
        refined_scores = (base_scores + refined_delta).masked_fill(~candidate_valid, -1e4)
        probabilities = F.softmax(refined_scores / self.temperature, dim=-1)
        if self.matchability_score_mode != "legacy":
            pair_matchability = (source_matchability * target_matchability).clamp_min(1e-6)
            probabilities = probabilities * pair_matchability.pow(self.matchability_score_power)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        refined_positions = target_positions + refined_offset
        if self.output_mode == "top1":
            best_indices = refined_scores.argmax(dim=-1, keepdim=True)
            selected_positions = torch.gather(
                refined_positions,
                dim=2,
                index=best_indices.unsqueeze(-1).expand(-1, -1, -1, 3),
            ).squeeze(2)
            final_probabilities = F.one_hot(best_indices.squeeze(-1), num_classes=topk).float()
        elif self.output_mode == "topm_reweighted":
            topm = min(max(self.topm, 1), topk)
            top_scores, top_indices = refined_scores.topk(topm, dim=-1)
            top_positions = torch.gather(
                refined_positions,
                dim=2,
                index=top_indices.unsqueeze(-1).expand(-1, -1, -1, 3),
            )
            top_weights = F.softmax(top_scores / self.temperature, dim=-1)
            selected_positions = (top_weights.unsqueeze(-1) * top_positions).sum(dim=2)
            final_probabilities = torch.zeros_like(probabilities)
            final_probabilities.scatter_(-1, top_indices, top_weights)
        else:
            raise ValueError(f"Unsupported refined matcher output mode: {self.output_mode}")
        final_probabilities = torch.nan_to_num(final_probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        source_positions_flat = flattened["source_positions"]
        expected_target_positions = selected_positions
        raw_displacement = (expected_target_positions - source_positions_flat).transpose(1, 2).contiguous()
        raw_displacement = raw_displacement.view(source.canonical_coords.shape[0], 3, *flattened["spatial_shape"])
        raw_displacement = torch.nan_to_num(raw_displacement, nan=0.0, posinf=0.0, neginf=0.0)

        # Confidence/entropy should describe the final sharpened correspondence distribution.
        confidence, entropy, margin = self._confidence_from_probabilities(
            probabilities=final_probabilities,
            source_matchability=flattened["source_matchability"],
            target_matchability=target_matchability,
        )
        confidence = confidence.view(source.canonical_coords.shape[0], 1, *flattened["spatial_shape"])
        margin = margin.view(source.canonical_coords.shape[0], 1, *flattened["spatial_shape"])
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0).view(
            source.canonical_coords.shape[0], 1, *flattened["spatial_shape"]
        )

        return MatchOutputs(
            expected_target_positions=expected_target_positions,
            raw_displacement=raw_displacement,
            probabilities=final_probabilities,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            source_positions=source_positions_flat,
        )
