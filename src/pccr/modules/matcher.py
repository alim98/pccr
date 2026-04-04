from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pccr.modules.pointmap import PointmapOutputs
from src.pccr.utils import flatten_spatial, softmax_entropy, voxel_grid


@dataclass
class MatchOutputs:
    expected_target_positions: torch.Tensor
    raw_displacement: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    source_positions: torch.Tensor


class CanonicalCorrelationMatcher(nn.Module):
    """Dense matcher guided by canonical coordinates and descriptor affinity."""

    def __init__(
        self,
        temperature: float = 0.07,
        topk: int = 24,
        canonical_radius: float = 0.45,
        matchability_score_mode: str = "legacy",
        matchability_score_power: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.topk = topk
        self.canonical_radius = canonical_radius
        self.matchability_score_mode = matchability_score_mode
        self.matchability_score_power = matchability_score_power

    def forward(
        self,
        source: PointmapOutputs,
        target: PointmapOutputs,
    ) -> MatchOutputs:
        batch_size = source.canonical_coords.shape[0]
        spatial_shape = tuple(source.canonical_coords.shape[2:])
        num_voxels = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]

        src_canonical = flatten_spatial(source.canonical_coords)
        tgt_canonical = flatten_spatial(target.canonical_coords)
        src_descriptor = flatten_spatial(source.descriptors)
        tgt_descriptor = flatten_spatial(target.descriptors)

        canonical_distance = torch.cdist(src_canonical, tgt_canonical)
        descriptor_similarity = torch.einsum("bid,bjd->bij", src_descriptor, tgt_descriptor)

        topk = min(self.topk, num_voxels)
        nearest_distances, nearest_indices = canonical_distance.topk(topk, dim=-1, largest=False)
        allowed_mask = torch.zeros_like(canonical_distance, dtype=torch.bool)
        allowed_mask.scatter_(-1, nearest_indices, True)
        allowed_mask = allowed_mask | (canonical_distance <= self.canonical_radius)

        target_matchability = flatten_spatial(target.matchability_logits).softmax(dim=-1)[..., 0]
        source_matchability = flatten_spatial(source.matchability_logits).softmax(dim=-1)[..., 0]
        uncertainty_penalty = flatten_spatial(source.uncertainty) + flatten_spatial(target.uncertainty).transpose(1, 2)
        scores = descriptor_similarity - canonical_distance - 0.1 * uncertainty_penalty
        scores = scores.masked_fill(~allowed_mask, -1e4)
        if self.matchability_score_mode == "legacy":
            scores = scores + target_matchability.unsqueeze(1).log().clamp_min(-20.0)
            probabilities = F.softmax(scores / self.temperature, dim=-1)
        else:
            probabilities = F.softmax(scores / self.temperature, dim=-1)
            pair_matchability = (
                source_matchability.unsqueeze(-1) * target_matchability.unsqueeze(1)
            ).clamp_min(1e-6)
            probabilities = probabilities * pair_matchability.pow(self.matchability_score_power)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        target_grid = voxel_grid(spatial_shape, source.canonical_coords.device)
        target_positions = flatten_spatial(target_grid.expand(batch_size, -1, -1, -1, -1))
        expected_target_positions = torch.einsum("bij,bjd->bid", probabilities, target_positions)

        source_positions = flatten_spatial(target_grid.expand(batch_size, -1, -1, -1, -1))
        raw_displacement = (expected_target_positions - source_positions).transpose(1, 2).contiguous()
        raw_displacement = raw_displacement.view(batch_size, 3, *spatial_shape)
        raw_displacement = torch.nan_to_num(raw_displacement, nan=0.0, posinf=0.0, neginf=0.0)

        confidence = probabilities.max(dim=-1).values * source_matchability
        confidence = confidence.view(batch_size, 1, *spatial_shape)
        entropy = softmax_entropy(probabilities, dim=-1).view(batch_size, 1, *spatial_shape)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

        return MatchOutputs(
            expected_target_positions=expected_target_positions,
            raw_displacement=raw_displacement,
            probabilities=probabilities,
            confidence=confidence,
            entropy=entropy,
            source_positions=source_positions,
        )
