from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.pccr.modules.matcher import (
    CandidateRefinedMatcher,
    MatchOutputs,
    _batched_gather,
    _batched_gather_scalar,
)
from src.pccr.modules.pointmap import PointmapOutputs


@dataclass
class StructuredMatchOutputs(MatchOutputs):
    structured_handoff_features: torch.Tensor


class StructuredCandidateRefinedMatcher(CandidateRefinedMatcher):
    """Candidate-refined matcher that preserves multi-hypothesis evidence for decoder handoff."""

    def __init__(self, *args, handoff_topm: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.handoff_topm = max(1, int(handoff_topm))

    def _build_structured_handoff_features(
        self,
        refined_positions: torch.Tensor,
        refined_scores: torch.Tensor,
        source_positions: torch.Tensor,
    ) -> torch.Tensor:
        topm = min(self.handoff_topm, refined_scores.shape[-1])
        top_scores, top_indices = refined_scores.topk(topm, dim=-1)
        top_positions = torch.gather(
            refined_positions,
            dim=2,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        top_weights = F.softmax(top_scores / max(self.temperature, 1e-6), dim=-1)
        top_displacements = top_positions - source_positions.unsqueeze(2)
        weighted_mean = (top_weights.unsqueeze(-1) * top_displacements).sum(dim=2, keepdim=True)
        weighted_var = (top_weights.unsqueeze(-1) * (top_displacements - weighted_mean).pow(2)).sum(dim=2)
        weighted_std = weighted_var.clamp_min(1e-6).sqrt()

        if topm > 1:
            top_delta = top_displacements[:, :, 0] - top_displacements[:, :, 1]
        else:
            top_delta = top_displacements.new_zeros(top_displacements.shape[0], top_displacements.shape[1], 3)

        features = torch.cat(
            [
                top_displacements.reshape(top_displacements.shape[0], top_displacements.shape[1], topm * 3),
                top_weights,
                weighted_std,
                top_delta,
            ],
            dim=-1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        source: PointmapOutputs,
        target: PointmapOutputs,
        stage_id: int | None = None,
    ) -> StructuredMatchOutputs | None:
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
        structured_handoff = self._build_structured_handoff_features(
            refined_positions=refined_positions,
            refined_scores=refined_scores,
            source_positions=flattened["source_positions"],
        )
        structured_handoff = structured_handoff.transpose(1, 2).contiguous()
        structured_handoff = structured_handoff.view(
            source.canonical_coords.shape[0],
            structured_handoff.shape[1],
            *flattened["spatial_shape"],
        )

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

        return StructuredMatchOutputs(
            expected_target_positions=expected_target_positions,
            raw_displacement=raw_displacement,
            probabilities=final_probabilities,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            source_positions=source_positions_flat,
            structured_handoff_features=structured_handoff,
        )
