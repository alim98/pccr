from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PointmapOutputs:
    canonical_coords: torch.Tensor
    descriptors: torch.Tensor
    uncertainty: torch.Tensor
    matchability_logits: torch.Tensor


class _PerScalePointmapHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        context_dim: int,
        descriptor_dim: int,
        num_matchability_classes: int,
    ) -> None:
        super().__init__()
        fused_channels = in_channels + context_dim
        self.backbone = nn.Sequential(
            nn.Conv3d(fused_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.GELU(),
        )
        self.pointmap = nn.Conv3d(in_channels, 3, kernel_size=1)
        self.descriptor = nn.Conv3d(in_channels, descriptor_dim, kernel_size=1)
        self.uncertainty = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.matchability = nn.Conv3d(in_channels, num_matchability_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> PointmapOutputs:
        context_map = context.expand(-1, -1, *features.shape[2:])
        hidden = self.backbone(torch.cat([features, context_map], dim=1))
        descriptors = F.normalize(self.descriptor(hidden), dim=1)
        return PointmapOutputs(
            canonical_coords=torch.tanh(self.pointmap(hidden)),
            descriptors=descriptors,
            uncertainty=F.softplus(self.uncertainty(hidden)),
            matchability_logits=self.matchability(hidden),
        )


class PairConditionedPointmapHead(nn.Module):
    """Predict canonical coordinates and matchability from pair-conditioned context."""

    def __init__(
        self,
        stage_channels: list[int],
        pointmap_stage_ids: list[int],
        context_dim: int,
        descriptor_dim: int,
        num_matchability_classes: int,
    ) -> None:
        super().__init__()
        self.pointmap_stage_ids = sorted(pointmap_stage_ids)
        deepest_channels = stage_channels[max(self.pointmap_stage_ids)]
        self.context_mlp = nn.Sequential(
            nn.Linear(deepest_channels * 4, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
        )
        self.per_scale = nn.ModuleDict(
            {
                str(stage_id): _PerScalePointmapHead(
                    in_channels=stage_channels[stage_id],
                    context_dim=context_dim,
                    descriptor_dim=descriptor_dim,
                    num_matchability_classes=num_matchability_classes,
                )
                for stage_id in self.pointmap_stage_ids
            }
        )

    def build_pair_context(
        self,
        source_features: dict[int, torch.Tensor],
        target_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        stage_id = max(self.pointmap_stage_ids)
        src = source_features[stage_id].mean(dim=(2, 3, 4))
        tgt = target_features[stage_id].mean(dim=(2, 3, 4))
        pair_stats = torch.cat([src, tgt, torch.abs(src - tgt), src * tgt], dim=1)
        return self.context_mlp(pair_stats).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def forward(
        self,
        source_features: dict[int, torch.Tensor],
        target_features: dict[int, torch.Tensor],
    ) -> tuple[dict[int, PointmapOutputs], dict[int, PointmapOutputs]]:
        pair_context = self.build_pair_context(source_features, target_features)
        source_outputs = {}
        target_outputs = {}
        for stage_id in self.pointmap_stage_ids:
            head = self.per_scale[str(stage_id)]
            source_outputs[stage_id] = head(source_features[stage_id], pair_context)
            target_outputs[stage_id] = head(target_features[stage_id], pair_context)
        return source_outputs, target_outputs
