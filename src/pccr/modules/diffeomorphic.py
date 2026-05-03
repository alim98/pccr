from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.model.transformation import SpatialTransformer
from src.pccr.utils import resize_displacement, softmax_entropy


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
    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
        max_velocity: float,
        extra_channels: int = 0,
    ) -> None:
        super().__init__()
        self.max_velocity = max_velocity
        self.extra_channels = extra_channels
        in_channels = feature_channels * 2 + 3 + 1 + 1 + extra_channels
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
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = [source_features, target_features, raw_displacement, confidence, entropy]
        if extra_features is not None:
            inputs.append(extra_features)
        elif self.extra_channels > 0:
            inputs.append(
                source_features.new_zeros(
                    source_features.shape[0],
                    self.extra_channels,
                    *source_features.shape[2:],
                )
            )
        velocity = self.net(torch.cat(inputs, dim=1))
        velocity = torch.tanh(velocity) * self.max_velocity
        return torch.nan_to_num(velocity, nan=0.0, posinf=self.max_velocity, neginf=-self.max_velocity)


class FinalResidualRefinementHead(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
        max_velocity: float,
        extra_channels: int = 0,
        num_blocks: int = 2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.max_velocity = max_velocity
        in_channels = feature_channels * 2 + 3 + 1 + 1 + extra_channels
        if activation == "leaky_relu":
            activation_layer = lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation == "gelu":
            activation_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported final refinement activation: {activation}")

        layers: list[nn.Module] = []
        current_in_channels = in_channels
        for _ in range(max(1, num_blocks)):
            layers.extend(
                [
                    nn.Conv3d(current_in_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(hidden_channels),
                    activation_layer(),
                ]
            )
            current_in_channels = hidden_channels
        layers.append(nn.Conv3d(hidden_channels, 3, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        displacement: torch.Tensor,
        confidence: torch.Tensor,
        entropy: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = [source_features, target_features, displacement, confidence, entropy]
        if extra_features is not None:
            inputs.append(extra_features)
        velocity = self.net(torch.cat(inputs, dim=1))
        velocity = torch.tanh(velocity) * self.max_velocity
        return torch.nan_to_num(velocity, nan=0.0, posinf=self.max_velocity, neginf=-self.max_velocity)


class ImageErrorEncoder(nn.Module):
    def __init__(self, out_channels: int, use_edge_inputs: bool) -> None:
        super().__init__()
        self.use_edge_inputs = use_edge_inputs
        in_channels = 3 + (1 if use_edge_inputs else 0)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
        dz = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dx = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]

        grad = image.new_zeros(image.shape)
        grad[:, :, 1:, :, :] += dz.abs()
        grad[:, :, :-1, :, :] += dz.abs()
        grad[:, :, :, 1:, :] += dy.abs()
        grad[:, :, :, :-1, :] += dy.abs()
        grad[:, :, :, :, 1:] += dx.abs()
        grad[:, :, :, :, :-1] += dx.abs()
        return grad

    def forward(self, moved_source: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        abs_diff = (moved_source - target_image).abs()
        inputs = [moved_source, target_image, abs_diff]
        if self.use_edge_inputs:
            edge_diff = (self._gradient_magnitude(moved_source) - self._gradient_magnitude(target_image)).abs()
            inputs.append(edge_diff)
        return self.net(torch.cat(inputs, dim=1))


@dataclass
class LocalResidualMatchOutputs:
    delta_displacement: torch.Tensor
    confidence: torch.Tensor
    margin: torch.Tensor
    entropy: torch.Tensor
    encoded_features: torch.Tensor


class LocalResidualMatcher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        radius: int,
        proj_channels: int,
        out_channels: int,
        temperature: float = 1.0,
        memory_efficient: bool = False,
        offset_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.temperature = temperature
        self.memory_efficient = memory_efficient
        self.offset_chunk_size = max(1, int(offset_chunk_size))
        # When this module is wrapped by an outer checkpoint in the decoder, we
        # disable the inner chunk checkpoints to avoid duplicate recomputation.
        self._outer_checkpoint_active = False
        # Auto guard: keep inner chunk checkpointing for large tensors to avoid OOM
        # during backward recomputation (notably full-size real training).
        self._disable_inner_chunk_max_voxels = int(
            os.getenv("PCCR_DISABLE_INNER_CHKPT_MAX_VOXELS", "2000000")
        )
        self.proj_source = nn.Conv3d(in_channels, proj_channels, kernel_size=1)
        self.proj_target = nn.Conv3d(in_channels, proj_channels, kernel_size=1)
        offsets = [
            (dz, dy, dx)
            for dz in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        self.offsets = offsets
        offset_components = torch.tensor(offsets, dtype=torch.float32)
        self.register_buffer("offset_components", offset_components, persistent=False)
        num_offsets = len(offsets)
        self.encoder = nn.Sequential(
            nn.Conv3d(num_offsets + 3 + 1 + 1, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def set_outer_checkpoint_active(self, active: bool) -> None:
        self._outer_checkpoint_active = bool(active)

    def _should_use_inner_chunk_checkpoint(self, spatial_shape: tuple[int, int, int]) -> bool:
        if not (self.training and torch.is_grad_enabled()):
            return False
        if not self._outer_checkpoint_active:
            return True
        if self._disable_inner_chunk_max_voxels < 0:
            return False
        D, H, W = spatial_shape
        return (D * H * W) > self._disable_inner_chunk_max_voxels

    @staticmethod
    def _conv3d_like(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None,
        reference_conv: nn.Conv3d,
    ) -> torch.Tensor:
        return F.conv3d(
            inputs,
            weights,
            bias=bias,
            stride=reference_conv.stride,
            padding=reference_conv.padding,
            dilation=reference_conv.dilation,
            groups=reference_conv.groups,
        )

    def _forward_streaming(
        self,
        source_proj: torch.Tensor,
        padded_target: torch.Tensor,
    ) -> LocalResidualMatchOutputs:
        """Memory-efficient forward that avoids materializing the full correlation volume.

        Vectorized: each chunk computes K correlations and applies a single conv3d call,
        then a vectorized online-softmax update consumes the same chunk.
        """
        first_conv = self.encoder[0]
        radius = self.radius
        num_offsets = len(self.offsets)
        T = max(self.temperature, 1e-6)
        B, _, D, H, W = source_proj.shape
        chunk_size = self.offset_chunk_size

        hidden = source_proj.new_zeros(B, first_conv.out_channels, D, H, W)

        def encode_offset_chunk(
            source: torch.Tensor,
            target: torch.Tensor,
            start_idx: int,
            end_idx: int,
        ) -> torch.Tensor:
            chunk_len = end_idx - start_idx
            corr_chunk = source.new_empty(B, chunk_len, D, H, W)
            for local_idx, idx in enumerate(range(start_idx, end_idx)):
                dz, dy, dx = self.offsets[idx]
                shifted = target[
                    :, :,
                    radius + dz : radius + dz + D,
                    radius + dy : radius + dy + H,
                    radius + dx : radius + dx + W,
                ]
                corr_chunk[:, local_idx : local_idx + 1] = (source * shifted).sum(dim=1, keepdim=True)
            chunk_weight = first_conv.weight[:, start_idx : start_idx + corr_chunk.shape[1]]
            return self._conv3d_like(
                corr_chunk,
                chunk_weight,
                bias=None,
                reference_conv=first_conv,
            )

        use_inner_chunk_checkpoint = self._should_use_inner_chunk_checkpoint((D, H, W))
        for start_idx in range(0, num_offsets, chunk_size):
            end_idx = min(start_idx + chunk_size, num_offsets)
            if use_inner_chunk_checkpoint:
                hidden = hidden + checkpoint(
                    encode_offset_chunk,
                    source_proj,
                    padded_target,
                    start_idx,
                    end_idx,
                    use_reentrant=False,
                )
            else:
                hidden = hidden + encode_offset_chunk(source_proj, padded_target, start_idx, end_idx)

        # Online softmax statistics. Detached during training to bound memory; vectorized
        # over chunks of K offsets at a time so the per-voxel recurrence runs in
        # num_offsets / K steps instead of num_offsets steps.
        stats_grad_enabled = not (self.training and torch.is_grad_enabled())
        with torch.set_grad_enabled(stats_grad_enabled):
            stats_source = source_proj if stats_grad_enabled else source_proj.detach()
            stats_target = padded_target if stats_grad_enabled else padded_target.detach()

            running_max = stats_source.new_full((B, 1, D, H, W), float("-inf"))
            running_sum = stats_source.new_zeros((B, 1, D, H, W))
            running_weighted_offset = stats_source.new_zeros((B, 3, D, H, W))
            running_weighted_scaled_corr = stats_source.new_zeros((B, 1, D, H, W))
            running_second_max = stats_source.new_full((B, 1, D, H, W), float("-inf"))

            for start_idx in range(0, num_offsets, chunk_size):
                end_idx = min(start_idx + chunk_size, num_offsets)
                chunk_len = end_idx - start_idx
                corr_chunk = stats_source.new_empty(B, chunk_len, D, H, W)
                for local_idx, idx in enumerate(range(start_idx, end_idx)):
                    dz, dy, dx = self.offsets[idx]
                    shifted = stats_target[
                        :, :,
                        radius + dz : radius + dz + D,
                        radius + dy : radius + dy + H,
                        radius + dx : radius + dx + W,
                    ]
                    corr_chunk[:, local_idx : local_idx + 1] = (
                        stats_source * shifted
                    ).sum(dim=1, keepdim=True)
                # [B, K, D, H, W]
                scaled_chunk = corr_chunk / T

                # Merge top-2 of past (running_max, running_second_max) with chunk values.
                cand = torch.cat([running_max, running_second_max, scaled_chunk], dim=1)
                top2 = torch.topk(cand, k=2, dim=1, sorted=True).values
                new_max = top2[:, :1]
                new_second_max = top2[:, 1:2]

                prev_scale = torch.exp(running_max - new_max)
                prev_scale = torch.nan_to_num(prev_scale, nan=0.0, posinf=0.0, neginf=0.0)
                chunk_exp = torch.exp(scaled_chunk - new_max)
                chunk_exp = torch.nan_to_num(chunk_exp, nan=0.0, posinf=0.0, neginf=0.0)

                chunk_sum = chunk_exp.sum(dim=1, keepdim=True)
                offset_vecs = self.offset_components[start_idx:end_idx].to(
                    dtype=chunk_exp.dtype, device=chunk_exp.device
                )
                chunk_weighted_offset = torch.einsum(
                    "bkdhw,kc->bcdhw", chunk_exp, offset_vecs
                )
                chunk_weighted_scorr = (chunk_exp * scaled_chunk).sum(dim=1, keepdim=True)

                running_max = new_max
                running_second_max = new_second_max
                running_sum = running_sum * prev_scale + chunk_sum
                running_weighted_offset = running_weighted_offset * prev_scale + chunk_weighted_offset
                running_weighted_scaled_corr = running_weighted_scaled_corr * prev_scale + chunk_weighted_scorr

        if self.training and torch.is_grad_enabled():
            running_max = running_max.detach()
            running_sum = running_sum.detach()
            running_weighted_offset = running_weighted_offset.detach()
            running_weighted_scaled_corr = running_weighted_scaled_corr.detach()
            running_second_max = running_second_max.detach()

        inv_sum = 1.0 / running_sum.clamp_min(1e-8)

        expected_offset = running_weighted_offset * inv_sum
        expected_offset = torch.nan_to_num(expected_offset, nan=0.0, posinf=0.0, neginf=0.0)

        confidence = inv_sum.clamp_max(1.0)
        confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)

        second_max_prob = torch.exp(running_second_max - running_max) * inv_sum
        margin = (confidence - second_max_prob).clamp_min(0.0)
        margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)

        # H = m + log(S) - (1/S) * sum_i exp(a_i - m) * a_i
        entropy = (
            running_max
            + torch.log(running_sum.clamp_min(1e-8))
            - running_weighted_scaled_corr * inv_sum
        )
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

        stats = torch.cat([expected_offset, confidence, entropy], dim=1)
        stats_weight = first_conv.weight[:, num_offsets:]
        hidden = hidden + self._conv3d_like(
            stats, stats_weight, bias=first_conv.bias, reference_conv=first_conv
        )

        hidden = self.encoder[1](hidden)
        hidden = self.encoder[2](hidden)
        hidden = self.encoder[3](hidden)
        hidden = self.encoder[4](hidden)
        hidden = self.encoder[5](hidden)
        encoded_features = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)

        return LocalResidualMatchOutputs(
            delta_displacement=expected_offset,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            encoded_features=encoded_features,
        )

    def forward(
        self,
        source_features: torch.Tensor,
        warped_target_features: torch.Tensor,
    ) -> LocalResidualMatchOutputs:
        source_proj = F.normalize(self.proj_source(source_features), dim=1)
        target_proj = F.normalize(self.proj_target(warped_target_features), dim=1)

        radius = self.radius
        padded_target = F.pad(target_proj, (radius, radius, radius, radius, radius, radius), mode="replicate")

        if self.memory_efficient:
            return self._forward_streaming(source_proj, padded_target)

        num_offsets = len(self.offsets)
        correlation_volume = source_proj.new_empty(
            source_proj.shape[0],
            num_offsets,
            source_proj.shape[2],
            source_proj.shape[3],
            source_proj.shape[4],
        )
        for offset_index, (dz, dy, dx) in enumerate(self.offsets):
            shifted = padded_target[
                :,
                :,
                radius + dz : radius + dz + source_proj.shape[2],
                radius + dy : radius + dy + source_proj.shape[3],
                radius + dx : radius + dx + source_proj.shape[4],
            ]
            correlation_volume[:, offset_index : offset_index + 1] = (source_proj * shifted).sum(dim=1, keepdim=True)

        probabilities = F.softmax(correlation_volume / max(self.temperature, 1e-6), dim=1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        expected_offset = torch.stack(
            [
                (
                    probabilities
                    * self.offset_components[:, component_index].view(1, -1, 1, 1, 1)
                ).sum(dim=1)
                for component_index in range(3)
            ],
            dim=1,
        )
        expected_offset = torch.nan_to_num(expected_offset, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = probabilities.max(dim=1, keepdim=True).values
        top2 = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
        if top2.shape[1] > 1:
            margin = top2[:, :1] - top2[:, 1:2]
        else:
            margin = top2[:, :1]
        margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = softmax_entropy(probabilities, dim=1).unsqueeze(1)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        encoded_features = self.encoder(torch.cat([correlation_volume, expected_offset, confidence, entropy], dim=1))
        encoded_features = torch.nan_to_num(encoded_features, nan=0.0, posinf=0.0, neginf=0.0)

        return LocalResidualMatchOutputs(
            delta_displacement=expected_offset,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            encoded_features=encoded_features,
        )


class StageLocalCorrelationRefiner(LocalResidualMatcher):
    """Lightweight mid-resolution local search around the current warped position."""

    def __init__(self, *, memory_efficient: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs, memory_efficient=memory_efficient)  # type: ignore[arg-type]


class LocalCostVolumeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        radius: int,
        proj_channels: int,
        out_channels: int,
        memory_efficient: bool = False,
        offset_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.memory_efficient = memory_efficient
        self.offset_chunk_size = max(1, offset_chunk_size)
        # See LocalResidualMatcher: avoid nested chunk + outer checkpointing.
        self._outer_checkpoint_active = False
        # Auto guard: keep inner chunk checkpointing for large tensors to avoid OOM
        # during backward recomputation (notably full-size real training).
        self._disable_inner_chunk_max_voxels = int(
            os.getenv("PCCR_DISABLE_INNER_CHKPT_MAX_VOXELS", "2000000")
        )
        self.proj_source = nn.Conv3d(in_channels, proj_channels, kernel_size=1)
        self.proj_target = nn.Conv3d(in_channels, proj_channels, kernel_size=1)
        num_offsets = (2 * radius + 1) ** 3
        self.offsets = [
            (dz, dy, dx)
            for dz in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        offset_components = torch.tensor(self.offsets, dtype=torch.float32)
        self.register_buffer("offset_components", offset_components, persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv3d(num_offsets + 3, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def set_outer_checkpoint_active(self, active: bool) -> None:
        self._outer_checkpoint_active = bool(active)

    def _should_use_inner_chunk_checkpoint(self, spatial_shape: tuple[int, int, int]) -> bool:
        if not (self.training and torch.is_grad_enabled()):
            return False
        if not self._outer_checkpoint_active:
            return True
        if self._disable_inner_chunk_max_voxels < 0:
            return False
        D, H, W = spatial_shape
        return (D * H * W) > self._disable_inner_chunk_max_voxels

    @staticmethod
    def _conv3d_like(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None,
        reference_conv: nn.Conv3d,
    ) -> torch.Tensor:
        return F.conv3d(
            inputs,
            weights,
            bias=bias,
            stride=reference_conv.stride,
            padding=reference_conv.padding,
            dilation=reference_conv.dilation,
            groups=reference_conv.groups,
        )

    @staticmethod
    def _update_running_offset_statistics(
        running_max: torch.Tensor | None,
        running_sum: torch.Tensor | None,
        running_weighted_offset: torch.Tensor | None,
        correlation: torch.Tensor,
        offset: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offset_tensor = correlation.new_tensor(offset).view(1, 3, 1, 1, 1)
        if running_max is None or running_sum is None or running_weighted_offset is None:
            return (
                correlation,
                torch.ones_like(correlation),
                offset_tensor.expand(
                    correlation.shape[0],
                    -1,
                    correlation.shape[2],
                    correlation.shape[3],
                    correlation.shape[4],
                ).clone(),
            )

        new_max = torch.maximum(running_max, correlation)
        previous_scale = torch.exp(running_max - new_max)
        current_scale = torch.exp(correlation - new_max)
        return (
            new_max,
            running_sum * previous_scale + current_scale,
            running_weighted_offset * previous_scale + offset_tensor * current_scale,
        )

    @staticmethod
    def _finalize_expected_offset(
        running_sum: torch.Tensor | None,
        running_weighted_offset: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if running_sum is None or running_weighted_offset is None:
            return reference.new_zeros(reference.shape[0], 3, *reference.shape[2:])
        return running_weighted_offset / running_sum.clamp_min(1e-8)

    def _encode_with_streamed_cost_volume(
        self,
        source_proj: torch.Tensor,
        padded_target: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized streamed cost volume: K correlations and one conv3d call per chunk,
        plus a vectorized chunked online-softmax for the expected-offset statistic."""
        first_conv = self.encoder[0]
        num_offsets = len(self.offsets)
        B, _, D, H, W = source_proj.shape
        radius = self.radius
        chunk_size = self.offset_chunk_size

        hidden = source_proj.new_zeros(
            B,
            first_conv.out_channels,
            D,
            H,
            W,
        )

        def encode_offset_chunk(
            source: torch.Tensor,
            target: torch.Tensor,
            start_idx: int,
            end_idx: int,
        ) -> torch.Tensor:
            chunk_len = end_idx - start_idx
            correlation_chunk = source.new_empty(B, chunk_len, D, H, W)
            for local_idx, offset_index in enumerate(range(start_idx, end_idx)):
                dz, dy, dx = self.offsets[offset_index]
                shifted = target[
                    :,
                    :,
                    radius + dz : radius + dz + D,
                    radius + dy : radius + dy + H,
                    radius + dx : radius + dx + W,
                ]
                correlation_chunk[:, local_idx : local_idx + 1] = (
                    source * shifted
                ).sum(dim=1, keepdim=True)
            chunk_weights = first_conv.weight[:, start_idx : start_idx + correlation_chunk.shape[1]]
            return self._conv3d_like(
                correlation_chunk,
                chunk_weights,
                bias=None,
                reference_conv=first_conv,
            )

        use_inner_chunk_checkpoint = self._should_use_inner_chunk_checkpoint((D, H, W))
        for start_idx in range(0, num_offsets, chunk_size):
            end_idx = min(start_idx + chunk_size, num_offsets)
            if use_inner_chunk_checkpoint:
                hidden = hidden + checkpoint(
                    encode_offset_chunk,
                    source_proj,
                    padded_target,
                    start_idx,
                    end_idx,
                    use_reentrant=False,
                )
            else:
                hidden = hidden + encode_offset_chunk(source_proj, padded_target, start_idx, end_idx)

        # Vectorized online-softmax over chunks of K offsets at a time. Detached during
        # training to bound memory.
        stats_grad_enabled = not (self.training and torch.is_grad_enabled())
        with torch.set_grad_enabled(stats_grad_enabled):
            stats_source = source_proj if stats_grad_enabled else source_proj.detach()
            stats_target = padded_target if stats_grad_enabled else padded_target.detach()

            running_max = stats_source.new_full((B, 1, D, H, W), float("-inf"))
            running_sum = stats_source.new_zeros((B, 1, D, H, W))
            running_weighted_offset = stats_source.new_zeros((B, 3, D, H, W))

            for start_idx in range(0, num_offsets, chunk_size):
                end_idx = min(start_idx + chunk_size, num_offsets)
                chunk_len = end_idx - start_idx
                corr_chunk = stats_source.new_empty(B, chunk_len, D, H, W)
                for local_idx, offset_index in enumerate(range(start_idx, end_idx)):
                    dz, dy, dx = self.offsets[offset_index]
                    shifted = stats_target[
                        :,
                        :,
                        radius + dz : radius + dz + D,
                        radius + dy : radius + dy + H,
                        radius + dx : radius + dx + W,
                    ]
                    corr_chunk[:, local_idx : local_idx + 1] = (
                        stats_source * shifted
                    ).sum(dim=1, keepdim=True)
                # [B, K, D, H, W]

                chunk_max = corr_chunk.max(dim=1, keepdim=True).values
                new_max = torch.maximum(running_max, chunk_max)
                prev_scale = torch.exp(running_max - new_max)
                prev_scale = torch.nan_to_num(prev_scale, nan=0.0, posinf=0.0, neginf=0.0)
                chunk_exp = torch.exp(corr_chunk - new_max)
                chunk_exp = torch.nan_to_num(chunk_exp, nan=0.0, posinf=0.0, neginf=0.0)

                chunk_sum = chunk_exp.sum(dim=1, keepdim=True)
                offset_vecs = self.offset_components[start_idx:end_idx].to(
                    dtype=chunk_exp.dtype, device=chunk_exp.device
                )
                chunk_weighted_offset = torch.einsum(
                    "bkdhw,kc->bcdhw", chunk_exp, offset_vecs
                )

                running_max = new_max
                running_sum = running_sum * prev_scale + chunk_sum
                running_weighted_offset = running_weighted_offset * prev_scale + chunk_weighted_offset

        expected_offset = running_weighted_offset / running_sum.clamp_min(1e-8)
        expected_offset = torch.nan_to_num(expected_offset, nan=0.0, posinf=0.0, neginf=0.0)
        if self.training and torch.is_grad_enabled():
            expected_offset = expected_offset.detach()

        hidden = hidden + self._conv3d_like(
            expected_offset,
            first_conv.weight[:, num_offsets:],
            bias=first_conv.bias,
            reference_conv=first_conv,
        )
        hidden = self.encoder[1](hidden)
        hidden = self.encoder[2](hidden)
        hidden = self.encoder[3](hidden)
        hidden = self.encoder[4](hidden)
        hidden = self.encoder[5](hidden)
        return hidden

    def _encode_legacy(
        self,
        source_proj: torch.Tensor,
        padded_target: torch.Tensor,
    ) -> torch.Tensor:
        num_offsets = len(self.offsets)
        correlation_volume = source_proj.new_empty(
            source_proj.shape[0],
            num_offsets,
            source_proj.shape[2],
            source_proj.shape[3],
            source_proj.shape[4],
        )
        running_max = None
        running_sum = None
        running_weighted_offset = None
        radius = self.radius

        for offset_index, (dz, dy, dx) in enumerate(self.offsets):
            shifted = padded_target[
                :,
                :,
                radius + dz : radius + dz + source_proj.shape[2],
                radius + dy : radius + dy + source_proj.shape[3],
                radius + dx : radius + dx + source_proj.shape[4],
            ]
            correlation = (source_proj * shifted).sum(dim=1, keepdim=True)
            correlation_volume[:, offset_index : offset_index + 1] = correlation
            running_max, running_sum, running_weighted_offset = self._update_running_offset_statistics(
                running_max,
                running_sum,
                running_weighted_offset,
                correlation,
                (dz, dy, dx),
            )

        expected_offset = self._finalize_expected_offset(running_sum, running_weighted_offset, source_proj)
        return self.encoder(torch.cat([correlation_volume, expected_offset], dim=1))

    def forward(self, source_features: torch.Tensor, warped_target_features: torch.Tensor) -> torch.Tensor:
        source_proj = F.normalize(self.proj_source(source_features), dim=1)
        target_proj = F.normalize(self.proj_target(warped_target_features), dim=1)

        radius = self.radius
        padded_target = F.pad(target_proj, (radius, radius, radius, radius, radius, radius), mode="replicate")
        if self.memory_efficient:
            return self._encode_with_streamed_cost_volume(source_proj, padded_target)
        return self._encode_legacy(source_proj, padded_target)


@dataclass
class DecoderOutputs:
    displacement: torch.Tensor
    moved_source: torch.Tensor
    stage_displacements: dict[int, torch.Tensor]
    stage_velocity_fields: dict[int, torch.Tensor]
    stage_target_displacements: dict[int, torch.Tensor]
    stage_target_confidences: dict[int, torch.Tensor]
    stage_target_margins: dict[int, torch.Tensor]
    stage_target_entropies: dict[int, torch.Tensor]
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
        final_refinement_memory_efficient_cost_volume: bool = False,
        final_refinement_cost_volume_offset_chunk_size: int = 8,
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
        stage1_local_refinement_memory_efficient: bool = True,
        stage1_local_refinement_offset_chunk_size: int = 32,
        use_stage2_local_refinement: bool = False,
        stage2_local_refinement_radius: int = 0,
        stage2_local_refinement_proj_channels: int = 16,
        stage2_local_refinement_feature_channels: int = 16,
        stage2_local_refinement_temperature: float = 1.0,
        stage2_local_refinement_memory_efficient: bool = True,
        stage2_local_refinement_offset_chunk_size: int = 32,
        use_gradient_checkpointing: bool = False,
        diagnostic_residual_only: bool = False,
    ) -> None:
        super().__init__()
        self.decoder_stage_ids = decoder_stage_ids
        self.image_size = tuple(image_size)
        self.use_fine_local_refinement = use_fine_local_refinement
        self.use_final_residual_refinement = use_final_residual_refinement
        self.diagnostic_residual_only = diagnostic_residual_only
        self.use_gradient_checkpointing = use_gradient_checkpointing
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
        self.use_stage2_local_refinement = use_stage2_local_refinement and stage2_local_refinement_radius > 0

        self.stage_decoders = nn.ModuleDict()
        self.integrators = nn.ModuleDict()
        self.transformers = nn.ModuleDict()
        self.stage_local_refiners = nn.ModuleDict()

        stage_sizes = self._stage_sizes(image_size, len(stage_channels))
        local_refinement_specs = {
            2: (
                self.use_stage2_local_refinement,
                stage2_local_refinement_radius,
                stage2_local_refinement_proj_channels,
                stage2_local_refinement_feature_channels,
                stage2_local_refinement_temperature,
                stage2_local_refinement_memory_efficient,
                stage2_local_refinement_offset_chunk_size,
            ),
            1: (
                self.use_stage1_local_refinement,
                stage1_local_refinement_radius,
                stage1_local_refinement_proj_channels,
                stage1_local_refinement_feature_channels,
                stage1_local_refinement_temperature,
                stage1_local_refinement_memory_efficient,
                stage1_local_refinement_offset_chunk_size,
            ),
        }
        for stage_id in decoder_stage_ids:
            size = tuple(stage_sizes[stage_id])
            extra_channels = 0
            enabled, radius, proj_channels, feature_channels, temperature, memory_efficient, offset_chunk_size = local_refinement_specs.get(
                stage_id,
                (False, 0, 16, 16, 1.0, True, 32),
            )
            if enabled:
                self.stage_local_refiners[str(stage_id)] = StageLocalCorrelationRefiner(
                    in_channels=stage_channels[stage_id],
                    radius=radius,
                    proj_channels=proj_channels,
                    out_channels=feature_channels,
                    temperature=temperature,
                    memory_efficient=memory_efficient,
                    offset_chunk_size=offset_chunk_size,
                )
                extra_channels = feature_channels
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
                    memory_efficient=final_refinement_memory_efficient_cost_volume,
                    offset_chunk_size=final_refinement_cost_volume_offset_chunk_size,
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
            edge_diff = (ImageErrorEncoder._gradient_magnitude(moved_source) - ImageErrorEncoder._gradient_magnitude(target_image)).abs()
            inputs.append(edge_diff)
        return torch.cat(inputs, dim=1)

    @staticmethod
    def _set_outer_checkpoint_flag(module: nn.Module, active: bool) -> None:
        setter = getattr(module, "set_outer_checkpoint_active", None)
        if callable(setter):
            setter(active)

    def _run_tensor_module(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        use_outer_checkpoint = self.use_gradient_checkpointing and self.training and torch.is_grad_enabled()

        if use_outer_checkpoint:
            def forward_fn(*inputs: torch.Tensor) -> torch.Tensor:
                self._set_outer_checkpoint_flag(module, True)
                try:
                    return module(*inputs)
                finally:
                    self._set_outer_checkpoint_flag(module, False)

            return checkpoint(forward_fn, *args, use_reentrant=False)

        self._set_outer_checkpoint_flag(module, False)
        return module(*args)

    def _run_local_refiner(
        self,
        module: nn.Module,
        *args: torch.Tensor,
    ) -> LocalResidualMatchOutputs:
        use_outer_checkpoint = self.use_gradient_checkpointing and self.training and torch.is_grad_enabled()

        def forward_fn(*inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            self._set_outer_checkpoint_flag(module, use_outer_checkpoint)
            try:
                outputs = module(*inputs)
                return (
                    outputs.delta_displacement,
                    outputs.confidence,
                    outputs.margin,
                    outputs.entropy,
                    outputs.encoded_features,
                )
            finally:
                self._set_outer_checkpoint_flag(module, False)

        if use_outer_checkpoint:
            delta_displacement, confidence, margin, entropy, encoded_features = checkpoint(
                forward_fn,
                *args,
                use_reentrant=False,
            )
        else:
            delta_displacement, confidence, margin, entropy, encoded_features = forward_fn(*args)
        return LocalResidualMatchOutputs(
            delta_displacement=delta_displacement,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            encoded_features=encoded_features,
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
                    stage_local = self._run_local_refiner(self.stage_local_refiners[str(stage_id)], source_stage, warped_target)
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
                extra_features.append(self._run_tensor_module(self.local_cost_volume_encoder, source_fine, target_fine))
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
