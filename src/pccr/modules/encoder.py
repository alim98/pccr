from __future__ import annotations

import torch
import torch.nn as nn

from src.model.hvit_light import EncoderCnnBlock


class SharedPyramidEncoder(nn.Module):
    """Shared-weight 3D encoder that reuses H-ViT encoder blocks."""

    def __init__(
        self,
        in_channels: int,
        stage_channels: list[int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        blocks = []
        current_in = in_channels
        for stage_id, channels in enumerate(stage_channels):
            stride = 1 if stage_id == 0 else 2
            blocks.append(
                EncoderCnnBlock(
                    in_channels=current_in,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            current_in = channels
        self.blocks = nn.ModuleList(blocks)

    def forward_single(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        features = image
        for stage_id, block in enumerate(self.blocks):
            features = block(features)
            outputs[stage_id] = features
        return outputs

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        return self.forward_single(source), self.forward_single(target)
