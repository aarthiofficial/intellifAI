# coding: utf-8

"""
Appearance extractor (F) defined in the paper, which maps the source image `s` to a 3D appearance feature volume.
"""

import torch
from torch import nn
from .util import SameBlock2d, DownBlock2d, ResBlock3d


class AppearanceFeatureExtractor(nn.Module):
    def __init__(
        self,
        image_channel: int,
        block_expansion: int,
        num_down_blocks: int,
        max_features: int,
        reshape_channel: int,
        reshape_depth: int,
        num_resblocks: int,
    ):
        super(AppearanceFeatureExtractor, self).__init__()

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.first = SameBlock2d(
            image_channel,
            block_expansion,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(
                DownBlock2d(
                    in_features,
                    out_features,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(
            in_channels=out_features,
            out_channels=max_features,
            kernel_size=1,
            stride=1,
        )

        self.resblocks_3d = nn.Sequential(
            *[
                ResBlock3d(reshape_channel, kernel_size=3, padding=1)
                for _ in range(num_resblocks)
            ]
        )

    def forward(self, source_image: torch.Tensor) -> torch.Tensor:
        out = self.first(source_image)  # -> Bx64x256x256

        for down_block in self.down_blocks:
            out = down_block(out)

        out = self.second(out)  # -> Bx512x64x64

        bs, c, h, w = out.shape
        f_s = out.view(bs, self.reshape_channel, self.reshape_depth, h, w)  # -> Bx32x16x64x64

        f_s = self.resblocks_3d(f_s)  # -> Bx32x16x64x64
        return f_s
