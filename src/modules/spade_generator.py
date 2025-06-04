# src/modules/spade_generator.py
# coding: utf-8

"""
SPADE Decoder (G): This module generates the animated image from the warped feature,
based on SPADE ResNet blocks and upsampling operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import SPADEResnetBlock


class SPADEDecoder(nn.Module):
    def __init__(self, upscale=1, max_features=256, block_expansion=64, out_channels=64, num_down_blocks=2):
        """
        Initialize the SPADE Decoder.

        Args:
            upscale (int): Upscale factor for output resolution (typically 1 or 2).
            max_features (int): Max number of channels in intermediate layers.
            block_expansion (int): Base multiplier for computing feature dimensions.
            out_channels (int): Output channels (usually 64).
            num_down_blocks (int): Determines the depth of the feature extraction.
        """
        super().__init__()

        self.upscale = upscale
        norm_G = 'spadespectralinstance'

        # Calculate initial input_channels based on num_down_blocks
        input_channels = min(max_features, block_expansion * (2 ** (num_down_blocks)))

        label_num_channels = input_channels

        # Initial processing
        self.fc = nn.Conv2d(input_channels, 2 * input_channels, kernel_size=3, padding=1)

        # Middle SPADE ResNet blocks
        self.G_middle_0 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)

        # Upsampling stages
        self.up = nn.Upsample(scale_factor=2)
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)

        # Final RGB image output
        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, kernel_size=3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)
            )

    def forward(self, feature):
        """
        Forward pass of the SPADE decoder.

        Args:
            feature (Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            Tensor: Output animated image of shape (B, 3, H_out, W_out)
        """
        seg = feature  # BxCxHxW, used as the SPADE condition input

        x = self.fc(feature)  # Bx2C
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x
