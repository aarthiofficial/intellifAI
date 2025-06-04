# coding: utf-8

"""
ConvNeXtV2 module adapted for extracting implicit keypoints, poses, and expression deformation.
"""

import torch
import torch.nn as nn
from .util import LayerNorm, DropPath, trunc_normal_, GRN

__all__ = ['convnextv2_tiny']


class Block(nn.Module):
    """
    ConvNeXtV2 Block

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depthwise convolution
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """
    ConvNeXtV2 backbone with keypoint, pose, and expression outputs.

    Args:
        in_chans (int): Input image channels.
        depths (list[int]): Number of blocks per stage.
        dims (list[int]): Feature dimensions per stage.
        drop_path_rate (float): Drop path rate.
        num_bins (int): Output bins for pose classification.
        num_kp (int): Number of keypoints.
    """

    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        **kwargs
    ):
        super().__init__()
        num_bins = kwargs.get("num_bins", 66)
        num_kp = kwargs.get("num_kp", 24)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            blocks = [Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # Output heads
        self.fc_kp = nn.Linear(dims[-1], 3 * num_kp)
        self.fc_scale = nn.Linear(dims[-1], 1)
        self.fc_pitch = nn.Linear(dims[-1], num_bins)
        self.fc_yaw = nn.Linear(dims[-1], num_bins)
        self.fc_roll = nn.Linear(dims[-1], num_bins)
        self.fc_t = nn.Linear(dims[-1], 3)
        self.fc_exp = nn.Linear(dims[-1], 3 * num_kp)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])  # Global average pooling
        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)

        return {
            "kp": self.fc_kp(x),          # Implicit keypoints
            "pitch": self.fc_pitch(x),    # Head pose
            "yaw": self.fc_yaw(x),
            "roll": self.fc_roll(x),
            "t": self.fc_t(x),            # Translation
            "scale": self.fc_scale(x),    # Scale
            "exp": self.fc_exp(x),        # Expression deltas
        }


def convnextv2_tiny(**kwargs):
    return ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
