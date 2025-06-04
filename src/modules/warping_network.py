# coding: utf-8

"""
Warping field estimator (W) as defined in the paper. It generates a warping field using
implicit keypoint representations (kp_source and kp_driving), and employs this flow field
to warp the source feature volume (feature_3d).
"""

from torch import nn
import torch.nn.functional as F
from .util import SameBlock2d
from .dense_motion import DenseMotionNetwork


class WarpingNetwork(nn.Module):
    def __init__(
        self,
        num_kp,
        block_expansion,
        max_features,
        num_down_blocks,
        reshape_channel,
        estimate_occlusion_map=False,
        dense_motion_params=None,
        **kwargs
    ):
        super(WarpingNetwork, self).__init__()

        self.upscale = kwargs.get('upscale', 1)
        self.flag_use_occlusion_map = kwargs.get('flag_use_occlusion_map', True)
        self.estimate_occlusion_map = estimate_occlusion_map

        # Initialize DenseMotionNetwork if parameters are provided
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp,
                feature_channel=reshape_channel,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params
            )
        else:
            self.dense_motion_network = None

        # Feature processing blocks after deformation
        feature_channels = block_expansion * (2 ** num_down_blocks)
        self.third = SameBlock2d(
            max_features,
            feature_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            lrelu=True
        )
        self.fourth = nn.Conv2d(
            in_channels=feature_channels,
            out_channels=feature_channels,
            kernel_size=1,
            stride=1
        )

    def deform_input(self, inp, deformation):
        """
        Applies deformation to the input tensor using grid_sample.

        Args:
            inp (Tensor): Input feature tensor of shape (B, C, D, H, W)
            deformation (Tensor): Deformation grid of shape (B, D, H, W, 3)

        Returns:
            Tensor: Deformed feature tensor of shape (B, C, D, H, W)
        """
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, feature_3d, kp_driving, kp_source):
        """
        Forward pass for the warping network.

        Args:
            feature_3d (Tensor): Source feature volume, shape (B, C, D, H, W)
            kp_driving (dict): Keypoints from driving image
            kp_source (dict): Keypoints from source image

        Returns:
            dict: Dictionary containing occlusion_map, deformation, and output features
        """
        occlusion_map = None
        deformation = None
        out = feature_3d

        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(
                feature=feature_3d,
                kp_driving=kp_driving,
                kp_source=kp_source
            )
            deformation = dense_motion['deformation']  # Expected shape: (B, D, H, W, 3)

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']  # Shape: (B, 1, H, W)

            # Warp the feature volume using the deformation grid
            out = self.deform_input(feature_3d, deformation)  # Shape: (B, C, D, H, W)

            bs, c, d, h, w = out.shape
            # Merge depth dimension into channel dimension for 2D conv processing
            out = out.view(bs, c * d, h, w)  # Shape: (B, C*D, H, W)

            out = self.third(out)  # Apply convolution block
            out = self.fourth(out)  # Final conv layer

            # Apply occlusion map if enabled and available
            if self.flag_use_occlusion_map and occlusion_map is not None:
                out = out * occlusion_map

        return {
            'occlusion_map': occlusion_map,
            'deformation': deformation,
            'out': out,
        }
