# src/modules/motion_extractor.py
# coding: utf-8

"""
Motion Extractor (M) module.

This module predicts canonical keypoints, head pose, and expression deformation 
directly from the input image using a ConvNeXtV2-based feature extractor.
"""

import torch
import torch.nn as nn

from .convnextv2 import convnextv2_tiny
from .util import filter_state_dict


# Extend this dictionary if you add more backbones
model_dict = {
    'convnextv2_tiny': convnextv2_tiny,
}


class MotionExtractor(nn.Module):
    def __init__(self, **kwargs):
        """
        Initialize the motion extractor.

        Args:
            backbone (str): Name of the backbone model to use.
            Other kwargs are passed to the backbone model.
        """
        super(MotionExtractor, self).__init__()

        backbone_name = kwargs.get('backbone', 'convnextv2_tiny')
        assert backbone_name in model_dict, f"Unsupported backbone: {backbone_name}"
        self.detector = model_dict[backbone_name](**kwargs)

    def load_pretrained(self, init_path: str):
        """
        Load pretrained weights into the detector backbone.

        Args:
            init_path (str): Path to the checkpoint file.
        """
        if init_path:
            checkpoint = torch.load(init_path, map_location=lambda storage, loc: storage)

            # Automatically handle checkpoints saved with or without top-level 'model' key
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            filtered_state_dict = filter_state_dict(state_dict, remove_name='head')

            ret = self.detector.load_state_dict(filtered_state_dict, strict=False)
            print(f"[INFO] Loaded pretrained model from {init_path}, result: {ret}")

    def forward(self, x):
        """
        Forward pass of the motion extractor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            dict: Output features, typically including keypoints, head pose, expression info, etc.
        """
        return self.detector(x)
