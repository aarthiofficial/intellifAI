# src/modules/stitching_retargeting_network.py
# coding: utf-8

"""
Stitching and Retargeting Modules (S + R) for animated portrait synthesis.

- Stitching module (S) blends the generated face back into the original image seamlessly.
- Eye and lip retargeting modules (R) address alignment issues and ensure realism in expression.
"""

import torch
import torch.nn as nn


class StitchingRetargetingNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        """
        Initialize a simple MLP for stitching and retargeting.

        Args:
            input_size (int): Dimensionality of the input features.
            hidden_sizes (list of int): Sizes of hidden layers.
            output_size (int): Dimensionality of the output.
        """
        super().__init__()

        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.mlp = nn.Sequential(*layers)

    def initialize_weights_to_zero(self):
        """
        Initializes all weights and biases of Linear layers to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): Input tensor of shape (B, input_size)

        Returns:
            Tensor: Output tensor of shape (B, output_size)
        """
        return self.mlp(x)
