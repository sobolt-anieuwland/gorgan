from typing import Tuple, Union, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ChannelAttentionBlock


class ResidualChannelAttentionBlock(nn.Module):
    """
    Residual Channel Attention Block as proposed in Zhang et al. (2018):
    https://arxiv.org/abs/1807.02758

    Apply series of convolutions plus a final Channel Attention Block and skip connection.
    """

    __attention: nn.Module
    __conv1: nn.Module
    __conv2: nn.Module
    __batch_norm: nn.Module

    def __init__(
        self,
        in_features: int,
        kernel_size: int = 3,
        reduction: int = 16,
        batch_norm: bool = True,
    ):
        """
        Initializes the ResidualInResidualDenseBlock class.

        Parameters
        ----------
        in_features: int
            The initial number of ingoing features expected by te graph.
        kernel_size: int
            Size of the kernel to use to initialize the convolutional layers.
        reduction: int
            Reduction factor to apply in the convolution downscaling inside the Channel
            Attention Block.
        batch_norm: bool
            Flag to decide whether batch normalization should be applied or not
        """
        super().__init__()

        self.__conv1 = nn.Conv2d(
            in_features, in_features, kernel_size, padding=1, bias=True
        )
        self.__conv2 = nn.Conv2d(
            in_features, in_features, kernel_size, padding=1, bias=True
        )
        self.__attention = ChannelAttentionBlock(in_features, reduction)
        if batch_norm:
            self.__batch_norm = nn.BatchNorm2d(in_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Push input features through graph to learn features for generating an image.

        Parameters
        ----------
        features: torch.Tensor
            The input features to be further processed by the graph.

        Returns
        -------
        torch.Tensor
            Learned features that can be used to generate an output image with.
        """
        features_out = self.__conv1(features)
        if self.__batch_norm:
            features_out = F.relu(self.__batch_norm(features_out))
        else:
            features_out = F.relu(features_out)
        features_out = self.__conv2(features_out)
        if self.__batch_norm:
            features_out = self.__batch_norm(features_out)
            features_out = self.__attention(features_out)
        else:
            features_out = self.__attention(features_out)

        return features_out + features
