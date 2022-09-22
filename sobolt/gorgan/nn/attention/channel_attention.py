from typing import Optional, List

import torch.nn.functional as F
import torch.nn as nn
import torch


class ChannelAttentionBlock(nn.Module):
    """Attention Block implementation as proposed in Zhang et al. (2018):
    https://arxiv.org/abs/1807.02758

    Apply channelwise convolution through global average pooling and a gate mechanism
    with sigmoid function.
    """

    def __init__(self, channel, reduction: int = 16):
        super(ChannelAttentionBlock, self).__init__()

        # Global average pooling: feature --> point
        self.__avg_pool = nn.AdaptiveAvgPool2d(1)

        # Feature channel downscale and upscale --> channel weight
        self.__conv1 = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)
        self.__conv2 = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Push input features through graph to learn features for generating an image.

        Parameters
        ----------
        features: torch.Tensor
            The input features to be further processed by the graph.

        Returns
        -------
        torch.Tensor
            Learned features that can be used to generate an output image with.
        """
        features_out = self.__avg_pool(features)
        features_out = self.__conv1(features_out)
        features_out = self.__conv2(F.relu(features_out))

        return features * torch.sigmoid(features_out)
