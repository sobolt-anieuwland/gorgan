from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class EsrganDenseBlock(nn.Module):
    """The ESRGAN's core is composed of feature learning blocks, which are a series of
    convolution and activations. The block is dense because all layers are connected to
    each other. Architecture comes from the ESRGAN paper (Wang et al., 2018).
    """

    __conv1: nn.Module
    __conv2: nn.Module
    __conv3: nn.Module
    __conv4: nn.Module
    __conv5: nn.Module

    __res_scale: float

    def __init__(self, in_features: int, out_features: int = 32, res_scale: float = 0.2):
        """Initializes the EsrganDenseBlock class.

        Parameters
        ----------
        in_features: int
            The number of inputting features for a convolution layer.
        out_features: int
            The number of outputting features for a convolution layer.
        res_scale: float
            The scalar to multiply to outgoing skip connections.
        """
        super().__init__()

        # Compute the iterative calculation of in features for each layer
        calc_feats = lambda mult: in_features + mult * out_features

        self.__conv1 = nn.Conv2d(calc_feats(0), out_features, 3, padding=1, bias=True)
        self.__conv2 = nn.Conv2d(calc_feats(1), out_features, 3, padding=1, bias=True)
        self.__conv3 = nn.Conv2d(calc_feats(2), out_features, 3, padding=1, bias=True)
        self.__conv4 = nn.Conv2d(calc_feats(3), out_features, 3, padding=1, bias=True)
        self.__conv5 = nn.Conv2d(calc_feats(4), in_features, 3, padding=1, bias=True)

        self.__res_scale = res_scale

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
        feature_out_1 = self.__conv1(features)
        feature_out_1 = checkpoint(F.leaky_relu, feature_out_1)

        feature_out_2 = self.__conv2(torch.cat((features, feature_out_1), 1))
        feature_out_2 = checkpoint(F.leaky_relu, feature_out_2)

        feature_out_3 = self.__conv3(
            torch.cat((features, feature_out_1, feature_out_2), 1)
        )
        feature_out_3 = checkpoint(F.leaky_relu, feature_out_3)

        feature_out_4 = self.__conv4(
            torch.cat((features, feature_out_1, feature_out_2, feature_out_3), 1)
        )

        feature_out_4 = checkpoint(F.leaky_relu, feature_out_4)

        feature_out_5 = self.__conv5(
            torch.cat(
                (features, feature_out_1, feature_out_2, feature_out_3, feature_out_4), 1
            )
        )
        feature_out_5 = checkpoint(F.leaky_relu, feature_out_5)
        return feature_out_5.mul(self.__res_scale) + features
