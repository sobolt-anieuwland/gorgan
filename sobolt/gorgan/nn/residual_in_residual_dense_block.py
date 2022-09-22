from typing import Optional, List

import torch
import torch.nn as nn

from . import EsrganDenseBlock


class ResidualInResidualDenseBlock(nn.Module):
    """A ESRGAN generator's body consists of multiple RRDB blocks. This is multiple fully
    connected blocks (dense), with skip connections betweenthem. Architecture comes
    from the ESRGAN paper (Wang et al., 2018).
    """

    __block1: EsrganDenseBlock
    __block2: EsrganDenseBlock
    __block3: EsrganDenseBlock

    __res_scale: float

    def __init__(self, in_features: int, out_features: int = 32, res_scale: float = 0.2):
        """Initializes the ResidualInResidualDenseBlock class.

        Parameters
        ----------
        in_features: int
            The initial number of ingoing features expected by te graph.
        out_features: int
            The outgoing number of features expected by the graph.
        res_scale: float
            The scalar to multiply to outgoing skip connections.
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.__block1 = EsrganDenseBlock(in_features, out_features)
        self.__block2 = EsrganDenseBlock(in_features, out_features)
        self.__block3 = EsrganDenseBlock(in_features, out_features)
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
        features_out = self.__block1(features)
        features_out = self.__block2(features_out)
        features_out = self.__block3(features_out)
        return features_out.mul(self.__res_scale) + features
