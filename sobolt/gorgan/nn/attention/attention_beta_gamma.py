from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBetaGamma(nn.Module):
    """Dense layer from the attention map (generated during attention block) that
    dynamically computes gamma and beta.

    These parameters are used for regularizing AdaLIN in decoder of generator,
    before upsampling happens.

    """

    # Graph components
    __dense_layer: nn.Sequential
    __gamma: nn.Linear
    __beta: nn.Linear

    def __init__(self, in_features: int, num_filters: int = 1):
        """Initializes the AttentionBetaGamma class.

        Parameters
        ----------
        in_features: int
            The shape of the incoming features from the processed tensor.
        num_filters: int
            The number of filters to create for a given layer.
        """
        super(AttentionBetaGamma, self).__init__()
        self.__dense_layer = nn.Sequential(
            nn.Linear(in_features * num_filters, in_features * num_filters, bias=False),
            nn.LeakyReLU(),
        )

        # Extract learned parameters from dense layers
        self.__gamma = nn.Linear(
            in_features * num_filters, in_features * num_filters, bias=False
        )
        self.__beta = nn.Linear(
            in_features * num_filters, in_features * num_filters, bias=False
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes feature tensors to learn beta and gamma parameters for adaptive
        normalization.

        Parameters
        ----------
        features: torch.Tensor
           The incoming feature maps to be further processed

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Learned parameters gamma and beta that are used in the attention-based
            adaptive normalization.
        """
        features = F.adaptive_avg_pool2d(features, [1, 1])
        features = self.__dense_layer(features.view(features.shape[0], -1))
        return self.__gamma(features), self.__beta(features)
