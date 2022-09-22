import torch
from torch import nn
import torch.nn.functional as F

from . import AdaptiveLayerInstanceNormalization


class ResidualWithAdaptiveNorm(nn.Module):
    """A residual convolution block with adaptive instance layer normalization.

    The normalization will either be instance or layer-based depending on learned
    parameters (beta/gamma) from the attention component.
    """

    # Graph components
    __pad_1: nn.Module
    __pad_2: nn.Module
    __conv_1: nn.Module
    __conv_2: nn.Module
    __norm_1: AdaptiveLayerInstanceNormalization
    __norm_2: AdaptiveLayerInstanceNormalization

    def __init__(self, in_features: int):
        """Initializes the ResidualWithAdaptiveNorm class.

        Parameters
        ----------
        in_features: int
            The number of in_features to shape the layer with.
        """
        super(ResidualWithAdaptiveNorm, self).__init__()

        self.__pad_1 = nn.ReflectionPad2d(1)
        self.__conv_1 = nn.Conv2d(in_features, in_features, kernel_size=3, bias=False)
        self.__norm_1 = AdaptiveLayerInstanceNormalization(in_features)

        self.__pad_2 = nn.ReflectionPad2d(1)
        self.__conv_2 = nn.Conv2d(in_features, in_features, kernel_size=3, bias=False)
        self.__norm_2 = AdaptiveLayerInstanceNormalization(in_features)

    def forward(
        self, features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Performs convolutions with skip connections and an adaptive normalization
        regularized with parameters learned from the attention component.

        Parameters
        ----------
        features: torch.Tensor
            Feature tensors we want to process with this residual block.
        gamma: torch.Tensor
            Regularization parameter γ, which can be learnt using an attention block.
        beta: torch.Tensor
            Regularization parameter β, which can be learnt using an attention block.

        Returns
        -------
        torch.Tensor
            The normalized features that can be further processed to generate an image.
        """
        features_out = self.__pad_1(features)
        features_out = self.__conv_1(features_out)
        features_out = self.__norm_1(features_out, gamma, beta)
        features_out = F.leaky_relu(features_out)
        features_out = self.__pad_2(features_out)
        features_out = self.__conv_2(features_out)
        features_out = self.__norm_2(features_out, gamma, beta)

        return features_out + features
