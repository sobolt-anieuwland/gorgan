import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdaptiveLayerInstanceNormalization(nn.Module):
    """Normalization to be called before the upsampling block of the generator.

    The normalization will either be instance based or layer based depending on the
    value of the learned regularization parameters (gamma / beta).
    """

    __eps: float
    __rho: nn.parameter.Parameter

    def __init__(self, in_features: int, eps: float = 1e-5):
        """Initializes the AdaptiveLayerInstanceNormalization class.

        Parameters
        ----------
        in_features: int
            The number of in_features to shape the layer with.
        eps
            A term to prevent 0-division.
        """
        super(AdaptiveLayerInstanceNormalization, self).__init__()
        self.__eps = eps
        self.__rho = Parameter(torch.Tensor(1, in_features, 1, 1))
        self.__rho.data.fill_(0.9)

    def forward(
        self, features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Applies adaptive normalization to feature inputs.

        Parameters
        ----------
        features: torch.Tensor
            The input to normalize.
        gamma: torch.Tensor
            Regularization parameter γ, which can be learnt using an attention block.
        beta: torch.Tensor
            Regularization parameter β, which can be learnt using an attention block.

        Returns
        -------
        torch.Tensor
            Normalization of the tensor will either be instance or layer based, depending
            on the given beta and gamma.
        """
        # Compute instance normalization
        instance_mean, instance_sigma = (
            torch.mean(features, dim=[2, 3], keepdim=True),
            torch.var(features, dim=[2, 3], keepdim=True),
        )
        instance_norm = (features - instance_mean) / torch.sqrt(
            instance_sigma + self.__eps
        )

        # Compute layer normalization
        layer_mean, layer_sigma = (
            torch.mean(features, dim=[1, 2, 3], keepdim=True),
            torch.var(features, dim=[1, 2, 3], keepdim=True),
        )
        layer_norm = (features - layer_mean) / torch.sqrt(layer_sigma + self.__eps)

        # Compute the instance-layer normalization
        rho = self.__rho.expand(features.shape[0], -1, -1, -1)
        normalization = rho * instance_norm + (1 - rho) * layer_norm

        # Regularize normalization based on gamma & beta parameters
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return normalization * gamma + beta
