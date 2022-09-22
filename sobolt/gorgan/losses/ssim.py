import torch
from torch import nn

from pytorch_msssim import SSIM


class SsimLoss(nn.Module):
    """The Structural Similarity Loss attempts to minimize the *perceived* quality
    difference between two pictures.
    """

    def __init__(self, num_channels: int):
        """Initializes the SSIM class."""
        super().__init__()
        self.__ssim = SSIM(data_range=1, channel=num_channels, size_average=True)

    def forward(self, generated: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the complement to structural similarity.

        Since SSIM a higher SSIM score is better, this loss takes the complement it by
        subtracting the computed result from 1, e.g. loss = 1 - ssim(generated, targets).

        Parameters
        ----------
        generated: torch.Tensor
            A generated input tensor of shape B x C x H x W.
        targets: torch.Tensor
            A generated input tensor of shape B x C x H x W, where each is the same
            as for generated.

        Returns
        -------
        torch.Tensor
            The perceived quality difference between the generated and targets for each
            batch item.
        """
        return 1 - self.__ssim(generated.clip(0, 1), targets.clip(0, 1))
