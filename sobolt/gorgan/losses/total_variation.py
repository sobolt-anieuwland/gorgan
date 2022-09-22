import torch
from torch import nn


class TotalVariationLoss(nn.Module):
    """The Total Variation Loss minimizes the amount of noise in a generated image.
    This loss computes the sum of absolute differences between neighboring
    pixels, which estimates the amount of noise in an image. Lower resolution input
    display a higher total variation over high resolution input.
    """

    def __init__(self):
        """Initializes the TotalVariationLoss class."""
        super(TotalVariationLoss, self).__init__()

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        """Computes the amount of noise present in generated input.

        Parameters
        ----------
        generated: torch.Tensor
            A generated input tensor of shape B x C x H x W.

        Returns
        -------
        torch.Tensor
            The sum of differences between the near pixels of generated input.
        """
        # Get input characteristics
        batch, channels, height, width = generated.size()

        # Compute total variation for height and width
        total_variation_height = torch.pow(
            generated[:, :, 1:, :] - generated[:, :, :-1, :], 2
        ).sum()
        total_variation_width = torch.pow(
            generated[:, :, :, 1:] - generated[:, :, :, :-1], 2
        ).sum()
        return (total_variation_height + total_variation_width) / (
            batch * channels * height * width
        )
