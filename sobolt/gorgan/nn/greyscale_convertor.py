import torch
import torch.nn as nn


class ConvertToGrey(nn.Module):
    """Converts multi channels tensor into single channel grayscale tensor."""

    def __init__(self):
        """Initializes the ConvertToGrey"""
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Greyscales input channels of a tensor.

        Parameters
        ----------
        inputs: torch.Tensor
            Input tensor (BxCxWxH) we want to convert to greyscale.

        Returns
        -------
        torch.Tensor
            Input channels averaged into single grayscale channel.
        """
        r, g, b = inputs[:, :3, :, :].unbind(dim=-3)
        gray_inputs = (0.2989 * r + 0.587 * g + 0.114 * b).to(inputs.dtype)
        return gray_inputs.unsqueeze(dim=-3)
