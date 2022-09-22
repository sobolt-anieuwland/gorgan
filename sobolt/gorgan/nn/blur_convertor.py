import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvertToBlur(nn.Module):
    """Applies gaussian blur to a tensor's channel independent of the number of
    channels present (BxCxHxW).
    """

    __weight: nn.Parameter

    def __init__(self):
        """Initializes the ConvertToBlur function"""
        super().__init__()
        gaussian_filter = [
            [0.0113, 0.0838, 0.0113],
            [0.0838, 0.6193, 0.0838],
            [0.0113, 0.0838, 0.0113],
        ]
        gaussian_filter = torch.FloatTensor(gaussian_filter).unsqueeze(0).unsqueeze(0)
        self.__weight = nn.Parameter(data=gaussian_filter, requires_grad=False)

    def __add_random_values(self, tensor):
        tensor = tensor + (0.015 * torch.randn(tensor.shape).to(tensor.device))
        return tensor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies a gaussian filter over each channel in an input tensor.

        Parameters
        ----------
        inputs: torch.Tensor
            The input tensor (BxCxHxW) we want to apply gaussian blur on.
        Returns
        -------
        torch.Tensor
            Gaussian blurred input channels of the tensor.
        """
        channels = inputs.unbind(dim=-3)
        channels = [
            F.conv2d(
                channel.unsqueeze(1), self.__add_random_values(self.__weight), padding=1
            )
            for channel in channels
        ]
        return torch.cat(channels, dim=-3)
