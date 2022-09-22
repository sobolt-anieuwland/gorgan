import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import SSIM


class Ssim(nn.Module):
    """Validation variant of the SSIM metric. It distinguishes itself in that it is
    calculated per batch item, instead of averaged over the whole batch.
    """

    def __init__(self, num_channels: int):
        """Initialize Ssim module."""
        super().__init__()
        self.__ssim = SSIM(
            data_range=1, channel=num_channels, size_average=False, nonnegative_ssim=True
        )

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the SSIM per batch item.

        See the documentation of VainF/pytorch-msssim's `SSIM()`.
        """
        return self.__ssim(generated.clip(0, 1), target.clip(0, 1))
