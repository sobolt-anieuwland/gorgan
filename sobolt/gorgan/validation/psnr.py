import torch
import torch.nn as nn
import torch.nn.functional as F


class Psnr(nn.Module):
    """Module to calculate PSNR per batch item."""

    def __init__(self):
        """Initialize PSNR module."""
        super().__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Calculate the PSNR per batch item.

        See the documentation of the function `calculate_psnr()`. This class is a thin
        wrapper around it to allow dynamically instantiating a callable instance that
        satisfies mypy.
        """
        return calculate_psnr(*args, **kwargs)


def calculate_psnr(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate PSNR per batch item.

    Parameters
    ----------
    generated: torch.Tensor
        The generated data to compare the PSNR for. Should have a batch dimension, e.g.
        the shape should be [B, other dimensions].
    target: torch.Tensor
        The label to compute the PSNR with. Should have the same dimensions as generated.

    Returns
    -------
    torch.Tensor
        Returns the calculated PSNR per batch item in a tensor with as many results as
        there were batch items. E.g. with a batch size of 4 there will be 4 PSNR values.
    """
    assert generated.shape[0] == target.shape[0]
    mse_reduced_per_item = torch.zeros((target.shape[0],), device=target.device)
    mse = F.mse_loss(generated, target, reduction="none")
    for batch_item in range(mse.shape[0]):
        mse_reduced_per_item[batch_item] = torch.mean(mse[batch_item])
    divisors = 1 / (mse_reduced_per_item + 1e-14)
    return 10 * torch.log10(divisors)
