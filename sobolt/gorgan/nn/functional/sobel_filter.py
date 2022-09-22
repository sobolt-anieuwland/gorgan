from typing import Tuple

import torch
import torch.nn.functional as F


def apply_sobel(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sobel operator, which is a filter to extract gradients from an image.

    Parameters
    ----------
    inputs: torch.Tensor
        A tensor we want to apply a sobel operator to extract vertical and horizontal
        gradients from.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Vertical and horizontal image gradients.
    """
    # Horizontal Sobel filter
    x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]).float().to(inputs.device)
    gradient_x = F.conv2d(inputs, x, bias=None, stride=1, padding=1, groups=1)

    # Vertical Sobel filter
    y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).float().to(inputs.device)
    gradient_y = F.conv2d(inputs, y, bias=None, stride=1, padding=1, groups=1)

    return gradient_x, gradient_y
