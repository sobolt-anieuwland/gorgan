from typing import Tuple
import numpy as np

import torch
import torch.nn as nn

from .functional import apply_sobel


class OrientationMagnitudeExtractor(nn.Module):
    """
    Computes a tensor's gradient orientation and magnitude with a sobel filter, which can
    then be used to plot a histogram of gradient orientation and magnitude. This gives
    us an estimation of the degree of sharpness of an input.
    """

    __gaussian_filter_2d: torch.Tensor
    __device: torch.device

    def __init__(self, device):
        """
        Initializes the OrientationMagnitudeExtractor class.
        """
        super(OrientationMagnitudeExtractor, self).__init__()

        self.__device = device

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the image tensor to convert it into the gradient space and extract
        the gradients' magnitude and orientation.

        Parameters
        ----------
        inputs: torch.Tensor
            A tensor in the form of (B x C x H x W).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple consisting of the magnitude and orientation of the gradients that
            make up the input.
        """
        # Convert input into cv2 format to grayscale it (facilitates subsequent gradient
        # conversion)
        r, g, b = inputs.unbind(dim=-3)
        gray_inputs = (0.2989 * r + 0.587 * g + 0.114 * b).to(inputs.dtype)
        gray_inputs = gray_inputs.unsqueeze(dim=-3)

        # Get vertical and horizontal gradients
        gradient_x, gradient_y = apply_sobel(gray_inputs)

        # Extract gradients' magnitude and orientation
        gradient_magnitude = (gradient_x ** 2 + gradient_y ** 2) ** 0.5
        gradient_orientation = torch.atan(gradient_y.float() / gradient_x.float())
        gradient_orientation = torch.round(gradient_orientation * (360 / np.pi) + 180)

        return gradient_magnitude, torch.round(gradient_orientation)
