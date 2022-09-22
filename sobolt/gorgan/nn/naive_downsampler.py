from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveDownsampler(nn.Module):
    """ Module that downsamples the given input with algorithms not relying on any special
        characteristics of the data.
    """

    def __init__(self, factor: int = 4):
        """ Initialize a naive downsampler.

            Parameters
            ----------
            factor: int, (default 2)
                How much the input should be downsampled.
        """
        super().__init__()
        self.__factor = factor  # FIXME rename to better name

    def forward(self, tensor: torch.Tensor, factor: int = 4):
        """ Feed a tensor for downsampling

            Parameters
            ----------
            tensor: torch.Tensor
                The input to downsample. Assumed to [B, C, W, H].

            Returns
            -------
            torch.Tensor
                A tensor of [B, C, W, H] downsampled using `factor` as set during
                initialization.
        """
        if factor is None:
            factor = self.__factor
        b, c, w, h = tensor.shape
        target_dims = (w // factor, h // factor)
        return {"generated": F.adaptive_avg_pool2d(tensor, target_dims)}

    def grow(self):
        pass  # raise ValueError("Implement!")
