from typing import Callable, Any

import torch
import torch.nn as nn


class SubpixelConvolution(nn.Module):
    """This class forms a sequence of layers for input upsampling."""

    def __init__(self, num_features: int):
        """This function initializes a sequence of layers for input upsampling.

        Parameters
        ----------
        num_features: int
            The shape of the incoming features from the processed tensor.
        """
        super().__init__()
        self.__main = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 1),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.ReplicationPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
        )

        # Initialize convolution with ICNR to prevent the checkerboard effect
        self.__main.apply(self.init_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply subpixel convolution to the input.

        Parameters
        ----------
        features: torch.Tensor
            The tensor to upsample using subpixel convolution. Should be of the shape
            [B, C, H, W].

        Returns
        -------
        torch.Tensor:
            The layers for processing a tensor of num_features with shape B x C x H x W
            into a shape of B x C x H * 2 x W * 2.
        """
        return self.__main(features)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            weight = self.__icnr(m.weight.data, mean=0.0, std=0.02)
            m.weight.data.copy_(weight)

    def __icnr(
        self,
        inputs: torch.Tensor,
        initializer=nn.init.normal_,
        upsample_factor: int = 2,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Initialize a layer with the convolution initialized to convolution NN resize
        (ICNR)from the paper Aitken et al. (2017).

        Parameters
        ----------
        inputs: torch.Tensor
            Layer to initialize
        initializer:
            The torch weight initialization distribution. Normal is default.
        upsample_factor: int
            The upsample factor applied on the outgoing features of the layer
        args
            Arguments for the initializer
        kwargs
            Key word arguments for the initializer

        Returns
        -------
        torch.Tensor
            The ICNR weight tensor to initialize a layer with.
        """
        upsample_squared = upsample_factor * upsample_factor

        assert inputs.shape[0] % upsample_squared == 0, (
            "The size of the first dimension: "
            f"tensor.shape[0] = {inputs.shape[0]}"
            " is not divisible by square of upsample_factor: "
            f"upscale_factor = {upsample_factor}"
        )

        features = inputs.shape[0] // upsample_squared
        sub_kernel = torch.empty(features, *inputs.shape[1:])
        sub_kernel = initializer(sub_kernel, *args, **kwargs)
        return sub_kernel.repeat_interleave(upsample_squared, dim=0)
