from typing import Dict, Any
import torch.nn as nn

from .residual_block import ResidualBlock


class ResidualIdentityBlock(ResidualBlock):
    @staticmethod
    def from_config(config: Dict[str, Any]):
        # TODO
        raise NotImplementedError("Please implement")

    def __init__(
        self,
        in_channels,
        mid_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        multiplier=1.0,
    ):
        # TODO Add names for layers to weights can be loaded
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                mid_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
                bias,
                dilation,
                padding_mode,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                mid_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
                bias,
                dilation,
                padding_mode,
            ),
            nn.BatchNorm2d(in_channels),
        ]

        res_graph = nn.Sequential(*layers)
        skip_conn = nn.Identity()
        super().__init__(res_graph, skip_conn, multiplier=multiplier)
