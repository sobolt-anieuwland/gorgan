from typing import Dict, Any
import torch.nn as nn

from .residual_block import ResidualBlock


class ResidualConvolutionalBlock(ResidualBlock):
    @staticmethod
    def from_config(config: Dict[str, Any]):
        # TODO
        raise NotImplementedError("Please implement")

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        multiplier: float = 1.0,
    ):
        # TODO Adjust to In1
        # * Add names for layers to weights can be loaded
        # * More configuration necessary? (Configurable amount of layers
        #   between residue and input?)
        # * Final check by Otto/Danny to ensure implementation is theoretically
        #   sound (see ResidualBlock class)
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
            nn.ReLU(True),
            nn.ConvTranspose2d(
                mid_channels,
                out_channels,
                kernel_size,
                1,  # stride
                padding,
                output_padding,
                groups,
                bias,
                dilation,
                padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
        ]

        res_graph = nn.Sequential(*layers)
        skip_conn = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
                bias,
                dilation,
                padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
        )
        super().__init__(res_graph, skip_conn, multiplier=multiplier)
