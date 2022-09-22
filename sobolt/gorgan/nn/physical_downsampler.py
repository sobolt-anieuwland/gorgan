import math
from typing import Dict, Any, Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional.physical_downsampling import physical_downsampling as PD
from .functional.satellite_parameters import SatParams
from ..nn.functional.satellite_parameters import SatParamsSuperView, SatParamsSentinel2


class PhysicalDownsampler(nn.Module):
    def __init__(self, factor: int = 4, satellite: Union[str, SatParams] = "sentinel-2"):
        super().__init__()
        available_params = {
            "sentinel-2": SatParamsSentinel2(),
            "superview": SatParamsSuperView(),
        }
        params = available_params[satellite] if isinstance(satellite, str) else satellite

        self.__pad = nn.ReflectionPad2d(3)
        self.__down = PhysicalDownsampling(params, resize=False)
        self.__factor = factor

    def forward(self, image):
        return {
            "generated": F.interpolate(
                self.__down(self.__pad(image), self.__factor)[:, :, 3:-3, 3:-3],
                scale_factor=(1 / self.__factor),
                mode="bicubic",
                recompute_scale_factor=False,
            )
        }


class PhysicalDownsampling(nn.Module):
    """Block to downsample satellite data taking into account sensors' phyical properties.
    """

    __sat_params: SatParams

    def __init__(self, sat_params: SatParams, resize: bool = True):
        """Initialize the downsampling block.

        Parameters
        ----------
        sat_params: SatParams
            The satellite parameters of the target images.
        """
        super().__init__()
        self.__sat_params = sat_params
        self.__resize = resize

    def forward(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        """Push a satellite imagery tensor through the block to downsample it.

        Parameters
        ----------
        image: torch.Tensor
            A tensor of shape (4, H, W), where the bands on axis 0 are RGBI values.
        factor: float
            The value with which the input `image` is downsampled.

        Returns
        -------
        torch.Tensor
            A downsample of the input.
        """
        downsampleds = []
        for batch_item_idx in range(image.shape[0]):
            downsampled = PD(
                image[batch_item_idx], self.__sat_params * factor, factor, self.__resize
            )
            downsampleds.append(downsampled)
        return torch.stack(downsampleds)
