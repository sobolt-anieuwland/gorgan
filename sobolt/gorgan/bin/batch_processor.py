from typing import Dict, Any
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml

import rasterio as rio
from rasterio.windows import Window

from sobolt.gorgan.graphs.factory import graph_factory
from sobolt.gorgan.data.transforms import (
    denormalization_factory,
    normalization_factory,
    only_rgbi,
    bgr_to_rgb,
)


def process_batches(
    in_image: str,
    out_image: str,
    config: Dict[str, Any],
    overlap: int,
    tiles_size: int,
    padding: int,
    bgr: bool,
    sensor: str,
):
    """Large rasters processor. This function crop a raster into multiple overlapping
    tiles, it then apply a given model to each tile and write them to an output raster
    with same profile updated when relevant (e.g. new resolution and size for super
    resolution).
    Input raster is cropped by defining a rectangular grid that fit the entire raster,
    multiple polygons are still not supported.

    Parameters
    ----------
    in_image: str
        Path of the input raster to process. Expected tif raster, but will open any file
        allowed by rasterio
    out_image: str
        Path of the output raster to save.
    config: Dict[str, Any]
        Dictionary containing setting required to initialize the graph to use to process
        the input raster.
    overlap: int
        Pixel overlap to use when cropping input raster. Overlap is performed on all
        possible side of the cropped tile, i.e. always with only exception of tiles sides
        which correspond to the raster edge.
    tiles_size: int
        Pixel size of tiles to crop. This values doesn not include overlap or padding,
        i.e. it correspond to the final size of the tile as written to the output raster.
    padding: int
        Pixel size of padding appllied to cropped tiles before processing them with the
        given graph. Used to smooth out edge artifacts.
    bgr: bool
        Whether the input raster format is bgr or not. Necessary to determine if the input
        raster should be converted to rgb or not.
    sensor: str
        Name of satellite / sensor, required to apply the correct normalization and
        denormalization functions.
    """
    # The Predictor class apply a given model to a given tile
    generate = PytorchPredictor(config)

    factor = config.get("upsample_factor", 4)
    clip = config.get("clipping", False)
    with rio.open(in_image) as src:

        # Get profile information and update them for the output raster profile
        max_width = src.width
        max_height = src.height
        dst_profile = src.profile.copy()
        dst_transform = src.profile["transform"] * src.profile["transform"].scale(
            1 / factor, 1 / factor
        )
        dst_profile.update(
            {
                "transform": dst_transform,
                "height": src.profile["height"] * factor,
                "width": src.profile["width"] * factor,
                "driver": "GTiff",
                "crs": src.profile["crs"],
            }
        )

        # Open input raster and iteratevily
        with rio.open(out_image, "w", **dst_profile) as dst:
            steps_x = range(0, max_width + tiles_size, tiles_size)
            steps_y = range(0, max_height + tiles_size, tiles_size)

            # Define grid over raster Polygon
            boxes = ((x, y) for x in steps_x for y in steps_y)

            # Progress bar initialization
            pbar_tot = (max_width + tiles_size) * (max_height + tiles_size)
            pbar_tot = round(pbar_tot / tiles_size ** 2)
            pbar = tqdm(total=pbar_tot)
            pbar_counter = 1
            for (x, y) in boxes:

                # Tile coordinates
                tile_x = x - overlap
                tile_y = y - overlap
                tile_width = tiles_size + (2 * overlap)
                tile_height = tiles_size + (2 * overlap)

                # Handle tiles on the raster edges
                tile_x = max(min(tile_x, max_width), 0)
                tile_y = max(min(tile_y, max_height), 0)

                # Read tile
                tile = src.read(window=Window(tile_x, tile_y, tile_width, tile_height))

                if not 0 in list(tile.shape):

                    # Apply graph to tile
                    tile = generate(tile, padding, factor, bgr, sensor, clip)

                    # Revert dtype to original input dtype
                    type_min = np.iinfo(src.dtypes[0]).min
                    type_max = np.iinfo(src.dtypes[0]).max
                    tile = np.clip(tile, type_min, type_max)
                    tile = tile.astype(src.dtypes[0])

                    # Update tile coordinates
                    left_x = 0 if tile_x * factor == 0 else overlap * factor
                    right_x = left_x + (tiles_size * factor)
                    top_y = 0 if tile_y * factor == 0 else overlap * factor
                    bottom_y = top_y + (tiles_size * factor)

                    # Crop padding
                    tile = tile[:, top_y:bottom_y, left_x:right_x]

                    # Final output tile coordinates
                    x_in = x * factor
                    y_in = y * factor

                    dst.write(
                        tile, window=Window(x_in, y_in, tile.shape[2], tile.shape[1])
                    )
                pbar.update(pbar_counter)
        pbar.close()


class PytorchPredictor:
    """Predictor using pytorch to load serve the model"""

    __device: torch.device
    __model: nn.Module

    def __init__(self, model_config: Dict[str, Any], gpu: bool = True):
        # Specify printing of detailed training information
        quiet = model_config.get("quiet", False)

        # Load model and set it to evaluation mode
        self.__device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )

        self.__model = PytorchPredictor.build_graph(model_config, quiet, **model_config)
        self.__model = self.__model.to(self.__device)
        self.__model.eval()
        torch.cuda.empty_cache()

        self.__transforms = model_config.get("transforms", [])

    def __call__(
        self,
        inputs: np.ndarray,
        padding: int,
        factor: int,
        bgr: bool,
        sensor: str,
        clip: bool,
    ) -> np.ndarray:
        """Predict using an initialized pytorch model. See parent class for
        details.
        """
        # Preprocessing pipeline
        inputs = only_rgbi(inputs, seed=0, config={})

        normalizations = [normalization_factory(name) for name in self.__transforms]

        for normalization in normalizations:
            inputs = normalization(inputs, 0, {})

        stats = 0
        if "zscore" in self.__transforms:
            inputs, stats = inputs[0], inputs[1]

        torch_tensors_in = (
            torch.tensor(inputs.astype(np.float32)).to(self.__device).unsqueeze(0)
        )
        torch_tensors_in = F.pad(
            torch_tensors_in, [padding, padding, padding, padding], mode="replicate"
        )
        if clip:
            torch_tensors_in = torch_tensors_in.clip(0, 1)

        # Apply model and remove padding afterwards
        with torch.no_grad():
            torch_tensors_out = self.__model(torch_tensors_in)["generated"]
        if clip:
            torch_tensors_out = torch_tensors_out.clip(0, 1)
        padding_upsampled = padding * factor
        height, width = torch_tensors_out.shape[2:4]
        torch_tensors_out = torch_tensors_out[
            :,
            :,
            padding_upsampled : height - padding_upsampled,
            padding_upsampled : width - padding_upsampled,
        ]
        torch_tensors_out = torch_tensors_out.squeeze(0)
        torch_tensors_out = torch_tensors_out.to("cpu")
        array_out = torch_tensors_out.numpy()

        # Denormalization
        if "zscore" in self.__transforms:
            array_out = (array_out, stats)

        denormalization = denormalization_factory(self.__transforms[0])
        array_out = denormalization(array_out, 0, {})

        if bgr:
            # Function to convert from bgr to rgb also works to convert from rgb to bgr
            array_out = bgr_to_rgb(array_out, seed=0, config={})

        return array_out

    @staticmethod
    def build_graph(
        graph_name: Dict[str, Any], quiet: bool, **args: Dict[str, Any]
    ) -> nn.Module:
        """Build the graph for prediction, meaning instantiate the correct class."""
        return graph_factory(graph_name, quiet, **args)

    @staticmethod
    def yaml_load(path: str) -> Dict[str, Any]:
        """Read the model definition yaml config file."""
        with open(path) as h:
            contents = h.read()
            return yaml.safe_load(contents)
