from pathlib import Path
from typing import Any, Callable, Dict, List, Union
import time

import numpy as np
import torch
import torchvision as torchv
import torch.nn.functional as F

from . import transform_factory


class ImageFolderDataset:
    @staticmethod
    def from_config(config: Dict[str, Any], normalization: List[float]):
        transforms = [
            transform_factory(name)
            for name in config.get("transforms", [])
            if name != "to_tensor"
        ]
        args = config["args"].copy()
        args["transforms"] = transforms
        args["normalization"] = normalization
        return ImageFolderDataset(**args)

    __images: torchv.datasets.ImageFolder
    __transforms: List[Callable[[Any, int, Any], Any]]

    def __init__(self, root: Union[str, Path], transforms, normalization: List[float]):
        self.__images = torchv.datasets.ImageFolder(root=root)
        self.__transforms = transforms
        self.__normalization = normalization

    def __getitem__(self, idx) -> torch.Tensor:
        seed = int(time.time() * 1000)
        img = np.array(self.__images[idx][0])
        img = np.swapaxes(img, 1, 2)  # Move channels first
        img = np.swapaxes(img, 0, 1)
        for transform in self.__transforms:
            img = transform(img, seed, {})
        return torch.tensor(img.copy())

    def __len__(self) -> int:
        return len(self.__images)
