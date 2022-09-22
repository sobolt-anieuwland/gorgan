from typing import Dict, Any

import torch.nn.functional as F
import torch

from . import Dataset


class ProgressiveUpsamplingDecorator(Dataset):
    """Class that wraps a Dataset to on-the-fly resize the targets output with a
    given resampling factor.
    """

    __dataset: Dataset
    __upsample_factor: float

    def __init__(self, dataset: Dataset, upsample_factor: float):
        super().__init__()
        self.__dataset = dataset
        self.__upsample_factor = upsample_factor
