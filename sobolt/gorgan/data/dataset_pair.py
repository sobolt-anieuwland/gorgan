from typing import Dict, Any, Tuple

import torch

from .dataset import Dataset


class DatasetPair(Dataset):
    """This interface defines what functions a class must support for
    consumption the GAN gorgan.

    Definitely implement `__getitem__()` and `__len__()`, where the first
    follows the dictionary key conventions as specified in its docstring.
    Each subclass is automatically available as a pytorch DataLoader thanks to
    the `as_dataloader()` method.
    """

    __originals: Any
    __targets: Any

    def __init__(self, originals, targets):
        self.__originals = originals
        self.__targets = targets

    def __getitem__(self, idx):
        """The function that allows indexed access to this dataset (e.g. ds[0]).

        Parameters
        ----------
        idx: int
            The index of the item to index. Should always be smaller than the
            length of this dataset.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the data for the GAN gorgan to train on.
        """
        idx_o = idx % len(self.__originals)
        idx_t = idx % len(self.__targets)

        return {"originals": self.__originals[idx_o], "targets": self.__targets[idx_t]}

    def __len__(self):
        """Returns the length of this dataset.

        Returns
        -------
        int
            The length of this dataset.
        """
        return max(len(self.__originals), len(self.__targets))
