from typing import Dict, Any, Tuple

import torch
from torch.utils import data


class Dataset(data.Dataset):
    """This interface defines what functions a class must support for
    consumption the GAN gorgan.

    Definitely implement `__getitem__()` and `__len__()`, where the first
    follows the dictionary key conventions as specified in its docstring.
    Each subclass is automatically available as a pytorch DataLoader thanks to
    the `as_dataloader()` method.
    """

    def __getitem__(self, idx):
        """The function that allows indexed access to this dataset (e.g. ds[0]).

        Implementation is left to subclasses to allow arbitrary data sources.

        Parameters
        ----------
        idx: int
            The index of the item to index. Should always be smaller than the
            length of this dataset.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the data for the GAN gorgan to train on.
            At the very least the keys `"originals"` and `"targets"` are
            required, which respectively contain the tensors from the original
            and target domain. Also accessed are `"originals_auxiliary"` and
            `"targets_auxiliary"` for auxiliary classes; these must be the class
            class index, not a 1-hot encoded vector. Furthermore, the keys
            `"originals_conditions"` and `"targets_conditions"` are used for
            conditional training and are expected to be tensors.
        """
        raise NotImplementedError()

    def __len__(self):
        """Returns the length of this dataset.

        Returns
        -------
        int
            The length of this dataset.
        """
        raise NotImplementedError()

    def as_dataloader(
        self, batch_size: int, data_workers: int, shuffle: bool = True
    ) -> data.DataLoader:
        return data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=data_workers,
        )
