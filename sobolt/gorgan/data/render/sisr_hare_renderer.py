from typing import Tuple, Dict, Any, Union, List

import PIL
import torch
import torch.nn.functional as F

from PIL.Image import Image

from sobolt.gorgan.data import add_title
from sobolt.gorgan.data.render import Renderer
from sobolt.gorgan.data.transforms import denormalize_zscore


class SisrHareRenderer(Renderer):
    """A class to render GAN panels for the SISR and HARE tasks."""

    __norm: str = "sigmoid"
    __infrared: bool = True
    __irg: bool = True
    __target_size: Tuple[int, int]

    def __init__(
        self, shape_targets: Union[List[int], Tuple[int, int, int]], norm: str = "sigmoid"
    ):
        """Initializes the SISR/HARE renderer.

        Parameters
        ----------
        shape_targets: Union[List[int], Tuple[int, int, int]]
            The shape of the target data of the form [C, H, W]. It is used to ensure all
            panel parts are of this shape.
        norm: The normalized used for the data. Valid values are sigmoid, tanh and zscore.
            This affects how the data is denormalized and then rendered.
        """
        super().__init__()
        self.__target_size = (shape_targets[1], shape_targets[2])
        self.__norm = norm

        assert self.__norm in ["sigmoid", "tanh", "zscore"]

    def __call__(
        self, batch: Dict[str, Any], batch_idx: int, renamings: Dict[str, str] = {}
    ) -> Dict[str, Image]:
        """Convert a training + inference batch to a dictionary of panels."""
        # fmt: off
        # Define order of batch items we want to render them in
        # Filter out those we don't want afterwards
        order = [
            "cycled_originals", "originals", "generated_originals",
            "cycled_targets", "targets", "generated_targets",
        ]
        order = [domain for domain in order if domain in batch.keys()]
        titles = [{"type": renamings.get(title, title)} for title in order]
        # fmt: on

        # Make a copy of the batch only containing tensors, because that's what the rest
        # operates on
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Preprocess the batch so it only contains the renders we want in the final panel
        # and their values are all between 0 and 1
        batch = self.prepare(batch)
        panels = self.combine(batch, order)
        return self.render(panels, batch_idx, titles)

    def render(
        self, panels: List[torch.Tensor], batch_idx: int, titles: List[Dict[str, str]]
    ) -> Dict[str, Image]:
        """Convert a list of tensors to pillow image, rendered to 8bit images."""
        rendered = [(tp * 255).clip(0, 255).byte() for tp in panels]
        num = len(panels)

        def to_image(tensor):
            image = PIL.Image.fromarray(tensor.cpu().numpy())
            return add_title(image, titles, self.__target_size[0])

        return {
            f"batch {batch_idx}, panel {i}.png": to_image(r)
            for (i, r) in enumerate(rendered, 1)
        }

    def combine(
        self, batch: Dict[str, torch.Tensor], order: List[str]
    ) -> List[torch.Tensor]:
        """Creates a list of panels out of the batch's tensors.

        To do so, the batch items of one batch index across the different values in the
        dictionary are combined into a single panel. E.g., if the dictionary contains
        a batch of tensors shaped [2, 3, 128, 128] for the keys "originals", "targets"
        and "generated_originals" this results in two panels because the batch is 2 large.
        Each panel will contain one batch item of all three keys.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
             The batch items to combine into a single panel. The tensors are assumed to
             all be of the same shape so they can be stacked / concatenated.
        order: List[str]
            A list of keys of items in batch. The order they occur in in the list is the
            order in which the data under those keys will be put in the panels.

        Returns
        -------
        List[torch.Tensor]
            A list of tensors, each list item being a panel of a single batch item.
        """
        assert len(order) > 0, "No renderable data present in batch"

        # Create a panel for each batch item
        panels = []
        num_batch_items = batch[order[0]].shape[0]
        for batch_item_idx in range(num_batch_items):
            panel_parts = [batch[domain][batch_item_idx] for domain in order]
            panel = torch.cat(panel_parts, dim=2)
            panel = panel.permute([1, 2, 0])  # Put channels last
            panels.append(panel)
        return panels

    def prepare(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns a batch of which all tensors are between 0 and 1 and which only
        contains tensor that should be included in a batch render.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            The dictionary of tensors produced by the data loader updated with the data
            produced by the GAN.

        Returns
        -------
        Dict[str, torch.Tensor]
            The input dictionary filtered to meet the following criteria:

            1. Only keys and tensors are present that must be included in the panels
            2. All tensors that we want included in the panels are in here
            3. All values of tensors are guaranteed to lie between 0 and 1
            4. All tensors are shaped such that they can be combined with `torch.cat()`
        """
        # Make a copy of the dictionary to prevent making modifications to a dictionary
        # other code also has a reference to
        batch = {k: v for k, v in batch.items()}

        # Define keys of interest but remove any not present in the batch
        combis: Dict[str, str] = {
            "originals": "originals_stats",
            "generated_originals": "originals_stats",
            "cycled_originals": "originals_stats",
            "targets": "targets_stats",
            "generated_targets": "targets_stats",
            "cycled_targets": "targets_stats",
        }
        combis = {k: v for k, v in combis.items() if k in batch}

        # Ensure all tensors are of the same shape, namely self.__target_shape
        # Condition: all keys in combis are present in the batch
        for k in combis.keys():
            data = batch[k]
            if (data.shape[2], data.shape[3]) == self.__target_size:
                continue
            size = self.__target_size
            data = F.interpolate(data, size=size, mode="bilinear", align_corners=False)
            batch[k] = data

        # TODO Haze mask?

        # Now:
        #   (1). Ensure all tensors are between 0 and 1
        #   (2). Ensure tensors include infrared and irg if so specified
        #   (3). Ensure return dict only contains tensors that should be part of the panel
        #
        # Condition: All batch items have the same shape: B × C × self.__target_size
        # Condition: All keys in combis are present in the batch
        result = {}
        for data_key, stats_source in combis.items():
            # Ensure point 1: All data between 0 and 1 using the helper method
            # Ensure point 3: Because we only return the tensors in combis
            # Parameter stats_source is ignored if self.__norm is not zscore
            result[data_key] = self.ensure_between_0_1(data_key, stats_source, batch)

            # If we want to render infrared / RGB:
            # We go from shape [B, C, H, W] to [B, C, 2H, W] or even [B, C, 3H, W]
            if (self.__infrared or self.__irg) and result[data_key].shape[1] == 4:
                h = self.__target_size[0]
                rgbi = result[data_key][:, :, :h, :]
                r, g, b, i = rgbi.unbind(1)

                results = [rgbi]
                # For now keep fourth band; necessary to be able to stack.
                # We remove it later
                if self.__irg:
                    irgb = torch.stack([i, r, g, b], dim=1)
                    results.append(irgb)
                if self.__infrared:
                    empty = torch.zeros_like(i)
                    infrared = torch.stack([i, empty, empty, empty], dim=1)
                    results.append(infrared)
                result[data_key] = torch.cat(results, dim=2)

        # Reduce all to 3 channels and return
        result = {k: v[:, :3] for k, v in result.items()}
        return result

    def ensure_between_0_1(
        self, data_key: str, stats_source: str, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Helper function to ensure a tensor is mapped to be between 0 and 1 for
        different normalization styles the right way.

        Parameters
        ----------
        data_key: str
            The key to access the right tensor with, such as 'originals', 'targets', or
            'generated_originals'. Accessed tensor should be [B, C, H, W].
        stats_source: str
            The key to access the zscore stats with. Ignored if the normalization method
            is not zscore. Examples are 'targets_stats' or 'originals_stats' for 'targets'
            and 'generated_originals' respectively. Accessed tensor should be
            [B, C, 2].
        batch: Dict[str, torch.Tensor]
            The batch to access the data from

        Returns
        -------
        torch.Tensor
            The tensor accessible under data_key with all values guaranteed to be between
            0 and 1, with a shape of [B, C, H, W].
        """
        if self.__norm == "sigmoid":
            data = batch[data_key].clip(0, 1)
        elif self.__norm == "tanh":
            data = ((batch[data_key] + 1) / 2).clip(0, 1)
        elif self.__norm == "zscore":
            data = batch[data_key].clone()
            for batch_item_idx in range(data.shape[0]):
                item = data[batch_item_idx]
                stats = batch[stats_source][batch_item_idx]
                data[batch_item_idx] = denormalize_zscore((item, stats), channelwise=True)
        else:
            raise ValueError(f"Unknown normalization {self.__norm}")

        return data
