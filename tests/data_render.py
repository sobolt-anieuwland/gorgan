from typing import List

import PIL
import pytest
import numpy as np
import torch
import torch.nn.functional as F

from sobolt.gorgan.data.render import SisrHareRenderer


# fmt: off
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_channels", [3, 4])
@pytest.mark.parametrize("norm", ["sigmoid", "tanh", "zscore"])
@pytest.mark.parametrize(
    "panel_parts",
    [
        ["originals", "generated_originals", "targets"],
        ["originals", "generated_originals", "cycled_originals", "targets"],
        ["originals", "generated_originals", "cycled_originals", "generated_targets",
         "cycled_targets", "targets"],
    ],
)
# fmt: on
def test_sisr_hare_renderer_basegan(
    batch_size: int, num_channels: int, norm: str, panel_parts: List[str]
):
    """Function to test the SisrHareRendering class.

    It tests the following properties
        * Does the function execute successfully (but not necessarily correctly) for
          sigmoid, tanh and zscore data?
        * If a given batch only includes certain data (e.g. only originals,
          generated_originals and targets) will all those and only those be in there?
          Does the same hold for other kinds of data?
        * Are infrared and IRG correctly included if it is 4 channel data?
        * Does it correctly render multiple panels if we have a batch size greater than 1?
          Does it also work for a batch size of 1?
        * Does the function execute successfully for both RGB and RGBI data?

    The greatest missing factor is whether or not the different normalization are
    correctly denormalized.
    """
    size = 64
    batch = {}

    # Preparing data for rendering
    inputs = torch.randint(0, 255, (num_channels, size, size)).float()
    if norm == "sigmoid":
        inputs = inputs / 255
    elif norm == "tanh":
        inputs == inputs / 255 * 2 - 1
    elif norm == "zscore":
        stats = torch.zeros((inputs.shape[0], 2))
        for c in range(inputs.shape[0]):
            mean = inputs[c].mean()
            std = inputs[c].std()
            inputs[c] = (inputs[c] - mean) / std
            stats[c][0] = mean
            stats[c][1] = std
        batch["originals_stats"] = stats
        batch["targets_stats"] = stats

    inputs = inputs.unsqueeze(dim=0)
    inputs = torch.cat([inputs] * batch_size, dim=0)

    for panel_part_id in panel_parts:
        # We'll simply include the same data multiple times to simulate having them all
        # in the batch
        batch[panel_part_id] = inputs

    # Execution of rendering
    render = SisrHareRenderer((num_channels, size, size), "sigmoid")
    outputs = render(batch, 1)

    # Tests section
    assert len(list(outputs.keys())) == batch_size  # As many images as batch items
    for k, v in outputs.items():
        assert isinstance(k, str)  # Are all the keys strings, as promised

        arr = np.array(v)
        # Pillow gives dimensions order of [H, W, C]
        # In torch code we normally follow [C, H, W]
        assert isinstance(v, PIL.Image.Image)  # Is it a pillow image?
        assert arr.shape[2] == 3  # Does the image contain 3 channels (RGB)?
        assert arr.shape[1] == size * len(panel_parts)  # Does each panel part occur once?
        assert arr.dtype == np.uint8 or arr.dtype == np.int8  # Does we have color data?
        assert (
            # Check if IR and IRG panels present if 4 channel data using the height
            # The height is the normal size plus 50 pixels for the titles
            (num_channels == 3 and arr.shape[0] == size + 50)
            or (num_channels == 4 and arr.shape[0] == 3 * size + 50)
        )
