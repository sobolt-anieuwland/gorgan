import os
from typing import List, Tuple, Dict, Any

import PIL.Image
from PIL import ImageFont, Image, ImageDraw
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import matplotlib.pyplot as plt

from .transforms import *


def tensor_to_picture(
    inputs: torch.Tensor, normalization: Tuple[float, float] = (0.0, 1.0)
) -> torch.Tensor:
    """Renders an input tensor into an image for visualization.

    Parameters
    ----------
    inputs: torch.Tensor
        The input tensor to be rendered. Expect tensor to have 3 bands (C x H x W).
    normalization: Tuple[float, float]
        The normalization range for colorizing tensors for validation.

    Returns
    -------
    torch.Tensor
        A rendered input tensor for visualization.
    """
    # Set tensor values between 0-255
    inputs = colorize_interval(inputs, normalization)
    return inputs


def tensor_to_heatmap(inputs: torch.Tensor) -> torch.Tensor:
    """Renders an attention tensor into a heatmap for visualization.

    Parameters
    ----------
    inputs: torch.Tensor
        The input tensor (C x H x W) to be rendered where C is 1.

    Returns
    -------
    torch.Tensor
        A tensor containing the attention map rendered to a RGB heatmap.
    """

    # Set tensor values between 0-255
    map = inputs.cpu().numpy()
    map = map - map.min()
    map = map / map.max()
    map = map.squeeze()
    # Apply the colormap like a function to any array:
    cm = plt.get_cmap("jet")
    map = cm(map) * 255
    img = map[:, :, :3]
    img = np.uint8(cwh_to_whc(img))
    img = torch.from_numpy(img)

    return img


def tensor_to_auxiliary(
    inputs: Dict[str, torch.Tensor], set_object, image: PIL.Image, img_nr: int
):
    """Adds to PIL image the corresponding class names for each rendered tensor in the
    image.

    Parameters
    ----------
    inputs: Dict[str, torch.Tensor]
    set_object: In1FlexDataset
        A dataset dictionary that can contain class labels for auxiliary.
    image: PIL.Image
        A stack of tensors processed into PIL image format.
    img_nr: int
        The index of a single tensor within a batch.
    """
    # Get class index for generated labels
    softmax_out = F.softmax(inputs["aux_preds"], 0)
    pred, idx = torch.max(softmax_out, 0)

    # Get class name from index value
    to_name = lambda i: set_object.translate_class_index_to_name(i)
    class_name_o = to_name(inputs["originals_auxiliary"].item())
    class_name_t = to_name(inputs["targets_auxiliary"].item())
    class_name_g = to_name(idx.item())

    # Add class names to the rendered stack of tensors
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = "Original: {0} vs. Generated: {1} vs. Target: {2}"
    text = text.format(class_name_o, class_name_g, class_name_t)
    w, h = draw.textsize(text, font=font)
    draw.text((10, 10), text, (255, 255, 255))


def pad_renders(imgs_combined: List[torch.Tensor]) -> List[torch.Tensor]:
    """Pad rendered tensors according to the max height and width in the list of
    rendered tensors.

    Parameters
    ----------
    imgs_combined: List[torch.Tensor]
        A list of rendered tensors of shape (C x H x W).

    Returns
    -------
    List[torch.Tensor]
        Padded tensors with 0s according to max height and width in the list pre-padding.
    """
    height = max(img.shape[-1] for img in imgs_combined)
    width = max(img.shape[-2] for img in imgs_combined)

    if width % 2 == 1 or height % 2 == 1:
        msg = "Expected even dimensions when rendering tensors, received odd {}"
        msg = msg.format((width, height))
        warnings.warn(msg)

    target_shape = [0, 0, height // 2, height // 2]
    return [
        F.pad(img, target_shape) if img.shape[-2:] != (width, height) else img
        for img in imgs_combined
    ]


def add_title(
    image: Image.Image, titles: List[Dict[str, str]], img_size: int
) -> Image.Image:
    """Add top padding and text to describe rendered images. Titles must be provided in
    the right order.

    Parameters
    ----------
    image: Image
        PIL image object, required since title are added using the PIL drawing function
    titles: List[Dict[str, str]]
        Titles to add on top of the render images.
    img_size: int
        Image size in pixels.

    Returns
    -------
    titled_image: Image
        PIL Image object with titles added.
    """
    # Add padding of 20px on the top
    titled_image = Image.new(image.mode, (image.size[0], image.size[1] + 50), (0, 0, 0))
    titled_image.paste(image, (0, 50))
    draw = ImageDraw.Draw(titled_image)

    # Path to more visible font, when not available just uncomment line 141 for default
    fnt_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    fnt = (
        ImageFont.truetype(fnt_path, 35)
        if os.path.isfile(fnt_path)
        else ImageFont.load_default()
    )

    for n, rend_type in enumerate(titles):
        title = rend_type["type"]
        title_x_shift = 10 + (img_size * n)  # Center title horizontally
        draw.text((title_x_shift, 5), title, font=fnt)
    return titled_image
