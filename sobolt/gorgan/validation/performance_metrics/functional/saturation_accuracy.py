from typing import Tuple

import torch


def intersection_over_union(
    originals: torch.Tensor, generated: torch.Tensor, threshold=5
) -> float:
    """Compute the amount of overlap between two tensors.

    Parameters
    ----------
    originals: torch.Tensor
        The original tensor of shape BxCxHxW we want to generate.
    generated: torch.Tensor
        The generated input tensor of shape BxCxHxW.
    threshold: minimum number of pixels required to calculate the metric, else return 1
    Returns
    -------
    float
        The intersection over union score between two input tensors.
    """
    # Set tensor values to 0-255
    originals = colorize_interval(originals)
    generated = colorize_interval(generated)

    # Set saturated values to true
    originals_filtered = torch.ge(originals, 255)
    generated_filtered = torch.ge(generated, 255)

    # Get metrics
    intersection = originals_filtered * generated_filtered
    if torch.sum(intersection) < threshold:
        return 1.0
    else:
        union = originals_filtered + generated_filtered
        return (torch.sum(intersection) / torch.sum(union).float()).item()


def colorize_interval(
    inputs: torch.Tensor, interval: Tuple[float, float] = (0.0, 1.0)
) -> torch.Tensor:
    """Colorizes an array by converting it the values to be between 0 and 255.
    No assumption is made about in which interval the values lie. This needs to be
    explicitly passed along. Every value within this interval is mapped to [0, 255],
    everything outside of it is clipped. The returned array is unsigned int8.

    Parameters
    ----------
    inputs: torch.Tensor
        The tensor to colorize given an interval.
    interval: Tuple[float, float]
        The interval for normalization.
    Returns
    -------
    torch.Tensor
        A transformed tensor with values in the range 0-255.
    """
    min_val, max_val = interval
    multiplier = 255.0 / (max_val - min_val)
    return torch.clamp((inputs - min_val) * multiplier, 0, 255).byte()
