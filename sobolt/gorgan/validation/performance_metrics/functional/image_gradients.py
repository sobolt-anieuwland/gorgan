import numpy as np
import torch


def earth_mover_distance(originals: torch.Tensor, generated: torch.Tensor) -> float:
    """Compute the earth mover distance between two histograms (1D tensors)

    Parameters
    ----------
    originals: torch.Tensor
        A frequency distribution for a metric relating to original input tensors. (i.e.
        gradient orientation or magnitudes)
    generated: torch.Tensor
        A frequency distribution for a metric relating to generated input tensors. (i.e.
        gradient orientation or magnitudes)

    Returns
    -------
    float
        The absolute distance between the two histograms (generated vs. generated)
    """
    # Set variables
    histogram_len = originals.shape[0]
    distance = torch.zeros_like(originals)

    # Compute distance
    for value in range(histogram_len - 1):
        distance[value + 1] = originals[value] - generated[value] + distance[value]
    return torch.sum(torch.abs(distance)).item()


def make_histogram(inputs: torch.Tensor) -> torch.Tensor:
    """Compute a normalized frequency distribution over a tensor input.

    Parameters
    ----------
    inputs: torch.Tensor
        The input to compute the frequency distribution of values.

    Returns
    -------
    torch.Tensor
        The array containing the normalized frequency distribution (values sum to 1).
    """
    inputs = (inputs.cpu().numpy()).astype(float)
    inputs = inputs.reshape(-1)
    inputs = np.nan_to_num(inputs)
    counts, limits = np.histogram(inputs, bins=10)
    return torch.from_numpy(counts / counts.sum())
