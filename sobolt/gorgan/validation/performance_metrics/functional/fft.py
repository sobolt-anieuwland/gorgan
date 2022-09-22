import torch


def median_frequency(shifted_fft: torch.Tensor) -> float:
    """Get median frequency from a shifted FFT applied to a tensor.

    Parameters
    ----------
    shifted_fft: torch.Tensor
        A tensor of frequencies from a FFTshift.

    Returns
    -------
    float
        The median frequency.
    """
    # Get frequency absolute value
    absolute_shifted_fft = torch.abs(shifted_fft)
    absolute_shifted_fft[absolute_shifted_fft == float("inf")] = 0

    # Get values necessary to compute threshold
    return torch.median(absolute_shifted_fft[:, :, :, :, 0]).item()
