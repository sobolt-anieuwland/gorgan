import torch


def dominant_frequency_percent(shifted_fft: torch.Tensor) -> float:
    """
    Compute the percent of high frequencies that fall above a threshold (set to
    half of maximum).

    Parameters
    ----------
    shifted_fft: torch.Tensor
        A tensor of frequencies from a FFTshift.

    Returns
    -------
    float
        The percent of frequencies that are above a certain threshold. This is an
        estimation of the level of detail within the input we performed the FFT.
    """
    # Frequencies are mirrored so take the absolute value
    absolute_shifted_fft = torch.abs(shifted_fft)
    absolute_shifted_fft[absolute_shifted_fft == float("inf")] = 0

    # Get values necessary to compute threshold
    max_frequency = torch.max(absolute_shifted_fft[:, :, :, :, 0])
    threshold = max_frequency / 1000

    # Select all values above threshold
    thresholded_frequencies = torch.masked_select(
        shifted_fft[:, :, :, :, 0], shifted_fft[:, :, :, :, 0].ge(threshold)
    )

    return thresholded_frequencies.shape[0] / (shifted_fft.shape[-2]) ** 2
