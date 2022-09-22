import torch
import torch.nn as nn
from .functional import fft_shift_2d


class FrequencyExtractor(nn.Module):
    """
    This class allows us to convert a signal into its frequencies. Since the input is
    at least 2D, we use fftshift to center the high frequency at the origin (0,
    0). By applying the 20 * natural log on the absolute value of the shifted
    frequencies, we get the magnitude spectrum of our frequencies, which can then be
    visualized in tensorboard's histogram.
    """

    def __init__(self):
        """
        Initializes the FrequencyExtractor class.
        """
        super(FrequencyExtractor, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Processes a tensor to convert it to the frequency domain to be able to extract
        the dominant frequency from a signal (magnitude spectrum).

        Parameters
        ----------
        inputs: torch.Tensor
            A tensor that has at least 2 dimensions, where the FFT is performed over
            its last two dimensions (W x H).

        Returns
        -------
        torch.Tensor
            The magnitude spectrum for a given input.
        """
        # Grayscale the image to facilitate subsequent computations.
        r, g, b = inputs.unbind(dim=-3)
        gray_inputs = (0.2989 * r + 0.587 * g + 0.114 * b).to(inputs.dtype)
        gray_inputs = gray_inputs.unsqueeze(dim=-3)

        # Perform FFT
        frequencies = torch.rfft(gray_inputs, signal_ndim=2, onesided=False)

        # Shift frequencies
        shifted_frequencies = fft_shift_2d(torch.abs(frequencies))

        # Magnitude spectrum
        return 20 * torch.log(shifted_frequencies)
