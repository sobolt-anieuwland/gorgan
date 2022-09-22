import torch


def shift_values(inputs: torch.Tensor, dimension: int, n_shift: int) -> torch.Tensor:
    """
    Rolling values within a tensor along a specified axis. Torch version of the "roll()"
    numpy function.

    Parameters
    ----------
    inputs: torch.Tensor
        The fft tensor we want to shift.
    dimension: int
        Tensor dimension we want perform the shift on.
    n_shift: int
        The number of shifts we want to perform

    Returns
    -------
    torch.Tensor:
        A tensor "inputs" with values shifted "n_shift" times along "dimension".
    """
    f_idx = tuple(
        slice(None, None, None) if i != dimension else slice(0, n_shift, None)
        for i in range(inputs.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != dimension else slice(n_shift, None, None)
        for i in range(inputs.dim())
    )
    front = inputs[f_idx]
    back = inputs[b_idx]
    return torch.cat([back, front], dimension)


def fft_shift_2d(inputs: torch.Tensor) -> torch.Tensor:
    """
    Shift the zero-frequency component to the center of the frequency spectrum.

    Parameters
    ----------
    inputs: torch.Tensor
        A fast fourier transformation of a 2d signal.

    Returns
    -------
    torch.Tensor
        A shifted FFT tensor, which now allows us to view high frequency that make up
        the signal at the origin of the tensor (0,0).
    """
    real_values, imaginary_values = torch.unbind(inputs, -1)

    real_values_shifted, imaginary_values_shifted = torch.Tensor([0]), torch.Tensor([0])

    # Shift each dimension of the fft tensor using shift_values function
    for dim in range(1, len(real_values.size())):
        n_shift = real_values.size(dim) // 2
        if real_values.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real_values_shifted = shift_values(real_values, dimension=dim, n_shift=n_shift)
        imaginary_values_shifted = shift_values(
            imaginary_values, dimension=dim, n_shift=n_shift
        )
    return torch.stack((real_values_shifted, imaginary_values_shifted), -1)
