from typing import Dict

import torch


def get_gradient_metrics(
    orientation: torch.Tensor, magnitude: torch.Tensor
) -> Dict[str, float]:
    """Compute summary statistics for an image's gradient magnitude and orientation.
    These distribution are summarized using statistical measures of dispersion (standard
    deviation - STD) and central tendency (mean). For orientation, we want to know the
    extent the distribution is uniform. We achieve this with skewness (measure of
    distribution asymmetry with 0 referring to normal) and kurtosis (measure of whether
    the distribution is heavy or light tailed, with 0 referring to normal).

    Parameters
    ----------
    orientation: torch.Tensor
        An images converted into its gradient orientations (0-360 angles)
    magnitude
        The magnitude of the gradients

    Returns
    -------
    Dict[str, float]
        A dictionary consisting of all the statistics for gradient orientation and
        magnitude computed in the function.
    """
    # Set NAN to 0 to prevent nan results
    magnitude[magnitude != magnitude] = 0
    orientation[orientation != orientation] = 0

    # Get gradient magnitude summary statistics
    gradient_magnitude_mean = torch.mean(magnitude)
    gradient_magnitude_std = torch.std(magnitude)

    # Get gradient orientation summary statistics
    gradient_orientation_mean = torch.mean(orientation)
    gradient_orientation_std = torch.std(orientation)
    gradient_orientation_kurtosis = (
        torch.mean(
            ((orientation - torch.mean(orientation)) / torch.std(orientation)) ** 4
        )
        - 3
    )
    gradient_orientation_skewness = torch.mean(
        ((orientation - torch.mean(orientation)) / torch.std(orientation)) ** 3
    )
    return {
        "val_magnitude_mean": gradient_magnitude_mean.item(),
        "val_magnitude_std": gradient_magnitude_std.item(),
        "val_orientation_mean": gradient_orientation_mean.item(),
        "val_orientation_std": gradient_orientation_std.item(),
        "val_orientation_kurtosis": gradient_orientation_kurtosis.item(),
        "val_orientation_skewness": gradient_orientation_skewness.item(),
    }
