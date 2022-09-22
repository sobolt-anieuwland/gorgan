from statistics import mean
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

from sobolt.gorgan.nn import (
    OrientationMagnitudeExtractor,
    FrequencyExtractor,
    PhysicalDownsampler,
)
from sobolt.gorgan.validation.performance_metrics.functional import (
    CosineSimilarity,
    median_frequency,
    earth_mover_distance,
    make_histogram,
    intersection_over_union,
    canny_edge_extractor,
)


def compute_cosine_similarity(
    originals: torch.Tensor, generated: torch.Tensor, extractor: CosineSimilarity
) -> float:
    """Computes the normalized dot product in the feature space to display the
    similarity between features.

    Parameters
    ----------
    originals: torch.Tensor
        The original tensor of shape CxHxW we want to generate.
    generated: torch.Tensor
        The generated input tensor of shape CxHxW.
    extractor: CosineSimilarity
        The cosine similarity class containing a pretrained model for feature extraction.

    Returns
    -------
    float
        A float between 0 & 1 indicating the similarity between two feature vectors
        extracted from a pretrained model (default Resnet18).
    """
    generated_features, original_features = extractor.extract_features(
        generated, originals
    )
    return F.cosine_similarity(
        generated_features[0].view(1, -1), original_features[0].view(1, -1)
    ).item()


def compute_radiometric_similarity(
    originals: torch.Tensor, generated: torch.Tensor
) -> Tuple[float, float]:
    """calculates the radiometric consistency between two data products with arbitrary
    number of bands.

    Parameters
    ----------
    originals: torch.Tensor
        The input tensor to generate of shape B x C x H x W
    generated: torch.Tensor
        The resulting tensor that was generated from original input of shape B x C x H x W

    Returns
    -------
    Tuple[float, float]
        The radiometric consistency using a naive downsampler (bilinear) & the maximum
        distance using the l1 loss between generated and original inputs.
    """
    _, _, h_o, w_o = originals.shape
    _, _, h_g, w_g = generated.shape
    factor = h_g // h_o

    # Downsample generated
    downsampler = PhysicalDownsampler(factor)
    generated_downsampled = downsampler(generated)["generated"]

    # Get metrics
    radiometric_consistency = F.mse_loss(generated_downsampled, originals)
    max_radiometric_distance = torch.max(
        F.l1_loss(generated_downsampled, originals, reduction="none")
    )
    return radiometric_consistency.item(), max_radiometric_distance.item()


def compute_saturation_accuracy(
    originals: torch.Tensor, generated: torch.Tensor
) -> float:
    """calculates intersection over union (IOU) of saturated pixels between two data
    products with arbitrary number of bands.

    Parameters
    ----------
    originals: torch.Tensor
        The input tensor to generate of shape B x C x H x W
    generated: torch.Tensor
        The resulting tensor that was generated from original input of shape B x C x H x W

    Returns
    -------
    float
        The IOU score of saturated pixels between originals and generated inputs.
    """
    out_size = generated.shape[-1]
    originals = F.interpolate(originals, size=out_size, mode="nearest")
    saturation_accuracy = intersection_over_union(originals, generated)
    return saturation_accuracy


def compute_gradient_stats(
    originals: torch.Tensor, generated: torch.Tensor, device: torch.device
) -> Tuple[float, float, float]:
    """calculates gradient magnitude between two data products with arbitrary number of
      bands.

     Parameters
     ----------
     originals: torch.Tensor
         The input tensor to generate of shape B x C x H x W
     generated: torch.Tensor
         The resulting tensor that was generated from original input of shape B x C x H x W
     device: torch.device
         The device to carry computation, cuda if available else CPU.

    Returns
     -------
     Tuple[float, float, float]
    """
    sobel_detector = OrientationMagnitudeExtractor(device)

    out_size = generated.shape[-1]
    originals = F.interpolate(originals, size=out_size, mode="bilinear")

    # Get input gradients magnitude & orientations
    grad_magn_o, grad_orient_o = sobel_detector(originals[:, :3, :, :])
    grad_magn_g, grad_orient_g = sobel_detector(generated[:, :3, :, :])

    # Get normalized frequency distribution
    grad_magn_g, grad_magn_o = (make_histogram(grad_magn_g), make_histogram(grad_magn_o))
    grad_orient_g, grad_orient_o = (
        make_histogram(grad_orient_g),
        make_histogram(grad_orient_o),
    )

    # Compute EMD for magnitude & orientations
    grad_magn_emd = earth_mover_distance(grad_magn_o, grad_magn_g)
    grad_orient_emd = earth_mover_distance(grad_orient_o, grad_orient_g)

    # Compute gradient magnitude increase in Generated from Originals
    mean_grad_mag_g = torch.mean(grad_magn_g)
    mean_grad_mag_o = torch.mean(grad_magn_o)
    grad_magn_increase = mean_grad_mag_g / mean_grad_mag_o

    return (grad_orient_emd, grad_magn_emd, grad_magn_increase.item())


def compute_median_frequency_increase(
    originals: torch.Tensor, generated: torch.Tensor
) -> float:
    """calculates gradient magnitude between two data products with arbitrary number of
    bands.

    Parameters
    ----------
    originals: torch.Tensor
    generated: torch.Tensor

    Returns
    -------
    float
        The difference in median fft frequency between generated and originals.
    """
    frequency_extractor = FrequencyExtractor()
    out_size = generated.shape[-1]
    originals = F.interpolate(originals, size=out_size, mode="bilinear")

    # Extract frequencies & compute 2d shift
    shifted_fft_g = frequency_extractor(generated[:, :3, :, :])
    shifted_fft_o = frequency_extractor(originals[:, :3, :, :])

    median_fft_g = median_frequency(shifted_fft_g)
    median_fft_o = median_frequency(shifted_fft_o)

    return (median_fft_g - median_fft_o) / out_size


def compute_canny_density_ratio(
    originals: torch.Tensor, generated: torch.Tensor
) -> float:
    """calculates ratio in canny density between two data products with arbitrary number
    of bands.

    Parameters
    ----------
    originals: torch.Tensor
        The input tensor to generate of shape B x C x H x W
    generated: torch.Tensor
        The resulting tensor that was generated from original input of shape B x C x H x W

    Returns
    -------
    float
        The ratio in number of canny edge pixels between two inputs.
    """
    canny_edges_o = canny_edge_extractor(originals[:, :3, :, :])
    canny_edges_g = canny_edge_extractor(generated[:, :3, :, :])

    total_o = np.sum(np.nonzero(canny_edges_o))
    total_g = np.sum(np.nonzero(canny_edges_g))

    if total_g >= 10 and total_o >= 10:
        return total_g / total_o
    else:
        return 1.0


def compute_total_variation_difference(
    originals: torch.Tensor, generated: torch.Tensor
) -> float:
    """Computes the difference in the amount of noise present in generated input
    versus original counterpart.

    Parameters
    ----------
    originals: torch.Tensor
        The input tensor to generate of shape B x C x H x W
    generated: torch.Tensor
        The resulting tensor that was generated from original input of shape B x C x H x W

    Returns
    -------
    float
        The difference between the sum of differences of nearby pixels of
        generated versus original input.
    """
    inputs: List = [originals, generated]

    # Get input characteristics
    batch: List = []
    channels: List = []
    height: List = []
    width: List = []

    for tensor in inputs:
        b, c, h, w = tensor.size()
        batch.append(b)
        channels.append(c)
        height.append(h)
        width.append(w)

    tva: List = []
    for (b, c, h, w, tensor) in zip(batch, channels, height, width, inputs):
        total_variation_height = torch.pow(
            tensor[:, :, 1:, :] - tensor[:, :, :-1, :], 2
        ).sum()
        total_variation_width = torch.pow(
            tensor[:, :, :, 1:] - tensor[:, :, :, :-1], 2
        ).sum()

        total_variation = (total_variation_height + total_variation_width) / (
            b * c * h * w
        )
        tva.append(total_variation)

    return (tva[0] - tva[1]).item()


def compute_fidelity_score(
    originals: torch.Tensor, generated: torch.Tensor, index: int, directory: str = ""
) -> Tuple[float, torch.Tensor]:
    """Calculates a fidelity score and a mask for visualization between two data products
    with arbitrary number of bands.

    Parameters
    ----------
    originals: torch.Tensor
        The input tensor to generate of shape B x C x H x W
    generated: torch.Tensor
        The resulting tensor that was generated from original input of shape B x C x H x W
    index: int
        The current index when iterating through a dataloader object.
    directory: int
        A directory to save images to.

    Returns
    -------
    Tuple[float, torch.Tensor]
        The average fidelity score (based on radiometric consistency using a naive
        downsampler (pixel average) across all bands & the corresponding fidelity input
        mask for visualization.
    """
    _, _, h_o, w_o = originals.shape
    _, _, h_g, w_g = generated.shape
    factor = h_g // h_o

    # Downsample generated
    downsampler = PhysicalDownsampler(factor)
    generated_downsampled = downsampler(generated)["generated"]

    # Get metrics
    fidelity = F.mse_loss(generated_downsampled, originals, reduction="none")
    fidelity = torch.sqrt(fidelity)

    # Get mask for visualization
    fidelity_mask = torch.mean(fidelity, dim=-3)
    fidelity_mask = F.interpolate(
        fidelity_mask.unsqueeze(0), scale_factor=factor, mode="nearest"
    )
    return torch.mean(fidelity).item(), fidelity_mask.squeeze(0).cpu()


def compute_enhancement_score(
    originals: torch.Tensor, generated: torch.Tensor, index: int, directory: str = ""
) -> Tuple[float, torch.Tensor]:
    """Calculates an enhancement score and an image mask for visualization between two
    data products with arbitrary number of bands.

    Parameters
    ----------
    originals: torch.Tensor
        The input tensor to generate of shape B x C x H x W
    generated: torch.Tensor
        The resulting tensor that was generated from original input of shape B x C x H x W
    index: int
        The current index when iterating through a dataloader object.
    directory: int
        A directory to save images to.

    Returns
    -------
    Tuple[float, torch.Tensor]
        The average enhancement score (based on radiometric consistency using a naive
        downsampler (pixel average) across all bands & the corresponding enhancement input
        mask for visualization.
    """
    _, _, h_o, w_o = originals.shape
    _, _, h_g, w_g = generated.shape
    factor = h_g // h_o

    # Upsample original
    originals_upsampled = F.interpolate(originals, scale_factor=factor, mode="nearest")

    # Get metrics
    enhancement = F.mse_loss(generated, originals_upsampled, reduction="none")
    enhancement = torch.sqrt(enhancement)

    # Get mask for visualization
    enhancement_mask = torch.mean(enhancement, dim=-3)
    return torch.mean(enhancement).item(), enhancement_mask.cpu()


def compute_resolution_score(resolution_metrics: Dict[str, Any]) -> float:
    """Compute an aggregate score to summarize all metrics relating to resolution.

    Parameters
    ----------
    resolution_metrics: Dict[str, Any]
        A dictionary containing the metric name and the corresponding values for which
        we want to aggregate into a resolution score.

    Returns
    -------
    float
        A normalized score indicating the quality resolution  of a generated input.
        Values range between 1-4 to indicate the resolution of a generated image,
        with 4 meaning perfect resolution.
    """
    gmd = resolution_metrics["grad_magnitude_emd"]
    gmd = [(value - min(gmd)) / ((max(gmd) + 0.000001) - min(gmd)) for value in gmd]

    mafi = resolution_metrics["median_absolute_frequency_increase"]
    mafi = [(value - min(mafi)) / ((max(mafi) + 0.000001) - min(mafi)) for value in mafi]

    cedi = resolution_metrics["canny_edge_density_increase"]
    cedi = [(value - min(cedi)) / ((max(cedi) + 0.000001) - min(cedi)) for value in cedi]

    resolution = np.concatenate([gmd, mafi, cedi])
    return mean(resolution)


def compute_similarity_score(similarity_metrics: Dict[str, Any]) -> float:
    """Compute an aggregate score to summarize all metrics relating to similarity.

    Parameters
    ----------
    similarity_metrics: Dict[str, Any]
        A dictionary containing the metric name and the corresponding values for which
        we want to aggregate into a similarity score.

    Returns
    -------
    float
        A normalized score indicating the similarity of a generated input to its
        original counterpart. Values range between 0-1, with a 1 indicating perfect
        similarity.
    """
    sat = similarity_metrics["saturation_accuracy"]
    cs = similarity_metrics["cosine_similarity"]

    goe = similarity_metrics["grad_orientation_emd"]
    goe = [1 - value for value in goe]
    goe = [value if value < 1 else 1 for value in goe]

    mrd = similarity_metrics["max_radiometric_distance"]
    mrd = [1 - (value / 3) for value in mrd]
    mrd = [value if value < 1 else 1 for value in mrd]

    rc = similarity_metrics["radiometric_consistency"]
    rc = [1 - (200 * value) for value in rc]

    similarity = np.concatenate([sat, cs, goe, mrd, rc])
    return mean(similarity)


def calc_variogram(orig, gen):
    """calculates ratio of critical distance of variagramof original and generated image"""
    # To be determined
    return 1


def calc_keypoint_increase(orig, gen):
    """calculates gradient magnitude between two data products with arbitrary number of bands"""
    # TODO: implement Otto/Ligia
    # 1 initiate SIFT keypoint detector
    # 2 calculate SIFT keypoints for original
    # 3 calculate for generated using same settings
    # if lowest number of keypoints is above threshold (i.e.10), return ratio. Else return 1

    keypoint_increase = 1.0
    return keypoint_increase
