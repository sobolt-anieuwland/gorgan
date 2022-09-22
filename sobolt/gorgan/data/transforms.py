from typing import Any, Dict, Callable, TypeVar, Tuple
from nptyping import Array

import torch
import numpy as np
from scipy.stats import norm, beta
from skimage.transform import resize
import random

_T = TypeVar("_T")


def to_tensor(sample: Array[Any], seed: int, config: Dict[str, Any]):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    # image = image.transpose((2, 0, 1))
    # val / 125.5 - 1
    # val / 2048 - 1
    # print(sample / 2048 - 1)
    return torch.from_numpy(sample)


def to_int32(sample: Array[Any], seed: int, config: Dict[str, Any]) -> Array[np.int32]:
    return to_dtype_np(np.int32, sample, seed, config)


def to_float32(
    sample: Array[Any], seed: int, config: Dict[str, Any]
) -> Array[np.float32]:
    return to_dtype_np(np.float32, sample, seed, config)


def to_float64(
    sample: Array[Any], seed: int, config: Dict[str, Any]
) -> Array[np.float64]:
    return to_dtype_np(np.float64, sample, seed, config)


def to_dtype_np(
    dtype: _T, sample: Array[Any], seed: int, config: Dict[str, Any]
) -> Array[_T]:
    return sample.astype(dtype)


def normalize_sentinel2_l1c(
    sample: Array[np.uint16, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.float32, ..., ..., ...]:
    # By looping over thousands of Sentinel 2 L1C images, I empirically
    # concluded they have a maximum value of 28000
    # Normalize to be between -1 and 1
    # return sample.astype(np.float32) / 14000 - 1
    sample = np.clip(sample.astype(np.float32), 0, 2000)
    return sample / 1000 - 1


def prepare_superview_esrgan(sample, seed: int, config: Dict[str, Any]):
    sample = sample[:3][[2, 1, 0]].astype(np.float32)
    return sample / np.max(sample)


def normalize_sentinel2_l2a(
    sample: Array[np.uint16, ..., ..., ...], seed: int, config: Dict[str, Any]
) -> Array[np.float32, ..., ..., ...]:
    """Maps Sentinel L2A colors to be between -1 and 1.

    Our Sentinel L2A tiles have colors in the range of 0 to 2000. To rescale,
    we first clip all values to be between 0 and 2000, then divide by 1000
    and finally substract 1.

    Does not work correctly for the infrared band, because those raw values
    are not between 0 and 2000, but somewhere around/between 0 and 3000. No
    normalization has been defined yet for that band.

    ESA: "The saturation level of 255 digital counts correspond to a level of 3558 for
        L1C products or 2000 for L2A products (0.3558 and 0.2 in reflectance value
        respectively)."

    Parameters
    ----------
    sample: Array[np.uint16, ..., ..., ...]
        An array with arbitrary dimensions for sized width, height, channels,
        but values of uint16. Values are assumed to be colors between 0 and
        2000.
    seed: int
        Value to set as random seed in order to apply same random transformation to input
        and target images.
    config: Dict[str, Any]
        A dictionary containing a training sessions specifications

    Returns
    -------
    Array[np.float32, ..., ..., ...]:
        A float array of arbitrary width, height, channels with all values
        between -1 and 1.
    """
    return np.clip(sample, 0, 2000).astype(np.float32) / 1000 - 1


def normalize_sentinel2_l2a_cran(
    sample: Array[np.uint16, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.float32, ..., ..., ...]:
    # According to this manual of S2 toolbox, to convert to rgbi we need
    # division by 2500 on RGB and 7500 for I.
    # https://cran.r-project.org/web/packages/sen2r/sen2r.pdf
    sample2 = sample.astype(np.float32) / (2500 / 2) - 1
    if sample2.shape[0] == 4:
        sample2[3] = sample[3].astype(np.float32) / (7500 / 2) - 1
    return sample2


def normalize_sentinel2_in1_zscore(
    sample: Array[np.uint16, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.float32, ..., ..., ...]:
    """Normalize sentinel 2 tiles with parameters based gather from the in1 dataset"""
    mean = [1313.7287, 1490.5601, 1657.8621, 2803.1157]
    std = [362.3934, 377.4156, 435.5199, 580.3479]
    sample2 = sample.astype(np.float32)
    for c in range(sample.shape[0]):
        sample2[c] = (sample2[c] - mean[c]) / std[c]
    return sample2


def normalize_sentinel2_in1_max(
    sample: Array[np.uint16, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.float32, ..., ..., ...]:
    """Normalize sentinel 2 tiles with parameters based gather from the in1 dataset"""
    max_ = [2818.2836, 3101.7645, 3471.5620, 5268.9062]
    sample2 = sample.astype(np.float32)
    for c in range(sample.shape[0]):
        sample2[c] = np.clip(sample2[c], 0, max_[c]) / max_[c]
    return sample2


def denormalize_sentinel2_l2a_cran(
    sample: Array[np.float32, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.uint16, ..., ..., ...]:
    sample2 = sample * 2500
    if sample2.shape[0] == 4:
        sample2[3] *= 3

    uint16_min = 0
    uint16_max = 65535
    return np.clip(sample2, uint16_min, uint16_max).astype(np.uint16)


def denormalize_sentinel2_in1_zscore(
    sample: Array[np.float32, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.uint16, ..., ..., ...]:
    # According to this manual of S2 toolbox, to convert to rgbi we need
    # division by 2500 on RGB and 7500 for I.
    # https://cran.r-project.org/web/packages/sen2r/sen2r.pdf
    mean = [1313.7287, 1490.5601, 1657.8621, 2803.1157]
    std = [362.3934, 377.4156, 435.5199, 580.3479]
    for c in range(sample.shape[0]):
        sample[c] = sample[c] * std[c] + mean[c]
    return sample.astype(np.uint16)


def denormalize_sentinel2_in1_max(
    sample: Array[np.float32, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.uint16, ..., ..., ...]:
    """Normalize sentinel 2 tiles with parameters based gather from the in1 dataset"""
    max_ = [2818.2836, 3101.7645, 3471.5620, 5268.9062]
    for c in range(sample.shape[0]):
        sample[c] *= max_[c]
    return sample.astype(np.uint16)


def normalize_8bit(sample, seed: int, config: Dict[str, Any]):
    return sample.astype(np.float32) / 127.5 - 1.0


def denormalize_8bit(sample, seed: int = 0, config: Dict[str, Any] = {}):
    return np.clip(sample * 255, 0, 255).astype(np.uint8)


def normalize_superview(
    sample: Array[np.uint16, ..., ..., ...],
    seed: int,
    config: Dict[str, Any],
    max_val=500.0,
) -> Array[np.float32, ..., ..., ...]:
    # While the data type is uint16, the SV documentation say it is 11bit, hence normalize
    # with 2048.
    #
    # Over 94999 / 186950 maxes
    # Minimal max: 45 - Median max: 879.0 - Mean max: 1002.1941599385257 - Maximal max: 2761
    # Histogram: (
    #   First row: counts, second row: bin edges
    #   array([ 1421, 15011, 20506, 19323, 13736,  8359,  5062,  3587,  4781, 3078,   134]),
    #   array([   0,  250,  500,  750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750]))
    # )
    # Percentages:
    # array([ 1.5, 15.8, 21.6, 20.3, 14.5, 8.8,  5.3,  3.8,  5.0,  3.2, 0.14])
    return sample.astype(np.float32) / max_val - 1.0


def normalize_superview_in1_zscore(
    sample: Array[np.uint16, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.float32, ..., ..., ...]:
    # According to this manual of S2 toolbox, to convert to rgbi we need
    # division by 2500 on RGB and 7500 for I.
    # https://cran.r-project.org/web/packages/sen2r/sen2r.pdf
    mean = [251.1667, 327.5683, 358.4528, 646.5957]
    std = [64.6789, 63.2482, 55.3749, 194.7362]
    sample2 = sample.astype(np.float32)
    for c in range(sample.shape[0]):
        sample2[c] = (sample2[c] - mean[c]) / std[c]
    return sample2


def normalize_superview_in1_max(
    sample: Array[np.uint16, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.float32, ..., ..., ...]:
    """Normalize sentinel 2 tiles with parameters based gather from the in1 dataset"""
    max_ = [937.3284, 999.4011, 940.6188, 646.5957]
    sample2 = sample.astype(np.float32)
    for c in range(sample.shape[0]):
        sample2[c] = np.clip(sample2[c], 0, max_[c]) / max_[c]
    return sample2


def denormalize_superview(
    sample: Array[np.float32, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.uint16, ..., ..., ...]:
    uint16_min = 0
    uint16_max = 65535
    return np.clip((sample * 1000), uint16_min, uint16_max).astype(np.uint16)


def denormalize_superview_in1_zscore(
    sample: Array[np.float32, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.uint16, ..., ..., ...]:
    # According to this manual of S2 toolbox, to convert to rgbi we need
    # division by 2500 on RGB and 7500 for I.
    # https://cran.r-project.org/web/packages/sen2r/sen2r.pdf
    mean = [251.1667, 327.5683, 358.4528, 646.5957]
    std = [64.6789, 63.2482, 55.3749, 194.7362]
    for c in range(sample.shape[0]):
        sample[c] = sample[c] * std[c] + mean[c]
    return sample.astype(np.uint16)


def denormalize_superview_in1_max(
    sample: Array[np.float32, ..., ..., 3], seed: int, config: Dict[str, Any]
) -> Array[np.uint16, ..., ..., ...]:
    """Normalize sentinel 2 tiles with parameters based gather from the in1 dataset"""
    max_ = [937.3284, 999.4011, 940.6188, 646.5957]
    for c in range(sample.shape[0]):
        sample[c] *= max_[c]
    return sample.astype(np.uint16)


def normalize_max_clipped(sample, seed: int, config: Dict[str, Any]):
    # return sample.astype(np.float32) / (np.max(sample) / 2) - 1.0
    return sample.astype(np.float32) / np.max(sample)


def interval_tanh_to_sigmoid(sample, seed: int, config: Dict[str, Any]):
    """ Translate values of [-1, 1] to [0, 1] by adding 1 and dividing by 2 """
    return (sample + 1) / 2


def interval_sigmoid_to_tanh(sample, seed: int, config: Dict[str, Any]):
    """ Translate values of [-1, 1] to [0, 1] by adding 1 and dividing by 2 """
    return sample * 2 - 1


def bgr_to_rgb(sample, seed: int, config: Dict[str, Any]):
    """Reorders the first three bands in the image such that, assuming the
    input is BGR, the output is RGB.

    Parameters
    ----------
    sample: Array[..., ..., ...]
        An input array with the following properties: (1) the first axis
        contains the channels; (2) there are at least 3 channels, more are
        allowed.
    seed: int
        Value to set as random seed in order to apply same random transformation to input
        and target images.
    config: Dict[str, Any]
        A dictionary containing a training sessions specifications.

    Returns
    -------
    Array[..., ..., ...]
        The same sample, but with indices 0 and 2 on axis 0 swapped.
    """
    if sample.shape[0] == 3:
        # If only three bands we can be more efficient by just returning a different view
        # E.g. no copying
        return sample[[2, 1, 0]]

    # TODO More efficient implementation with views and no copying
    sample[0], sample[2] = sample[2].copy(), sample[0].copy()
    return sample


def cwh_to_whc(sample, seed: int = 0, config: Dict[str, Any] = {}):
    return sample.transpose([2, 1, 0])


def colorize(
    sample: Array[np.float32, ..., ..., ...], seed: int, config: Dict[str, Any]
) -> Array[np.uint8, ..., ..., ...]:
    """Colorizes an array by converting it the values to be between 0 and 255.
    Assumed is that the array is normalized to be between -1 and 1. Returned
    array is unsigned int8.
    """
    return np.clip((sample + 1) * 127.5, 0, 255).astype(np.uint8)


def colorize_interval(
    sample: torch.Tensor, interval: Tuple[float, float]
) -> torch.Tensor:
    """Colorizes an array by converting it the values to be between 0 and 255.
    No assumption is made about in which interval the values lie. This needs to be
    explicitly passed along. Every value within this interval is mapped to [0, 255],
    everything outside of it is clipped. The returned array is unsigned int8.
    """
    min_val, max_val = interval
    multiplier = 255.0 / (max_val - min_val)
    return torch.clamp((sample - min_val) * multiplier, 0, 255).byte()


def augment_contrast(
    sample: Array[np.float32, ..., ..., ...],
    seed: int,
    config: Dict[str, Any],
    interval: Tuple[float, float] = (0.7, 1.3),
) -> Array[np.float32, ..., ..., ...]:
    """Apply random contrast change to a given input image.
    The input image is transformed channelwise & pixelwise according to the equation:
    O = I**gamma

    Parameters
    ----------
    sample: Array[float32, ..., ..., ...]
        Input image array
    seed: Int
        Value to set as random seed in order to apply same random transformation to input
        and target images.
    config: Dict[str, Any]
        A dictionary containing a training sessions specifications.
    interval: Tuple[float, float]
        Interval from which the correction value gamma is sampled.
        The interval endpoints must be positive.

    Returns
    -------
    Array[float, ..., ..., ...]
        Same input image with random contrast applied
    """
    random.seed(seed)
    gamma = random.uniform(interval[0], interval[1])
    return sample ** gamma


def augment_brightness(
    sample: Array[np.float32, ..., ..., ...],
    seed: int,
    config: Dict[str, Any],
    interval: Tuple[float, float] = (1.4, 1.6),
) -> Array[np.float32, ..., ..., ...]:
    """Apply random brightness change to a given input image.
    The input image is transformed channelwise & pixelwise according to the equation:
    O = I*gain

    Parameters
    ----------
    sample: Array[float32, ..., ..., ...]
        Input image array
    seed: Int
        Value to set as random seed in order to apply same random transformation to input
        and target images.
    config: Dict[str, Any]
        A dictionary containing a training sessions specifications.
    interval: Tuple[float, float]
        Interval from which the correction value gain is sampled.
        No restriction regarding the interval endpoints.

    Returns
    -------
    Array[float, ..., ..., ...]
        Same input image with randomly sampled brightness applied
    """
    random.seed(seed)
    gain = random.uniform(interval[0], interval[1])
    return sample * gain


def augment_offset(
    sample: Array[np.float32, ..., ..., ...],
    seed: int,
    config: Dict[str, Any],
    interval: Tuple[float, float] = (-0.16, -0.14),
) -> Array[np.float32, ..., ..., ...]:
    """Apply random offset to a given input image.
    The input image is transformed channelwise & pixelwise according to the equation:
    O = I + offset

    Parameters
    ----------
    sample: Array[float32, ..., ..., ...]
        Input image array
    seed: Int
        Value to set as random seed in order to apply same random transformation to input
        and target images.
    config: Dict[str, Any]
        A dictionary containing a training sessions specifications.
    interval: Tuple[float, float]
        Interval from which the correction offset value is sampled.
        No restriction regarding the interval endpoints.

    Returns
    -------
    Array[float, ..., ..., ...]
        Same input image with randomdomly sampled offset applied
    """
    random.seed(seed)
    offset = random.uniform(interval[0], interval[1])
    return sample + offset


def only_rgb(sample, seed: int, config: Dict[str, Any]):
    return sample[:3, :, :]


def only_rgbi(sample, seed: int, config: Dict[str, Any]):
    return sample[:4, :, :]


def crop_to_center_originals(sample, seed: int, config: Dict[str, Any]):
    _, h, w = sample.shape
    shape = config["shape_originals"]
    target_height, target_width = shape[-2], shape[-1]

    start_h = h // 2 - target_height // 2
    start_w = w // 2 - target_width // 2

    return sample[:, start_h : start_h + target_height, start_w : start_w + target_width]


def crop_to_center_targets(sample, seed: int, config: Dict[str, Any]):
    _, h, w = sample.shape

    shape = config["shape_targets"]
    target_height, target_width = shape[-2], shape[-1]

    start_h = h // 2 - target_height // 2
    start_w = w // 2 - target_width // 2
    return sample[:, start_h : start_h + target_height, start_w : start_w + target_width]


def scale_to_128(sample, seed: int, config: Dict[str, Any]):
    shape = sample.shape
    newshape = (shape[0], 128, 128)
    return resize(sample, newshape, anti_aliasing=True)


def scale_to_256(sample, seed: int, config: Dict[str, Any]):
    shape = sample.shape
    newshape = (shape[0], 256, 256)
    return resize(sample, newshape, anti_aliasing=True)


def scale_to_128_nominal(sample, seed: int, config: Dict[str, Any]):
    """Scales the input down to 128x128 and converts to the byte/uint8 datatype.
    Used for scaling class (nominal) data down. Don't rely on this, because it
    doesn't do this correctly. Implemented this way because we don't use our
    conditional / class masks.
    """
    shape = sample.shape
    newshape = (shape[0], 128, 128)
    return resize(sample, newshape, anti_aliasing=False).astype(np.uint8)


def scale_to_512(sample, seed: int, config: Dict[str, Any]):
    shape = sample.shape
    newshape = (shape[0], 512, 512)
    return resize(sample, newshape, anti_aliasing=True)


def scale_to_1024(sample, seed: int, config: Dict[str, Any]):
    shape = sample.shape
    newshape = (shape[0], 1024, 1024)
    return resize(sample, newshape, anti_aliasing=True)


def scale_to_16(sample, seed: int, config: Dict[str, Any]):
    shape = sample.shape
    newshape = (shape[0], 16, 16)
    return resize(sample, newshape, anti_aliasing=True)


def scale_to_64(sample, seed: int, config: Dict[str, Any]):
    shape = sample.shape
    newshape = (shape[0], 64, 64)
    return resize(sample, newshape, anti_aliasing=True)


def normalize_zscore(
    sample, seed: int = 0, config: Dict[str, Any] = {}, channelwise: bool = True
):
    """Converts sample to channel wise z score"""
    no_data = sample[0, :, :] == 0
    for i in range(sample.shape[0] - 1):
        no_data = no_data & (sample[i + 1, :, :] == 0)
    no_data_mask = np.repeat(np.expand_dims(no_data, 0), sample.shape[0], axis=0)

    normalized = sample.copy().astype(np.float32)
    if channelwise:
        stats = np.zeros((sample.shape[0], 2))
        for channel in range(sample.shape[0]):
            mean = np.mean(sample[channel][~no_data])
            std = np.std(sample[channel][~no_data])
            normalized[channel] = sample[channel] - mean
            normalized[channel] = normalized[channel] / std
            normalized[channel][no_data] = 0
            stats[channel] = np.stack((mean, std))
    else:
        stats = np.zeros((1, 2))
        mean = np.mean(sample[~no_data_mask])
        std = np.std(sample[~no_data_mask])
        normalized = (normalized - mean) / std
        normalized[no_data_mask] = 0
        stats = np.stack((mean, std))
    return (normalized, stats)


def denormalize_zscore(
    sample, seed: int = 1, config: Dict[str, Any] = {}, channelwise: bool = True
):
    no_data = sample[0, :, :] == 0
    for i in range(sample.shape[0] - 1):
        no_data = no_data & (sample[i + 1, :, :] == 0)
    no_data_mask = np.repeat(np.expand_dims(no_data, 0), sample.shape[0], axis=0)

    sample, stats = sample[0], sample[1]
    if channelwise:
        for channel in range(sample.shape[0]):
            mean = stats[channel][0]
            std = stats[channel][1]
            sample[channel] = (sample[channel] * std) + mean
            min_ = mean - (2 * std)
            max_ = mean + (2 * std)
            sample[channel] = np.clip(sample[channel], min_, max_)
            sample[channel] = (sample[channel] - min_) / (max_ - min_)
            sample[channel][no_data] = 0
    else:
        mean = stats[0]
        std = stats[1]
        sample = (sample * std) + mean
        sample[no_data_mask] = 0
    return sample


def normalize_beta(sample, seed: int = 1, config: Dict[str, Any] = {}):
    no_data = sample[0, :, :] == 0
    for i in range(sample.shape[0] - 1):
        no_data = no_data & (sample[i + 1, :, :] == 0)
    no_data_mask = np.repeat(np.expand_dims(no_data, 0), sample.shape[0], axis=0)

    sample2 = sample.copy().astype(np.float32)
    sample2, stats = normalize_zscore(sample2, channelwise=False)
    sample2 = norm.cdf(sample2)
    sample2[sample2 == 0.5] = 0  # masking no data in cdf space, norm.cdf(0) = 0.5
    sample2 = beta.ppf(sample2, 2, 2)
    sample2[no_data_mask] = 0
    return (sample2, stats)


def denormalize_beta(sample, seed: int = 1, config: Dict[str, Any] = {}):
    no_data = sample[0, :, :] == 0
    for i in range(sample.shape[0] - 1):
        no_data = no_data & (sample[i + 1, :, :] == 0)
    no_data_mask = np.repeat(np.expand_dims(no_data, 0), sample.shape[0], axis=0)

    sample, stats = sample[0], sample[1]
    sample = beta.cdf(sample, 2, 2)
    sample[no_data_mask] = 0.5  # putting back no data in cdf space, norm.ppf(0.5) = 0
    sample = norm.ppd(sample)
    sample = denormalize_zscore((sample, stats), channelwise=False)
    sample[no_data_mask] = 0
    return sample


def flip_updown(sample, seed: int = 1, config: Dict[str, Any] = {}):
    return np.flip(sample, axis=1)


def transform_factory(name: str) -> Callable[[Any, int, Dict[str, Any]], Any]:
    mapping: Dict[str, Callable[[Any, int, Dict[str, Any]], Any]] = {
        "to_int32": to_int32,
        "to_float32": to_float32,
        "to_float64": to_float64,
        "to_tensor": to_tensor,
        "bgr_to_rgb": bgr_to_rgb,
        "colorize": colorize,
        "scale_to_256": scale_to_256,
        # "colorize_interval": colorize_interval,
        "normalize_sentinel2_l2a": normalize_sentinel2_l2a,
        "normalize_sentinel2_l2a_cran": normalize_sentinel2_l2a_cran,
        "normalize_8bit": normalize_8bit,
        "normalize_superview": normalize_superview,
        "normalize_max_clipped": normalize_max_clipped,
        "only_bgr": only_rgb,
        "only_rgb": only_rgb,  # Correct mapping: operations are the same
        "only_rgbi": only_rgbi,
        "cwh_to_whc": cwh_to_whc,
        "whc_to_cwh": cwh_to_whc,  # Correct mapping: operations are the same
        "scale_to_16": scale_to_16,
        "scale_to_64": scale_to_64,
        "crop_to_center_originals": crop_to_center_originals,
        "crop_to_center_targets": crop_to_center_targets,
        "scale_to_128": scale_to_128,
        "scale_to_128_nominal": scale_to_128_nominal,
        "scale_to_512": scale_to_512,
        "prepare_superview_esrgan": prepare_superview_esrgan,
        "interval_tanh_to_sigmoid": interval_tanh_to_sigmoid,
        "interval_sigmoid_to_tanh": interval_sigmoid_to_tanh,
        "augment_brightness": augment_brightness,
        "augment_contrast": augment_contrast,
        "augment_offset": augment_offset,
        "normalize_sentinel2_in1_zscore": normalize_sentinel2_in1_zscore,
        "denormalize_sentinel2_in1_zscore": denormalize_sentinel2_in1_zscore,
        "normalize_superview_in1_zscore": normalize_superview_in1_zscore,
        "denormalize_superview_in1_zscore": denormalize_superview_in1_zscore,
        "normalize_sentinel2_in1_max": normalize_sentinel2_in1_max,
        "denormalize_sentinel2_in1_max": denormalize_sentinel2_in1_max,
        "normalize_superview_in1_max": normalize_sentinel2_in1_max,
        "denormalize_superview_in1_max": denormalize_superview_in1_max,
        "normalize_beta": normalize_beta,
        "denormalize_beta": denormalize_beta,
        "flip_updown": flip_updown,
    }
    return mapping[name]


def normalization_factory(name: str) -> Callable[[Any, int, Dict[str, Any]], Any]:
    mapping: Dict[str, Callable[[Any, int, Dict[str, Any]], Any]] = {
        "sentinel2_l2a": normalize_sentinel2_l2a,
        "sentinel2": normalize_sentinel2_l2a_cran,
        "8bit": normalize_8bit,
        "superview": normalize_superview,
        "max_clipped": normalize_max_clipped,
        "tanh_sigmoid": interval_tanh_to_sigmoid,
        "zscore": normalize_zscore,
        "sentinel2_in1_zscore": normalize_sentinel2_in1_zscore,
        "superview_in1_zscore": normalize_superview_in1_zscore,
        "sentinel2_in1_max": normalize_sentinel2_in1_max,
        "superview_in1_max": normalize_superview_in1_max,
        "beta": normalize_beta,
    }
    return mapping[name]


def denormalization_factory(name: str) -> Callable[[Any, int, Dict[str, Any]], Any]:
    mapping: Dict[str, Callable[[Any, int, Dict[str, Any]], Any]] = {
        "sentinel2": denormalize_sentinel2_l2a_cran,
        "8bit": denormalize_8bit,
        "superview": denormalize_superview,
        "zscore": denormalize_zscore,
        "sentinel2_in1_zscore": denormalize_sentinel2_in1_zscore,
        "superview_in1_zscore": denormalize_superview_in1_zscore,
        "sentinel2_in1_max": denormalize_sentinel2_in1_max,
        "superview_in1_max": denormalize_superview_in1_max,
        "beta": denormalize_beta,
    }
    return mapping[name]


# class RandomCrop(object):
#     """Crop randomly the image in a sample.

#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size

#     def __call__(self, sample):
#         image = sample['image']

#         h, w = image.shape[1::]
#         new_h, new_w = self.output_size
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)

#         image = image[:,top: top + new_h,
#                       left: left + new_w]

#         return {'image': image}
