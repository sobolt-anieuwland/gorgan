import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .satellite_parameters import SatParams


def physical_downsampling(
    image: torch.Tensor, sat_params: SatParams, factor: float, resize: bool = True
) -> torch.Tensor:
    """This function downsamples a tensor after applying a modulation transfer function
    approximated for a speficic satellite camera sensors.

    Parameters
    ----------
    image: torch.Tensor
        An image of the shape [4, H, W] where the bands are RGBI order.
    sat_params: SatParams
        The set of satellite parameters of the input image.
    factor: float
        The factor with which the image must be downsampled. This number is also used to
        calculate the satellite parameters of downsampled image

    Returns
    -------
    torch.Tensor
        A pytorch tensor of the shape [4, H // factor, W // factor], where the bands
        are in RGBI order.
    """
    # Reorder channels last, PD expects that
    image = image.squeeze().permute(1, 2, 0)
    image = physical_modulation_transfer(image, sat_params / factor, sat_params)

    # Reorder channels first, interpolate requires batch and channels as first dimensions
    image = image.permute(2, 0, 1).unsqueeze(0)
    if resize:
        _, height, width, _ = image.shape
        downsampled_size = (round(height / factor), round(width / factor))
        image = F.interpolate(image, size=downsampled_size, mode="nearest")
    return image.squeeze(0)


def physical_modulation_transfer(x: torch.Tensor, sat_lr: SatParams, sat_hr: SatParams):
    """Explanation on how to use the RectangularAperture function for physically downsampling Sentinel-2 data.

    RectangularAperture -- input parameters:
        x = input tensor of shape (image_width, image_height, 3), where 3 are the RGB bands (type = float)
        Lx_ratio = ratio of the aperture size of the original sensor (Sentinel 2 = 150 mm) and target sensor in the x-direction
        Ly_ratio = ratio of the aperture size of the original sensor (Sentinel 2 = 150 mm) and target sensor in the y-direction
        f_ratio = ratio of the focal distance of the original sensor (Sentinel 2 = 600 mm) and target sensor

    The function is divided into 4 sections to guide your through:
    Part 1: Definition of the input parameters, Sentinel 2 instrument parameters and frequency limits of the Modulation Transfer Functions (MTF) and the input data
    Part 2: Calculation of the MTF for the original sensor and target sensor
    Part 3: Calculation of the ratio between the original sensor MTF and target sensor MTF per colour band
    part 4: Multiplication of the  MTF ratio and Fourier spectrum of the input data

    Caluclations are done for each band seperately with the same apporach. Variables are anotated with R, G, or B, refering to its spectral band.
    Variables refering to the original sensor, target sensor and the ratio between them (effective sensor), are anotated with 1, 2 and 3, respectively.

    Notes are added to the parameters and variables to inform on what's being done. For the sake of simplicity, steps that are repeated for each band are only added to the steps related to the R-band.

    WARNING: The height and width dimensions are probably handled incorrectly here. This
        is only a problem if you have non-square imagery or apertures. This code treats
        a tensor as being shaped [W, H, C], while typically tensors are [H, W, C].
    """
    # Part 1: Parameter defenition
    # Required parameters
    xsize = x.shape[0]  # x-dimension of the input data
    ysize = x.shape[1]  # y-dimension of the input data

    # Low resolution satellite parameters
    resolution = sat_lr.resolution  # resolution
    h = sat_lr.altitude  # altitude
    Lx = sat_lr.aperture_x  # aperture size in mm (x-coordinate)
    Ly = sat_lr.aperture_y  # aperture size in mm (y-coordinate)
    f = sat_lr.focal_distance  # focal distance
    transform = f / h  # transform object to image plane frequencies

    # High resolution satellite parameters
    L2x = sat_hr.aperture_x
    L2y = sat_hr.aperture_y

    # Numerical apertures
    Nx = f / Lx
    Ny = f / Ly
    N2x = f / L2x
    N2y = f / L2y

    # Central wavelenghts of the Sentinel 2 RGB bands
    lambdaR = sat_hr.wavelength_red  # band4 (red)
    lambdaG = sat_hr.wavelength_green  # band3 (green)
    lambdaB = sat_hr.wavelength_blue  # band2 (blue)
    lambdaI = sat_hr.wavelength_infrared  # band8 (infrared)

    # Frequency limits of MTF of optical systems (frequencies beyond this limit are not
    # eing modulated)
    vlimRx = 1 / (lambdaR * Nx)
    vlimGx = 1 / (lambdaG * Nx)
    vlimBx = 1 / (lambdaB * Nx)
    vlimIx = 1 / (lambdaI * Nx)
    vlimRy = 1 / (lambdaR * Ny)
    vlimGy = 1 / (lambdaG * Ny)
    vlimBy = 1 / (lambdaB * Ny)
    vlimIy = 1 / (lambdaI * Ny)

    vlimR2x = 1 / (lambdaR * N2x)
    vlimG2x = 1 / (lambdaG * N2x)
    vlimB2x = 1 / (lambdaB * N2x)
    vlimI2x = 1 / (lambdaI * N2x)
    vlimR2y = 1 / (lambdaR * N2y)
    vlimG2y = 1 / (lambdaG * N2y)
    vlimB2y = 1 / (lambdaB * N2y)
    vlimI2y = 1 / (lambdaI * N2y)

    # frequency limits of MTF of 10m resolution data
    maxfreq = math.pi / (resolution) / transform

    ######################################################################################
    ######################################################################################

    # Part 2: Calculation MTFs

    # Range of spatial frequencies (v) in the x- and y-directions (x or y) for which the
    # TFs will be calculated for the red (r), green (g) and blue (b) band
    vRx = torch.arange(-maxfreq, maxfreq, 2 * maxfreq / xsize, device=x.device)
    vRx = torch.abs(vRx)
    vRx = vRx.unsqueeze(1)
    vRy = torch.arange(-maxfreq, maxfreq, 2 * maxfreq / ysize, device=x.device)
    vRy = torch.abs(vRy)
    vRy = vRy.unsqueeze(0)

    vGx = torch.arange(-maxfreq, maxfreq, 2 * maxfreq / xsize, device=x.device)
    vGx = torch.abs(vGx)
    vGx = vGx.unsqueeze(1)
    vGy = torch.arange(-maxfreq, maxfreq, 2 * maxfreq / ysize, device=x.device)
    vGy = torch.abs(vGy)
    vGy = vGy.unsqueeze(0)

    vBx = torch.arange(-maxfreq, maxfreq, 2 * maxfreq / xsize, device=x.device)
    vBx = torch.abs(vBx)
    vBx = vBx.unsqueeze(1)
    vBy = torch.arange(-maxfreq, maxfreq, 2 * maxfreq / ysize, device=x.device)
    vBy = torch.abs(vBy)
    vBy = vBy.unsqueeze(0)

    if x.shape[2] == 4:
        vIx = torch.arange(-maxfreq, maxfreq, 2 * maxfreq / xsize, device=x.device)
        vIx = torch.abs(vIx)
        vIx = vIx.unsqueeze(1)
        vIy = torch.arange(-maxfreq, maxfreq, 2 * maxfreq / ysize, device=x.device)
        vIy = torch.abs(vIy)
        vIy = vIy.unsqueeze(0)

    # Calculations of the Modulation Transfer Function (MTF) of the original sensor and
    # target sensor (2) for the red (r), green (g) and blue (b) band
    MTFR = 1 - 1 / vlimRy * vRy - 1 / vlimRx * vRx + 1 / (vlimRx * vlimRy) * vRx * vRy
    MTFR2 = (
        1 - 1 / vlimR2y * vRy - 1 / vlimR2x * vRx + 1 / (vlimR2x * vlimR2y) * vRx * vRy
    )

    MTFG = 1 - 1 / vlimGy * vGy - 1 / vlimGx * vGx + 1 / (vlimGx * vlimGy) * vGx * vGy
    MTFG2 = (
        1 - 1 / vlimG2y * vGy - 1 / vlimG2x * vGx + 1 / (vlimG2x * vlimG2y) * vGx * vGy
    )

    MTFB = 1 - 1 / vlimBy * vBy - 1 / vlimBx * vBx + 1 / (vlimBx * vlimBy) * vBx * vBy
    MTFB2 = (
        1 - 1 / vlimB2y * vBy - 1 / vlimB2x * vBx + 1 / (vlimB2x * vlimB2y) * vBx * vBy
    )

    if x.shape[2] == 4:
        MTFI = 1 - 1 / vlimIy * vIy - 1 / vlimIx * vIx + 1 / (vlimIx * vlimIy) * vIx * vIy
        MTFI2 = (
            1
            - 1 / vlimI2y * vIy
            - 1 / vlimI2x * vIx
            + 1 / (vlimI2x * vlimI2y) * vIx * vIy
        )

    ######################################################################################
    ######################################################################################

    # Part 3: Calulating MTF ratios

    # Wiener filter value to overcome division by 0
    w = torch.tensor([0.0005], device=x.device)

    # Adjusting the  MTFs to adhere the criteria of no negative values and all values
    # beyond frequency to 0
    MTFR[(vRx >= vlimRx) | (vRy >= vlimRy)] = 0  # Zero values beyond the frequency limit
    MTFR[MTFR != MTFR] = 0  # Setting all NaN values to 0
    MTFR[MTFR < 0] = 0  # No negative values allowed for MTF values
    MTFR[MTFR == 0] = w  # Overcome division by 0

    MTFG[(vGx >= vlimGx) | (vGy >= vlimGy)] = 0
    MTFG[MTFG != MTFG] = 0
    MTFG[MTFG < 0] = 0
    MTFG[MTFG == 0] = w

    MTFB[(vBx >= vlimBx) | (vBy >= vlimBy)] = 0
    MTFB[MTFB != MTFB] = 0
    MTFB[MTFB < 0] = 0
    MTFB[MTFB == 0] = w

    if x.shape[2] == 4:
        MTFI[(vIx >= vlimIx) | (vIy >= vlimIy)] = 0
        MTFI[MTFI != MTFI] = 0
        MTFI[MTFI < 0] = 0
        MTFI[MTFI == 0] = w

    MTFR2[(vRx > vlimR2x) | (vRy > vlimR2y)] = 0  # Zero values beyond the frequency limit
    MTFR2[MTFR2 != MTFR2] = 0  # Setting all NaN values to 0
    MTFR2[MTFR2 < 0] = 0  # No negative values allowed for MTF values

    MTFG2[(vGx > vlimG2x) | (vGy > vlimG2y)] = 0
    MTFG2[MTFG2 != MTFG2] = 0
    MTFG2[MTFG2 < 0] = 0

    MTFB2[(vBx > vlimB2x) | (vBy > vlimB2y)] = 0
    MTFB2[MTFB2 != MTFB2] = 0
    MTFB2[MTFB2 < 0] = 0

    if x.shape[2] == 4:
        MTFI2[(vIx > vlimI2x) | (vIy > vlimI2y)] = 0
        MTFI2[MTFI2 != MTFI2] = 0
        MTFI2[MTFI2 < 0] = 0

    # Calculations of the inverse of the MTF of the original sensor and effective MTF by
    # multiplication with the MTF of the target sensor
    recMTFR = 1 / ((MTFR))  # Inverse of MTF value
    recMTFR = recMTFR / torch.max(recMTFR)  # Normalization
    MTFR3 = recMTFR * MTFR2  # Effective MTF, ratio between original and target aperture
    MTFR3 = MTFR3 / torch.max(MTFR3)  # Normalization
    MTFR3[(vRx > vlimR2x) | (vRy > vlimR2y)] = 0  # Zero values beyond the frequency limit
    MTFR3[MTFR2 != MTFR2] = 0  # Setting all NaN values to 0
    MTFR3[MTFR2 < 0] = 0  # No negative values allowed for MTF values

    recMTFG = 1 / ((MTFG))
    recMTFG = recMTFG / torch.max(recMTFG)
    MTFG3 = recMTFG * MTFG2
    MTFG3 = MTFG3 / torch.max(MTFG3)
    MTFG3[(vGx > vlimG2x) | (vGy > vlimG2y)] = 0
    MTFG3[MTFG2 != MTFG2] = 0
    MTFG3[MTFG2 < 0] = 0

    recMTFB = 1 / ((MTFB))
    recMTFB = recMTFB / torch.max(recMTFB)
    MTFB3 = recMTFB * MTFB2
    MTFB3 = MTFB3 / torch.max(MTFB3)
    MTFB3[(vBx > vlimB2x) | (vBy > vlimB2y)] = 0
    MTFB3[MTFB2 != MTFB2] = 0
    MTFB3[MTFB2 < 0] = 0

    if x.shape[2] == 4:
        recMTFI = 1 / ((MTFI))
        recMTFI = recMTFI / torch.max(recMTFI)
        MTFI3 = recMTFI * MTFI2
        MTFI3 = MTFI3 / torch.max(MTFI3)
        MTFI3[(vIx > vlimI2x) | (vIy > vlimI2y)] = 0
        MTFI3[MTFI2 != MTFI2] = 0
        MTFI3[MTFI2 < 0] = 0

    ######################################################################################
    ######################################################################################

    # Part 4: Multiplication of effective MTF with input Fourier spectrum

    # Add dimension for real and imaginary values, as the torch.rfft function gives 1
    # dimension for real values and 1 dimension for imaginary values as output
    # torch.fft results in an extra dimension for real and imaginary values
    MTF_R3 = torch.zeros(xsize, ysize, 2, device=x.device)
    MTF_R3[:, :, 0] = MTFR3  # MTF_real is set equal to MTF_imaginary
    MTF_R3[:, :, 1] = MTFR3  # MTF_real is set equal to MTF_imaginary
    MTF_G3 = torch.zeros(xsize, ysize, 2, device=x.device)
    MTF_G3[:, :, 0] = MTFG3
    MTF_G3[:, :, 1] = MTFG3
    MTF_B3 = torch.zeros(xsize, ysize, 2, device=x.device)
    MTF_B3[:, :, 0] = MTFB3
    MTF_B3[:, :, 1] = MTFB3

    if x.shape[2] == 4:
        MTF_I3 = torch.zeros(xsize, ysize, 2, device=x.device)
        MTF_I3[:, :, 0] = MTFI3
        MTF_I3[:, :, 1] = MTFI3

    # Get RGBIbands of input data for seperate multiplication with the MTF of the
    # corresponding color band
    img_a1R = x[:, :, 0]
    img_a1G = x[:, :, 1]
    img_a1B = x[:, :, 2]

    if x.shape[2] == 4:
        img_a1I = x[:, :, 3]

    # Calculating the fourier spectra of the input image, the downsampled image and the
    # target image
    # 1. Caluclating Fourier spectrum of input data
    # 2. Roll data for fftshift (not available in Pytorch package)
    # 3. Rotate data around centerpoint by 180 degrees for fftshift (not available)
    # 4. Calculate new Fourier spectrum
    # 5. Roll data back to original position
    # 6. Rotate data around centerpoint by 180 degrees to original position
    # 7. Inverse fft
    # fmt: off
    fimg_a1R = torch.rfft(img_a1R, 2, onesided=False)
    imshape = torch.tensor(([xsize / 2, ysize / 2]), device=x.device).int()
    fimg_a1R = torch.roll(fimg_a1R, shifts=(imshape[0], imshape[1]), dims=(0, 1))  # type: ignore
    fimg_a1R = torch.rot90(fimg_a1R, 2, [0, 1])
    fimg_a2Rshift = fimg_a1R * MTF_R3
    fimg_a2R = torch.roll(fimg_a2Rshift, shifts=(-imshape[0], -imshape[1]), dims=(0, 1))  # type: ignore
    fimg_a2R = torch.rot90(fimg_a2R, 2, [0, 1])
    img_a2R = torch.irfft(fimg_a2R, 2, onesided=False)

    fimg_a1G = torch.rfft(img_a1G, 2, onesided=False)
    fimg_a1G = torch.roll(fimg_a1G, shifts=(imshape[0], imshape[1]), dims=(0, 1))  # type: ignore
    fimg_a1G = torch.rot90(fimg_a1G, 2, [0, 1])
    fimg_a2Gshift = fimg_a1G * MTF_G3
    fimg_a2G = torch.roll(fimg_a2Gshift, shifts=(-imshape[0], -imshape[1]), dims=(0, 1))  # type: ignore
    fimg_a2G = torch.rot90(fimg_a2G, 2, [0, 1])
    img_a2G = torch.irfft(fimg_a2G, 2, onesided=False)

    fimg_a1B = torch.rfft(img_a1B, 2, onesided=False)
    fimg_a1B = torch.roll(fimg_a1B, shifts=(imshape[0], imshape[1]), dims=(0, 1))  # type: ignore
    fimg_a1B = torch.rot90(fimg_a1B, 2, [0, 1])
    fimg_a2Bshift = fimg_a1B * MTF_B3
    fimg_a2B = torch.roll(fimg_a2Bshift, shifts=(-imshape[0], imshape[1]), dims=(0, 1))  # type: ignore
    fimg_a2B = torch.rot90(fimg_a2B, 2, [0, 1])
    img_a2B = torch.irfft(fimg_a2B, 2, onesided=False)

    if x.shape[2] == 4:
        fimg_a1I = torch.rfft(img_a1I, 2, onesided=False)
        fimg_a1I = torch.roll(fimg_a1I, shifts=(imshape[0], imshape[1]), dims=(0, 1))  # type: ignore
        fimg_a1I = torch.rot90(fimg_a1I, 2, [0, 1])
        fimg_a2Ishift = fimg_a1I * MTF_I3
        fimg_a2I = torch.roll(fimg_a2Ishift, shifts=(-imshape[0], imshape[1]), dims=(0, 1))  # type: ignore
        fimg_a2I = torch.rot90(fimg_a2I, 2, [0, 1])
        img_a2I = torch.irfft(fimg_a2I, 2, onesided=False)
    # fmt: on

    # Stack bands to RGBI format
    if x.shape[2] == 4:
        img_a2 = torch.stack((img_a2R, img_a2G, img_a2B, img_a2I), dim=-1)
    else:
        img_a2 = torch.stack((img_a2R, img_a2G, img_a2B), dim=-1)
    return img_a2
