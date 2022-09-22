import pytest

import torch

import sobolt.gorgan.nn.functional as F
from sobolt.gorgan.nn.functional.satellite_parameters import SatParamsSentinel2
from sobolt.gorgan.nn import PhysicalDownsampler

@pytest.mark.parametrize(
    "image",
    [
        torch.rand(1, 4, 128, 128, requires_grad=True),  # Square image
        torch.rand(1, 4, 128, 256, requires_grad=True),  # Rectangular image
    ],
)
@pytest.mark.parametrize("factor", [2, 4])
def test_physical_downsampling(image: torch.Tensor, factor: int):
    downsampler = PhysicalDownsampler(factor)
    downsampled_image = downsampler(image)["generated"]

    assert image.shape[1] == downsampled_image.shape[1]
    assert image.shape[2] == downsampled_image.shape[2] * factor
    assert image.shape[3] == downsampled_image.shape[3] * factor
    assert isinstance(downsampled_image, torch.Tensor)

@pytest.mark.parametrize(
    "image",
    [
        torch.rand(128, 128, 4, requires_grad=True),  # Square image
        torch.rand(128, 256, 4, requires_grad=True),  # Rectangular image
    ],
)
def test_physical_modulation_transfer(image: torch.Tensor):
    hr = SatParamsSentinel2() * 4
    lr = SatParamsSentinel2()
    image.retain_grad()
    modulated_image = F.physical_modulation_transfer(image, lr, hr)

    assert image.shape == modulated_image.shape
    assert isinstance(modulated_image, torch.Tensor)
    assert image.grad == modulated_image.grad
