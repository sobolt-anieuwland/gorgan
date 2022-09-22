from math import nan
import pytest

import torch
from sobolt.gorgan.nn.functional import get_gradient_metrics, apply_sobel, dominant_frequency_percent
from sobolt.gorgan.nn.functional.satellite_parameters import SatParams

# Tests if all dict keys are present
def test_gradient_metrics_dict():
    tensor = torch.tensor(1.0)
    d = get_gradient_metrics(tensor, tensor)
    print(d)
    dict_mappings = [
        'val_magnitude_mean',
        'val_magnitude_std',
        'val_orientation_mean',
        'val_orientation_std',
        'val_orientation_kurtosis',
        'val_orientation_skewness'
        ]
    assert isinstance(d, dict)

    for mapping in dict_mappings:
        assert mapping in d.keys()

# Tests that the mean and std are calculated properly
def test_gradient_metrics_results():
    tensor = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    d = get_gradient_metrics(tensor, tensor)
    assert d['val_magnitude_mean'] == 0.5
    assert d['val_magnitude_std'] == 0
    assert d['val_orientation_mean'] == 0.5
    assert d['val_orientation_std'] == 0

# Tests that a nan value gets turned into a 0
def test_gradient_metrics_nan_check():
    tensor = torch.tensor([[nan, nan, nan]])
    d = get_gradient_metrics(tensor, tensor)
    assert d['val_magnitude_mean'] == 0
    assert d['val_magnitude_std'] == 0
    assert d['val_orientation_mean'] == 0
    assert d['val_orientation_std'] == 0

# Tests if the sobel filter is implemented correctly
def test_sobel_filter():
    t = torch.tensor(
        [[[[0.5,  0,  0],
          [0, 0.5, 0],
          [0, 0, 0.5]]]])
    sobel = apply_sobel(t)
    result_x = torch.tensor(
        [[[[-0.5, 1.0, 0.5],
        [-1.0, 0.0, 1.0],
        [-0.5, -1.0, 0.5]]]]
    )
    result_y = torch.tensor(
        [[[[-0.5, -1.0, -0.5],
        [1.0, 0.0, -1.0],
        [0.5, 1.0, 0.5]]]]
    )

    assert torch.all(torch.eq(sobel[0], result_x))
    assert torch.all(torch.eq(sobel[1], result_y))

# Tests if the outcome type is correct
def test_sobel_filter_type():
    t = torch.tensor(
        [[[[0.5,  0,  0],
          [0, 0.5, 0],
          [0, 0, 0.5]]]])
    sobel = apply_sobel(t)
    assert isinstance(sobel, tuple)
    assert torch.is_tensor(sobel[0])
    assert torch.is_tensor(sobel[1])

# Tests wrong input with an error
def test_sobel_filter_incorrect_tensor():
    t = torch.tensor([[1,2]])
    with pytest.raises(RuntimeError):
        apply_sobel(t)

# Tests output type
def test_dominant_frequence_percentage_output_type():
    tensor = torch.tensor(
        [[[[0.5,  0,  0],
          [0, 0.5, 0],
          [0, 0, 0.5]]]])
    tensor = torch.rand((1,1,1,4,4))
    percentage = dominant_frequency_percent(tensor)
    assert isinstance(percentage, float)


# Dummy class to test SatParam
class SatParamDummy(SatParams):

    def __init__(self):
        super().__init__(
            1, 2, 3, 4, 5, 6, 7, 8, 9
        )

# Tests initialisation of parameters
def test_SatParam_parameters():
    c = SatParamDummy()
    assert c.altitude == 1
    assert c.aperture_x == 2
    assert c.aperture_y == 3
    assert c.focal_distance == 4
    assert c.resolution == 5
    assert c.wavelength_blue == 6
    assert c.wavelength_green == 7
    assert c.wavelength_red == 8
    assert c.wavelength_infrared == 9

# Tests multiplication of the parameters
def test_SatParam_parameters_mul():
    c = SatParamDummy()
    new_param = c * 2
    assert new_param.altitude == 1
    assert new_param.aperture_x == 1.0
    assert new_param.aperture_y == 1.5
    assert new_param.focal_distance == 4
    assert new_param.resolution == 2.5
    assert new_param.wavelength_blue == 6
    assert new_param.wavelength_green == 7
    assert new_param.wavelength_red == 8
    assert new_param.wavelength_infrared == 9

# Tests divide of the parameters
def test_SatParam_parameters_div():
    c = SatParamDummy()
    factor = 2
    new_param = c / 2
    assert new_param.altitude == 1
    assert new_param.aperture_x == 4
    assert new_param.aperture_y == 6
    assert new_param.focal_distance == 4
    assert new_param.resolution == 10
    assert new_param.wavelength_blue == 6
    assert new_param.wavelength_green == 7
    assert new_param.wavelength_red == 8
    assert new_param.wavelength_infrared == 9
