import pytest
import numpy as np
from nptyping import Array
from typing import Any

from sobolt.gorgan.data.transforms import (
    normalize_sentinel2_l2a_cran,
    denormalize_sentinel2_l2a_cran,
    normalize_superview,
    denormalize_superview,
    interval_tanh_to_sigmoid,
    normalize_zscore,
    denormalize_zscore,
    normalize_beta,
    denormalize_beta,
    normalize_superview_in1_max,
    normalize_superview_in1_zscore,
)


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 2500, 5000, 7500], dtype=np.uint16),
    ],
)
@pytest.mark.parametrize("target", [np.array([-1, 1, 3, 1], dtype=np.float32)])
# Test normalization and denormalization on dummy input.
def test_normalize_sentinel2_l2a_cran(sample: Array[Any], target):
    sample_normalized = normalize_sentinel2_l2a_cran(sample, seed=0, config={})
    assert sample_normalized[0] == target[0]
    assert sample_normalized[1] == target[1]
    assert sample_normalized[2] == target[2]
    assert sample_normalized[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 1, 3, 1], dtype=np.uint16),
    ],
)
@pytest.mark.parametrize("target", [np.array([0, 2500, 7500, 7500], dtype=np.float32)])
# Test normalization and denormalization on dummy input.
def test_denormalize_sentinel2_l2a_cran(sample: Array[Any], target):
    sample_denormalized = denormalize_sentinel2_l2a_cran(sample, seed=0, config={})
    assert sample_denormalized[0] == target[0]
    assert sample_denormalized[1] == target[1]
    assert sample_denormalized[2] == target[2]
    assert sample_denormalized[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 1, 3, -1], dtype=np.float32),
    ],
)
@pytest.mark.parametrize("target", [np.array([0.5, 1, 2, 0], dtype=np.float32)])
# Test normalization and denormalization on dummy input.
def test_interval_tanh_to_sigmoid(sample: Array[Any], target):
    sample_tanh_to_sigmoid = interval_tanh_to_sigmoid(sample, seed=0, config={})
    assert sample_tanh_to_sigmoid[0] == target[0]
    assert sample_tanh_to_sigmoid[1] == target[1]
    assert sample_tanh_to_sigmoid[2] == target[2]
    assert sample_tanh_to_sigmoid[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 2500, 5000, 7500], dtype=np.uint16),
    ],
)
def test_processing_pipeline_s2_cran(sample: Array[Any]):
    normalized_tanh = normalize_sentinel2_l2a_cran(sample, seed=0, config={})
    assert (normalized_tanh.dtype == np.float32) or (normalized_tanh.dtype == np.float64)
    normalized_sigmoid = interval_tanh_to_sigmoid(normalized_tanh, seed=0, config={})
    denormalized = denormalize_sentinel2_l2a_cran(normalized_sigmoid, seed=0, config={})
    assert denormalized.dtype == np.uint16
    assert (sample == denormalized).all()


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 500, 1000, 2000], dtype=np.uint16),
    ],
)
@pytest.mark.parametrize("target", [np.array([-1, 0, 1, 3], dtype=np.float32)])
# Test normalization and denormalization on dummy input.
def test_normalize_superview(sample: Array[Any], target):
    sample_normalized = normalize_superview(sample, seed=0, config={})
    assert sample_normalized[0] == target[0]
    assert sample_normalized[1] == target[1]
    assert sample_normalized[2] == target[2]
    assert sample_normalized[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 1, 2, 3], dtype=np.uint16),
    ],
)
@pytest.mark.parametrize("target", [np.array([0, 1000, 2000, 3000], dtype=np.float32)])
# Test normalization and denormalization on dummy input.
def test_denormalize_superview(sample: Array[Any], target):
    sample_denormalized = denormalize_superview(sample, seed=0, config={})
    assert sample_denormalized[0] == target[0]
    assert sample_denormalized[1] == target[1]
    assert sample_denormalized[2] == target[2]
    assert sample_denormalized[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 500, 1000, 2000], dtype=np.uint16),
    ],
)
def test_processing_pipeline_superview(sample: Array[Any]):
    normalized_tanh = normalize_superview(sample, seed=0, config={})
    assert (normalized_tanh.dtype == np.float32) or (normalized_tanh.dtype == np.float64)
    normalized_sigmoid = interval_tanh_to_sigmoid(normalized_tanh, seed=0, config={})
    denormalized = denormalize_superview(normalized_sigmoid, seed=0, config={})
    assert denormalized.dtype == np.uint16
    assert (sample == denormalized).all()


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 2500, 5000, 7500], dtype=np.uint16),
    ],
)
@pytest.mark.parametrize(
    "target",
    [np.array([0.18482805, 0.3827582, 0.6172418, 0.81517195], dtype=np.float32)],
)
def test_normalize_beta(sample: Array[Any], target):
    normalzied_beta = normalize_beta(sample, seed=0, config={})
    assert normalzied_beta[0] == target[0]
    assert normalzied_beta[1] == target[1]
    assert normalzied_beta[2] == target[2]
    assert normalzied_beta[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0.18482805, 0.3827582, 0.6172418, 0.81517195], dtype=np.uint16),
    ],
)
@pytest.mark.parametrize(
    "target",
    [np.array([0, 2500, 5000, 7500], dtype=np.float32)],
)
def test_denormalize_beta(sample: Array[Any], target):
    denormalzied_beta = denormalize_beta(sample, seed=0, config={})
    assert denormalzied_beta[0] == target[0]
    assert denormalzied_beta[1] == target[1]
    assert denormalzied_beta[2] == target[2]
    assert denormalzied_beta[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 500, 1000, 2000], dtype=np.uint16),
    ],
)
def test_normalize_beta_pipeline(sample: Array[Any]):
    normalized_beta = normalize_beta(sample, seed=0, config={})
    assert normalized_beta.dtype == np.float32
    denormalized = denormalize_beta(normalized_beta, seed=0, config={})
    assert (sample == denormalized).all()


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 2500, 5000, 7500], dtype=np.uint16),
    ],
)
@pytest.mark.parametrize(
    "target",
    [np.array([-1.34164079, -0.4472136, 0.4472136, 1.34164079], dtype=np.float32)],
)
def test_normalize_zscore(sample: Array[Any], target):
    normalzied_z = normalize_zscore(sample, seed=0, config={})
    assert normalzied_z[0] == target[0]
    assert normalzied_z[1] == target[1]
    assert normalzied_z[2] == target[2]
    assert normalzied_z[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([-1.34164079, -0.4472136, 0.4472136, 1.34164079], dtype=np.uint16),
    ],
)
@pytest.mark.parametrize(
    "target",
    [np.array([0, 2500, 5000, 7500], dtype=np.float32)],
)
def test_denormalize_zscore(sample: Array[Any], target):
    denormalzied_z = denormalize_zscore(sample, seed=0, config={})
    assert denormalzied_z[0] == target[0]
    assert denormalzied_z[1] == target[1]
    assert denormalzied_z[2] == target[2]
    assert denormalzied_z[3] == target[3]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([0, 500, 1000, 2000], dtype=np.uint16),
    ],
)
def test_normalize_zscore_pipeline(sample: Array[Any]):
    normalized_beta = normalize_zscore(sample, seed=0, config={})
    assert normalized_beta.dtype == np.float32
    denormalized = denormalize_zscore(normalized_beta, seed=0, config={})
    assert (sample == denormalized).all()
