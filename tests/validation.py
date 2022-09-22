import math
import numpy as np
import pytest

from sobolt.gorgan.validation.performance_metrics import MetricsCollector


@pytest.mark.parametrize(
    "lores",
    np.random.randn(1, 2, 3, 24, 24),
)
@pytest.mark.parametrize(
    "hires",
    np.random.randn(1, 2, 3, 48, 48),
)
def test_api(lores, hires):
    validator = MetricsCollector()
    metrics = validator(lores, hires)

    # Check if output is a dictionary
    assert isinstance(metrics, list)

    # Check if main metrics are in dict
    for sub_dict in metrics:
        assert isinstance(sub_dict, dict), "No dictionary found"
        assert (
            "fidelity" in sub_dict.keys() and "enhancement" in sub_dict.keys()
        ), "Metric dictionary is incomplete"

        # Check if metric values are floats
        assert isinstance(sub_dict["fidelity"], float)

        # Check no value are NaN or INF
        for k, v in sub_dict.items():
            assert not math.isnan(v), "NaN value found for metric {}".format(k)
            assert not math.isinf(v), "INF value found for metric {}".format(k)
