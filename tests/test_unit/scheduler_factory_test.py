import torch.nn as nn
import pytest

from sobolt.gorgan.optim.scheduler_adapter import (
    ReduceLrOnPlateauAdapter,
    LrThresholdSchedulerAdapter,
    CosineAnnealingWarmRestartsAdapter,
)
from sobolt.gorgan.optim import scheduler_factory
from sobolt.gorgan.optim.optimizer_factory import optimizer_factory


# Tests if all the mappings that should work, work
@pytest.mark.parametrize(
    "name, expected_output",
    [
        ("plateau", ReduceLrOnPlateauAdapter),
        ("threshold", LrThresholdSchedulerAdapter),
        ("cosine", CosineAnnealingWarmRestartsAdapter),
        ("none", None),
    ],
)
def test_scheduler_factory_mappings(name, expected_output):
    graph = nn.Conv2d(4, 4, 3)

    d = {"lr": 0.00005}

    optimizer = optimizer_factory(graph.parameters(), "Adam", d)
    scheduler = scheduler_factory(name, optimizer)
    if name == "none":
        assert scheduler == None
    else:
        assert isinstance(scheduler, expected_output)


# Tests if incorrect mappings give an eror
@pytest.mark.parametrize(
    "name",
    [
        "kdjwchddwa",
        "-/2=+",
        "",
        4,
        None,
    ],
)
def test_scheduler_factory_invalid_scheduler(name):
    graph = nn.Conv2d(4, 4, 3)

    d = {"lr": 0.00005}

    optimizer = optimizer_factory(graph.parameters(), "Adam", d)
    with pytest.raises(ValueError, match=r"Invalid LR scheduler chosen*"):
        scheduler_factory(name, optimizer)
