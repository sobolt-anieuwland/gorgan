import torch
import torch.nn as nn
import pytest

from sobolt.gorgan.optim.lr_scheduler_threshold import LrThresholdScheduler

# Tests if the decrease works when threshold is reached
def test_thresholder_step_decrease_lr():
    graph = nn.Conv2d(4, 4, 3)
    lr_start = 0.001

    optim = torch.optim.Adam(graph.parameters(), lr=lr_start)
    thresholder = LrThresholdScheduler(optim, factor=0.25, patience=-1)

    # for param_group in optim.param_groups:
    #     lr_start = float(param_group["lr"])
    print("lr-start: ",lr_start)
    optim.step()
    thresholder.step(0.5, 10000001)
    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])
    assert lr_start > lr_end
    assert lr_end == 0.00025


# Tests if the decrease in lr works with two steps
def test_thresholder_step_decrease_lr_two_steps_condition():
    graph = nn.Conv2d(4, 4, 3)
    lr_start = 0.001

    optim = torch.optim.Adam(graph.parameters(), lr=lr_start)
    thresholder = LrThresholdScheduler(optim, factor=0.25, patience=0)

    optim.step()

    thresholder.step(0.5, 10000001)
    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])
    assert lr_start == lr_end

    thresholder.step(0.5, 10000001)

    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])
    assert lr_start > lr_end
    assert lr_end == 0.00025

# Tests that there is no change when the threshold isn't reached
def test_thresholder_no_lr_change():
    graph = nn.Conv2d(4, 4, 3)
    lr_start = 0.001

    optim = torch.optim.Adam(graph.parameters(), lr=lr_start)
    thresholder = LrThresholdScheduler(optim, factor=0.25)

    optim.step()
    thresholder.step(0.5, 1)
    for param_group in optim.param_groups:
        lr_end = param_group["lr"]
    assert lr_start == lr_end


# Tests if the decrease works when threshold is reached
@pytest.mark.parametrize(
    "factor, result",
    [
        (0.25, 0.00025),
        (0.75, 0.00075),
    ],
)
def test_thresholder_step_decrease_lr_var_factor(factor, result):
    graph = nn.Conv2d(4, 4, 3)

    optim = torch.optim.Adam(graph.parameters())
    thresholder = LrThresholdScheduler(optim, factor=factor, patience=-1)

    optim.step()
    thresholder.step(0.5, 10000001)
    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])

    assert lr_end == pytest.approx(result)


# Tests that a wrong factor gives a eror message
def test_thresholder_invalid_factor():
    graph = nn.Conv2d(4, 4, 3)
    optim = torch.optim.Adam(graph.parameters())
    with pytest.raises(ValueError):
        LrThresholdScheduler(optim, factor=10, patience=-1)


# The lr when it needs to go lower than min_lr works as a max
@pytest.mark.parametrize(
    "min_lr, result",
    [
        (0.00051, 0.00051),
        (0.0009, 0.0009),
    ],
)
def test_thresholder_step_min_lr_is_new_lr(min_lr, result):
    graph = nn.Conv2d(4, 4, 3)

    optim = torch.optim.Adam(graph.parameters())
    thresholder = LrThresholdScheduler(optim, min_lr=min_lr, patience=-1)

    optim.step()
    thresholder.step(0.5, 10000001)
    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])

    assert lr_end == pytest.approx(result)


# Tests if the eps works correctly, when the lr needs to be changed
@pytest.mark.parametrize(
    "eps, result",
    [
        (0.00051, 0.001),
        (0.00049, 0.0005),
    ],
)
def test_thresholder_eps(eps, result):
    graph = nn.Conv2d(4, 4, 3)

    optim = torch.optim.Adam(graph.parameters())
    thresholder = LrThresholdScheduler(optim, patience=-1, eps=eps)

    optim.step()
    thresholder.step(0.5, 10000001)
    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])
    assert lr_end == result


# Tests that the lr changes when it reaches certain thresholds
@pytest.mark.parametrize(
    "threshold, result",
    [
        (1, 0.0005),
        (3, 0.001),
    ],
)
def test_thresholder_threshold_var_reached(threshold, result):
    graph = nn.Conv2d(4, 4, 3)

    optim = torch.optim.Adam(graph.parameters())
    thresholder = LrThresholdScheduler(optim, patience=-1, threshold=threshold)

    optim.step()

    # second argument is function_to_threshold if that is higher than threshold it should change
    thresholder.step(0.5, 2)
    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])
    assert lr_end == result


# Tests if cooldown works correctly with correct ranges
@pytest.mark.parametrize(
    "cooldown, _range, result",
    [
        (3, 3, 0.0005),
        (3, 5, 0.0005),
        (3, 6, 0.00025),
    ],
)
def test_thresholder_cooldown(cooldown, _range, result):
    graph = nn.Conv2d(4, 4, 3)

    optim = torch.optim.Adam(graph.parameters())
    thresholder = LrThresholdScheduler(optim, patience=2, cooldown=cooldown, threshold=1)

    optim.step()

    # Prepare the threshold vars so there will be a lr change
    for i in range(4):
        thresholder.step(0.5, 2)

    for i in range(_range):
        thresholder.step(0.5, 2)

    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])
    assert lr_end == result


# Tests that changing the factor decay changes the lr accordingly
@pytest.mark.parametrize(
    "factor_decay, result",
    [
        (1, 0.00025),
        (0.1, 0.00000025),
    ],
)
def test_thresholder_factor_decay(factor_decay, result):
    graph = nn.Conv2d(4, 4, 3)

    optim = torch.optim.Adam(graph.parameters())
    thresholder = LrThresholdScheduler(
        optim, patience=-1, threshold=1, factor_decay=factor_decay
    )

    optim.step()

    # second argument is function_to_threshold if that is higher than threshold it should change
    thresholder.step(0.5, 2)
    thresholder.step(0.5, 2)

    for param_group in optim.param_groups:
        lr_end = float(param_group["lr"])
    assert lr_end == pytest.approx(result)
