from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torch.nn as nn
import pytest

from sobolt.gorgan.optim.scheduler_adapter import (
    SchedulerAdapter,
    ReduceLrOnPlateauAdapter,
    LrThresholdSchedulerAdapter,
    CosineAnnealingWarmRestartsAdapter,
)
from sobolt.gorgan.optim.lr_scheduler_threshold import LrThresholdScheduler
from sobolt.gorgan.optim.optimizer_factory import optimizer_factory


# Tests calling the interface without a scheduler
def test_scheduler_adapter():
    scheduler = SchedulerAdapter
    with pytest.raises(NotImplementedError):
        scheduler.step({}, 1, 1, 1, 1)


# Tests decrease in learning rate
def test_plateau_adapter():
    graph = nn.Conv2d(3, 4, 3)
    lr = 0.00005
    d = {"lr": lr, "betas": (0.0, 0.999)}
    optimizer = optimizer_factory(graph.parameters(), "Adam", d)
    scheduler = ReduceLROnPlateau(optimizer)
    scheduler = ReduceLrOnPlateauAdapter(scheduler)
    logs = {"D_loss": 15}

    for epoch in range(1, 2):
        for idx in range(1, 100):
            optimizer.step()
            scheduler.step(logs, epoch, idx, 10)
        for param_group in optimizer.param_groups:
            new_lr = param_group["lr"]
    assert new_lr < lr


# Tests decrease in learning rate
def test_LrThresholdScheduler_adapter():
    graph = nn.Conv2d(3, 4, 3)
    lr = 0.00005
    d = {"lr": lr, "betas": (0.0, 0.999)}
    optimizer = optimizer_factory(graph.parameters(), "Adam", d)
    scheduler = LrThresholdScheduler(optimizer)
    scheduler = LrThresholdSchedulerAdapter(scheduler)
    logs = {"D_loss": 0.5, "GP": 10000001}

    for epoch in range(1, 4):
        for idx in range(1, 10):
            optimizer.step()
            scheduler.step(logs, epoch, idx, 10)
        for param_group in optimizer.param_groups:
            new_lr = param_group["lr"]
    assert new_lr < lr


# Tests decrease in learning rate
def test_cosine_adapter():
    graph = nn.Conv2d(3, 4, 3)
    lr = 0.00005
    d = {"lr": lr, "betas": (0.0, 0.999)}
    optimizer = optimizer_factory(graph.parameters(), "Adam", d)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 1)
    scheduler = CosineAnnealingWarmRestartsAdapter(scheduler)

    for epoch in range(1, 2):
        for idx in range(1, 10):
            optimizer.step()
            scheduler.step({}, epoch, idx, 10)
        for param_group in optimizer.param_groups:
            new_lr = param_group["lr"]
    assert new_lr < lr
