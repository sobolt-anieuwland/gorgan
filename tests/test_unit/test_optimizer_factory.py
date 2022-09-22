import torch.nn as nn
import pytest

from sobolt.gorgan.optim.optimizer_factory import optimizer_factory

# Checks initialisation
def test_optimizer_factory():

    graph = nn.Conv2d(4, 4, 3)

    d = {"lr": 0.00005, "betas": (0.0, 0.999)}

    optimizer = optimizer_factory(graph.parameters(), "Adam", d)
    for param in optimizer.param_groups:
        learning_rate = param["lr"]
        beta_check = param["betas"]
        grad = param["amsgrad"]
        eps = param["eps"]
        weight_decay = param["weight_decay"]

    assert learning_rate == 0.00005
    assert beta_check == (0.0, 0.999)
    assert grad == False
    assert eps == 0.00000001
    assert weight_decay == 0


# This tests if all the mappings that should work, work
@pytest.mark.parametrize(
    "name",
    [
        "Adadelta",
        "Adagrad",
        "Adam",
        "Adamax",
        "AdamW",
        "ASGD",
        "LBFGS",
        "RMSprop",
        "Rprop",
        "SGD",
        # "SparseAdam", Not included because it is never used
    ],
)
def test_optimizer_factory_mappings(name):
    graph = nn.Conv2d(4, 4, 3)

    d = {"lr": 0.00005}
    print(graph.parameters())
    optimizer = optimizer_factory(graph.parameters(), name=name, kwargs=d)
    assert optimizer


# This test is for when mappings that don't exist are entered
@pytest.mark.parametrize(
    "name",
    [
        "1",
        "ajdjwkjch",
        "--///+@",
    ],
)
def test_optimizer_factory_wrong_mappings(name):
    graph = nn.Conv2d(4, 4, 3)

    d = {"lr": 0.00005}
    print(graph.parameters())
    with pytest.raises(KeyError):
        optimizer_factory(graph.parameters(), name=name, kwargs=d)
