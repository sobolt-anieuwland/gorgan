from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class DiscriminatorWrapper(nn.Module):
    """
    Wrapper to use in combination with the test function Integrated Gradients.
    Takes a discriminator object as an input but return a tensor object rather
    than a dictionary.
    """

    def __init__(self, d_graph: nn.Module, base_loss: str = "minimax"):
        super(DiscriminatorWrapper, self).__init__()

        self.__d_graph = d_graph
        self.__base_loss = base_loss

    def forward(self, *args, **kwargs):
        decision = self.__d_graph(*args, **kwargs)["discriminated"]
        if self.__base_loss == "wasserstein":
            decision = torch.sigmoid(decision)
        return decision
