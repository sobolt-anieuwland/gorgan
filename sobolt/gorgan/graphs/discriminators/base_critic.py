from typing import Tuple, Dict, Any

import torch.nn as nn


class BaseCritic(nn.Module):
    def __init__(self, shape_targets, **kwargs):
        super().__init__()
        in_shape = shape_targets
        prod_in_shape = in_shape[0] * in_shape[1] * in_shape[2]

        self.__main = nn.Sequential(
            nn.Linear(prod_in_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, inputs):
        inputs_flat = inputs.view(inputs.shape[0], -1)
        o = self.__main(inputs_flat)
        return {"discriminated": o}
