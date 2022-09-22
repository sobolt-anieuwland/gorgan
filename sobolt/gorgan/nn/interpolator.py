import torch.nn as nn


class Interpolator(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.__out_size = out_size

    def forward(self, inputs):
        return nn.functional.interpolate(
            inputs, size=self.__out_size, mode="bilinear", align_corners=False
        )
