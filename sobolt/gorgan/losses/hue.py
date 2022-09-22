from typing import Callable, Dict

import torch
import torch.nn as nn


class HueLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(HueLoss, self).__init__()
        self.__mae = nn.L1Loss()
        self.__eps = eps

    def get_hue(self, input, device):
        img = input * 0.5 + 0.5
        hue = torch.Tensor(input.shape[0], input.shape[2], input.shape[3]).to(device)

        hue[img[:, 2] == img.max(1)[0]] = (
            4.0
            + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.__eps))[
                img[:, 2] == img.max(1)[0]
            ]
        )

        hue[img[:, 1] == img.max(1)[0]] = (
            2.0
            + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.__eps))[
                img[:, 1] == img.max(1)[0]
            ]
        )

        hue[img[:, 0] == img.max(1)[0]] = (
            0.0
            + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.__eps))[
                img[:, 0] == img.max(1)[0]
            ]
        ) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        return hue / 6

    def forward(self, generated: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = generated.device
        generated_hue = self.get_hue(generated, device)
        target_hue = self.get_hue(targets, device)
        return self.__mae(generated_hue, target_hue)
