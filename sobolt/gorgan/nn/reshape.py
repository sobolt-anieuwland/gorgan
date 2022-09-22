import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, target_size):
        super(Reshape, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        return x.view(-1, self.target_size)
