import torch
import torch.nn as nn


class In1DataParallel(nn.DataParallel):
    def grow(self):
        self.module.grow()
