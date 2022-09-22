import torch
import torch.nn as nn


class RgbLoss(nn.Module):
    """
    RGB loss first computes the channel-wise (RGB) frequency of each pixel value for
    generated and original images. Second, the loss is computed between generated and
    original frequency distributions, which reflects the extent the two
    color distributions differ. The loss function to compute the difference is the mean squared error.

    """

    def __init__(self):
        pass
