import torch.nn as nn

from sobolt.gorgan.nn import Interpolator


class DcganGenerator(nn.Module):
    __main: nn.Module

    def __init__(self, shape_originals, shape_targets, attention: bool = False, **kwargs):
        super().__init__()
        num_channels_in = shape_originals[0]
        num_channels_out = shape_targets[0]
        ngf = 64
        ndf = 64

        self.__main = nn.Sequential(  # SISR
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_channels_in, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, num_channels_out, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels_out) x 64 x 64
        )

    def forward(self, inputs):
        return {"generated": self.__main(inputs)}
