from collections import OrderedDict
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self, shape_originals, in_channels=4, out_channels=4, init_features=32, **kwargs
    ):
        super().__init__()
        in_channels = shape_originals[0]
        out_channels = shape_originals[0]
        features = init_features

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc42")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.__upconv4 = nn.Conv2d(features * 16, features * 8, 1)
        self.__activation4 = nn.LeakyReLU()
        self.__pixel_shuffle4 = nn.PixelShuffle(2)
        self.__conv4 = nn.Conv2d(features * 2, features * 8, 1, 1)
        self.__pad4 = nn.ReplicationPad2d((1, 0, 1, 0))
        self.__average_pool4 = nn.AvgPool2d(2, stride=1)
        self.__decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.__upconv3 = nn.Conv2d(features * 8, features * 4, 1)
        self.__activation3 = nn.LeakyReLU()
        self.__pixel_shuffle3 = nn.PixelShuffle(2)
        self.__conv3 = nn.Conv2d(features, features * 4, 1, 1)
        self.__pad3 = nn.ReplicationPad2d((1, 0, 1, 0))
        self.__average_pool3 = nn.AvgPool2d(2, stride=1)
        self.__decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.__upconv2 = nn.Conv2d(features * 4, features * 2, 1)
        self.__activation2 = nn.LeakyReLU()
        self.__pixel_shuffle2 = nn.PixelShuffle(2)
        self.__conv2 = nn.Conv2d((features * 2) // 4, features * 2, 1, 1)
        self.__pad2 = nn.ReplicationPad2d((1, 0, 1, 0))
        self.__average_pool2 = nn.AvgPool2d(2, stride=1)
        self.__decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.__upconv1 = nn.Conv2d(features * 2, features, 1)
        self.__activation1 = nn.LeakyReLU()
        self.__pixel_shuffle1 = nn.PixelShuffle(2)
        self.__conv1 = nn.Conv2d(features // 4, features, 1, 1)
        self.__pad1 = nn.ReplicationPad2d((1, 0, 1, 0))
        self.__average_pool1 = nn.AvgPool2d(2, stride=1)
        self.__decoder1 = UNet._block(features * 2, features, name="dec1")
        self.__conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, conditional_masks: Optional[torch.Tensor] = None):

        ret: Dict = {}

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec = self.__upconv4(bottleneck)
        dec = self.__activation4(dec)
        dec = self.__pixel_shuffle4(dec)
        dec = self.__conv4(dec)
        dec = self.__pad4(dec)
        dec = self.__average_pool4(dec)
        dec = torch.cat((dec, enc4), dim=1)
        dec = self.__decoder4(dec)
        dec = self.__upconv3(dec)
        dec = self.__activation3(dec)
        dec = self.__pixel_shuffle3(dec)
        dec = self.__conv3(dec)
        dec = self.__pad3(dec)
        dec = self.__average_pool3(dec)
        dec = torch.cat((dec, enc3), dim=1)
        dec = self.__decoder3(dec)
        dec = self.__upconv2(dec)
        dec = self.__activation2(dec)
        dec = self.__pixel_shuffle2(dec)
        dec = self.__conv2(dec)
        dec = self.__pad2(dec)
        dec = self.__average_pool2(dec)
        dec = torch.cat((dec, enc2), dim=1)
        dec = self.__decoder2(dec)
        dec = self.__upconv1(dec)
        dec = self.__activation1(dec)
        dec = self.__pixel_shuffle1(dec)
        dec = self.__conv1(dec)
        dec = self.__pad1(dec)
        dec = self.__average_pool1(dec)
        dec = torch.cat((dec, enc1), dim=1)
        dec = self.__decoder1(dec)

        ret["generated"] = F.relu(self.__conv(dec))
        return ret

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
