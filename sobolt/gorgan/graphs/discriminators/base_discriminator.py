from typing import Tuple, Dict, Any

import torch.nn as nn


class BaseDiscriminator(nn.Module):
    def __init__(self, shape_targets, aux_gan=False, num_classes=0, **kwargs):
        super().__init__()
        self.__target_shape = shape_targets

        self.aux_gan = aux_gan

        in_shape = shape_targets

        num_channels = in_shape[0]
        prod_in_shape = in_shape[0] * in_shape[1] * in_shape[2]
        prod_convolved_shape = int(20 * (in_shape[1] / 16) * (in_shape[2] / 16))
        prod_convolved_shape = 20

        self.__main = nn.Sequential(
            nn.Conv2d(num_channels, 8, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #            nn.Conv2d(8, 12, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            #            nn.LeakyReLU(0.2, inplace=True),
            #            nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False),
            #            nn.LeakyReLU(0.2, inplace=True),
            #            nn.Conv2d(16, 20, 3, stride=2, padding=1, bias=False),
            #            nn.BatchNorm2d(20),
            #            nn.LeakyReLU(0.2, inplace=True),
            #            nn.Flatten(),
            #        )
            nn.Flatten(),
            nn.Linear(2 * 512 * 512, prod_convolved_shape),
        )

        self.__dis_seq = nn.Sequential(nn.Linear(prod_convolved_shape, 1), nn.Sigmoid())

        self.__aux_seq = nn.Sequential(
            nn.Linear(prod_convolved_shape, num_classes), nn.Softmax()
        )

    def forward(self, inputs):
        o = self.__main(inputs)
        ret = {}
        ret["discriminated"] = self.__dis_seq(o)
        if self.aux_gan:
            a = self.__aux_seq(o)
            ret["auxiliary"] = a
        return ret


## Alternative designs:
##        self.__main = nn.Sequential(
##            Reshape(2 * 512 * 512),
##            nn.Linear(2 * 512 * 512, 20),
##            nn.ReLU(),
##            nn.Linear(20, 1),
##            nn.Sigmoid(),
##        )

#        # TODO make graph dependent on config["discriminator"]
#        # FIXME automatically calculate correct sizes from initial parameters
#        # Discriminator that works for dummy dataset
#        self.__main = nn.Sequential(
#            nn.Conv2d(num_channels, 8, 3, stride=2, padding=1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Conv2d(8, 12, 3, stride=2, padding=1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Conv2d(16, 20, 3, stride=2, padding=1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Flatten(),
#            nn.Linear(327680, 1),
#            nn.Sigmoid(),
#        )

##        self.__main = nn.Sequential(
##            nn.ConvTranspose2d(num_channels, 16, 3, 1, 1, padding_mode='zeros'),
##            nn.ReLU(True),
##            nn.ConvTranspose2d(16, 32, 3, 1, 1, padding_mode='zeros'),
##            nn.ReLU(True),
##            nn.ConvTranspose2d(32, 16, 3, 1, 1, padding_mode='zeros'),
##            nn.ReLU(True),
##            nn.ConvTranspose2d(16, num_channels, 3, 1, 1, padding_mode='zeros'),
##            nn.ReLU(True),
##            nn.AvgPool2d(512),   FIXME
##            nn.Linear(num_channels, 1),
##        )
