import torch.nn as nn

from sobolt.gorgan.nn import AttentionBlock


class DcganDiscriminator(nn.Module):
    # __target_shape  # unclear, but existent member variable
    __decision: nn.Module
    __use_auxiliary: bool
    __auxiliary: nn.Module

    __use_attention: bool
    __attention: nn.Module

    def __init__(
        self,
        shape_targets,
        use_auxiliary: bool = False,
        aux_num_classes: int = -1,
        use_attention: bool = False,
        base_loss: str = "minimax",
        **kwargs,
    ):
        super().__init__()

        self.__target_shape = shape_targets

        in_shape = shape_targets

        num_channels = in_shape[0]
        ndf = 64
        prod_in_shape = in_shape[0] * in_shape[1] * in_shape[2]
        prod_convolved_shape = int(20 * (in_shape[1] / 16) * (in_shape[2] / 16))
        prod_convolved_shape = 20

        self.__main = nn.Sequential(
            nn.Conv2d(num_channels, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)),
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False)),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
        )

        # Final decision component of the discriminator
        # Modify graph depending on Wasserstein or Minimax
        self.__decision = nn.Linear(36, 1)
        if base_loss == "minimax":
            self.__decision = nn.Sequential(self.__decision, nn.Sigmoid())

        # Enable auxiliary classification if so configured
        self.__use_auxiliary = use_auxiliary
        if self.__use_auxiliary:
            if aux_num_classes < 1:  # Guard against invalid number of classes
                raise ValueError("Invalid value for num_aux_classes: {num_aux_classes}")

            # Softmax is done implicitly in the cross_entropy loss function
            self.__auxiliary = nn.Linear(36, aux_num_classes)

        # Enable attention if so configured
        self.__use_attention = use_attention
        self.__use_attention = False
        if self.__use_attention:
            self.__attention = AttentionBlock(1)

    def forward(self, inputs):
        ret = {}

        # Get the inputs' features
        o_features = self.__main(inputs)

        # If enabled, use attention to guide the discriminator
        if self.__use_attention:
            o_features, o_att, _ = self.__attention(o_features)
            ret["att_preds"] = o_att  # attention predictions

        # If enabled, do auxiliary classification to guide the discriminator
        if self.__use_auxiliary:
            ret["aux_preds"] = self.__auxiliary(o_features)  # auxiliary predictions

        # Final discriminator decision
        ret["discriminated"] = self.__decision(o_features).view(-1)
        return ret

    def grow(self):
        pass
