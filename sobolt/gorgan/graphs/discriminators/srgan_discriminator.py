import torch.nn as nn

from sobolt.gorgan.nn import AttentionBlock


class SrganDiscriminator(nn.Module):
    # __target_shape  # unclear, but existent member variable

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
        ndf = 16

        self.__main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
            ),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            ),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            ),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.utils.spectral_norm(nn.Conv2d(512, 1024, kernel_size=1)),
            nn.LeakyReLU(0.2),
        )

        # Final decision component of the discriminator
        # FIXME remove Linear. DCGAN's definition is to not have fully connected layers
        # Modify graph depending on Wasserstein or Minimax
        decision_layers = [nn.Conv2d(1024, 1, kernel_size=1), nn.Flatten()]

        if base_loss == "wasserstein":
            decision_layers.append(nn.Linear(1, 1))
        else:
            decision_layers.append(nn.Sigmoid())
        self.__dis_seq = nn.Sequential(*decision_layers)

        # Enable auxiliary classification if so configured
        self.__use_auxiliary = use_auxiliary
        if self.__use_auxiliary:
            if aux_num_classes < 1:  # Guard against invalid number of classes
                raise ValueError("Invalid value for num_aux_classes: {num_aux_classes}")

            # Softmax is done implicitly in the cross_entropy loss function
            self.__auxiliary = nn.Sequential(
                nn.Flatten(), nn.Linear(1024, aux_num_classes)
            )

        # Enable attention if so configured
        self.__use_attention = use_attention
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

        # If enabled, do auxiliary classification to guide the discriminator. Both the
        # auxiliary task and the final discriminator decision require flattened input.
        # o_features = o_features.view(o_features.shape[0], -1)
        if self.__use_auxiliary:
            ret["aux_preds"] = self.__auxiliary(o_features)  # auxiliary predictions

        # Final discriminator decision
        ret["discriminated"] = self.__dis_seq(o_features)  # .view(-1)
        return ret
