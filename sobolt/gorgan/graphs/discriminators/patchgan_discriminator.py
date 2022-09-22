from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sobolt.gorgan.nn import AttentionBlock


class PatchganDiscriminator(nn.Module):
    """ defines a PatchGAN discriminator, adopt from CycleGAN """

    def __init__(
        self,
        shape_targets: Tuple[int, int, int],
        use_auxiliary: bool = False,
        aux_num_classes: int = -1,
        use_attention: bool = False,
        base_loss: str = "minimax",
        in_features: int = 64,
        adaptive_pool: int = 20,
        num_blocks: int = 5,
        **kwargs,
    ):
        """Initializes PatchGAN discriminator class.

        Parameters
        ----------
        shape_targets: Tuple[int, int, int]
            Shape of the target input - C,W,H (ex. 3,512,512).
        use_auxiliary: bool
            Enables a classification task (i.e. landcovers) by adding a linear layer to
            the graph, which outputs class probability.
        aux_num_classes: int
            The number of possible classes to be classified.
        use_attention: bool
            Enables the addition of a CAM-based attention module with learnable
            parameters.
        base_loss: str
            The main discriminator loss to be minimized, consists of either "minimax" or
            "wasserstein".
        in_features: int
            The number of in filters for a convolution layer. Default is 64.
        adaptive_pool: int
            Sets the shape (H x W) of a given layer. Default is 20.
        num_blocks: int
            The amount intermediate convolutional blocks. Changing this setting changes
            the discriminator's receptive field. Default is 5, because it worked well for
            our single-image-super-resolution tasks, but the original paper used 3.
        kwargs: Any
            Allows class to take arbitrary number of keyword arguments.
        """
        super().__init__()
        # Declare graphs's main feature extraction blocks
        in_shape = shape_targets[0]
        main_blocks: List[nn.Module] = []
        tail_blocks: List[nn.Module] = []

        main_blocks += [
            nn.Conv2d(in_shape, in_features, 4, 2, 1, padding_mode="replicate"),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
        ]

        feat_mult = 1
        for block in range(1, num_blocks):
            # gradually increase the number of filters
            feat_mult_prev = feat_mult
            feat_mult = min(2 ** block, 8)
            main_blocks += [
                nn.Conv2d(
                    in_features * feat_mult_prev,
                    in_features * feat_mult,
                    4,
                    2,
                    1,
                    padding_mode="replicate",
                    bias=True,
                ),
                nn.InstanceNorm2d(in_features * feat_mult),
                nn.LeakyReLU(0.2, True),
                nn.ReplicationPad2d((1, 0, 1, 0)),
                nn.AvgPool2d(2, stride=1),
            ]

        feat_mult_prev = feat_mult
        feat_mult = min(2 ** num_blocks, 8)
        main_blocks += [
            nn.Conv2d(
                in_features * feat_mult_prev,
                in_features * feat_mult,
                4,
                1,
                1,
                padding_mode="replicate",
                bias=True,
            ),
            nn.InstanceNorm2d(in_features * feat_mult),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
        ]
        # 64 * 4 -> 64 * 8
        # 64x64  -> 63x63

        # Apply attention if specified
        self.__use_attention = use_attention
        if self.__use_attention:
            self.__attention = AttentionBlock(in_features * feat_mult)

        tail_blocks += [
            nn.Conv2d(in_features * feat_mult, 1, 4, 1, 1, padding_mode="replicate"),
            nn.ReplicationPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
        ]
        # 64 * 8 -> 1
        # 63x63  -> 62x62

        self.__main_blocks = nn.Sequential(*main_blocks)
        self.__tail_blocks = nn.Sequential(*tail_blocks)

        # Final decision component of the discriminator
        # Modify graph depending on Wasserstein or Minimax
        if base_loss == "minimax":
            self.__tail_blocks = nn.Sequential(self.__tail_blocks, nn.Sigmoid())

        # Enable auxiliary classification if so configured
        self.__use_auxiliary = use_auxiliary
        if self.__use_auxiliary:
            if aux_num_classes < 1:  # Guard against invalid number of classes
                raise ValueError("Invalid value for num_aux_classes: {num_aux_classes}")

            # Softmax is done implicitly in the cross_entropy loss function
            self.__auxiliary = nn.Linear(
                in_features * adaptive_pool * adaptive_pool, aux_num_classes
            )

    def forward(self, inputs) -> Dict[str, torch.Tensor]:
        """Push input data through graph to receive decision on realness.

        Parameters
        ----------
        inputs: torch.Tensor
            Generated and Target input tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of the discriminators decision (a tensor) on generated and target
            inputs. Depending on enabled components, the dictionary will also include
            the discriminator's decision for those components (i.e. auxiliary).

        """
        ret = {}
        # Get the inputs' features
        o_features = self.__main_blocks(inputs)

        # If enabled, use attention to guide the discriminator
        if self.__use_attention:
            _, ret["att_preds"], ret["att_heatmap"] = self.__attention(o_features)

        # If enabled, do auxiliary classification to guide the discriminator
        if self.__use_auxiliary:
            ret["aux_preds"] = self.__auxiliary(o_features)  # auxiliary predictions

        # Final discriminator decision
        ret["discriminated"] = self.__tail_blocks(o_features)
        return ret

    def grow(self):
        pass
