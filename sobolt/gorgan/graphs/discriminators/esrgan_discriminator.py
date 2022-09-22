from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sobolt.gorgan.nn import AttentionBlock


class EsrganDiscriminator(nn.Module):
    """
    Discriminator following the overall architecture (number of filters and feature
    extraction blocks) of the ESRGAN (enhanced super resolution) paper. This model is
    deeper (blocks) and wider (512 features) then the DCGAN discriminator.
    """

    __main_blocks: nn.Module

    __use_auxiliary: bool
    __auxiliary: nn.Module

    __use_attention: bool
    __attention: nn.Module

    def __init__(
        self,
        shape_targets: Tuple[int, int, int],
        use_auxiliary: bool = False,
        aux_num_classes: int = -1,
        use_attention: bool = False,
        base_loss: str = "minimax",
        in_features: int = 64,
        adaptive_pool: int = 20,
        **kwargs,
    ):
        """Initializes ESRGAN discriminator class.

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
        kwargs: Any
            Allows class to take arbitrary number of keyword arguments.
        """
        super().__init__()

        in_shape = shape_targets[0]
        main_blocks: List[nn.Module] = []

        # Declaring graph’s main feature extraction blocks
        # A block refers to a sequence of layers: padding, convolution, activation and
        # normalization. Features are expanded and shrank as a result of the for loop (
        # *2 and //2 subsequently)
        for _ in range(4):
            main_blocks += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_shape, in_features, 3),
                nn.LeakyReLU(),
                nn.BatchNorm2d(in_features),
            ]
            in_shape = in_features

            main_blocks += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_shape, in_features, 3, 2),
                nn.LeakyReLU(),
            ]
            in_features *= 2

        in_features //= 2
        in_shape = in_features

        main_blocks += [nn.Conv2d(in_shape, in_features, 3), nn.LeakyReLU(0.2)]

        # Apply attention if specified
        self.__use_attention = use_attention
        if self.__use_attention:
            self.__attention = AttentionBlock(in_features)

        # Last block for feature learning, adaptive pool used for flexible input size,
        # but may hinder performance for large inputs
        tail_blocks = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.AdaptiveAvgPool2d((adaptive_pool, adaptive_pool)),
        ]

        self.__main_blocks = nn.Sequential(*main_blocks)
        self.__tail_blocks = nn.Sequential(*tail_blocks)

        # Creating D’s classification/decision layer
        self.__decision = nn.Sequential(
            nn.Linear(in_features * adaptive_pool * adaptive_pool, 1)
        )

        # Final decision component of the discriminator
        # Modify graph depending on Wasserstein or Minimax
        if base_loss == "minimax":
            self.__decision = nn.Sequential(self.__decision, nn.Sigmoid())

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

        # Get correct shape for decision layer
        o_features = self.__tail_blocks(o_features)

        # If enabled, do auxiliary classification to guide the discriminator
        if self.__use_auxiliary:
            ret["aux_preds"] = self.__auxiliary(o_features)  # auxiliary predictions

        # Final discriminator decision
        o_features = o_features.view(o_features.size(0), -1)
        ret["discriminated"] = self.__decision(o_features)
        return ret

    def grow(self):
        pass
