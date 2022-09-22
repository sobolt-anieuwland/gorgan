import math
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sobolt.gorgan.nn import (
    ResidualSpadeBlock,
    ResidualInResidualDenseBlock,
    AttentionBlock,
    AttentionBetaGamma,
    ResidualWithAdaptiveNorm,
)


class SimpleEsrganGenerator(nn.Module):
    """A generator that can learn to incrementally upsample and increase the number of
    feature layers depending on the upsample factor and init_prog_step specified. The
    main architecture is inspired from the enhanced super resolution (ESRGAN) paper
    (Wang et al., 2018). The generator wraps multiple residual in residual blocks for its
    feature learning, where each RRDB further consists of densely connected convolution
    blocks. The head and reconstructor layers are what allows us to go from an image to
    the feature space and back to the image respectively.
    """

    # Graph components
    __head: nn.Conv2d
    __feature_layers: nn.ModuleList
    __up1: nn.Sequential
    __up2: nn.Sequential
    __up_conv1: nn.Conv2d
    __up_conv2: nn.Conv2d
    __hr_conv: nn.Conv2d
    __final_conv: nn.Conv2d

    def __init__(self, shape_originals, n_rrdb: int = 23, **kwargs):
        """Initializer for the progressive ESRGAN-based generator.

        Parameters
        ----------
        """
        super().__init__()

        # Set Generator init variables
        in_features = shape_originals[0]
        filter_feats = 48

        # Feature encoders that are resolution (progressive step) specific
        # Programmed like as a module list to be compatible with a previous
        # implementations' weights that had heads and reconstructors per progressive step.
        self.__head = nn.Conv2d(
            in_features, filter_feats, 3, 1, 1, padding_mode="reflect", bias=True
        )

        # Layers that learn features
        self.__feature_layers = nn.ModuleList(
            [ResidualInResidualDenseBlock(filter_feats) for _ in range(n_rrdb)]
        )

        self.__feature_conv = nn.Conv2d(
            filter_feats, filter_feats, 3, 1, 1, padding_mode="reflect", bias=True
        )

        # Upsample and reconstruct image here
        self.__up1 = upsampler(in_features=filter_feats)
        self.__up2 = upsampler(in_features=filter_feats)
        self.__up_conv1 = nn.Conv2d(
            filter_feats, filter_feats, 3, 1, 1, padding_mode="reflect", bias=True
        )
        self.__up_conv2 = nn.Conv2d(
            filter_feats, filter_feats, 3, 1, 1, padding_mode="reflect", bias=True
        )
        self.__hr_conv = nn.Conv2d(
            filter_feats, filter_feats, 3, 1, 1, padding_mode="reflect", bias=True
        )
        self.__final_conv = nn.Conv2d(
            filter_feats, 3, 3, 1, 1, padding_mode="reflect", bias=True
        )

    def forward(self, image) -> Dict[str, torch.Tensor]:
        """Push input features through graph to learn features for generating an image.

        Parameters
        ----------
        image: torch.Tensor
            The input features to be further processed by the graph.
        conditional_masks: Optional[torch.Tensor]

        Returns
        -------
        torch.Tensor
            The generated image based on the learned features.
        """
        ret: Dict = {}  # return dictionary

        # Convert to feature space
        features_head = self.__head(image)

        # Get features
        features = features_head.clone()
        for feature_layer in self.__feature_layers:
            features = feature_layer(features)

        # Reconstruct an image from feature space
        features = features + features_head
        del features_head  # Remove so more RAM for HR data

        features = self.__up1(features)
        features = F.leaky_relu(self.__up_conv1(features))  # Upsample here in future
        features = self.__up2(features)
        features = F.leaky_relu(self.__up_conv2(features))
        features = self.__final_conv(F.leaky_relu(self.__hr_conv(features)))

        # Register the resulting tensor to the main return dictionary
        ret["generated"] = features
        return ret

    def grow(self):
        pass


def upsampler(in_features: int) -> nn.Sequential:
    """This function initializes a pytorch sequential of layers for input upsampling.

    Parameters
    ----------
    in_features: int
        The shape of the incoming features from the processed tensor.
    num_prog_steps: int
        The number of progressive step to complete within a training experiment.

    Returns
    -------
    nn.Sequential
        The layers for processing a tensor of in_features with shape B x C x H x W
        into a shape of B x C x H * num_prog_step x W * num_prog_step.
    """
    block: List = []

    # Add upsampling blocks based on the specified number of progressive steps
    for _ in range(1):
        block += [
            nn.Conv2d(in_features, in_features * (2 ** 2), 1),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.ReplicationPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
        ]
    return nn.Sequential(*block)
