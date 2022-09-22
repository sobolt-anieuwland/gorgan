import math
from typing import Optional, List, Dict, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sobolt.gorgan.nn import ResidualInResidualDenseBlock, SubpixelConvolution


class AlternativeEsrganGenerator(nn.Module):
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
    __feature_last: nn.Conv2d
    __reconstructors: nn.ModuleList
    __up: nn.Sequential
    __hr_conv: nn.Conv2d
    __last_conv: nn.Conv2d

    def __init__(
        self,
        shape_originals,
        n_rrdb: int = 16,
        upsample_factor: int = 2,
        activation: str = "none",
        **kwargs,
    ):
        """Initializer for the progressive ESRGAN-based generator.

        Parameters
        ----------
        shape_originals: List[int]
            The shape of the input data. Expected to be a list with information [C, H, W].
        n_rrdb: int = 16
            Configures how many RRDB blocks the graph's feature learning trunk has.
        upsample_factor: int = 2
            The upsampling factor to support. The parameters affects how many upsampling
            layers the graph contains.
        activation: str
            The final activation layer name, ex. tanh, sigmoid. Default none
        """
        super().__init__()

        # Set Generator init variables
        in_features = shape_originals[0]
        num_feats = 48
        conv_args: Dict[str, Union[int, bool, str]] = {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "padding_mode": "reflect",
            "bias": True,
        }

        # The input layer of the graph
        self.__head = nn.Conv2d(in_features, num_feats, **conv_args)  # type: ignore

        # Layers that learn features
        trunk = [ResidualInResidualDenseBlock(num_feats) for _ in range(n_rrdb)]
        self.__feature_layers = nn.ModuleList(trunk)
        self.__feature_last = nn.Conv2d(num_feats, num_feats, **conv_args)  # type: ignore

        # Upsample if specified
        up_blocks: List[nn.Module] = [nn.Identity()]
        num_up_blocks = int(math.log2(upsample_factor))
        if num_up_blocks > 0:
            up_blocks = [build_esrgan_up_block(num_feats) for _ in range(num_up_blocks)]
        self.__up = nn.Sequential(*up_blocks)

        # Reconstruct image
        self.__hr_conv = nn.Conv2d(num_feats, num_feats, **conv_args)  # type: ignore
        self.__last_conv = nn.Conv2d(num_feats, in_features, **conv_args)  # type: ignore

        activations = {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "none": nn.Identity()}
        self.__activation = activations[activation]

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Push input features through graph to learn features for generating an image.

        Parameters
        ----------
        image: torch.Tensor
            The image to be super resolved.

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
        features = self.__feature_last(features)

        # Connect learnt features with the residue from feautres_head
        features = features + features_head
        del features_head  # Remove so more RAM for HR data

        # Upsample and return
        features = self.__up(features)
        features = self.__last_conv(F.leaky_relu(self.__hr_conv(features)))

        # Final activation if specified
        features = self.__activation(features)

        # Register the resulting tensor to the main return dictionary
        ret["generated"] = features
        return ret

    def grow(self):
        pass


def build_esrgan_up_block(
    num_feats: int, conv_args: Optional[Dict[str, Any]] = None
) -> nn.Sequential:
    # Use default convolutional arguments if left unspecified
    if conv_args is None:
        conv_args = {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "padding_mode": "reflect",
            "bias": True,
        }

    # Return upsampling block
    layers = [
        SubpixelConvolution(num_feats),
        nn.Conv2d(num_feats, num_feats, **conv_args),
        nn.LeakyReLU(negative_slope=0.2),
    ]
    return nn.Sequential(*layers)
