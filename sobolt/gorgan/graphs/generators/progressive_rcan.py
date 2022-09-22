import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sobolt.gorgan.nn import ResidualChannelAttentionBlock


class ResidualChannelAttentionGenerator(nn.Module):
    """A generator that can learn to incrementally upsample and increase the number of
    feature layers depending on the upsample factor and init_prog_step specified.

    The main architecture is inspired from the Residual Channel Attention Network (RCAN)
    paper (Zhang et al. 2018). The generator wraps multiple Residual Channel Attention
    Blocksfor its feature learning, where each RCAB further consists of convolutions and
    Channel Attention Blocks. The head and reconstructor layers are what allows us to go
    from an image to the feature space and back to the image respectively.
    """

    __head: nn.Module
    __conv: nn.Module

    __use_progressive: bool
    __use_attention: bool
    __use_condition: bool
    __attention: nn.Module

    __prog_depth: int

    def __init__(
        self,
        shape_originals,
        upsample_factor: int = 4,
        use_progressive: bool = False,
        use_condition: bool = False,
        num_conditional_channels: int = 0,
        conditional_mask_indices: List[int] = [],
        init_prog_step: int = 2,
        n_blocks: int = 25,
        **kwargs,
    ):
        """
        Initializer for this generator.

        Parameters
        ----------
        shape_originals
            An object of three integers indicating the shape ([B, W, H]) of a single
            input sample, which can be accessed using numeric indices:
            `shape_originals[0]`.
        upsample_factor: int
            The factor with which to upsample the input. If the input tensor is shaped
            [B, C, W, H], this generator outputs a tensor of [B, C, uf * W, uf * H].
            The value must be a power of 2. Passing in 1 is also legal and results in
            no upsampling.
        use_progressive: bool
            Whether or not to train this generator progressively.
        use_condition: bool
            A boolean settings whether or not to use conditional masks
            for generation.
        init_prog_step: int
            The initial step in the multiple progressive steps. The total number of
            progressive steps is calculated based on `upsample_factor`.
        n_blocks: int
            Number of RCAB blocks to be stacked in the network body.
        **kwargs
            Catch-all parameter that allows passing in arguments not part of this
            generator and which is not used. This allows defining different generators
            with different important parameters.
        """
        assert upsample_factor > 0 and math.log2(upsample_factor).is_integer()
        assert isinstance(use_progressive, bool)
        super().__init__()

        self.__use_condition = use_condition and conditional_mask_indices != []
        num_conditional_channels = (
            len(conditional_mask_indices) if self.__use_condition else 0
        )

        num_prog_steps = int(math.log2(upsample_factor)) + 1
        in_features = shape_originals[0]

        # Feature encoders that are resolution (progressive step) specific
        # Programmed like as a module list to be compatible with a previous
        # implementations' weights that had heads and reconstructors per progressive step.
        self.__n_blocks = n_blocks
        self.__heads = nn.ModuleList(
            [nn.Conv2d(in_features, 16 * in_features, 3, padding=1) for _ in range(1)]
        )

        # Layers that learn features
        self.__feature_layers = nn.ModuleList(
            [
                ResidualChannelAttentionBlock(16 * in_features)
                for _ in range(1 + self.__n_blocks * num_prog_steps)
            ]
        )

        # Reconstruction layer
        # See note above self.__heads regarding the ModuleList
        self.__reconstructors = nn.ModuleList(
            [ReconstructionLayer(in_features) for _ in range(1)]
        )

        self.__prog_depth = init_prog_step

    def forward(
        self, image, conditional_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Push input features through graph to learn features for generating an image.

        Parameters
        ----------
        image: torch.Tensor
            The input features to be further processed by the graph.
        conditional_masks: Optional[torch.Tensor]
            The list containing any mask specified. Used in combination with
            conditional gan. Default is None.

        Returns
        -------
        torch.Tensor
            The generated image based on the learned features.
        """
        ret = {}  # return dictionary

        # Normally SPADE generators use a separate segmentation mask. We, however, use
        # the input image itself as its "segmentation mask"
        segm_mask = image.clone()
        if self.__use_condition and isinstance(conditional_masks, torch.Tensor):
            segm_mask = torch.cat([segm_mask, conditional_masks], 1)

        # Progressively go through each learning phase
        # NOTE: Step variable updated by for loop used to get the right feature layers
        step = 0
        image_step = image
        for step in range(1, self.__prog_depth + 1):
            image_step = self.__up(image_step)

        # Convert to feature space
        features = self.__heads[0](image_step)

        # Get features
        for feature_layer in self.__feature_layers[: 1 + self.__n_blocks * step]:
            features = feature_layer(features)

        # Reconstruct an image from feature space
        image_step = self.__reconstructors[0](features)
        ret["generated"] = image_step
        return ret

    def grow(self):
        self.__prog_depth += 1

    def __up(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class ReconstructionLayer(nn.Module):
    __to_image_1: nn.Module
    __to_image_2: nn.Module
    __to_image_3: nn.Module
    __to_image_4: nn.Module

    def __init__(self, in_features: int):
        """
        Initializes the ReconstructionLayer class.

        Parameters
        ----------
        in_features: int
            The shape of the incoming features from the processed tensor.
        """
        super().__init__()

        self.__to_image_1 = nn.Conv2d(
            16 * in_features, 8 * in_features, 3, padding=1, padding_mode="reflect"
        )

        self.__to_image_2 = nn.Conv2d(
            8 * in_features, 4 * in_features, 3, padding=1, padding_mode="reflect"
        )

        self.__to_image_3 = nn.Conv2d(
            4 * in_features, 2 * in_features, 3, padding=1, padding_mode="reflect"
        )

        self.__to_image_4 = nn.Conv2d(
            2 * in_features, in_features, 3, padding=1, padding_mode="reflect"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes a tensor to its image space.

        Parameters
        ----------
        features: torch.Tensor
        A feature tensor to reconstruct into its image space.
        feature_maps: Optional[List[torch.Tensor]]
            The list to append feature maps for visualization during test. Default is
            None.
        Returns
        -------
        torch.Tensor
        """
        features_out = F.relu(self.__to_image_1(features))

        # features_out = upsample_skip(features_out + features) #currently ignored
        features_out = F.relu(self.__to_image_2(features_out))

        features_out = self.__to_image_3(features_out)
        features_out = F.relu(features_out)

        image = F.relu(self.__to_image_4(features_out))

        return image
