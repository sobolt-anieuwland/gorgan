import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


from sobolt.gorgan.nn import ResidualSpadeBlock, AttentionBlock


class ProgressiveSpadeGenerator(nn.Module):
    # Graph components
    __head: nn.ModuleList
    __spade_middle_0: nn.Module
    __spade_middle_1: nn.Module
    __spade_up_1: nn.Module
    __spade_up_2: nn.Module
    __spade_up_3: nn.Module
    __spade_up_4: nn.Module
    __conv: nn.Module
    __feature_layers: nn.ModuleList

    # Optional graph component activation
    __use_progressive: bool
    __use_attention: bool
    __use_condition: bool
    __attention: nn.Module

    # General variables
    __prog_depth: int

    def __init__(
        self,
        shape_originals,
        upsample_factor: int = 4,
        use_progressive: bool = False,
        use_attention: bool = False,
        use_condition: bool = False,
        conditional_mask_indices: List[int] = [],
        init_prog_step: int = 1,
        **kwargs,
    ):
        """Initializer for this generator.

        Parameters
        ----------
        shape_originals
            An object of three integers indicating the shape ([B, W, H]) of a single
            input sample, which can be accessed using numeric indices:
            `shape_originals[0]`.
        upsample_factor: int (Default is 4)
            The factor with which to upsample the input. If the input tensor is shaped
            [B, C, W, H], this generator outputs a tensor of [B, C, uf * W, uf * H].
            The value must be a power of 2. Passing in 1 is also legal and results in
            no upsampling.
        use_progressive: bool (Default False)
            Whether or not to train this generator progressively.
        use_attention: bool (Default False)
            A boolean that enables or disables the use of attention.
        use_condition: bool (Default False)
            A boolean settings whether or not to use conditional masks
            for generation.
        conditional_mask_indices: int (Default None)
            The number of conditional masks to use for conditional gan. Value is
            specified from the config.
        init_prog_step: int (Default 1)
            The initial step in the multiple progressive steps. The total number of
            progressive steps is calculated based on `upsample_factor`.
        **kwargs
            Catch-all parameter that allows passing in arguments not part of this
            generator and which is not used. This allows defining different generators
            with different important parameters.
        """
        assert upsample_factor > 0 and math.log2(upsample_factor).is_integer()
        assert isinstance(use_progressive, bool)
        assert isinstance(use_attention, bool)
        super().__init__()

        self.__use_condition = use_condition and conditional_mask_indices != []
        num_conditional_channels = (
            len(conditional_mask_indices) if self.__use_condition else 0
        )

        num_prog_steps = int(math.log2(upsample_factor)) + 1
        in_features = shape_originals[0]
        mask_features = in_features + num_conditional_channels

        # Feature encoders that are resolution (progressive step) specific
        # Programmed like as a module list to be compatible with a previous
        # implementations' weights that had heads and reconstructors per progressive step.
        self.__heads = nn.ModuleList(
            [
                nn.Conv2d(in_features, 16 * in_features, 3, padding=1)
                for _ in range(1)  # num_prog_steps)
            ]
        )

        # Layers that learn features
        self.__feature_layers = nn.ModuleList(
            [
                ResidualSpadeBlock(
                    16 * in_features, 16 * in_features, num_bands_segm_mask=mask_features
                )
                for _ in range(1 + 2 * num_prog_steps)
            ]
        )

        # Reconstruction layer
        # See note above self.__heads regarding the ModuleList
        self.__reconstructors = nn.ModuleList(
            [ReconstructionLayer(in_features, mask_features) for _ in range(1)]
        )

        # Optional component: Attention
        self.__use_attention = use_attention
        if self.__use_attention:
            self.__attention = AttentionBlock(16 * in_features)

        self.__prog_depth = init_prog_step

    def forward(
        self, image, conditional_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Push input features through graph to learn features for generating an image.

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
        for step in range(1, self.__prog_depth + 1):
            image = self.__up(image)

        # Convert to feature space
        features = self.__heads[0](image)

        # Get features
        for feature_layer in self.__feature_layers[: 1 + 2 * step]:
            features = feature_layer(features, segm_mask)

        # If attention is enabled, use it just before reconstruction
        if self.__use_attention:
            features, ret["att_preds"], ret["att_heatmap"] = self.__attention(features)

        # Reconstruct an image from feature space
        features = self.__activate(self.__reconstructors[0](features, segm_mask))

        ret["generated"] = features
        return ret

    def grow(self):
        self.__prog_depth += 1

    def __activate(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.tanh(x)
        return F.relu(x)

    def __up(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class ReconstructionLayer(nn.Module):
    def __init__(self, in_features, mask_features):
        """Initializes the ReconstructionLayer class.

        Parameters
        ----------
        in_features: int
            The shape of the incoming features from the processed tensor.
        mask_features: int
            The sum of in_features and number of conditional masks.
        """
        super().__init__()
        self.__to_image_1 = ResidualSpadeBlock(
            16 * in_features, 8 * in_features, num_bands_segm_mask=mask_features
        )
        self.__to_image_2 = ResidualSpadeBlock(
            8 * in_features, 4 * in_features, num_bands_segm_mask=mask_features
        )
        self.__to_image_3 = ResidualSpadeBlock(
            4 * in_features, 2 * in_features, num_bands_segm_mask=mask_features
        )
        self.__to_image_4 = ResidualSpadeBlock(
            2 * in_features, in_features, num_bands_segm_mask=mask_features
        )

    def forward(self, features: torch.Tensor, segm_mask: List[Any]):
        """Processes a tensor to its image space.

        Parameters
        ----------
        features: torch.Tensor
        A feature tensor to reconstruct into its image space.
        segm_mask: List[Any]
            The segmentation mask required by SPADEGAN. The default is an input image.

        Returns
        -------
        torch.Tensor
        """
        features = self.__to_image_1(features, segm_mask)
        features = self.__to_image_2(features, segm_mask)
        features = self.__to_image_3(features, segm_mask)
        features = self.__to_image_4(features, segm_mask)
        return features
