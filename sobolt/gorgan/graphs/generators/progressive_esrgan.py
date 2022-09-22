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


class EsrganProgressiveGenerator(nn.Module):
    """A generator that can learn to incrementally upsample and increase the number of
    feature layers depending on the upsample factor and init_prog_step specified. The
    main architecture is inspired from the enhanced super resolution (ESRGAN) paper
    (Wang et al., 2018). The generator wraps multiple residual in residual blocks for its
    feature learning, where each RRDB further consists of densely connected convolution
    blocks. The head and reconstructor layers are what allows us to go from an image to
    the feature space and back to the image respectively.
    """

    # Graph components
    __heads: nn.ModuleList
    __feature_layers: nn.ModuleList
    __reconstructors: nn.ModuleList

    # Optional graph components
    __attention: nn.Module
    __get_gamma_beta: nn.Module
    __residual_alin_block: nn.Module

    # Optional graph component activation
    __use_progressive: bool
    __use_attention: bool
    __use_condition: bool

    # General variables
    __prog_depth: int
    __n_rrdb: int
    __num_prog_steps: int
    __upsample_factor: int

    def __init__(
        self,
        shape_originals,
        upsample_factor: int = 4,
        use_progressive: bool = False,
        use_attention: bool = False,
        use_condition: bool = False,
        conditional_mask_indices: List[int] = [],
        init_prog_step: int = 2,
        n_rrdb: int = 23,
        block_def: List[Tuple[int, int]] = [(2, 1), (2, 1)],
        block_shapes: List[Tuple[int, int, int]] = [(48, 48, 32), (48, 20, 10)],
        **kwargs,
    ):
        """Initializer for the progressive ESRGAN-based generator.

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
        n_rrdb: int (Default 23)
            The number of RRDB for feature learning.
        block_def: List[Tuple[int, int]]
            A list of tuple to define the layers in an upsampling block.
            The index 0 of the tuple specifies the upsampling factor and index 1 the
            number of RRDB that follows the upsampling layer. Upsampling blocks can be
            stacked sequentially (specified by the list index. Default is [(2, 1), (2,
            1)], where layers are 2x upsampling - 1 RRDB - 2x upsampling - 1 RRDB.
        block_shapes: List[Tuple[int, int, int]]  (Default [(48, 48, 32), (48, 20, 10)])
            Tuples define the number of filters (in/out) for each network layers,
            where tuple indices specifies filter shape for (in_upsampling_layer,
            in_rrdb, out_rrdb).List index specifies the shapes within a block,
            where multiple list elements construct multiple upsampling blocks.
        **kwargs
            Catch-all parameter that allows passing in arguments not part of this
            generator and which is not used. This allows defining different generators
            with different important parameters.
        """
        assert upsample_factor > 0 and math.log2(upsample_factor).is_integer()
        assert isinstance(use_progressive, bool)
        assert isinstance(use_attention, bool)
        super().__init__()

        # Set Generator init variables
        in_features = shape_originals[0]
        filter_feats = block_shapes[0][0]

        # Set variables for progressive training
        self.__num_prog_steps = int(math.log2(upsample_factor))
        self.__num_prog_steps = self.__num_prog_steps if self.__num_prog_steps else 1
        self.__n_rrdb = n_rrdb
        self.__prog_depth = init_prog_step

        # Prepare conditional information for ResidualSpadeBlock
        self.__use_condition = use_condition and len(conditional_mask_indices) > 0
        num_conditional_channels = (
            len(conditional_mask_indices) if self.__use_condition else 0
        )
        mask_features = in_features + num_conditional_channels

        # Feature encoders that are resolution (progressive step) specific
        # Programmed like as a module list to be compatible with a previous
        # implementations' weights that had heads and reconstructors per progressive step.
        self.__heads = nn.ModuleList(
            [
                nn.Conv2d(in_features, filter_feats, 3, padding=1, padding_mode="reflect")
                for _ in range(1)
            ]
        )

        # Layers that learn features
        self.__feature_layers = nn.ModuleList(
            [ResidualInResidualDenseBlock(filter_feats) for _ in range(self.__n_rrdb)]
        )

        # Layers that learn features using SPADE module for conditional gan
        if self.__use_condition:
            self.__spade_feature_layers = nn.ModuleList(
                [
                    ResidualSpadeBlock(
                        filter_feats, filter_feats, num_bands_segm_mask=mask_features
                    )
                    for _ in range(1)  # Can be made configurable
                ]
            )

        # Reconstruction layer
        # See note above self.__heads regarding the ModuleList
        self.__reconstructors = nn.ModuleList(
            [
                ReconstructionLayer(
                    in_features,
                    self.__num_prog_steps,
                    use_attention,
                    block_def=block_def,
                    block_shapes=block_shapes,
                )
                for _ in range(1)
            ]
        )

        self.__use_progressive = use_progressive
        self.__upsample_factor = upsample_factor

        # Compute number of block to train depending on progressive train settings
        # If statement to prevent going beyond total number of blocks
        self.__num_train_blocks = (
            round(self.__n_rrdb / self.__num_prog_steps) * self.__prog_depth
        )
        if not self.__use_progressive or self.__num_train_blocks > self.__n_rrdb:
            self.__num_train_blocks = self.__n_rrdb

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
        ret: Dict = {}  # return dictionary

        # Normally SPADE generators use a separate segmentation mask. We, however, use
        # the input image itself as its "segmentation mask"
        segm_mask = image.clone()
        if self.__use_condition and isinstance(conditional_masks, torch.Tensor):
            segm_mask = torch.cat([segm_mask, conditional_masks], 1)

        # Progressively go through each learning phase
        # NOTE: Step variable updated by for loop used to get the right feature layers
        step = self.__prog_depth

        # Convert to feature space
        features_head = self.__heads[0](image)

        # Get features
        features = features_head.clone()
        for feature_layer in self.__feature_layers[: self.__num_train_blocks]:
            features = feature_layer(features)

        # Get SPADE-processed features - conditional information included if activated
        if self.__use_condition:
            features = self.__spade_feature_layers[0](features, segm_mask)

        # Reconstruct an image from feature space
        features, ret = self.__reconstructors[0](
            features, return_dict=ret, skip_connection=features_head
        )  # Return dictionary passed to get attention tensors

        # Register the resulting tensor to the main return dictionary
        ret["generated"] = features
        return ret

    def grow(self):
        self.__prog_depth += 1


def upsampler(in_features: int, num_prog_steps: int = 1) -> nn.Sequential:
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
    for _ in range(num_prog_steps):
        block += [
            nn.Conv2d(in_features, in_features * (2 ** 2), 1),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.ReplicationPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
        ]
    return nn.Sequential(*block)


class ReconstructionLayer(nn.Module):
    __num_prog_step: int
    __up_blocks: nn.ModuleList
    __to_image_1: nn.Conv2d
    __to_image_2: nn.Conv2d
    __to_image_3: nn.Conv2d

    # Optional variables
    __use_attention: bool
    __attention: AttentionBlock
    __get_gamma_beta: AttentionBetaGamma
    __residual_alin_block: ResidualWithAdaptiveNorm

    def __init__(
        self,
        in_features: int,
        num_prog_step: int = 1,
        use_attention: bool = False,
        block_def: List[Tuple[int, int]] = [(2, 1), (2, 1)],
        block_shapes: List[Tuple[int, int, int]] = [(48, 48, 32), (48, 20, 10)],
    ):
        """Initializes the ReconstructionLayer class.

        Parameters
        ----------
        in_features: int
            The shape of the incoming features from the processed tensor.
        num_prog_step: int (Default 1)
            The current progressive step the graph is in. The log2(upsample_factor) is
            used to determine the current step.
        use_attention: bool (Default False)
            A boolean that enables or disables the use of attention.
        block_def: List[Tuple[int, int]]
            A list of tuple to define the layers in an upsampling block.
            The index 0 of the tuple specifies the upsampling factor and index 1 the
            number of RRDB that follows the upsampling layer. Upsampling blocks can be
            stacked sequentially (specified by the list index. Default is [(2, 1), (2,
            1)], where layers are 2x upsampling - 1 RRDB - 2x upsampling - 1 RRDB.
        block_shapes: List[Tuple[int, int, int]]  (Default [(48, 48, 32), (48, 20, 10)])
            Tuples define the number of filters (in/out) for each network layers,
            where tuple indices specifies filter shape for (in_upsampling_layer,
            in_rrdb, out_rrdb).List index specifies the shapes within a block,
            where multiple list elements construct multiple upsampling blocks.
        """
        super().__init__()

        # Set Progressive training variables
        num_features: Tuple[int, int, int]
        self.__num_prog_step = num_prog_step

        # Optional component: Attention
        self.__use_attention = use_attention
        if self.__use_attention:
            self.__attention = AttentionBlock(block_shapes[0][0])
            self.__get_gamma_beta = AttentionBetaGamma(block_shapes[0][0])
            self.__residual_alin_block = ResidualWithAdaptiveNorm(block_shapes[0][0])

        # Define number of upsampling blocks (up layer + RRDB)
        __up_blocks: List = []
        for up_block, num_features in zip(block_def, block_shapes):
            __up_blocks += [
                UpsamplingBlock(
                    up_block[0],
                    up_block[1],
                    num_features[0],
                    num_features[1],
                    num_features[2],
                )
            ]
        self.__up_blocks: nn.ModuleList = nn.ModuleList(__up_blocks)

        # Define convolutions to process tensor back to image space
        self.__to_image_1 = nn.Conv2d(
            num_features[1], num_features[1], 3, padding=1, padding_mode="reflect"
        )

        self.__to_image_2 = nn.Conv2d(
            num_features[1], num_features[1], 3, padding=1, padding_mode="reflect"
        )

        self.__to_image_3 = nn.Conv2d(
            num_features[1], in_features, 3, padding=1, padding_mode="reflect"
        )

    def forward(
        self,
        features: torch.Tensor,
        return_dict: Dict[str, torch.Tensor],
        skip_connection: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Processes a tensor to its image space.

        Parameters
        ----------
        features: torch.Tensor
            A feature tensor to reconstruct into its image space.
        return_dict: Dict[str, torch.Tensor]
            The main return dictionary passed through the framework during a training run.
        skip_connection: torch.Tensor
            The default uses the head convolution to form a skip connection during the
            upsampling of the input.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
           A tuple of features that have been reconstructed into a B x 3 x H x W shape
           and the main results dictionary which now contains the output for attention
           and heatmaps.
        """
        # If attention is enabled, use it just before reconstruction
        if self.__use_attention:
            (
                features_out,
                return_dict["att_preds"],
                return_dict["att_heatmap"],
            ) = self.__attention(features)

            # Generator specific adaptive normalization
            gamma, beta = self.__get_gamma_beta(features_out)
            features = self.__residual_alin_block(features, gamma, beta)

        for up_block in self.__up_blocks:
            features, skip_connection = checkpoint(up_block, features, skip_connection)

        features = self.__to_image_1(features)
        features = checkpoint(F.relu, features)

        features = self.__to_image_2(features)
        features = checkpoint(F.relu, features)

        features = self.__to_image_3(features)
        features = checkpoint(F.relu, features)

        return features, return_dict


class UpsamplingBlock(nn.Module):
    """A flexible architecture for upsampling layers. The default option defines a
    block as an upsampling layer (defined by factor) followed by a Residual in Residual Dense
    Block (RRDB). The RRDB is optional (defined by num_rrdb).
    """

    __num_rrdb: int
    __conv_layer: nn.Conv2d
    __upsampler: nn.Sequential
    __feature_layer_up: nn.ModuleList

    def __init__(
        self,
        factor: int,
        num_rrdb: int,
        features_up: int = 48,
        features_rrdb: int = 48,
        features_rrdb_out: int = 32,
    ):
        """Initializes the UpsamplingBlock class.

        Parameters
        ----------
        factor: int
            The factor to upsample an input tensor with.
        num_rrdb: int
            The number of RRDB to apply after upsampling. Default=1.
        features_up: int
            The number of feature filters going in the upsampling layer. Default=48.
        features_rrdb: int
            The number of feature filters going in the RRDB. Default=48.
        features_rrdb_out: int
            The number of feature filters going out the RRDB block. Default=32.
        """
        super().__init__()

        # Sets UpsamplingBlock variables
        self.__num_rrdb = num_rrdb
        self.__upsampler = upsampler(in_features=features_up, num_prog_steps=factor // 2)

        self.__conv_layer = nn.Conv2d(
            features_up, features_rrdb, 3, padding=1, padding_mode="reflect"
        )

        self.__feature_layer_up = nn.ModuleList(
            [
                ResidualInResidualDenseBlock(
                    in_features=features_rrdb, out_features=features_rrdb_out
                )
                for _ in range(num_rrdb)
            ]
        )

    def forward(
        self, features: torch.Tensor, skip_connection: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes a tensor into its desired shape by specified factor and learn
        features using RRDB.

        Parameters
        ----------
        features: torch.Tensor
            The tensor to process.
        skip_connection: torch.Tensor
            The skip connection to pass in earlier learned features to a specific layer.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing tensor of learned feature maps and upsampled into the
            desired shape and skip connection to add to another layer of choice.
        """
        # Upsample tensor with skip connection
        features = self.__upsampler(features + skip_connection)

        # Convolution to get correct dimension
        features = self.__conv_layer(features)
        features = F.relu(features)

        features_skip = features.clone()

        # Process tensor with RRDB
        for feature_layer in self.__feature_layer_up[: self.__num_rrdb]:
            features = feature_layer(features)
        return features, features_skip
