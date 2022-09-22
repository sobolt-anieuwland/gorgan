from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import AttentionBetaGamma
from .residual_adalinorm import ResidualWithAdaptiveNorm


class AttentionBlock(nn.Module):
    """Attention in the generator is extracted by an auxiliary classifier.

    This MLP learns the weights of the kth feature map for the source domain.

    To be implemented within a generator (between downsampling/upsampling) and
    a discriminator (post initial convolutions).
    Example:
    if config['discriminator']['graph']['AttentionBlock']:
        self.attention = AttentionBlock(nc_current, feature_map_mult, discriminator=True)

    Feature map multiplier is different for G and D. Original paper uses for
    G = 2 ** n_downsampling_block and for D = 2 ** (n_layers_in_D_before_attention - 2).
    In other words, `feature_map_mult` is related to the amount of (blocks of) layers in
    the graphs.
    """

    # Graph components
    __gap_fc: nn.Linear
    __gmp_fc: nn.Linear
    __conv: nn.Conv2d

    # General variables
    __graph: str

    def __init__(self, in_features: int):
        """Initializes the attention block.

        Parameters
        ----------
        in_features: int
            The shape of the incoming features from the processed tensor.
        """
        super(AttentionBlock, self).__init__()
        self.__gap_fc = nn.Linear(in_features, 1, bias=False)
        self.__gmp_fc = nn.Linear(in_features, 1, bias=False)
        self.__conv = nn.Conv2d(
            in_features * 2, in_features, kernel_size=1, stride=1, bias=True
        )

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generating a class-activation-map (CAM) based attention.

        Takes feature maps, processes them through GAP and GMP. The concatenation of
        both tensors consists of attention. In the generator only, this implementation of
        attention allows for the extraction of 2 parameters from dense layers to
        normalize a residual convolutional block.

        Parameters
        ----------
        features: torch.Tensor
            Feature tensors we want to apply attention to.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The predictions from the forward pass, the attention features and the
            heatmaps to visualize where attention is being applied on a given input.
        """
        # Global Average Pool
        global_avg_pool = F.adaptive_avg_pool2d(features, [1, 1])
        global_avg_pool = global_avg_pool.view(features.shape[0], -1)
        gap_logits = self.__gap_fc(global_avg_pool)  # get features * w + b

        # get weights
        gap_weights: torch.Tensor = self.__gap_fc.weight
        gap_weights = gap_weights.unsqueeze(2).unsqueeze(3)
        global_avg_pool = features * gap_weights  # w * Encoder --> this is attention

        # Global Max Pool
        global_max_pool = F.adaptive_max_pool2d(features, 1)
        global_max_pool = global_max_pool.view(features.shape[0], -1)
        gmp_logits = self.__gmp_fc(global_max_pool)  # get features * w + b

        # get weights
        gmp_weights: torch.Tensor = self.__gmp_fc.weight
        gmp_weights = gmp_weights.unsqueeze(2).unsqueeze(3)
        global_max_pool = features * gmp_weights  # w * Encoder  --> this is attention

        # Get out features, attention features, & attention heatmaps for visualization
        attention_features = torch.cat([gap_logits, gmp_logits], 1)
        features_out = torch.cat([global_avg_pool, global_max_pool], 1)
        features_out = F.leaky_relu(self.__conv(features_out))
        attention_heatmap = torch.sum(features_out, dim=1, keepdim=True)

        return features_out, attention_features, attention_heatmap
