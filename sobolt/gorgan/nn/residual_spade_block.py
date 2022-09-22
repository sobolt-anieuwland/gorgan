from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import SpadeBlock


class ResidualSpadeBlock(nn.Module):
    __spade1: SpadeBlock
    __spade2: SpadeBlock

    __conv1: nn.Module
    __conv2: nn.Module

    __skip_spade: SpadeBlock
    __skip_conv: nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bands_segm_mask: int = 3,
        mask_interpolation_mode: str = "bilinear",
    ):
        """ Two SPADE sequential blocks with a third SPADE block serving as a skip
            connection.

            Parameters
            ----------
            in_features: int
                The number of features in the input
            out_features: int
                The number of features from the output
            num_bands_segm_mask: int
                Optional. How many bands the input mask has. By default 3.
            mask_interpolation_mode: str (Default: bilinear)
                See explanation in `SpadeBlock`.
        """
        super().__init__()
        mid_features = min(in_features, out_features)

        # Create layers forming a residual block
        self.__spade1 = SpadeBlock(
            in_features,
            num_bands_segm_mask=num_bands_segm_mask,
            mask_interpolation_mode=mask_interpolation_mode,
        )
        self.__conv1 = nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1)
        self.__conv1 = nn.utils.spectral_norm(self.__conv1)

        self.__spade2 = SpadeBlock(
            mid_features,
            num_bands_segm_mask=num_bands_segm_mask,
            mask_interpolation_mode=mask_interpolation_mode,
        )
        self.__conv2 = nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1)
        self.__conv2 = nn.utils.spectral_norm(self.__conv2)

        # Create skip connection
        self.__skip_spade = SpadeBlock(
            in_features, num_bands_segm_mask=num_bands_segm_mask
        )
        self.__skip_conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.__skip_conv = nn.utils.spectral_norm(self.__skip_conv)

    def forward(self, features, segm_mask):
        """
        Pushes the given features through this residual SPADE block.

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
        features_out = self.__spade1(features, segm_mask)
        features_out = F.relu(features_out)
        features_out = self.__conv1(features_out)

        features_out = self.__spade2(features_out, segm_mask)
        features_out = F.relu(features_out)
        features_out = self.__conv2(features_out)

        skip_features = self.__skip_spade(features, segm_mask)
        skip_features = F.relu(skip_features)
        skip_features = self.__skip_conv(skip_features)
        return features_out + skip_features
