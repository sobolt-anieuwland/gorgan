from typing import Dict, Any, Tuple

import torch.nn as nn
import torch.nn.functional as F


class SpadeBlock(nn.Module):
    __conv_base: nn.Module
    __conv_gamma: nn.ConvTranspose2d
    __conv_beta: nn.ConvTranspose2d
    __batch_norm: nn.BatchNorm2d

    def __init__(
        self,
        num_features: int,  # norm_nc
        num_bands_segm_mask: int = 3,  # probably 3 or 4, label_nc
        kernel_size: int = 3,
        num_embeddings: int = 128,
        mask_interpolation_mode: str = "bilinear",
    ):
        """ A SPADE block: normalizes using a segmentation map with learnt parameters
            after normalizing using a parameter free method (BatchNorm for now).

            Parameters
            ----------
            num_features: int
                The amount input's feature count
            num_bands_segm_mask: int
                The amount of bands the segmentation mask carries. Since we will pass in
                our own images as labels, by default the amount of bands is 3.
            kernel_size: int = 3
                The kernel size for the convolutions
            num_embeddings: int
                The hidden embedding size. Was hardcoded to 128 in the original paper.
            mask_interpolation_mode: str (Default bilinear)
                The mode to interpolate the mask to the input tensor's size with.

            Returns
            -------
            torch.Tensor
                A tensor of the spade [num_features, W, H]
        """
        super().__init__()
        padding = kernel_size // 2
        self.__mode = mask_interpolation_mode

        self.__conv_base = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    in_channels=num_bands_segm_mask,
                    out_channels=num_embeddings,
                    kernel_size=(kernel_size, kernel_size),
                    padding=padding,
                )
            ),
            nn.ReLU(),
        )
        self.__conv_gamma = nn.ConvTranspose2d(
            in_channels=num_embeddings,
            out_channels=num_features,
            kernel_size=(kernel_size, kernel_size),
            padding=padding,
        )
        self.__conv_gamma = nn.utils.spectral_norm(self.__conv_gamma)

        self.__conv_beta = nn.ConvTranspose2d(
            in_channels=num_embeddings,
            out_channels=num_features,
            kernel_size=(kernel_size, kernel_size),
            padding=padding,
        )
        self.__conv_beta = nn.utils.spectral_norm(self.__conv_beta)

        self.__batch_norm = nn.BatchNorm2d(num_features, affine=False)
        self.__num_features = num_features

    def forward(self, features, segm_mask):
        # Calculate affine properties
        # 1. Resize sem_mask to same size as features
        # 2. Put features through spade block to calculate gamma and beta
        # 3. Apply calculated affine to features

        size = features.size()[-2:]
        align = {"align_corners": False} if "linear" in self.__mode else {}
        resized_segm_mask = F.interpolate(segm_mask, size=size, mode=self.__mode, **align)
        resized_segm_mask = self.__conv_base(resized_segm_mask)
        gamma = self.__conv_gamma(resized_segm_mask)
        beta = self.__conv_beta(resized_segm_mask)

        # Apply affine to normalized features
        return self.__batch_norm(features) * (1 + gamma) + beta
