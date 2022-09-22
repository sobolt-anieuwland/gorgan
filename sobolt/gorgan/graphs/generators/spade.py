from typing import Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sobolt.gorgan.nn import ResidualSpadeBlock, AttentionBlock


class SpadeSisrGenerator(nn.Module):
    __head: nn.Module
    __spade_middle_0: nn.Module
    __spade_middle_1: nn.Module
    __spade_up_1: nn.Module
    __spade_up_2: nn.Module
    __spade_up_3: nn.Module
    __spade_up_4: nn.Module
    __conv: nn.Module

    __upsample_factor: int

    __use_condition: bool
    __use_attention: bool
    __attention: nn.Module

    def __init__(
        self,
        shape_originals,
        upsample_factor: int = 1,
        use_attention: bool = False,
        use_condition: bool = False,
        num_conditional_channels: int = 0,
        conditional_mask_indices: List[int] = [],
        **kwargs,
    ):
        super().__init__()
        self.__upsample_factor = upsample_factor
        self.__upsample_steps = int(math.log2(upsample_factor))
        self.__use_condition = use_condition and conditional_mask_indices != []
        num_conditional_channels = (
            len(conditional_mask_indices) if self.__use_condition else 0
        )

        in_features = shape_originals[0]
        mask_features = in_features + num_conditional_channels

        self.__head = nn.Conv2d(in_features, 16 * in_features, 3, padding=1)

        # Encoder
        self.__spade_middle_0 = ResidualSpadeBlock(
            16 * in_features,
            16 * in_features,
            num_bands_segm_mask=mask_features,
            mask_interpolation_mode="nearest",
        )
        self.__spade_middle_1 = ResidualSpadeBlock(
            16 * in_features,
            16 * in_features,
            num_bands_segm_mask=mask_features,
            mask_interpolation_mode="nearest",
        )

        # Decoder
        self.__spade_up_1 = ResidualSpadeBlock(
            16 * in_features,
            8 * in_features,
            num_bands_segm_mask=mask_features,
            mask_interpolation_mode="nearest",
        )
        self.__spade_up_2 = ResidualSpadeBlock(
            8 * in_features,
            4 * in_features,
            num_bands_segm_mask=mask_features,
            mask_interpolation_mode="nearest",
        )
        self.__spade_up_3 = ResidualSpadeBlock(
            4 * in_features,
            2 * in_features,
            num_bands_segm_mask=mask_features,
            mask_interpolation_mode="nearest",
        )
        self.__spade_up_4 = ResidualSpadeBlock(
            2 * in_features,
            in_features,
            num_bands_segm_mask=mask_features,
            mask_interpolation_mode="nearest",
        )

        # Final convolution before returning
        self.__conv = nn.Conv2d(in_features, in_features, kernel_size=(3, 3), padding=1)
        self.__conv = nn.utils.spectral_norm(self.__conv)

        self.__use_attention = use_attention
        if self.__use_attention:
            self.__attention = AttentionBlock(16 * in_features)

    def forward(
        self, original: torch.Tensor, conditional_masks: Optional[torch.Tensor] = None
    ):
        ret = {}  # return dictionary

        # Up sample as often as specified then learn to deblur
        for up_step in range(self.__upsample_steps):
            original = self.__up(original)

        # Normally SPADE generators use a separate segmentation mask as conditional
        # information. We use the original image, plus optional extra conditional
        # masks.
        segm_mask = original.clone()
        if self.__use_condition and isinstance(conditional_masks, torch.Tensor):
            segm_mask = torch.cat([segm_mask, conditional_masks], 1)

        inputs = self.__head(original)
        inputs = self.__spade_middle_0(inputs, segm_mask)
        inputs = self.__spade_middle_1(inputs, segm_mask)

        if self.__use_attention:  # If enabled, use attention to guide the generator
            inputs, ret["att_preds"], ret["att_heatmap"] = self.__attention(inputs)

        inputs = self.__spade_up_1(inputs, segm_mask)
        inputs = self.__spade_up_2(inputs, segm_mask)
        inputs = self.__spade_up_3(inputs, segm_mask)
        inputs = self.__spade_up_4(inputs, segm_mask)

        inputs = F.leaky_relu(inputs, 2e-1)
        inputs = self.__conv(inputs)

        ret["generated"] = torch.tanh(inputs)
        return ret

    def __up(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
