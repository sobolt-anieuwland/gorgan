from typing import Dict, Any, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    HueLoss,
    TotalVariationLoss,
    LocalPhaseCoherenceLoss,
    PerceptualLoss,
    SsimLoss,
)


class CompositeLoss(nn.Module):
    """Class that calculates and combines losses, specifically which depending on the
    settings used during construction.

    It can combine losses calculated only on the generated data and losses that are
    comparisons of the generated data vs targets, functioning similar to a content loss.
    """

    @staticmethod
    def from_config(config: Dict[str, Any], num_channels: int):
        loss_weights = config.get("loss_weights", {})

        return CompositeLoss(
            num_channels,
            composite_weight=loss_weights.get("composite", 1.0),
            content=config.get("content", False),
            content_weight=loss_weights.get("content", 1.0),
            perceptual=config.get("perceptual", False),
            perceptual_weight=loss_weights.get("perceptual", 1.0),
            total_variation=config.get("total_variation", False),
            total_variation_weight=loss_weights.get("total_variation", 1.0),
            lp_coherence=config.get("lp_coherence", False),
            lp_coherence_weight=loss_weights.get("lp_coherence", 1.0),
            ssim=config.get("ssim", False),
            ssim_weight=loss_weights.get("ssim", 1.0),
            hue=config.get("hue", False),
            hue_weight=loss_weights.get("hue", 1.0),
        )

    __loss_functions: nn.ModuleDict
    __loss_weights: Dict[str, float] = {}
    __composite_weight: float = 1.0

    def __init__(
        self,
        num_channels: int,
        composite_weight: float = 1.0,
        content: bool = False,
        content_weight: float = 1.0,
        perceptual: bool = False,
        perceptual_weight: float = 1.0,
        total_variation: bool = False,
        total_variation_weight: float = 1.0,
        lp_coherence: bool = False,
        lp_coherence_weight: float = 1.0,
        ssim: bool = False,
        ssim_weight: float = 1.0,
        hue: bool = False,
        hue_weight: float = 1.0,
    ):
        """ Instantiates the class CompositeLoss"""
        super().__init__()

        # Initialize ModuleDict after super().__init__() to ensure modules are properly
        # registered with nn.Module.
        self.__loss_functions = nn.ModuleDict()
        self.__composite_weight = composite_weight

        if content and content_weight != 0.0:
            l1_func = nn.L1Loss()
            self.__loss_functions["content"] = l1_func
            self.__loss_weights["content"] = content_weight

        if perceptual and perceptual_weight != 0.0:
            perc_func = PerceptualLoss()
            self.__loss_functions["perc"] = perc_func
            self.__loss_weights["perc"] = perceptual_weight

        if ssim and ssim_weight != 0.0:
            ssim_func = SsimLoss(num_channels)
            self.__loss_functions["ssim"] = ssim_func
            self.__loss_weights["ssim"] = ssim_weight

        if hue and hue_weight != 0.0:
            hue_func = HueLoss()
            self.__loss_functions["hue"] = hue_func
            self.__loss_weights["hue"] = hue_weight

        if total_variation and total_variation_weight != 0.0:
            tva_func = SingleParamAdapter(TotalVariationLoss())
            self.__loss_functions["tva"] = tva_func
            self.__loss_weights["tva"] = total_variation_weight

        if lp_coherence and lp_coherence_weight != 0.0:
            lpc_func = SingleParamAdapter(LocalPhaseCoherenceLoss())
            self.__loss_functions["lpc"] = lpc_func
            self.__loss_weights["lpc"] = lp_coherence_weight

    def forward(
        self, generated: torch.Tensor, targets: torch.Tensor, postfix: str = ""
    ) -> Dict[str, float]:
        """Computes and backpropagates losses specified in config.

        Parameters
        ----------
        generated:
            Generated tensor of shape (BxCxHxW).
        targets:
            Target tensor of shape (BxCxHxW) to compare with the generated tensor.
        postfix: str
            Name of the graph to append to loss name.

        Returns
        -------
        Dict[str, Any]
            A dictionary with all losses under the `"losses"` key.
        """
        losses: Dict = {}
        total = 0.0

        # Calculate all losses
        for name, loss_function in self.__loss_functions.items():
            weight = self.__loss_weights.get(name, 1.0)

            loss = loss_function(generated, targets)
            loss = loss * weight * self.__composite_weight
            loss.backward(retain_graph=True)

            total = total + loss.item()
            losses[f"G_{name}{postfix}"] = loss.item()

        losses[f"G_composite{postfix}"] = total
        return losses


class SingleParamAdapter(nn.Module):
    """Decorator class to adapt loss functions only operating on a single tensor to have
    two parameters. This way loss functions like TVA can be used mixed together with loss
    functions like perceptual or content loss, without needing to know which is which.
    """

    __criterion: nn.Module

    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.__criterion = criterion

    def forward(self, generated: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        """Calculate the wrapped loss. The second parameter is ignored."""
        return self.__criterion(generated)
