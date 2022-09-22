from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adversarial import AdversarialDiscriminatorLoss, AdversarialGeneratorLoss


class MinimaxDiscriminatorLoss(AdversarialDiscriminatorLoss):
    __adversarial_weight: float

    def __init__(self, adversarial_weight: float = 1.0):
        """Initialize the adversarial loss.

        Parameters
        ----------
        adversarial_weight: float
            Default is 1.0. The coefficient to multiply the calculated loss with.
        """
        super(MinimaxDiscriminatorLoss, self).__init__()
        self.__adversarial_weight = adversarial_weight

    def forward(
        self,
        targets_discriminated: torch.Tensor,
        generated_discriminated: torch.Tensor,
        losses: Dict[str, float] = {},
        name: str = "",
        **kwargs,
    ) -> Dict[Any, Any]:
        """Calculate and backpropagate the loss.

        Parameters
        ----------
        targets_discriminated: torch.Tensor
            A discriminator's output.
        generated_discriminated: torch.Tensor
            A discriminator's output.
        losses: Dict[str, float]
            The dictionary to put the calculated loss in.
        name: str
            Default is the empty string. A postfix to include in the return dictionary.

        Returns
        -------
        Dict[str, float]
            A dictionary with the calculated loss.
        """
        # Train on target (real) images
        labels = torch.ones_like(targets_discriminated)  # teach targets are real (1)
        loss_targets = F.binary_cross_entropy(targets_discriminated, labels)
        loss_targets = self.__adversarial_weight * loss_targets
        loss_targets.backward(retain_graph=True)

        # Train on generated (fake) images
        labels = torch.zeros_like(generated_discriminated)  # generated imgs are fake (0)
        loss_generated = F.binary_cross_entropy(generated_discriminated, labels)
        loss_generated = self.__adversarial_weight * loss_generated
        loss_generated.backward(retain_graph=True)

        # Store values for reporting and return
        losses["D_loss"] = loss_targets.item() + loss_generated.item()
        losses["D_target"] = targets_discriminated.mean().item()
        losses["D_generated_d"] = generated_discriminated.mean().item()
        return losses


class MinimaxGeneratorLoss(AdversarialGeneratorLoss):
    __adversarial_weight: float

    def __init__(self, adversarial_weight: float = 1.0):
        """Initialize the adversarial loss.

        Parameters
        ----------
        adversarial_weight: float
            Default is 1.0. The coefficient to multiply the calculated loss with.
        """
        super(MinimaxGeneratorLoss, self).__init__()
        self.__adversarial_weight = adversarial_weight

    def forward(
        self,
        discriminated: torch.Tensor,
        losses: Dict[str, float] = {},
        name: str = "",
        **kwargs,
    ) -> Dict[str, float]:
        """Calculate and backpropagate the loss.

        Parameters
        ----------
        discriminated: torch.Tensor
            A discriminator's output.
        losses: Dict[str, float]
            The dictionary to put the calculated loss in.
        name: str
            Default is the empty string. A postfix to include in the return dictionary.

        Returns
        -------
        Dict[str, float]
            A dictionary with the calculated loss.
        """
        # Calculate loss and backpropagate. Keep graph, because other losses might use it.
        # Pretend the generated images are real (1) by comparing to ones.
        loss = F.binary_cross_entropy(discriminated, torch.ones_like(discriminated))
        loss = self.__adversarial_weight * loss
        loss.backward(retain_graph=True)

        # Store values for reporting and return
        losses["G_loss"] = loss.item()
        losses["D_generated_g"] = discriminated.mean().item()
        return losses
