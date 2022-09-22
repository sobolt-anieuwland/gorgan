from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adversarial import AdversarialDiscriminatorLoss, AdversarialGeneratorLoss


class LeastSquaredErrorDiscriminatorLoss(AdversarialDiscriminatorLoss):
    """Least squared error loss.

    MAO e.a. (2017), Least Squares Generative Adversarial Networks.
    MAO e.a. (2017), On the Effectiveness of Least Squares Generative Adversarial Networks.
    """

    __adversarial_weight: float

    def __init__(self, adversarial_weight: float = 1.0):
        """Initialize the adversarial loss.

        Parameters
        ----------
        adversarial_weight: float
            Default is 1.0. The coefficient to multiply the calculated loss with.
        """
        super().__init__()
        self.__adversarial_weight = adversarial_weight

    def forward(
        self,
        targets_discriminated: torch.Tensor,
        generated_discriminated: torch.Tensor,
        losses: Dict[str, float] = {},
        name: str = "",
        **kwargs,
    ) -> Dict[str, float]:
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
        loss = ((targets_discriminated - 1) ** 2).mean()
        loss = loss + (generated_discriminated ** 2).mean()
        loss = loss * 0.5
        loss = self.__adversarial_weight * loss
        loss.backward(retain_graph=True)

        # Store values for reporting and return
        losses["D_loss{}".format(name)] = loss.item()
        return losses


class LeastSquaredErrorGeneratorLoss(AdversarialGeneratorLoss):
    """Least squared error loss.

    From:
    MAO e.a. (2017), Least Squares Generative Adversarial Networks.
    MAO e.a. (2017), On the Effectiveness of Least Squares Generative Adversarial Networks.
    """

    __adversarial_weight: float

    def __init__(self, adversarial_weight: float = 1.0):
        """Initialize the adversarial loss.

        Parameters
        ----------
        adversarial_weight: float
            Default is 1.0. The coefficient to multiply the calculated loss with.
        """
        super().__init__()
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
        # Store values for reporting and return
        loss = 0.5 * torch.mean((discriminated - 1) ** 2)
        loss = self.__adversarial_weight * loss
        loss.backward(retain_graph=True)

        losses["G_loss{}".format(name)] = loss.item()
        return losses
