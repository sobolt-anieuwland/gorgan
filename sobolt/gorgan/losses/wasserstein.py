from typing import Callable, Dict, Any

import torch
import torch.nn as nn

from .adversarial import AdversarialDiscriminatorLoss, AdversarialGeneratorLoss


class WassersteinCriticLoss(AdversarialDiscriminatorLoss):
    """Wasserstein loss.

    ARJOVSKY e.a. (2017), Wasserstein GAN.
    GULRAJANI e.a. (2017), Improved Training of Wasserstein GANs.
    """

    __adversarial_weight: float
    __gp_weight: int
    __critic: Callable[[torch.Tensor], Dict[str, Any]]

    def __init__(
        self,
        critic: Callable[[torch.Tensor], Dict[str, Any]],
        gradient_penalty: bool = False,
        gp_weight: int = 10,
        adversarial_weight: float = 1.0,
    ):
        """Initialize the adversarial loss.

        Parameters
        ----------
        critic: Callable[[torch.Tensor], Dict[str, Any]]
            A function allowing the loss to apply the critic to the discriminated data.
            Necessary for the gradient penalty.
        gradient_penalty: bool
            Whether or not to use gradient penalty (False by default). It is very strongly
            recommended to always set this to True.
        gp_weight: int
            Coefficient of how strong the gradient penalty (default is 10).
        adversarial_weight: float
            Default is 1.0. The coefficient to multiply the calculated loss with.
        """
        super(WassersteinCriticLoss, self).__init__()
        self.__adversarial_weight = adversarial_weight

        self.__critic = critic  # type: ignore
        self.__gp_weight = gp_weight

    def forward(
        self,
        targets: torch.Tensor,
        generated: torch.Tensor,
        targets_discriminated: torch.Tensor,
        generated_discriminated: torch.Tensor,
        losses: Dict[str, float] = {},
        name: str = "",
        **kwargs,
    ) -> Dict[str, float]:
        """Calculate and backpropagate the loss.

        Parameters
        ----------
        targets: torch.Tensor
            The targets to be discriminated. Used for the gradient penalty.
        generated: torch.Tensor
            The generated data to be discriminated. Used for the gradient penalty.
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
        loss = -(torch.mean(targets_discriminated) - torch.mean(generated_discriminated))
        loss = self.__adversarial_weight * loss
        loss.backward(retain_graph=True)
        losses["D_loss"] = loss.item()

        gp = self.gradient_penalty(generated, targets) * self.__gp_weight
        gp.backward(retain_graph=True)
        losses["GP"] = gp.item()
        return losses

    def gradient_penalty(
        self, generated: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """ Gradient penalty is defined as (l2norm(partial_derivatives) -1) ** 2 """
        device = generated.device
        targets = targets.to(device)
        alpha = torch.rand(targets.size(0), 1, 1, 1, device=device)
        interpolation = alpha * targets + (1 - alpha) * generated.detach()
        interpolation = interpolation.requires_grad_(True)
        criticism = self.__critic(interpolation)["discriminated"]  # type: ignore

        gradient = torch.autograd.grad(
            outputs=criticism,
            inputs=interpolation,
            grad_outputs=torch.ones_like(criticism, device=device),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        gradient = gradient.view(gradient.size(0), -1)
        gradient_penalty = torch.sqrt(torch.sum(gradient ** 2, dim=1))

        return ((gradient_penalty - 1) ** 2).mean()

    def classify(self, discriminated: torch.Tensor) -> torch.Tensor:
        """Convert discriminator's outputs to classes.

        For Wasserstein, the inputs are mapped to be between 0 and 1 using sigmoid and then
        thresholded with 0.5, values below it becoming class 0 and above it class 1.

        Returns
        -------
        torch.Tensor
            The input tensors converted to classes 0 and 1.
        """
        discriminated = torch.sigmoid(discriminated)
        return (discriminated > 0.5).float()


class WassersteinGeneratorLoss(AdversarialGeneratorLoss):
    def __init__(self, adversarial_weight: float = 1.0):
        """Initialize the adversarial loss.

        Parameters
        ----------
        adversarial_weight: float
            Default is 1.0. The coefficient to multiply the calculated loss with.
        """
        super(WassersteinGeneratorLoss, self).__init__()
        self.__adversarial_weight = adversarial_weight

    def forward(
        self, discriminated: torch.Tensor, losses={}, name: str = "", **kwargs
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
        loss = -torch.mean(discriminated)
        loss.backward(retain_graph=True)
        losses["G_loss"] = loss.item()
        return losses
