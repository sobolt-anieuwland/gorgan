from typing import Tuple, Any, Dict
import numpy as np
import torch
import torch.nn as nn

from sobolt.gorgan.nn import AttentionBlock


class LinearDiscriminator(nn.Module):
    """
    A dummy discriminator that consists of only a linear layer and activation. This is
    can be used as a benchmark to compare actual runs for. Auxiliary and Attention are
    added also to allow benchmarking of these respective components.
    """

    __main: nn.Module
    __decision: nn.Module

    __use_auxiliary: bool
    __auxiliary: nn.Module

    __use_attention: bool
    __attention: nn.Module

    def __init__(
        self,
        shape_targets,
        use_auxiliary: bool = False,
        aux_num_classes: int = -1,
        use_attention: bool = False,
        base_loss: str = "minimax",
        **kwargs,
    ):
        """
        Initializes the LineaDiscriminator class.

        Parameters
        ----------
        shape_targets: Tuple[int, int, int]
            Shape of the target input - C,W,H (ex. 3,512,512).
        use_auxiliary: bool
            Enables a classification task (i.e. landcovers) by adding a linear layer to
            the graph, which outputs class probability.
        aux_num_classes: int
            The number of possible classes to be classified.
        use_attention: bool
            Enables the addition of a CAM-based attention module with learnable
            parameters.
        base_loss:
            The main discriminator loss to be minimized, consists of either "minimax" or
            "wasserstein".
        **kwargs
            Catch-all parameter that allows passing in arguments not part of this
            generator and which is not used. This allows defining different generators
            with different important parameters.
        """
        super(LinearDiscriminator, self).__init__()

        # A dummy discriminator that only activates the input
        self.__main = nn.Sequential(
            nn.Linear(np.prod(shape_targets), 100), nn.LeakyReLU(0.2, inplace=True)
        )

        # Final decision component of the discriminator
        # Modify graph depending on Wasserstein or Minimax
        self.__decision = nn.Linear(100, 1)
        if base_loss == "minimax":
            self.__decision = nn.Sequential(self.__decision, nn.Sigmoid())

        # Enable auxiliary classification if so configured
        self.__use_auxiliary = use_auxiliary
        if self.__use_auxiliary:
            if aux_num_classes < 1:  # Guard against invalid number of classes
                raise ValueError("Invalid value for num_aux_classes: {num_aux_classes}")

            # Softmax is done implicitly in the cross_entropy loss function
            self.__auxiliary = nn.Linear(100, aux_num_classes)

        # Optional component: Attention
        self.__use_attention = use_attention
        self.__use_attention = False
        if self.__use_attention:
            self.__attention = AttentionBlock(1)

    def forward(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        Push input data through graph to receive decision on realness.

        Parameters
        ----------
        inputs: torch.Tensor
            Generated and Target input tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of the discriminators decision (a tensor) on generated and target
            inputs. Depending on enabled components, the dictionary will also include
            the discriminator's decision for those components (i.e. auxiliary).
        """
        ret: Dict[str, Any] = {}  # return dictionary

        # Get the inputs' features
        o_features = self.__main(inputs.view(-1))

        # If enabled, use attention to guide the discriminator
        if self.__use_attention:
            o_features, o_att, _ = self.__attention(o_features)
            ret["att_preds"] = o_att  # attention predictions

        # If enabled, do auxiliary classification to guide the discriminator
        if self.__use_auxiliary:
            ret["aux_preds"] = self.__auxiliary(o_features)  # auxiliary predictions

        # Final discriminator decision
        ret["discriminated"] = self.__decision(o_features).view(-1)
        return ret

    def grow(self):
        pass
