from typing import Any, Dict
import torch
import torch.nn as nn

from sobolt.gorgan.nn import AttentionBlock, Interpolator


class LinearGenerator(nn.Module):
    """
    A dummy generator that consists of linear upsampling. This is
    can be used as a benchmark to compare actual runs for. Attention is
    added to also allow benchmarking of this components.
    """

    __main: nn.Module

    def __init__(
        self,
        shape_originals,
        use_attention: bool = False,
        upsample_factor: int = 4,
        **kwargs,
    ):
        """
        Initializes the LinearGenerator class.

        Parameters
        ----------
        shape_targets: torch.Tensor
            Shape of the target input - C,W,H (ex. 3,512,512).
        use_attention: bool (Default False)
            A boolean that enables or disables the use of attention.
        **kwargs
            Catch-all parameter that allows passing in arguments not part of this
            generator and which is not used. This allows defining different generators
            with different important parameters.
        """
        super(LinearGenerator, self).__init__()

        # Interpolates the size of an input into the desired shape (targets)
        shape_targets = [
            shape_originals[1] * upsample_factor,
            shape_originals[2] * upsample_factor,
        ]
        self.__main = Interpolator(shape_targets)

        # Optional component: Attention
        self.__use_attention = use_attention
        if self.__use_attention:
            self.__attention = AttentionBlock(shape_originals[0])

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
            Dictionary of the Generator decision (a tensor) on generated and target
            inputs. Depending on enabled components, the dictionary will also include
            the discriminator's decision for those components (i.e. auxiliary).
        """
        ret: Dict[str, Any] = {}  # return dictionary

        # Linear upsampling
        ret["generated"] = self.__main(inputs)

        # If attention is enabled
        if self.__use_attention:
            _, ret["att_preds"], ret["att_heatmap"] = self.__attention(ret["generated"])

        return ret
