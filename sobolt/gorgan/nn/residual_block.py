from typing import Optional
import torch.nn as nn


class ResidualBlock(nn.Module):
    _multiplier: float = 1.0
    _res_graph: nn.Module
    _skip_conn: nn.Module
    _activator: Optional[nn.Module]

    def __init__(
        self,
        res_graph: nn.Module,
        skip_conn: nn.Module,
        act: Optional[nn.Module] = nn.ReLU(True),
        multiplier: float = 1.0,
    ):
        """ A helper class allowing to easily implement skip connection.

            The skip principle is implemented in the forward function.

            Parameters
            ----------
            res_graph: nn.Module
                The block of layers whose residue must be reinforced  with the
                skip connection. Typically is a sequence of layers combined with
                `nn.Sequential`, but could just as well be a single
                convolutional layer.
            skip_conn: nn.Module
                The block of layers that are the skip connection
            act: nn.Module
                (Optional, `nn.ReLU` by default). The activation function to
                apply to the residue + skip connection output. Disable by
                passing in `nn.Identity` or `None`.
            multiplier: float
                (Optional, 1.0 by default). A value to scale the residue with.
        """
        super().__init__()
        self._multiplier = multiplier
        self._activator = act
        self._res_graph = res_graph
        self._skip_conn = skip_conn

    def forward(self, features):
        """ This class' forward function implements the actual principle of a
            residual block, namely feeding the input through a block of layers
            and so getting the residue. Subsequently, the original data is again
            added to this residue, before feeding both through the activation
            function.

            Returns
            -------
            torch.Tensor:
                The result of:
                `act(multiplier * res_graph(input_) + skip_conn(input_))`
        """
        skip_features = self._skip_conn(features)
        features = self._res_graph(features)
        features *= self._multiplier
        intermediate = features + skip_features

        if self._activator is not None:
            return self._activator(intermediate)
        else:
            return intermediate
