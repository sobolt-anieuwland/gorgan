from typing import List, Dict
import torch

from .functional import middle_index


def convolution_layer_extractor(
    state_dict: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """This function extracts from the current state-dict the first, middle and last
    convolution layer. The weights will then be plotted in.

    Parameters
    ----------
    state_dict: Dict[str, Dict[str, torch.Tensor]]
        The current graph nested state-dict that contains the "layer_name" and
        corresponding
        weights for both generator and discriminator

    Returns
    -------
    Dict[str, torch.Tensor]
        A filtered layer_name: weights dictionary that can then be used by tensorboard
        for weight plotting through a histogram.
    """
    # Functions to filter the graph state dict for specific convolutional layers.
    is_conv_g = (
        lambda name: name[-11:] == "weight_orig" and "conv1" in name and "_SpadeBlock"
    )
    is_skip_g = (
        lambda name: name[-11:] == "weight_orig"
        and "skip" in name
        and "_SpadeBlock" not in name
    )
    is_conv_d = lambda name: name[-6:] == "weight" and "main" in name

    state_dict_g = state_dict["generator"]
    state_dict_d = state_dict["discriminator"]

    if len(state_dict_g) == 0 or len(state_dict_d) == 0:
        return {}

    # Select specific layer: weights for the generator
    layers_g = {
        layer_name: weights
        for layer_name, weights in state_dict_g.items()
        if is_conv_g(layer_name) or is_skip_g(layer_name)
    }
    middle_g = middle_index(layers_g)
    list_layers_g = (
        list(layers_g.keys())[0:2]
        + list(layers_g.keys())[middle_g[0] : middle_g[1]]
        + list(layers_g.keys())[-3:-1]
    )

    # Generator dict
    layers_g = {layer_name: layers_g[layer_name] for layer_name in list_layers_g}

    # Select specific layer: weights for the discriminator
    layers_d = {
        layer_name: weights
        for layer_name, weights in state_dict_d.items()
        if is_conv_d(layer_name)
    }
    middle_d = middle_index(layers_d)
    list_layers_d = (
        [list(layers_d.keys())[0]]
        + [list(layers_d.keys())[middle_d[0]]]
        + [list(layers_d.keys())[-1]]
    )

    # Discriminator dict
    layers_d = {layer_name: layers_d[layer_name] for layer_name in list_layers_d}
    layers_g.update(layers_d)

    return layers_g
