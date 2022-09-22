import os
from collections import OrderedDict
from typing import Dict, Any

import torch
import torch.nn as nn

from sobolt.gorgan.nn import NaiveDownsampler, PhysicalDownsampler
from .discriminators import (
    BaseDiscriminator,
    BaseCritic,
    DcganDiscriminator,
    PatchganDiscriminator,
    EsrganDiscriminator,
    LinearDiscriminator,
)
from .generators import (
    SpadeSisrGenerator,
    ProgressiveSpadeGenerator,
    DcganGenerator,
    LinearGenerator,
    EsrganProgressiveGenerator,
    AlternativeEsrganGenerator,
    SimpleEsrganGenerator,
    ResidualChannelAttentionGenerator,
    UNet,
)


def graph_factory(graph_def: Dict[str, Any], quiet: bool = False, **kwargs) -> nn.Module:
    # Complain if no component type is set
    if "type" not in graph_def:
        raise ValueError("Missing component choice under key `type`")

    mappings = {
        "SpadeSisrGenerator": SpadeSisrGenerator,
        "ProgressiveSpadeGenerator": ProgressiveSpadeGenerator,
        "EsrganProgressiveGenerator": EsrganProgressiveGenerator,
        "EsrganGenerator": SimpleEsrganGenerator,
        "AlternativeEsrganGenerator": AlternativeEsrganGenerator,
        "ResidualChannelAttentionGenerator": ResidualChannelAttentionGenerator,
        "BaseDiscriminator": BaseDiscriminator,
        "BaseCritic": BaseCritic,
        "DcganGenerator": DcganGenerator,
        "DcganDiscriminator": DcganDiscriminator,
        "PatchganDiscriminator": PatchganDiscriminator,
        "EsrganDiscriminator": EsrganDiscriminator,
        "LinearDiscriminator": LinearDiscriminator,
        "LinearGenerator": LinearGenerator,
        "NaiveDownsampler": NaiveDownsampler,
        "PhysicalDownsampler": PhysicalDownsampler,
        "UNet": UNet,
    }

    # Complain if specified component doesn't exist
    choice = graph_def["type"]
    if choice not in mappings:
        raise ValueError(f"Component {choice} does not exist. Options: {mappings.keys()}")

    # Set arguments and create component
    args = graph_def.get("arguments", {})
    args.update(graph_def.get("args", {}))
    args.update(kwargs)
    g = mappings[choice](**args)
    if not quiet:
        print(g)
    initialize_weights(graph_def, g, quiet)
    torch.cuda.empty_cache()

    return g


def initialize_weights(config: Dict[str, Any], graph: nn.Module, quiet: bool):
    init = config.get("weights", "none")
    if init == "none":
        if not quiet:
            print("Not initializing weights")

        return

    if init == "random":
        if not quiet:
            print("Randomly initializing weights")

        graph.apply(randomize_weights)
        return

    if init == "xavier_uniform":
        if not quiet:
            print("Initializing weights with Xavier uniform distribution")

        graph.apply(xavier_uniform)
        return

    if init == "xavier_normal":
        if not quiet:
            print("Initializing weights with Xavier normal distribution")

        graph.apply(xavier_normal)
        return

    if init == "kaiming_uniform":
        if not quiet:
            print("Initializing weights with Kaiming uniform distribution")

        graph.apply(kaiming_uniform)
        return

    if init == "kaiming_normal":
        if not quiet:
            print("Initializing weights with Kaiming normal distribution")

        graph.apply(kaiming_normal)
        return

    if init == "orthogonal":
        print("Initializing weights as orthogonal matrices.")
        graph.apply(orthogonal)
        return

    # If none of the above cases, assume it's a file that needs be loaded (potentially
    # from the internet
    if init[:7] == "http://" or init[:8] == "https://" and len(init) > 9:
        # TODO Download file available on the internet
        # Check if file not already available in predefined dir
        # If not download weights to that dir
        # Update init to point to that file in that dir
        if not quiet:
            print(f"Downloading {init}")

        raise NotImplementedError("Downloading weights not yet implemented")

    if os.path.isfile(init):
        # uri is assumed to be a path to a pytorch state dict
        if not quiet:
            print(f"Initializing weights with {init}")

        state_dict = torch.load(init, map_location="cpu")
        if any("module" in layer for layer in list(state_dict.keys())):
            state_dict = clean_dict(state_dict)
        graph.load_state_dict(state_dict, strict=True)

    else:
        raise ValueError(f"Unexpected setting for weights: {init}")


def clean_dict(old_state_dict):
    new_state_dict = OrderedDict()

    for key, value in old_state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    return new_state_dict


def randomize_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "SubpixelConvolution":
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.05)
        nn.init.constant_(m.bias.data, 0)


def xavier_uniform(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "SubpixelConvolution":
        nn.init.xavier_uniform_(m.weight.data, gain=0.85)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.constant_(m.bias.data, 0)


def xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "SubpixelConvolution":
        nn.init.xavier_normal_(m.weight.data, gain=0.95)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.constant_(m.bias.data, 0)


def kaiming_uniform(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "SubpixelConvolution":
        nn.init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.constant_(m.bias.data, 0)


def kaiming_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "SubpixelConvolution":
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.constant_(m.bias.data, 0)


def orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "SubpixelConvolution":
        nn.init.orthogonal_(m.weight.data)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.constant_(m.bias.data, 0)
