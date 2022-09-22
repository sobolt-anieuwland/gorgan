from typing import Dict, Any

from .gan import Gan
from .dummy_gan import DummyGan
from .base_gan import BaseGan
from .ade_gan import AdeGan
from .cycle_gan import CycleGan


def gan_factory(config: Dict[str, Any], rank: int = 0, world_size: int = 1) -> Gan:
    # if isinstance(rank, int):
    #     dist.init_process_group(
    #         "nccl", init_method="tcp://127.0.0.1:23456", rank=rank, world_size=world_size
    #     )

    gan_type = config.get("gan", "BaseGan")
    if gan_type == "BaseGan":
        return BaseGan.from_config(config, rank, world_size)
    elif gan_type == "AdeGan":
        return AdeGan.from_config(config, rank, world_size)
    elif gan_type == "CycleGan":
        return CycleGan.from_config(config, rank, world_size)
    elif gan_type == "DummyGan":
        return DummyGan.from_config(config, rank, world_size)
    else:
        raise ValueError("Unknown GAN type ", gan_type)
