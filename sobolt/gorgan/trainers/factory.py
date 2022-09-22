from typing import Dict, Any

from . import BaseTrainer


def trainer_factory(
    config: Dict[str, Any], device_rank: int, num_devices: int, quiet: bool
):
    return BaseTrainer.from_config(
        config, device_rank=device_rank, num_devices=num_devices, quiet=quiet
    )
