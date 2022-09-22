from typing import Union, Optional

from torch import optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    MultiStepLR,
)
from . import (
    LrThresholdScheduler,
    SchedulerAdapter,
    ReduceLrOnPlateauAdapter,
    LrThresholdSchedulerAdapter,
    CosineAnnealingWarmRestartsAdapter,
    MultiStepLrAdapter,
)


def scheduler_factory(
    scheduler_type: str, optimizer: optim.Optimizer, factor_decay: float = 1.0
) -> Optional[SchedulerAdapter]:
    """
    Initializes the learning rate scheduler depending on the technique specified in the
    config file.

    Available techniques:
    Plateau: if an objective function (default D_loss) reaches a plateau for x
    iterations, then the learning decreases by a factor of .75

    Threshold: The learning rate decreases by a factor of .75 when a threshold value is
    reached for an objective function (default: gradient penalty)

    Cosine: the learning rate decreases and increases with a warm restart according
    to a cosine function.
    iterations, then the learning decreases by a factor of .75

    Parameters
    ----------
    scheduler_type: str
        The scheduler technique we want to set for a given optimizer.
    optimizer: optim.Optimizer
        The optimizer we want to wrap in a LR schedule.
    factor_decay:
        The scalar to decay the decrease in LR over iterations, default is 1.0.

    Returns
    -------
    Optional[Union[ReduceLROnPlateau, LRThresholdScheduler, CosineAnnealingWarmRestarts]]
         The schedule we want to set for decreasing the learning rate during training.
    """
    scheduler: Union[
        ReduceLROnPlateau, LrThresholdScheduler, CosineAnnealingWarmRestarts, MultiStepLR
    ]
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.75,
            min_lr=0.0,
            patience=1,
            cooldown=5,
            verbose=True,
        )
        print("LR Scheduler: {} ".format(scheduler_type))
        return ReduceLrOnPlateauAdapter(scheduler)
    elif scheduler_type == "threshold":
        scheduler = LrThresholdScheduler(
            optimizer, factor=0.75, factor_decay=factor_decay, patience=1, cooldown=5
        )
        print("LR Scheduler: {} ".format(scheduler_type))
        return LrThresholdSchedulerAdapter(scheduler)
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, 1)
        print("LR Scheduler: {} ".format(scheduler_type))
        return CosineAnnealingWarmRestartsAdapter(scheduler)
    elif scheduler_type == "multistep":
        scheduler = MultiStepLR(
            optimizer, milestones=[50000, 100000, 200000, 300000], gamma=0.5
        )
        print("LR Scheduler: {} ".format(scheduler_type))
        return MultiStepLrAdapter(scheduler)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Invalid LR scheduler chosen: {scheduler_type}")
